#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms, datasets
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# ---------- utils ----------
def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def softmax1(logits: torch.Tensor) -> np.ndarray:
    return torch.softmax(logits, dim=1)[:,1].detach().cpu().numpy()

# ---------- TEXT: gradient saliency ----------
def text_saliency_examples(csv_path, model_dir, out_html,
                           text_col="text", max_len=160, device="cpu", k=6):
    tok = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir).to(device).eval()

    df = pd.read_csv(csv_path)
    if text_col not in df.columns:
        raise SystemExit(f"[error] '{text_col}' column missing in {csv_path}")

    # compute probabilities for all
    probs, enc_cache = [], []
    with torch.no_grad():
        for i in range(0, len(df), 32):
            chunk = df.iloc[i:i+32]
            enc = tok(chunk[text_col].astype(str).tolist(),
                      truncation=True, padding=True, max_length=max_len, return_tensors="pt")
            enc_cache.append((i, enc))  # keep offsets
            logits = model(**{k:v.to(device) for k,v in enc.items()}).logits
            probs.extend(softmax1(logits))
    probs = np.array(probs)

    # choose examples: top-3 confident disaster + top-3 confident non_disaster
    idx_dis = np.argsort(-probs)[: math.ceil(k/2)]
    idx_non = np.argsort(probs)[: math.floor(k/2)]
    chosen = list(idx_dis) + list(idx_non)
    chosen = sorted(set(chosen))[:k]

    # Build saliency for chosen
    html_parts = [
        "<html><head><meta charset='utf-8'><style>body{font-family:Arial,sans-serif} .ex{margin:16px 0} .p{color:#555} .tok{padding:2px 3px;margin:1px;border-radius:3px;display:inline-block}</style></head><body>",
        "<h2>Text Saliency (DistilBERT)</h2>",
        "<p>Background intensity ≈ token importance for the predicted class (via gradient on input embeddings).<br/>Red = disaster importance; Blue = non-disaster importance.</p>"
    ]

    for idx in chosen:
        text = str(df.iloc[idx][text_col])
        enc = tok([text], truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)

        # embed-level gradients
        emb_layer = model.get_input_embeddings()
        emb = emb_layer(input_ids)               # (1, L, H)
        emb.retain_grad()
        emb.requires_grad_(True)

        out = model(inputs_embeds=emb, attention_mask=attn)
        logits = out.logits                       # (1,2)
        p = float(torch.softmax(logits, dim=1)[0,1].detach().cpu().item())
        pred = 1 if p >= 0.5 else 0

        # grad w.r.t predicted logit
        model.zero_grad(set_to_none=True)
        target_logit = logits[0, pred]
        target_logit.backward()

        grads = emb.grad.detach().cpu().numpy()[0]      # (L,H)
        sal = np.linalg.norm(grads, axis=1)             # token saliency
        sal = sal / (sal.max() + 1e-8)

        toks = tok.convert_ids_to_tokens(input_ids[0].cpu().tolist())
        # build colored HTML for visible tokens (skip [CLS]=[CLS]/[SEP]/[PAD])
        vis = []
        for t, s, m in zip(toks, sal, attn[0].cpu().tolist()):
            if m == 0: break
            if t in ("[CLS]","[SEP]"): continue
            # styling
            if pred == 1:
                # disaster → red scale
                bg = f"rgba(214,39,40,{0.15 + 0.75*float(s)})"
            else:
                # non_disaster → blue scale
                bg = f"rgba(31,119,180,{0.15 + 0.75*float(s)})"
            t_clean = t.replace("##","")
            vis.append(f"<span class='tok' style='background:{bg}'>{t_clean}</span>")

        label = "disaster" if pred==1 else "non_disaster"
        html_parts.append(f"<div class='ex'><div class='p'><b>Pred:</b> {label} &nbsp; <b>p(disaster)</b>= {p:.3f}</div>")
        html_parts.append("<div style='line-height:1.8'>" + " ".join(vis) + "</div></div>")

    html_parts.append("</body></html>")
    out_html = Path(out_html); out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text("\n".join(html_parts), encoding="utf-8")
    print(f"[saved] {out_html}")

# ---------- IMAGE: Grad-CAM ----------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.h1 = target_layer.register_forward_hook(self._forward_hook)
        self.h2 = target_layer.register_backward_hook(self._backward_hook)  # for torchvision<=0.15

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove(self):
        self.h1.remove(); self.h2.remove()

    def __call__(self, scores, idx_class=1):
        # scores: output logits (B,2); we expect backward already set up
        B = scores.size(0)
        cams = []
        for b in range(B):
            self.model.zero_grad(set_to_none=True)
            scores[b, idx_class].backward(retain_graph=True)

            A = self.activations[b]   # (C,H,W)
            G = self.gradients[b]     # (C,H,W)
            weights = torch.mean(G, dim=(1,2))   # (C,)
            cam = torch.relu(torch.sum(weights[:,None,None] * A, dim=0))  # (H,W)
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
            cams.append(cam.cpu().numpy())
        return cams

def gradcam_on_images(image_root, ckpt_path, out_dir, img_size=224, device="cpu", n=6):
    tf = transforms.Compose([transforms.Resize((img_size,img_size)), transforms.ToTensor()])
    ds = datasets.ImageFolder(Path(image_root), transform=tf)
    dl = torch.utils.data.DataLoader(ds, batch_size=8)

    model = models.resnet18(weights=None)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, 2)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    # target last conv layer
    target_layer = model.layer4[-1].conv2
    cam_engine = GradCAM(model, target_layer)

    # compute probabilities for selection
    all_probs, all_paths, all_ys = [], [], []
    with torch.no_grad():
        for x,y in dl:
            x = x.to(device)
            logits = model(x)
            all_probs.extend(softmax1(logits))
    all_paths = [p for p,_ in ds.samples]
    all_ys = [y for _,y in ds.samples]

    # pick top-k confident disasters + top-k confident non
    probs = np.array(all_probs)
    idx_dis = np.argsort(-probs)[: math.ceil(n/2)]
    idx_non = np.argsort(probs)[: math.floor(n/2)]
    chosen = list(idx_dis) + list(idx_non)
    chosen = sorted(set(chosen))[:n]

    # run Grad-CAM and save overlays
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("jet")

    # helper to overlay
    def overlay_heatmap(img_arr, cam):
        cam_up = cam
        # resize cam to image
        cam_up = Image.fromarray((cam_up*255).astype(np.uint8)).resize((img_arr.shape[1], img_arr.shape[0]))
        cam_up = np.array(cam_up)/255.0
        heat = cmap(cam_up)[:,:,:3]  # RGB
        overlay = (0.45*heat + 0.55*(img_arr/255.0))
        overlay = np.clip(overlay, 0, 1)
        return (overlay*255).astype(np.uint8)

    # second pass: get cams
    model.zero_grad(set_to_none=True)
    for idx in chosen:
        path = all_paths[idx]
        y_true = all_ys[idx]
        img = Image.open(path).convert("RGB").resize((img_size,img_size))
        x = transforms.ToTensor()(img).unsqueeze(0).to(device)
        x.requires_grad_(True)
        logits = model(x)
        cams = cam_engine(logits, idx_class=1)   # class 1 = disaster
        cam = cams[0]

        base = np.array(img)
        overlay = overlay_heatmap(base, cam)

        # stack side-by-side
        fig, ax = plt.subplots(1,2, figsize=(8,4), dpi=140)
        ax[0].imshow(base); ax[0].axis("off"); ax[0].set_title("Image")
        ax[1].imshow(overlay); ax[1].axis("off"); ax[1].set_title("Grad-CAM (disaster)")
        fig.tight_layout()

        out_png = out_dir / f"gradcam_{Path(path).stem}.png"
        fig.savefig(out_png); plt.close(fig)
        print(f"[saved] {out_png}")

    cam_engine.remove()

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-csv", type=str, default="text_data/test.csv")
    ap.add_argument("--text-model-dir", type=str, default="artifacts/distilbert_ft")
    ap.add_argument("--text-maxlen", type=int, default=160)
    ap.add_argument("--text-k", type=int, default=6)
    ap.add_argument("--image-root", type=str, default="image_data/test")
    ap.add_argument("--image-ckpt", type=str, default="artifacts/resnet18_ft/resnet18.pt")
    ap.add_argument("--image-n", type=int, default=6)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--out-dir", type=str, default="docs/interpretability")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # Text saliency
    text_html = out / "text_saliency.html"
    try:
        text_saliency_examples(args.text_csv, args.text_model_dir, text_html,
                               text_col="text", max_len=args.text_maxlen, device=device, k=args.text_k)
    except Exception as e:
        print(f"[warn] text saliency skipped: {e}")

    # Image Grad-CAM
    try:
        gradcam_on_images(args.image_root, args.image_ckpt, out/"gradcam_examples",
                          img_size=args.img_size, device=device, n=args.image_n)
    except Exception as e:
        print(f"[warn] image grad-cam skipped: {e}")

    print(f"[done] outputs in: {out}")

if __name__ == "__main__":
    main()
