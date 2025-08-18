#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# ---------------- utils ----------------
def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def softmax1(logits):
    return torch.softmax(logits, dim=1)[:,1].cpu().numpy()

def tune_threshold(y, p, lo=0.30, hi=0.70, steps=41):
    best = (0.5, -1.0, 0, 0, 0, 0)  # thr, f1, tp, fp, fn, tn
    thr_vals = np.linspace(lo, hi, steps)
    for thr in thr_vals:
        pred = (p >= thr).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        if f1 > best[1]:
            best = (float(thr), float(f1), int(tp), int(fp), int(fn), int(tn))
    return {"thr": best[0], "f1": best[1], "tp": best[2], "fp": best[3], "fn": best[4], "tn": best[5]}

def summarize_errors(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {"tp":int(tp),"fp":int(fp),"fn":int(fn),"tn":int(tn),
            "precision_recall_f1": tuple(precision_recall_fscore_support(y, pred, average="binary", zero_division=0)[:3])}

# ------------- TEXT (DistilBERT) -------------
def text_probs(split_csv, model_dir, batch_size=32, max_len=160, device="cpu"):
    tok = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    df = pd.read_csv(split_csv)
    labmap = {"non_disaster":0,"disaster":1}
    y = df["label"].map(labmap).values.astype(int)

    probs = []
    with torch.no_grad():
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            enc = tok(batch["text"].tolist(), truncation=True, padding=True, max_length=max_len, return_tensors="pt")
            enc = {k:v.to(device) for k,v in enc.items()}
            logits = model(**enc).logits
            probs.append(softmax1(logits))
    return df, np.concatenate(probs), y

def print_text_examples(df, y, p, thr, k=5):
    pred = (p >= thr).astype(int)
    # FP: predicted 1 but true 0
    fp_idx = np.where((pred==1) & (y==0))[0][:k]
    fn_idx = np.where((pred==0) & (y==1))[0][:k]
    print("\n=== TEXT: Top False Positives (pred=disaster, true=non_disaster) ===")
    for i in fp_idx:
        print(f"[p={p[i]:.2f}] {df.iloc[i]['text'][:180]}")
    print("\n=== TEXT: Top False Negatives (pred=non_disaster, true=disaster) ===")
    for i in fn_idx:
        print(f"[p={p[i]:.2f}] {df.iloc[i]['text'][:180]}")

# ------------- IMAGE (ResNet18) -------------
def image_probs(split_dir, ckpt_path, batch_size=32, img_size=224, device="cpu"):
    tf = transforms.Compose([transforms.Resize((img_size,img_size)), transforms.ToTensor()])
    ds = datasets.ImageFolder(split_dir, transform=tf)
    dl = DataLoader(ds, batch_size=batch_size)
    model = models.resnet18(weights=None)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, 2)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model = model.to(device); model.eval()
    probs, ys, paths = [], [], []
    with torch.no_grad():
        for x,y in dl:
            x = x.to(device)
            logits = model(x)
            probs.append(softmax1(logits))
            ys.extend(y.numpy().tolist())
    # map indices to file paths
    paths = [p for p,_ in ds.samples]
    return np.concatenate(probs), np.array(ys, dtype=int), paths, ds.classes

def save_image_error_grid(paths, y, p, thr, cls_names, out_png, title, k=8):
    pred = (p >= thr).astype(int)
    # pick FN or FP depending on title keyword
    if "False Positives" in title:
        idx = np.where((pred==1) & (y==0))[0][:k]
    else:
        idx = np.where((pred==0) & (y==1))[0][:k]
    if len(idx) == 0:
        print(f"[warn] No examples for {title}")
        return
    cols = 4
    rows = int(np.ceil(len(idx)/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.2*rows), dpi=140)
    axes = np.array(axes).reshape(-1)
    for ax in axes: ax.axis("off")
    for j,i in enumerate(idx):
        img = plt.imread(paths[i])
        axes[j].imshow(img)
        axes[j].set_title(f"p={p[i]:.2f}  true={cls_names[y[i]]}")
        axes[j].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    print(f"[saved] {out_png}")

# ------------- FUSED (from saved arrays) -------------
def fused_arrays(fusion_dir):
    d = Path(fusion_dir)
    y = np.load(d/"_labels_test.npy")
    p_txt  = np.load(d/"_probs_text_test.npy")
    p_img  = np.load(d/"_probs_image_test.npy")
    p_fuse = np.load(d/"_probs_fused_test.npy")
    alpha_thr = json.loads((d/"alpha.json").read_text())
    return y, p_txt, p_img, p_fuse, alpha_thr

# ------------- main -------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-data-dir", type=str, default="text_data")
    ap.add_argument("--distil-dir", type=str, default="artifacts/distilbert")
    ap.add_argument("--image-data-dir", type=str, default="image_data")
    ap.add_argument("--resnet-ckpt", type=str, default="artifacts/resnet18/resnet18.pt")
    ap.add_argument("--fusion-dir", type=str, default="artifacts/fusion")
    ap.add_argument("--out-dir", type=str, default="docs/errors")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--max-len", type=int, default=160)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # ---------- TEXT ----------
    df_test, p_text_test, y_text_test = text_probs(Path(args.text_data_dir)/"test.csv",
                                                   args.distil_dir, args.batch_size, args.max_len, device)
    df_val, p_text_val, y_text_val = text_probs(Path(args.text_data_dir)/"val.csv",
                                                args.distil_dir, args.batch_size, args.max_len, device)
    best_text = tune_threshold(y_text_val, p_text_val)
    m_text = summarize_errors(y_text_test, p_text_test, thr=best_text["thr"])
    print("\n[TEXT] tuned on val:", best_text)
    print("[TEXT] test summary:", m_text)
    print_text_examples(df_test, y_text_test, p_text_test, best_text["thr"], k=6)

    # ---------- IMAGE ----------
    p_img_test, y_img_test, img_paths_test, cls_names = image_probs(Path(args.image_data_dir)/"test",
                                                                    args.resnet_ckpt, args.batch_size, args.img_size, device)
    p_img_val,  y_img_val,  img_paths_val,  _ = image_probs(Path(args.image_data_dir)/"val",
                                                             args.resnet_ckpt, args.batch_size, args.img_size, device)
    best_img = tune_threshold(y_img_val, p_img_val)
    m_img = summarize_errors(y_img_test, p_img_test, thr=best_img["thr"])
    print("\n[IMAGE] tuned on val:", best_img)
    print("[IMAGE] test summary:", m_img)

    # save grids
    save_image_error_grid(img_paths_test, y_img_test, p_img_test, best_img["thr"],
                          cls_names, out/"image_fp_grid.png", "IMAGE – False Positives (pred=disaster, true=non)")
    save_image_error_grid(img_paths_test, y_img_test, p_img_test, best_img["thr"],
                          cls_names, out/"image_fn_grid.png", "IMAGE – False Negatives (pred=non, true=disaster)")

    # ---------- FUSED ----------
    # Uses synthetic pairing saved by late_fusion.py (y from text stream)
    if Path(args.fusion_dir, "_probs_fused_test.npy").exists():
        y_f, p_txt_f, p_img_f, p_fuse, alpha_thr = fused_arrays(args.fusion_dir)
        thr_f = float(alpha_thr.get("threshold", 0.5))
        m_fuse = summarize_errors(y_f, p_fuse, thr=thr_f)
        print("\n[FUSED] alpha & thr from late_fusion.py:", alpha_thr)
        print("[FUSED] test summary:", m_fuse)
        # top FP/FN indices for fused
        pred_f = (p_fuse >= thr_f).astype(int)
        fp_idx = np.where((pred_f==1) & (y_f==0))[0][:6]
        fn_idx = np.where((pred_f==0) & (y_f==1))[0][:6]
        print("\n[FUSED] sample FP idx:", fp_idx.tolist())
        print("[FUSED] sample FN idx:", fn_idx.tolist())

    # ---------- Write a short notes file ----------
    notes = {
        "text_threshold": best_text,
        "image_threshold": best_img,
        "text_summary": m_text,
        "image_summary": m_img,
        "fusion_alpha_thr": alpha_thr if Path(args.fusion_dir, "alpha.json").exists() else None
    }
    (out/"error_summary.json").write_text(json.dumps(notes, indent=2))
    print(f"\n[saved] {out/'error_summary.json'}")
    print(f"[info] Image FP/FN grids saved under: {out}")
    print("\nTip: skim printed TEXT FP/FN lines above and jot patterns (sarcasm, puns, ambiguous 'fire', etc.)")
    
if __name__ == "__main__":
    main()
