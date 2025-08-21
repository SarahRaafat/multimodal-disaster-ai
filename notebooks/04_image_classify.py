#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


def softmax1(logits):
    return torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()


ap = argparse.ArgumentParser()
ap.add_argument("--image-root", required=True, help="folder with images")
ap.add_argument("--ckpt", required=True, help="resnet ckpt (.pt)")
ap.add_argument("--out-csv", required=True)
ap.add_argument("--img-size", type=int, default=224)
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
tf = transforms.Compose(
    [transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor()]
)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
state = torch.load(args.ckpt, map_location=device)
model.load_state_dict(state, strict=True)
model = model.to(device).eval()

rows = []
with torch.no_grad():
    for fn in sorted(os.listdir(args.image_root)):
        if not fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
            continue
        path = Path(args.image_root) / fn
        x = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        logits = model(x)
        p = float(softmax1(logits)[0])
        pred = "disaster" if p >= 0.5 else "non_disaster"
        rows.append({"image": fn, "p_disaster": p, "pred": pred})

out = Path(args.out_csv)
out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out, index=False)
print(f"[saved] {out}")
