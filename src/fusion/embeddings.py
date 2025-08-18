#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, json
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

def set_seed(seed=42):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def load_text_split(path):
    df = pd.read_csv(path)
    label_map = {"non_disaster":0, "disaster":1}
    df["y"] = df["label"].map(label_map)
    return df[["id","text","y"]] if "id" in df.columns else df.reset_index().rename(columns={"index":"id"})[["id","text","y"]]

def text_embeddings(split_csv, distil_dir, batch_size=32, max_len=160, device="cpu"):
    tok = DistilBertTokenizerFast.from_pretrained(distil_dir)
    model = DistilBertForSequenceClassification.from_pretrained(distil_dir).to(device)
    model.eval()
    # get the base transformer to read hidden states
    base = model.distilbert
    def encode_batch(texts):
        enc = tok(texts, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
        return {k:v.to(device) for k,v in enc.items()}
    embs, ys, ids = [], [], []
    df = load_text_split(split_csv)
    with torch.no_grad():
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            inputs = encode_batch(batch["text"].tolist())
            out = base(**inputs)               # last_hidden_state [B, L, H]
            cls = out.last_hidden_state[:,0]   # use CLS token embedding
            embs.append(cls.cpu().numpy())
            ys.extend(batch["y"].tolist())
            ids.extend(batch["id"].tolist())
    return np.vstack(embs), np.array(ys), np.array(ids)

def image_embeddings(root_split_dir, resnet_ckpt, batch_size=32, img_size=224, device="cpu"):
    from torchvision import models, datasets, transforms
    import torch.nn as nn
    from torch.utils.data import DataLoader
    import torch

    # transforms
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    # dataset + loader
    ds = datasets.ImageFolder(root_split_dir, transform=tf)
    dl = DataLoader(ds, batch_size=batch_size)
    classes = ds.classes

    # build model same way as training
    model = models.resnet18(weights=None)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, len(classes))   # 2 classes (flooded/non_flooded)

    # load trained checkpoint strictly (includes fc weights)
    state = torch.load(resnet_ckpt, map_location=device)
    model.load_state_dict(state, strict=True)

    # now replace head with Identity to extract penultimate features
    model.fc = nn.Identity()
    model = model.to(device)
    model.eval()

    # collect features
    feats, ys, ids = [], [], []
    with torch.no_grad():
        for i, (x,y) in enumerate(dl):
            x = x.to(device)
            f = model(x)   # [B, 512]
            feats.append(f.cpu().numpy())
            ys.extend(y.cpu().numpy().tolist())
            ids.extend([f"img_{i*batch_size+j}" for j in range(len(y))])

    return np.vstack(feats), np.array(ys), np.array(ids), classes

def save_npz(path, **arrays):
    np.savez_compressed(path, **arrays)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-data-dir", type=str, default="text_data")
    ap.add_argument("--distil-dir", type=str, default="artifacts/distilbert")
    ap.add_argument("--image-data-dir", type=str, default="image_data")
    ap.add_argument("--resnet-ckpt", type=str, default="artifacts/resnet18/resnet18.pt")
    ap.add_argument("--out-dir", type=str, default="artifacts/embeddings")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    # text
    for split in ["train","val","test"]:
        te, ty, tid = text_embeddings(Path(args.text_data_dir)/f"{split}.csv", args.distil_dir, args.batch_size, args.max_len, device)
        save_npz(Path(args.out_dir)/f"text_{split}.npz", X=te, y=ty, ids=tid)
        print(f"[text {split}] {te.shape}")

    # images
    for split in ["train","val","test"]:
        ie, iy, iid, classes = image_embeddings(Path(args.image_data_dir)/split, args.resnet_ckpt, args.batch_size, args.img_size, device)
        save_npz(Path(args.out_dir)/f"image_{split}.npz", X=ie, y=iy, ids=iid, classes=np.array(classes))
        print(f"[image {split}] {ie.shape}, classes={classes}")

    with open(Path(args.out_dir)/"meta.json","w") as f:
        json.dump({"text_model":args.distil_dir, "image_model":args.resnet_ckpt, "device":device}, f, indent=2)

if __name__ == "__main__":
    main()
