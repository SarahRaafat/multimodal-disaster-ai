#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Early-Fusion MLP for Multimodal Disaster Classification

Inputs:
  - artifacts/embeddings/text_{train,val,test}.npz  -> X:(N,768), y:(N,)
  - artifacts/embeddings/image_{train,val,test}.npz -> X:(N,512), y:(N,)

Three modes to handle (lack of) pairing:
  1) concat_naive  : pair by index up to min(n_text, n_image); X=[text;image] (1280-d)
  2) text_only_pad : X=[text; zeros(512)]  (uses text y)
  3) image_only_pad: X=[zeros(768); image] (uses image y)

Trains a small MLP: 1280 -> 512 (ReLU, Dropout) -> 2

Saves:
  - artifacts/fusion/mlp.pt
  - artifacts/fusion/val_metrics.json
  - artifacts/fusion/test_metrics.json
  - artifacts/fusion/history.json
"""

import argparse, json, os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# --------------------------
# Utils
# --------------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_npz(p: Path):
    d = np.load(p)
    # expected keys: X, y
    return d["X"], d["y"]

def make_split(mode: str, txt_npz: Path, img_npz: Path):
    Xt, yt = load_npz(txt_npz)      # (Nt, 768)
    Xi, yi = load_npz(img_npz)      # (Ni, 512)

    if mode == "concat_naive":
        n = min(len(yt), len(yi))
        X = np.concatenate([Xt[:n], Xi[:n]], axis=1)  # (n, 1280)
        y = yt[:n].astype(int)
    elif mode == "text_only_pad":
        zeros = np.zeros((Xt.shape[0], 512), dtype=Xp_dtype(Xt))
        X = np.concatenate([Xt, zeros], axis=1)
        y = yt.astype(int)
    elif mode == "image_only_pad":
        zeros = np.zeros((Xi.shape[0], 768), dtype=Xp_dtype(Xi))
        X = np.concatenate([zeros, Xi], axis=1)
        y = yi.astype(int)
    else:
        raise ValueError("Unknown mode: " + mode)

    return X, y

def Xp_dtype(X):
    # helper to pick a float dtype (np.float32 preferred)
    return np.float32 if X.dtype == np.float32 else np.float32

def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

class FusionMLP(nn.Module):
    def __init__(self, in_dim=1280, hid=512, out=2, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hid, out),
        )
    def forward(self, x): return self.net(x)

def evaluate(model, dl, device):
    model.eval()
    preds, gold = [], []
    with torch.no_grad():
        for X, y in dl:
            X = X.to(device)
            out = model(X)
            preds += out.argmax(1).cpu().tolist()
            gold  += y.tolist()
    acc = accuracy_score(gold, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(gold, preds, average="binary", zero_division=0)
    cm = confusion_matrix(gold, preds).tolist()
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "confusion_matrix": cm}

# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", type=str, default="artifacts/embeddings", help="Embeddings root folder")
    ap.add_argument("--mode", type=str, default="concat_naive",
                    choices=["concat_naive","text_only_pad","image_only_pad"],
                    help="How to build fusion input")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-dir", type=str, default="artifacts/fusion")
    ap.add_argument("--class-weights", action="store_true", help="Use class weights from train split")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}  mode={args.mode}")

    emb_dir = Path(args.emb_dir)
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)

    # Build splits
    txt_train = emb_dir/"text_train.npz"; img_train = emb_dir/"image_train.npz"
    txt_val   = emb_dir/"text_val.npz";   img_val   = emb_dir/"image_val.npz"
    txt_test  = emb_dir/"text_test.npz";  img_test  = emb_dir/"image_test.npz"

    Xtr, ytr = make_split(args.mode, txt_train, img_train)
    Xva, yva = make_split(args.mode, txt_val,   img_val)
    Xte, yte = make_split(args.mode, txt_test,  img_test)

    print(f"[data] train={Xtr.shape} val={Xva.shape} test={Xte.shape}")

    # Dataloaders
    train_dl = make_loader(Xtr, ytr, args.batch_size, shuffle=True)
    val_dl   = make_loader(Xva, yva, args.batch_size, shuffle=False)
    test_dl  = make_loader(Xte, yte, args.batch_size, shuffle=False)

    # Model
    model = FusionMLP(in_dim=1280, hid=512, out=2, p=args.dropout).to(device)

    # Criterion (optional class weights)
    if args.class_weights:
        # weights inversely proportional to class frequency in *train*
        from collections import Counter
        c = Counter(map(int, ytr.tolist()))
        w0 = 1.0 / max(c.get(0,1), 1)
        w1 = 1.0 / max(c.get(1,1), 1)
        # normalize so larger class doesn't explode the loss scale
        s = w0 + w1; w = torch.tensor([w0/s, w1/s], dtype=torch.float32, device=device)
        print(f"[loss] Using class weights = [{w[0].item():.6f}, {w[1].item():.6f}]  (from counts={dict(c)})")
        criterion = nn.CrossEntropyLoss(weight=w)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Train
    history = {"train_loss": [], "val_f1": []}
    best_f1 = -1.0
    best_path = save_dir/"mlp.pt"

    for epoch in range(1, args.epochs+1):
        # one epoch
        model.train()
        total = 0.0
        for X, y in train_dl:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total += loss.item()
        tr_loss = total / max(1, len(train_dl))

        # eval
        val_metrics = evaluate(model, val_dl, device)
        history["train_loss"].append(tr_loss)
        history["val_f1"].append(val_metrics["f1"])
        print(f"[epoch {epoch}] loss={tr_loss:.4f} val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1']:.4f}")

        # checkpoint best
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), best_path)
            with open(save_dir/"val_metrics.json","w") as f: json.dump(val_metrics, f, indent=2)
            print(f"[save] best model -> {best_path}")

    # Load best and test
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics = evaluate(model, test_dl, device)
    with open(save_dir/"test_metrics.json","w") as f: json.dump(test_metrics, f, indent=2)
    with open(save_dir/"history.json","w") as f: json.dump(history, f, indent=2)

    print("[test-best]", json.dumps(test_metrics, indent=2))
    print(f"[done] artifacts at: {save_dir}")


if __name__ == "__main__":
    main()
