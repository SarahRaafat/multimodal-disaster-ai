#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def set_seed(seed: int = 42):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_loaders(root, img_size, batch_size):
    tf_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.ToTensor(),
    ])
    tf_eval = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    train_ds = datasets.ImageFolder(root=Path(root)/"train", transform=tf_train)
    val_ds   = datasets.ImageFolder(root=Path(root)/"val",   transform=tf_eval)
    test_ds  = datasets.ImageFolder(root=Path(root)/"test",  transform=tf_eval)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size)
    return train_dl, val_dl, test_dl, train_ds.classes


def evaluate(model, dataloader, device):
    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for x,y in dataloader:
            x,y = x.to(device), y.to(device)
            out = model(x)
            y_hat = out.argmax(1)
            preds.extend(y_hat.cpu().numpy().tolist())
            golds.extend(y.cpu().numpy().tolist())
    acc = accuracy_score(golds, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(golds, preds, average="binary", pos_label=0)
    cm = confusion_matrix(golds, preds).tolist()
    return {"accuracy":acc,"precision":prec,"recall":rec,"f1":f1,"confusion_matrix":cm}


def train_epoch(model, dl, criterion, optim, device):
    model.train()
    total = 0.0
    for x,y in dl:
        x,y = x.to(device), y.to(device)
        optim.zero_grad()
        out = model(x)
        loss = criterion(out,y)
        loss.backward()
        optim.step()
        total += loss.item()
    return total/len(dl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="image_data")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-dir", type=str, default="artifacts/resnet18")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    train_dl, val_dl, test_dl, classes = get_loaders(args.data_dir, args.img_size, args.batch_size)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for p in model.parameters(): p.requires_grad=False
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, len(classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.fc.parameters(), lr=args.lr)

    best_f1, history = -1, {"train_loss":[],"val_f1":[]}
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1,args.epochs+1):
        tr_loss = train_epoch(model, train_dl, criterion, optim, device)
        val_metrics = evaluate(model,val_dl,device)
        history["train_loss"].append(tr_loss); history["val_f1"].append(val_metrics["f1"])
        print(f"[epoch {epoch}] loss={tr_loss:.4f} val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1']:.4f}")
        if val_metrics["f1"]>best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), Path(args.save_dir)/"resnet18.pt")
            with open(Path(args.save_dir)/"val_metrics.json","w") as f: json.dump(val_metrics,f,indent=2)
            print(f"[save] best model saved to {args.save_dir}")

    # reload best
    model.load_state_dict(torch.load(Path(args.save_dir)/"resnet18.pt"))
    test_metrics = evaluate(model,test_dl,device)
    with open(Path(args.save_dir)/"test_metrics.json","w") as f: json.dump(test_metrics,f,indent=2)
    with open(Path(args.save_dir)/"history.json","w") as f: json.dump(history,f,indent=2)
    print("[test-best]", json.dumps(test_metrics, indent=2))


if __name__=="__main__":
    main()
