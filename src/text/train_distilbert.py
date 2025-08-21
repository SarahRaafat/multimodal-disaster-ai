#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    f1_score,
)


# ---------------------
# Utilities
# ---------------------
def set_seed(seed: int = 42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_split_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{path} must contain 'text' and 'label' columns")
    label_map = {"non_disaster": 0, "disaster": 1}
    if not set(df["label"].unique()).issubset(set(label_map.keys())):
        raise ValueError(
            f"Unexpected labels in {path}. Expected only {list(label_map.keys())}"
        )
    df = df.copy()
    df["y"] = df["label"].map(label_map)
    return df[["text", "y"]]


class TextClsDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: DistilBertTokenizerFast,
        max_len: int = 128,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        txt = str(self.texts[idx])
        y = int(self.labels[idx])
        enc = self.tokenizer(
            txt,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(y, dtype=torch.long)
        return item


def evaluate(model, dataloader, device) -> Dict[str, float]:
    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            y_hat = torch.argmax(outputs.logits, dim=1)
            preds.extend(y_hat.cpu().numpy().tolist())
            golds.extend(batch["labels"].cpu().numpy().tolist())
    acc = accuracy_score(golds, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        golds, preds, average="binary", pos_label=1, zero_division=0
    )
    cm = confusion_matrix(golds, preds).tolist()
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
    }


# ---- Threshold helper
def eval_with_threshold(model, dataloader, device, thr=0.5):
    model.eval()
    probs, golds = [], []
    with torch.no_grad():
        for batch in dataloader:
            b = {k: v.to(device) for k, v in batch.items()}
            logits = model(**b).logits
            p1 = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # P(disaster=1)
            probs.extend(p1)
            golds.extend(b["labels"].cpu().numpy())
    probs = np.array(probs)
    golds = np.array(golds)
    preds = (probs >= thr).astype(int)
    f1 = f1_score(golds, preds, pos_label=1, zero_division=0)
    return f1, probs, golds, preds


def train_one_epoch(model, dataloader, optimizer, scheduler, device, criterion=None):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        if criterion is None:
            out = model(**batch)
            loss = out.loss
        else:
            out = model(**{k: v for k, v in batch.items() if k != "labels"})
            loss = criterion(out.logits, batch["labels"])
        loss.backward()
        total_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    return total_loss / max(1, len(dataloader))


def main():
    ap = argparse.ArgumentParser(
        description="Train DistilBERT on disaster tweets (binary classification)."
    )
    ap.add_argument(
        "--data-dir", type=str, default="text_data", help="train/val/test CSV folder"
    )
    ap.add_argument("--model-name", type=str, default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-dir", type=str, default="artifacts/distilbert")
    ap.add_argument(
        "--class-weights",
        action="store_true",
        help="Use inverse-frequency class weights in CrossEntropyLoss",
    )
    ap.add_argument(
        "--finetune",
        action="store_true",
        help="Unfreeze top layers of DistilBERT and fine-tune with smaller LR",
    )
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # ---------------------------
    # Load data
    # ---------------------------
    data_dir = Path(args.data_dir)
    train_df = load_split_csv(data_dir / "train.csv")
    val_df = load_split_csv(data_dir / "val.csv")
    test_df = load_split_csv(data_dir / "test.csv")
    print(f"[data] train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    # ---------------------------
    # Tokenizer, Datasets, Loaders
    # ---------------------------
    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_name)
    model = DistilBertForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    ).to(device)

    train_ds = TextClsDataset(
        train_df["text"].tolist(),
        train_df["y"].tolist(),
        tokenizer,
        max_len=args.max_len,
    )
    val_ds = TextClsDataset(
        val_df["text"].tolist(), val_df["y"].tolist(), tokenizer, max_len=args.max_len
    )
    test_ds = TextClsDataset(
        test_df["text"].tolist(), test_df["y"].tolist(), tokenizer, max_len=args.max_len
    )

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size)

    # ---------------------------
    # Optimizer (finetune or not)
    # ---------------------------
    if args.finetune:
        for p in model.parameters():
            p.requires_grad = False
        # unfreeze top-2 transformer layers + head
        for i in [-2, -1]:
            for p in model.distilbert.transformer.layer[i].parameters():
                p.requires_grad = True
        for p in model.pre_classifier.parameters():
            p.requires_grad = True
        for p in model.classifier.parameters():
            p.requires_grad = True

        head_params = list(model.pre_classifier.parameters()) + list(
            model.classifier.parameters()
        )
        base_params = []
        for i in [-2, -1]:
            base_params += list(model.distilbert.transformer.layer[i].parameters())

        optimizer = AdamW(
            [
                {"params": base_params, "lr": 5e-6},  # tiny LR for backbone
                {"params": head_params, "lr": args.lr},  # e.g., 2e-5
            ],
            weight_decay=0.01,
        )
        print(f"[finetune] Unfroze top-2 layers + head. LRs: base=5e-6, head={args.lr}")
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    # ---------------------------
    # Scheduler
    # ---------------------------
    total_steps = len(train_dl) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # ---------------------------
    # Optional class weights
    # ---------------------------
    criterion = None
    if args.class_weights:
        counts = train_df["y"].value_counts().to_dict()
        w_non = 1.0 / counts.get(0, 1)
        w_pos = 1.0 / counts.get(1, 1)
        weights = torch.tensor([w_non, w_pos], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        print(
            f"[loss] Using class weights = {weights.tolist()}  (from counts={counts})"
        )

    # ---------------------------
    # Training loop
    # ---------------------------
    history, best_f1 = {"train_loss": [], "val_f1": []}, -1.0
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(
            model, train_dl, optimizer, scheduler, device, criterion
        )
        val_metrics = evaluate(model, val_dl, device)
        history["train_loss"].append(tr_loss)
        history["val_f1"].append(val_metrics["f1"])
        print(
            f"[epoch {epoch}] loss={tr_loss:.4f}  val_acc={val_metrics['accuracy']:.4f}  val_f1={val_metrics['f1']:.4f}"
        )
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            model.save_pretrained(args.save_dir)
            tokenizer.save_pretrained(args.save_dir)
            with open(Path(args.save_dir) / "val_metrics.json", "w") as f:
                json.dump(val_metrics, f, indent=2)
            print(f"[save] Best model saved to {args.save_dir}")

    # ---------------------------
    # Reload best model
    # ---------------------------
    print("[info] Reloading best checkpoint...")
    best_model = DistilBertForSequenceClassification.from_pretrained(
        args.save_dir, num_labels=2
    ).to(device)

    # ---------------------------
    # Threshold tuning
    # ---------------------------
    best_thr, best_thr_f1 = 0.5, 0.0
    for t in np.linspace(0.30, 0.70, 21):
        f1_t, _, _, _ = eval_with_threshold(best_model, val_dl, device, thr=t)
        if f1_t > best_thr_f1:
            best_thr, best_thr_f1 = t, f1_t
    print(f"[val] tuned threshold={best_thr:.2f} f1={best_thr_f1:.4f}")
    with open(Path(args.save_dir) / "best_threshold.json", "w") as f:
        json.dump(
            {"threshold": float(best_thr), "val_f1": float(best_thr_f1)}, f, indent=2
        )

    # ---------------------------
    # Test evaluation
    # ---------------------------
    f1_test, probs, golds, preds = eval_with_threshold(
        best_model, test_dl, device, thr=best_thr
    )
    acc = accuracy_score(golds, preds)
    prec, rec, f1v, _ = precision_recall_fscore_support(
        golds, preds, average="binary", pos_label=1, zero_division=0
    )
    cm = confusion_matrix(golds, preds).tolist()
    test_metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1v,
        "confusion_matrix": cm,
        "threshold": float(best_thr),
    }
    with open(Path(args.save_dir) / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    with open(Path(args.save_dir) / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("[test-best-threshold]", json.dumps(test_metrics, indent=2))
    print(f"[done] artifacts at: {args.save_dir}")


if __name__ == "__main__":
    main()
