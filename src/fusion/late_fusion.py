"""Late fusion of text + image probabilities with simple calibration utilities.

- Platt scaling on validation
- ECE & Brier computation
- Grid search over alpha (blend) and decision threshold
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


def set_seed(seed=42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def softmax_probs(logits):
    p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return p


def f1_from_probs(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    _, _, f1, _ = precision_recall_fscore_support(
        y, yhat, average="binary", zero_division=0
    )
    return f1


def metrics_from_probs(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    acc = accuracy_score(y, yhat)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y, yhat, average="binary", zero_division=0
    )
    cm = confusion_matrix(y, yhat).tolist()
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "threshold": thr,
    }


def brier_score(y, p):
    y = y.astype(float)
    return float(np.mean((p - y) ** 2))


def ece_score(y, p, bins=10):
    # reliability diag expected calibration error
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        m = (p >= lo) & (p < hi) if i < bins - 1 else (p >= lo) & (p <= hi)
        if not np.any(m):
            continue
        conf = float(np.mean(p[m]))
        acc = float(np.mean((p[m] >= 0.5).astype(int) == y[m]))
        ece += (m.mean()) * abs(acc - conf)
    return float(ece)


# -------- text probs --------
def text_probs(split_csv, model_dir, batch_size=32, max_len=160, device="cpu"):
    import pandas as pd

    tok = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    df = pd.read_csv(split_csv)
    labmap = {"non_disaster": 0, "disaster": 1}
    y = df["label"].map(labmap).values.astype(int)

    probs = []
    with torch.no_grad():
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i : i + batch_size]
            enc = tok(
                batch["text"].tolist(),
                truncation=True,
                padding=True,
                max_length=max_len,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            probs.append(softmax_probs(logits))
    return np.concatenate(probs), y


# -------- image probs --------
def image_probs(root_split_dir, ckpt_path, batch_size=32, img_size=224, device="cpu"):
    tf = transforms.Compose(
        [transforms.Resize((img_size, img_size)), transforms.ToTensor()]
    )
    ds = datasets.ImageFolder(root_split_dir, transform=tf)
    dl = DataLoader(ds, batch_size=batch_size)
    model = models.resnet18(weights=None)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, 2)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()

    probs, ys = [], []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            logits = model(x)
            probs.append(softmax_probs(logits))
            ys.extend(y.numpy().tolist())
    return np.concatenate(probs), np.array(ys, dtype=int)


def fit_platt(p_val: np.ndarray, y_val: np.ndarray):
    """Fit Platt scaling on validation probabilities.

    Args:
        p_val: Validation probs for class=1.
        y_val: Validation labels (0/1).

    Returns:
        (A, B) such that calibrated p = sigmoid(A * logit(p) + B).
    """
    eps = 1e-6
    logits = np.log(
        np.clip(p_val, eps, 1 - eps) / np.clip(1 - p_val, eps, 1 - eps)
    ).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver="lbfgs")
    lr.fit(logits, y_val)
    A = float(lr.coef_[0, 0])
    B = float(lr.intercept_[0])
    return A, B


def apply_platt(p, A, B):
    eps = 1e-6
    logits = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps))
    z = A * logits + B
    return 1.0 / (1.0 + np.exp(-z))


# ------------- main -------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-data-dir", type=str, default="text_data")
    ap.add_argument("--distil-dir", type=str, default="artifacts/distilbert")
    ap.add_argument("--image-data-dir", type=str, default="image_data")
    ap.add_argument("--resnet-ckpt", type=str, default="artifacts/resnet18/resnet18.pt")
    ap.add_argument("--out-dir", type=str, default="artifacts/fusion")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- VAL probs
    p_txt_val, y_txt_val = text_probs(
        Path(args.text_data_dir) / "val.csv",
        args.distil_dir,
        args.batch_size,
        args.max_len,
        device,
    )
    p_img_val, y_img_val = image_probs(
        Path(args.image_data_dir) / "val",
        args.resnet_ckpt,
        args.batch_size,
        args.img_size,
        device,
    )

    n_val = min(len(p_txt_val), len(p_img_val))
    p_txt_val, p_img_val = p_txt_val[:n_val], p_img_val[:n_val]
    y_val_a, y_val_b = y_txt_val[:n_val], y_img_val[:n_val]
    mismatch = int(np.sum(y_val_a != y_val_b))
    y_val = y_val_a  # choose text labels
    print(f"[val] paired n={n_val}, label mismatches={mismatch}")

    # --- TEST probs
    p_txt_test, y_txt_test = text_probs(
        Path(args.text_data_dir) / "test.csv",
        args.distil_dir,
        args.batch_size,
        args.max_len,
        device,
    )
    p_img_test, y_img_test = image_probs(
        Path(args.image_data_dir) / "test",
        args.resnet_ckpt,
        args.batch_size,
        args.img_size,
        device,
    )

    n_test = min(len(p_txt_test), len(p_img_test))
    p_txt_test, p_img_test = p_txt_test[:n_test], p_img_test[:n_test]
    y_test_a, y_test_b = y_txt_test[:n_test], y_img_test[:n_test]
    mismatch_t = int(np.sum(y_test_a != y_test_b))
    y_test = y_test_a
    print(f"[test] paired n={n_test}, label mismatches={mismatch_t}")

    # --- calibrate on val (Platt)
    A_txt, B_txt = fit_platt(p_txt_val, y_val)
    A_img, B_img = fit_platt(p_img_val, y_val)  # calibrate to same labels
    p_txt_val_cal = apply_platt(p_txt_val, A_txt, B_txt)
    p_img_val_cal = apply_platt(p_img_val, A_img, B_img)
    p_txt_test_cal = apply_platt(p_txt_test, A_txt, B_txt)
    p_img_test_cal = apply_platt(p_img_test, A_img, B_img)

    # --- tune alpha & threshold on val
    best = {"f1": -1, "alpha": 0.5, "thr": 0.5}
    for alpha in np.linspace(0.0, 1.0, 21):
        p_f_val = alpha * p_txt_val_cal + (1 - alpha) * p_img_val_cal
        for thr in np.linspace(0.30, 0.70, 21):
            f1 = f1_from_probs(y_val, p_f_val, thr)
            if f1 > best["f1"]:
                best = {"f1": float(f1), "alpha": float(alpha), "thr": float(thr)}
    print(
        f"[val] best fusion: alpha={best['alpha']:.2f}, thr={best['thr']:.2f}, f1={best['f1']:.4f}"
    )

    # --- final test eval
    p_f_test = best["alpha"] * p_txt_test_cal + (1 - best["alpha"]) * p_img_test_cal
    test_metrics = metrics_from_probs(y_test, p_f_test, thr=best["thr"])

    # also compute calibration scores
    cal = {
        "text": {
            "ece": ece_score(y_test, p_txt_test_cal),
            "brier": brier_score(y_test, p_txt_test_cal),
        },
        "image": {
            "ece": ece_score(y_test, p_img_test_cal),
            "brier": brier_score(y_test, p_img_test_cal),
        },
        "fused": {
            "ece": ece_score(y_test, p_f_test),
            "brier": brier_score(y_test, p_f_test),
        },
    }

    # save artifacts
    with open(out / "alpha.json", "w") as f:
        json.dump({"alpha": best["alpha"], "threshold": best["thr"]}, f, indent=2)
    with open(out / "late_val_best.json", "w") as f:
        json.dump(
            {"alpha": best["alpha"], "thr": best["thr"], "val_f1": best["f1"]},
            f,
            indent=2,
        )
    with open(out / "late_test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    with open(out / "calibration_test.json", "w") as f:
        json.dump(cal, f, indent=2)

    # store arrays for plotting later
    np.save(out / "_probs_text_test.npy", p_txt_test_cal)
    np.save(out / "_probs_image_test.npy", p_img_test_cal)
    np.save(out / "_probs_fused_test.npy", p_f_test)
    np.save(out / "_labels_test.npy", y_test)

    print("[test-late-fusion]", json.dumps(test_metrics, indent=2))
    print("[calibration]", json.dumps(cal, indent=2))
    print(f"[done] artifacts at: {out}")


if __name__ == "__main__":
    main()
