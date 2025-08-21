#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification


def softmax1(logits):
    return torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()


ap = argparse.ArgumentParser()
ap.add_argument("--text-csv", required=True)
ap.add_argument("--model-dir", default="artifacts/distilbert_ft")
ap.add_argument("--out-csv", required=True)
ap.add_argument("--max-len", type=int, default=160)
args = ap.parse_args()

df = pd.read_csv(args.text_csv)
tok = DistilBertTokenizerFast.from_pretrained(args.model_dir)
model = DistilBertForSequenceClassification.from_pretrained(args.model_dir).eval()

try:
    thr = json.loads((Path(args.model_dir) / "best_threshold.json").read_text()).get(
        "threshold", 0.5
    )
except Exception:
    thr = 0.5

probs = []
with torch.no_grad():
    for i in range(0, len(df), 32):
        enc = tok(
            df["text"].astype(str).iloc[i : i + 32].tolist(),
            truncation=True,
            padding=True,
            max_length=args.max_len,
            return_tensors="pt",
        )
        logits = model(**enc).logits
        probs.extend(softmax1(logits))

df_out = df.copy()
df_out["p_disaster"] = probs
df_out["pred"] = (df_out["p_disaster"] >= thr).map(
    {True: "disaster", False: "non_disaster"}
)
Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
df_out.to_csv(args.out_csv, index=False)
print(f"[saved] {args.out_csv}  (thr={thr:.2f})")
