#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path
from typing import Set, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
MULTISPACE_RE = re.compile(r"\s+")
CONTROL_RE = re.compile(r"[\u0000-\u001F\u007F]")

def clean_text(s: str, keep_hashtags_token: bool = True) -> str:
    """Basic tweet cleaning suitable for fast ML baselines.
    - remove URLs
    - strip @mentions
    - convert #hashtag -> 'hashtag' (or keep as token '#hashtag' if keep_hashtags_token)
    - collapse whitespace
    - strip control chars
    """
    if not isinstance(s, str):
        return ""
    s = CONTROL_RE.sub(" ", s)
    s = URL_RE.sub(" ", s)
    s = MENTION_RE.sub(" ", s)
    if keep_hashtags_token:
        s = HASHTAG_RE.sub(lambda m: f"{m.group(0)}", s)
    else:
        s = HASHTAG_RE.sub(lambda m: m.group(1), s)
    s = MULTISPACE_RE.sub(" ", s).strip()
    return s

def normalize_label(x, positives: Set[str], neg_label="non_disaster", pos_label="disaster"):
    """Map various dataset label encodings into {'disaster','non_disaster'}."""
    if pd.isna(x):
        return neg_label
    v = str(x).strip().lower()
    return pos_label if v in positives else neg_label

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure required columns exist: id, text, label, source, lang, created_at, lat, lon."""
    if "id" not in df.columns:
        df = df.copy()
        df["id"] = [f"row-{i}" for i in range(len(df))]
    for c in ["source", "lang"]:
        if c not in df.columns:
            df[c] = ""
    for c in ["created_at"]:
        if c not in df.columns:
            df[c] = ""
    for c in ["lat", "lon"]:
        if c not in df.columns:
            df[c] = np.nan
    cols = ["id", "text", "label", "source", "lang", "created_at", "lat", "lon"]
    return df[cols]

def stratified_sample(df: pd.DataFrame, label_col: str, target_size: int) -> pd.DataFrame:
    if len(df) <= target_size:
        return df
    frac = target_size / len(df)
    return (df.groupby(label_col, group_keys=False)
              .apply(lambda g: g.sample(frac=frac, random_state=42))
              .reset_index(drop=True))

def split_train_val_test(df: pd.DataFrame, label_col: str,
                         val_size: float = 0.1, test_size: float = 0.1,
                         seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified split into train/val/test with given fractions (sum <= 1)."""
    assert 0 < val_size < 1 and 0 < test_size < 1 and (val_size + test_size) < 1
    df_train, df_tmp = train_test_split(
        df, test_size=(val_size + test_size), stratify=df[label_col], random_state=seed
    )
    rel_test = test_size / (val_size + test_size)
    df_val, df_test = train_test_split(
        df_tmp, test_size=rel_test, stratify=df_tmp[label_col], random_state=seed
    )
    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser(description="Prepare disaster text data (clean, normalize, sample, split).")
    ap.add_argument("--input", type=str, required=True, help="Path to input CSV (raw).")
    ap.add_argument("--text-col", type=str, default="text", help="Column containing text.")
    ap.add_argument("--label-col", type=str, default="label", help="Column containing labels.")
    ap.add_argument("--lang-col", type=str, default=None, help="Optional language column to filter by en.")
    ap.add_argument("--lang", type=str, default="en", help="Language code to keep if lang-col is provided.")
    ap.add_argument("--positive-labels", type=str, default="disaster,urgent,1,positive",
                    help="Comma-separated values treated as POSITIVE (disaster). Lowercased.")
    ap.add_argument("--keep-hashtags-token", action="store_true", help="Keep '#hashtag' tokens as-is.")
    ap.add_argument("--target-size", type=int, default=5000, help="Target total dataset size after sampling.")
    ap.add_argument("--val-size", type=float, default=0.1, help="Validation fraction (0,1).")
    ap.add_argument("--test-size", type=float, default=0.1, help="Test fraction (0,1).")
    ap.add_argument("--output-dir", type=str, required=True, help="Output directory for splits.")
    ap.add_argument("--source-name", type=str, default="", help="Optional name of source dataset for metadata.")
    ap.add_argument("--lowercase", action="store_true", help="Lowercase text after cleaning.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    if args.text_col not in df.columns or args.label_col not in df.columns:
        raise SystemExit(f"Expected columns '{args.text_col}' and '{args.label_col}' not found. Found: {list(df.columns)}")

    # Optional language filter
    if args.lang_col and args.lang_col in df.columns:
        before = len(df)
        df = df[df[args.lang_col].astype(str).str.lower() == args.lang.lower()].copy()
        print(f"[lang-filter] kept {len(df)}/{before} rows with {args.lang} in '{args.lang_col}'")

    # Clean text
    df = df.copy()
    df["text"] = df[args.text_col].astype(str).map(lambda s: clean_text(s, keep_hashtags_token=args.keep_hashtags_token))
    if args.lowercase:
        df["text"] = df["text"].str.lower()

    # Normalize labels
    positives = set([s.strip().lower() for s in args.positive_labels.split(",") if s.strip()])
    df["label"] = df[args.label_col].map(lambda x: normalize_label(x, positives=positives))

    # Attach optional columns
    if args.source_name:
        df["source"] = args.source_name

    # Ensure standard columns exist
    df = ensure_columns(df)

    # Drop empty texts
    df = df[df["text"].str.len() > 0].reset_index(drop=True)

    # Class distribution prior to sampling
    class_counts_before = df["label"].value_counts().to_dict()
    print(f"[counts-before] {class_counts_before}")

    # Stratified sample
    df_sampled = stratified_sample(df, label_col="label", target_size=args.target_size).reset_index(drop=True)

    # Class distribution after sampling
    class_counts_after = df_sampled["label"].value_counts().to_dict()
    print(f"[counts-after] {class_counts_after}")

    # Split
    train_df, val_df, test_df = split_train_val_test(
        df_sampled, "label", val_size=args.val_size, test_size=args.test_size, seed=args.seed
    )

    # Save
    train_path = outdir / "train.csv"
    val_path = outdir / "val.csv"
    test_path = outdir / "test.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Report
    report = {
        "input_path": str(Path(args.input).resolve()),
        "output_dir": str(outdir.resolve()),
        "rows_input": int(len(df)),
        "rows_sampled": int(len(df_sampled)),
        "class_counts_before": class_counts_before,
        "class_counts_after": class_counts_after,
        "splits": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "config": {
            "text_col": args.text_col,
            "label_col": args.label_col,
            "lang_col": args.lang_col,
            "lang": args.lang,
            "positive_labels": sorted(list(positives)),
            "keep_hashtags_token": bool(args.keep_hashtags_token),
            "target_size": args.target_size,
            "val_size": args.val_size,
            "test_size": args.test_size,
            "lowercase": bool(args.lowercase),
            "seed": args.seed,
            "source_name": args.source_name,
        },
    }
    with open(outdir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[done] wrote: {train_path}, {val_path}, {test_path}")
    print(f"[report] {outdir / 'report.json'}")

if __name__ == "__main__":
    main()
