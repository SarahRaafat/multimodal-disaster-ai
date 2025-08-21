"""
src/data/text_dataset.py
Minimal text dataset loader for disaster classification
"""

import pandas as pd
import argparse
from pathlib import Path
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast


def load_text_split(path: str):
    """
    Load a split CSV into a DataFrame with columns ['text','y'].
    Prints size & class balance.
    """
    df = pd.read_csv(path)

    # Some CSVs may have 'label' instead of 'y' → normalize
    if "label" in df.columns and "y" not in df.columns:
        df = df.rename(columns={"label": "y"})

    counts = df["y"].value_counts().to_dict()
    print(f"[load_text_split] {path}: {len(df)} rows, balance={counts}")

    return df


LABEL2ID = {"non_disaster": 0, "disaster": 1}


class TextClsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer: DistilBertTokenizerFast, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        raw_label = self.labels[idx]
        label = LABEL2ID[raw_label] if isinstance(raw_label, str) else int(raw_label)

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": label,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="Path to split file (e.g., text_data/train.csv)",
    )
    args = parser.parse_args()

    df = load_text_split(args.split)
    for i in range(3):
        print(f"\nExample {i+1}:")
        print("Text:", df.loc[i, "text"])
        print("Label:", df.loc[i, "y"])
