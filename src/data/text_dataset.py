"""Minimal text dataset loaders for disaster tweet classification (binary)."""

import pandas as pd
import argparse
from pathlib import Path
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast


def load_text_split(path):
    """Load a CSV split and map labels to integers.

    The CSV must contain:
      - `text` (str): raw tweet/content
      - `label` (str): {"non_disaster","disaster"}

    Args:
        path: Path-like to CSV file.

    Returns:
        pandas.DataFrame with columns:
            - text (str)
            - y (int; 0=non_disaster, 1=disaster)

    Raises:
        ValueError: if required columns/labels are missing.
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
    """Tokenized text dataset for binary classification.

    Wraps a HuggingFace tokenizer to produce input_ids, attention_mask, and label tensors.
    """

    def __init__(self, texts, labels, tokenizer: DistilBertTokenizerFast, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        """Return a tokenized example.

        Args:
            idx: Index into the dataset.

        Returns:
            Dict[str, torch.Tensor]: keys include
              - input_ids (LongTensor)
              - attention_mask (LongTensor)
              - labels (LongTensor: 0/1)
        """
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
