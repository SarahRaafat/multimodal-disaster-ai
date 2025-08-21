from pathlib import Path
from transformers import DistilBertTokenizerFast
from torch.utils.data import DataLoader

# Your actual module + class names from this repo:
from src.data.text_dataset import load_text_split, TextClsDataset


def test_load_text_split_and_dataset():
    p = Path("text_data/train.csv")
    assert p.exists(), "text_data/train.csv missing"

    df = load_text_split(p)  # returns df with columns ["text","y"]
    assert {"text", "y"} <= set(df.columns)
    assert len(df) > 0

    tok = DistilBertTokenizerFast.from_pretrained("artifacts/distilbert_ft")
    ds = TextClsDataset(df["text"].tolist()[:8], df["y"].tolist()[:8], tok, max_len=32)
    dl = DataLoader(ds, batch_size=2)

    batch = next(iter(dl))
    assert batch["input_ids"].shape == (2, 32)
    assert batch["attention_mask"].shape == (2, 32)
    assert batch["labels"].shape == (2,)
