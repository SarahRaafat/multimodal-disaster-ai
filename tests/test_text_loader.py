import pytest
from pathlib import Path
from transformers import DistilBertTokenizerFast
from torch.utils.data import DataLoader
from src.data.text_dataset import load_text_split, TextClsDataset

# Skip on CI if the dataset isn’t present
pytestmark = pytest.mark.skipif(
    not Path("text_data/train.csv").exists(),
    reason="No text_data folder on CI",
)


def test_load_text_split_and_dataset():
    p = Path("text_data/train.csv")
    # From here on we can assume it exists locally
    df = load_text_split(p)
    assert {"text", "y"} <= set(df.columns)
    assert len(df) > 0

    tok = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    ds = TextClsDataset(df["text"].tolist()[:8], df["y"].tolist()[:8], tok, max_len=32)
    dl = DataLoader(ds, batch_size=2)

    batch = next(iter(dl))
    assert batch["input_ids"].shape == (2, 32)
    assert batch["attention_mask"].shape == (2, 32)
    assert batch["labels"].shape == (2,)
