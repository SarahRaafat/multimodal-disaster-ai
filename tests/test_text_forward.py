from pathlib import Path
import pytest
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification


def test_distilbert_forward_shape():
    model_dir = Path("artifacts/distilbert_ft")
    if not model_dir.exists():
        pytest.skip("Fine-tuned artifacts not present on CI; skipping.")

    tok = DistilBertTokenizerFast.from_pretrained(model_dir)
    enc = tok(
        ["test tweet about floods"],
        truncation=True,
        padding="max_length",
        max_length=16,
        return_tensors="pt",
    )
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    out = model(**enc)
    assert out.logits.shape == (1, 2)
