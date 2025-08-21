# tests/test_text_forward.py
from pathlib import Path
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast


def _has_local_checkpoint(d: Path) -> bool:
    return (
        d.exists()
        and (d / "config.json").exists()
        and ((d / "pytorch_model.bin").exists() or (d / "model.safetensors").exists())
    )


def test_distilbert_forward_shape():
    model_dir = Path("artifacts/distilbert_ft")

    if _has_local_checkpoint(model_dir):
        tok = DistilBertTokenizerFast.from_pretrained(model_dir)
        model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    else:
        # CI fallback: just verify forward pass shape with the base model
        tok = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2
        )

    enc = tok(
        ["test tweet about floods"],
        truncation=True,
        padding="max_length",
        max_length=16,
        return_tensors="pt",
    )
    out = model(**enc)
    assert out.logits.shape == (1, 2)
