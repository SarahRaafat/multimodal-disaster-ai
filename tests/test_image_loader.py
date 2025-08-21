import pytest
import torch
from pathlib import Path
from src.data.image_dataset import get_torchvision_dataloaders


def _get_loaders(result):
    if isinstance(result, dict):
        return result["train"], result["val"], result["test"]
    # tuple-style (train, val, test)
    return result


@pytest.mark.skipif(
    not Path("image_data").exists(), reason="No image_data folder in CI"
)
def test_image_loaders_one_batch():
    res = get_torchvision_dataloaders(root_dir="image_data", img_size=224, batch_size=4)
    train_dl, val_dl, test_dl = _get_loaders(res)

    batch = next(iter(train_dl))
    # Accept (x, y) or dict-like batches
    if isinstance(batch, (list, tuple)):
        xb, yb = batch
    elif isinstance(batch, dict):
        if "images" in batch and "labels" in batch:
            xb, yb = batch["images"], batch["labels"]
        elif "x" in batch and "y" in batch:
            xb, yb = batch["x"], batch["y"]
        else:
            raise AssertionError(f"Unexpected batch keys: {batch.keys()}")
    elif torch.is_tensor(batch):
        xb, yb = batch, torch.zeros(batch.size(0), dtype=torch.long)
    else:
        raise AssertionError(f"Unsupported batch type: {type(batch)}")

    assert xb.shape[1:] == (3, 224, 224)
    assert yb.ndim == 1

    classes = getattr(train_dl.dataset, "classes", None)
    assert classes is not None, "classes attribute not found on dataset"
    assert set(classes) == {"flooded", "non_flooded"}
