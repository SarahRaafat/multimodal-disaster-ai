"""Utilities to build torchvision DataLoaders for the image dataset (flooded vs non_flooded)."""

import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_torchvision_dataloaders(
    root_dir: str, img_size: int = 224, batch_size: int = 8, num_workers: int = 2
):
    """Create train/val/test dataloaders from an `image_data/` folder.

    Expects the following structure:
        root_dir/
          train/{flooded,non_flooded}/...
          val/{flooded,non_flooded}/...
          test/{flooded,non_flooded}/...

    Args:
        root_dir: Path to dataset root.
        img_size: Final square size for images (HxW).
        batch_size: Samples per batch.
        num_workers: DataLoader workers.

    Returns:
        (train_dl, val_dl, test_dl): three `torch.utils.data.DataLoader` objects.

    Raises:
        AssertionError: if expected split folders do not exist.
    """
    root = Path(root_dir)
    assert root.exists(), f"{root} not found"

    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )

    splits = {}
    for split in ["train", "val", "test"]:
        ds = datasets.ImageFolder(root / split, transform=transform)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"))
        splits[split] = dl
        print(f"[{split}] {len(ds)} images, classes={ds.classes}")

    return splits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=str, required=True, help="Root image_data folder"
    )
    parser.add_argument(
        "--show-batch", action="store_true", help="If set, shows 1 batch shapes"
    )
    args = parser.parse_args()

    dls = get_torchvision_dataloaders(args.root)

    if args.show_batch:
        import matplotlib.pyplot as plt
        import torchvision

        imgs, labels = next(iter(dls["train"]))
        print("Batch shapes:", imgs.shape, labels.shape)

        grid = torchvision.utils.make_grid(imgs[:8], nrow=4)
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis("off")
        plt.show()
