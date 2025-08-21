"""
src/data/image_dataset.py
Minimal image dataset loader with torchvision
"""

import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_torchvision_dataloaders(root_dir, img_size=224, batch_size=8):
    """
    Creates train/val/test dataloaders from image_data/ folder.
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
