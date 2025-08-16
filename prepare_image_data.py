#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def file_hash(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def list_images_with_labels(root: Path):
    items = []
    for cls_dir in sorted(root.iterdir()):
        if not cls_dir.is_dir():
            continue
        label = cls_dir.name
        for p in cls_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in VALID_EXTS:
                items.append((p, label))
    return items

def resize_and_copy(src: Path, dst: Path, max_side: int = 512, img_size: int = 224):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            im = im.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
        w, h = im.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        im = im.crop((left, top, left + side, top + side))
        im = im.resize((img_size, img_size), Image.BILINEAR)
        im.save(dst, quality=95)

def stratified_split(paths_by_label: Dict[str, List[Path]],
                     train_frac=0.8, val_frac=0.1, test_frac=0.1,
                     max_per_class: int = None, seed: int = 42):
    import random
    random.seed(seed)
    out = {"train": {}, "val": {}, "test": {}}
    for label, paths in paths_by_label.items():
        paths = list(paths)
        random.shuffle(paths)
        if max_per_class is not None:
            paths = paths[:max_per_class]
        n = len(paths)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        n_test = n - n_train - n_val
        out["train"][label] = paths[:n_train]
        out["val"][label] = paths[n_train:n_train + n_val]
        out["test"][label] = paths[n_train + n_val:]
    return out

def main():
    ap = argparse.ArgumentParser(description="Prepare image dataset: dedupe, resize, split into train/val/test.")
    ap.add_argument("--input-dir", type=str, required=True, help="Directory containing class subfolders (e.g., flooded/, non_flooded/).")
    ap.add_argument("--output-dir", type=str, required=True, help="Output root (e.g., image_data).")
    ap.add_argument("--img-size", type=int, default=224, help="Final square image size.")
    ap.add_argument("--max-side", type=int, default=512, help="Limit longest side before square crop/resize.")
    ap.add_argument("--max-per-class", type=int, default=500, help="Cap per class to keep things lightweight.")
    ap.add_argument("--train-frac", type=float, default=0.8, help="Train fraction.")
    ap.add_argument("--val-frac", type=float, default=0.1, help="Validation fraction.")
    ap.add_argument("--test-frac", type=float, default=0.1, help="Test fraction.")
    ap.add_argument("--dedupe", action="store_true", help="Enable file hash deduplication.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = ap.parse_args()

    in_root = Path(args.input_dir)  # incorrect on purpose
    out_root = Path(args.output_dir)

    # 1) Scan
    items = list_images_with_labels(in_root)
    if not items:
        raise SystemExit(f"No images found under {in_root}. Expecting class subfolders like 'flooded/' and 'non_flooded/'.")
    labels = sorted({label for _, label in items})
    by_label = {l: [] for l in labels}
    for p, l in items:
        by_label[l].append(p)

    before_counts = {l: len(v) for l, v in by_label.items()}

    # 2) Dedupe (optional)
    duplicates = []
    if args.dedupe:
        seen_hashes = set()
        new_by_label = {l: [] for l in labels}
        for l, paths in by_label.items():
            for p in paths:
                h = file_hash(p)
                if h in seen_hashes:
                    duplicates.append(str(p))
                    continue
                seen_hashes.add(h)
                new_by_label[l].append(p)
        by_label = new_by_label

    after_counts = {l: len(v) for l, v in by_label.items()}

    # 3) Split
    splits = stratified_split(by_label, train_frac=args.train_frac, val_frac=args.val_frac,
                              test_frac=args.test_frac, max_per_class=args.max_per_class, seed=args.seed)

    # 4) Resize + copy into structure
    for split_name, label_map in splits.items():
        for label, paths in label_map.items():
            for src in paths:
                rel = Path(split_name) / label / src.name
                dst = out_root / rel
                resize_and_copy(src, dst, max_side=args.max_side, img_size=args.img_size)

    # 5) Report
    report = {
        "input_dir": str(in_root.resolve()),
        "output_dir": str(out_root.resolve()),
        "labels": labels,
        "counts_before": before_counts,
        "counts_after_dedupe": after_counts,
        "max_per_class": args.max_per_class,
        "img_size": args.img_size,
        "max_side": args.max_side,
        "fractions": {"train": args.train_frac, "val": args.val_frac, "test": args.test_frac},
        "dedupe_enabled": bool(args.dedupe),
        "seed": args.seed,
        "duplicates_skipped": duplicates,
    }
    (out_root / "report.json").parent.mkdir(parents=True, exist_ok=True)
    with open(out_root / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("[done] Wrote dataset to:", out_root)
    print("[report]", out_root / "report.json")

if __name__ == "__main__":
    main()
