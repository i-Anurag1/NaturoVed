"""
Dataset preparation utilities.

Expected directory layout AFTER you run `organize_dataset()` (or do it
manually per data/README.md):

    data/
      raw/                        <- original PlantVillage download, untouched
      processed/
        train/
          Tomato___Healthy/*.jpg
          Tomato___Early_Blight/*.jpg
          ...
        val/
          <same class subfolders>
        test/
          <same class subfolders>

We use a class-preserving 70/15/15 split. Splitting is done by COPYING
filenames into split folders (not physically duplicating pixel data
across environments) so `tf.keras.utils.image_dataset_from_directory`
can be used directly with zero custom Dataset code.
"""
import os
import shutil
import random
from pathlib import Path
from app.ml.labels import CLASS_NAMES

RANDOM_SEED = 42
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def organize_dataset(raw_dir: str, processed_dir: str, class_names=None, seed: int = RANDOM_SEED):
    """
    Split a flat `raw_dir/<ClassName>/*.jpg` dataset into
    `processed_dir/{train,val,test}/<ClassName>/*.jpg`.

    Only folders whose name is in `class_names` are processed, so you
    can point `raw_dir` at the FULL PlantVillage download and this
    function will automatically cherry-pick just the classes this MVP
    supports (see app/ml/labels.py).
    """
    class_names = class_names or CLASS_NAMES
    random.seed(seed)
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"raw_dir '{raw_dir}' does not exist. See data/README.md for download steps."
        )

    summary = {}
    for cls in class_names:
        src = raw_dir / cls
        if not src.exists():
            print(f"[WARN] class folder not found, skipping: {src}")
            continue

        images = [p for p in src.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        random.shuffle(images)

        n = len(images)
        n_train = int(n * SPLIT_RATIOS["train"])
        n_val = int(n * SPLIT_RATIOS["val"])

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        for split_name, split_files in splits.items():
            dst_dir = processed_dir / split_name / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                shutil.copy2(f, dst_dir / f.name)

        summary[cls] = {k: len(v) for k, v in splits.items()}
        print(f"{cls}: total={n} -> train={len(splits['train'])} "
              f"val={len(splits['val'])} test={len(splits['test'])}")

    return summary


def dataset_stats(processed_dir: str):
    """Print per-class, per-split image counts. Useful sanity check
    before training and a nice table to paste into the project report."""
    processed_dir = Path(processed_dir)
    stats = {}
    for split in ["train", "val", "test"]:
        split_dir = processed_dir / split
        if not split_dir.exists():
            continue
        stats[split] = {}
        for cls_dir in sorted(split_dir.iterdir()):
            if cls_dir.is_dir():
                count = len(list(cls_dir.glob("*")))
                stats[split][cls_dir.name] = count
    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Organize raw PlantVillage-style dataset into train/val/test splits")
    parser.add_argument("--raw_dir", default="../../data/raw", help="Path to raw dataset (class-per-folder)")
    parser.add_argument("--processed_dir", default="../../data/processed", help="Output path for split dataset")
    args = parser.parse_args()

    organize_dataset(args.raw_dir, args.processed_dir)
    print("\nFinal dataset stats:")
    print(dataset_stats(args.processed_dir))
