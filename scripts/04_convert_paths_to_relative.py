"""
Helper after Step 4: make the saved split paths portable for Kaggle.

Input:  `data/splits/train.csv`, `val.csv`, `test.csv`.
Output: the same three files with `path` stored relative to the dataset root,
        always using forward slashes (for example `Glaucoma/Glaucoma131.jpg`).

Split membership never changes: only the path strings are rewritten.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow running this file directly from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROJECT_ROOT, SPLITS_DIR  # noqa: E402
from src.utils import to_dataset_relative_string  # noqa: E402

SPLIT_NAMES = ["train", "val", "test"]


def convert_split_file(csv_path, dry_run):
    """Rewrite the path column in one split CSV without changing its rows."""
    frame = pd.read_csv(csv_path)
    original_paths = frame["path"].copy()

    other_columns = [column for column in frame.columns if column != "path"]
    original_rows = frame[other_columns].copy()

    frame["path"] = original_paths.map(to_dataset_relative_string)
    changed = int((original_paths != frame["path"]).sum())

    if not frame[other_columns].equals(original_rows):
        raise RuntimeError(f"Split membership changed unexpectedly in {csv_path}")

    print(f"  {csv_path.name}: {len(frame)} rows, {changed} path strings changed")
    if not dry_run:
        frame.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

    return frame


def main():
    """Convert all three split files, then show a few example paths."""
    parser = argparse.ArgumentParser(description="Make split paths portable.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing any file.",
    )
    args = parser.parse_args()

    print("Converting split paths to dataset-relative form...")
    if args.dry_run:
        print("DRY RUN: no CSV files will be changed.")

    for name in SPLIT_NAMES:
        convert_split_file(SPLITS_DIR / f"{name}.csv", args.dry_run)

    train = pd.read_csv(SPLITS_DIR / "train.csv")
    print("\nFirst 3 lines of train.csv:")
    print(train.head(3).to_csv(index=False).strip())

    print("\nDone. Split membership was not changed.")


if __name__ == "__main__":
    main()
