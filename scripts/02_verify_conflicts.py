"""
Verification helper for Step 2: check label-conflict groups.

Input:  `outputs/reports/exact_duplicates.csv`
        `data/splits/dedup_flags.csv`
Output: printed byte comparisons and class-count table,
        `outputs/figures/conflict_examples.png`

This helper only reads the reports and images. It does NOT change any drop flags.
"""

import filecmp
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

# Allow running this file directly from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FIGURES_DIR, PROJECT_ROOT, REPORTS_DIR, SPLITS_DIR  # noqa: E402
from src.utils import ensure_directories  # noqa: E402

NUMBER_TO_COMPARE = 5
NUMBER_FOR_FIGURE = 6


def load_conflicts():
    """Load only the duplicate groups that have more than one class."""
    duplicates = pd.read_csv(REPORTS_DIR / "exact_duplicates.csv")
    conflicts = duplicates[duplicates["label_conflict"]].copy()
    if conflicts.empty:
        raise RuntimeError("No label conflicts found in exact_duplicates.csv")
    return conflicts


def split_group(conflict_row):
    """Return the class names and relative paths stored in one report row."""
    classes = [text.strip() for text in str(conflict_row["classes"]).split("|")]
    paths = [text.strip() for text in str(conflict_row["paths"]).split("|")]
    return classes, paths


def verify_bytes(conflicts, count):
    """Compare the first two files in `count` conflict groups byte by byte."""
    print("\nByte-by-byte check with filecmp.cmp(a, b, shallow=False)")
    print("-" * 72)

    for index, (_, conflict_row) in enumerate(conflicts.head(count).iterrows(), start=1):
        classes, paths = split_group(conflict_row)
        path_a = PROJECT_ROOT / paths[0]
        path_b = PROJECT_ROOT / paths[1]
        same_bytes = filecmp.cmp(path_a, path_b, shallow=False)

        print(f"Group {index}  hash={conflict_row['hash']}")
        print(f"  {classes[0]}: {path_a}")
        print(f"  {classes[1]}: {path_b}")
        print(f"  identical bytes: {same_bytes}")
        print()

        if not same_bytes:
            raise RuntimeError("A reported conflict did not match byte for byte.")


def plot_conflict_pairs(conflicts, count, output_path):
    """Save a grid of `count` conflict pairs, with class names above each image."""
    figure, axes = plt.subplots(nrows=count, ncols=2, figsize=(10, 2.8 * count))

    for row_index, (_, conflict_row) in enumerate(conflicts.head(count).iterrows()):
        classes, paths = split_group(conflict_row)
        for column_index in range(2):
            axis = axes[row_index, column_index]
            image_path = PROJECT_ROOT / paths[column_index]
            with Image.open(image_path) as image:
                image.thumbnail((320, 240))
                axis.imshow(image)
            axis.set_title(classes[column_index], fontsize=8)
            axis.axis("off")

    figure.suptitle("Exact-duplicate label conflicts (identical bytes, different folders)", fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"Saved: {output_path}")


def print_class_counts():
    """Print per-class image counts before and after applying drop flags."""
    flags = pd.read_csv(SPLITS_DIR / "dedup_flags.csv")

    summary = flags.groupby("class").agg(
        before=("path", "size"),
        dropped=("drop", "sum"),
    )
    summary["after"] = summary["before"] - summary["dropped"]
    summary["percent_removed"] = 100.0 * summary["dropped"] / summary["before"]
    summary = summary.reset_index()

    print("\nClass counts before vs after Step 2 drops")
    print("-" * 72)
    print(
        summary.to_string(
            index=False,
            formatters={"percent_removed": "{:.1f}%".format},
        )
    )

    total_before = int(summary["before"].sum())
    total_dropped = int(summary["dropped"].sum())
    total_after = int(summary["after"].sum())
    total_percent = 100.0 * total_dropped / total_before
    print("-" * 72)
    print(
        f"TOTAL before={total_before}  dropped={total_dropped}  "
        f"after={total_after}  percent_removed={total_percent:.1f}%"
    )


def main():
    """Run the Step 2 verification checks."""
    print("Verifying Step 2 label conflicts...")
    ensure_directories(FIGURES_DIR)

    conflicts = load_conflicts()
    verify_bytes(conflicts, NUMBER_TO_COMPARE)
    plot_conflict_pairs(conflicts, NUMBER_FOR_FIGURE, FIGURES_DIR / "conflict_examples.png")
    print_class_counts()

    print("\nVerification finished. Drop flags were not changed.")


if __name__ == "__main__":
    main()
