"""
Step 2 of 8: find exact duplicate images with MD5 hashes.

Input:  `data/splits/all_images.csv` made by Step 1 (original images only).
Output: `outputs/reports/exact_duplicates.csv`
        `data/splits/dedup_flags.csv`

An MD5 hash is a short "fingerprint" of the file bytes. Two files with the
same hash are byte-for-byte identical. This script only reads images; it never
deletes or changes them.
"""

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

# Allow running this file directly from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROJECT_ROOT, REPORTS_DIR, SPLITS_DIR  # noqa: E402
from src.utils import ensure_directories, save_dataframe, set_seed  # noqa: E402

DUPLICATE_COLUMNS = [
    "hash",
    "count",
    "classes",
    "paths",
    "label_conflict",
    "kept_path",
]
FLAG_COLUMNS = [
    "path",
    "class",
    "md5",
    "keep",
    "drop",
    "drop_reason",
    "label_conflict",
]


def md5_hash_file(path):
    """Return the MD5 fingerprint of one file by reading it in small pieces."""
    hasher = hashlib.md5()
    with path.open("rb") as image_file:
        while True:
            chunk = image_file.read(1024 * 1024)  # 1 MB at a time.
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def load_original_images(max_images=None):
    """Load the original-image list from Step 1, optionally limited for testing."""
    images_csv = SPLITS_DIR / "all_images.csv"
    images = pd.read_csv(images_csv)

    required_columns = {"path", "class"}
    missing = required_columns - set(images.columns)
    if missing:
        raise RuntimeError(f"{images_csv} is missing columns: {sorted(missing)}")

    if max_images is not None:
        images = images.head(max_images)

    return images


def hash_all_images(images):
    """Add an `md5` column to every image row and print progress."""
    rows = []
    total = len(images)

    for index, image_row in images.iterrows():
        image_path = PROJECT_ROOT / image_row["path"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Image from all_images.csv not found: {image_path}")

        rows.append(
            {
                "path": image_row["path"],
                "class": image_row["class"],
                "md5": md5_hash_file(image_path),
            }
        )

        if (index + 1) % 500 == 0 or index + 1 == total:
            print(f"  hashed {index + 1}/{total} images")

    return pd.DataFrame(rows)


def build_reports(hashed_images):
    """
    Make the duplicate report and the per-image keep/drop flags.

    Rules:
    - Same hash, same class: keep the first path (alphabetically), drop the rest.
    - Same hash, different classes: label conflict, drop every copy because we
      cannot know which class label is correct.
    """
    duplicate_rows = []
    flag_rows = []

    # sort=False keeps the original order, which makes the report easier to read.
    for md5_value, group in hashed_images.groupby("md5", sort=False):
        group = group.sort_values("path")
        classes = sorted(group["class"].unique())
        is_conflict = len(classes) > 1
        is_duplicate = len(group) > 1
        kept_path = "" if is_conflict else group.iloc[0]["path"]

        for _, image_row in group.iterrows():
            should_keep = not is_conflict and image_row["path"] == kept_path
            if is_conflict:
                reason = "label_conflict"
            elif should_keep:
                reason = ""
            else:
                reason = "exact_duplicate"

            flag_rows.append(
                {
                    "path": image_row["path"],
                    "class": image_row["class"],
                    "md5": md5_value,
                    "keep": should_keep,
                    "drop": not should_keep,
                    "drop_reason": reason,
                    "label_conflict": is_conflict,
                }
            )

        if is_duplicate:
            duplicate_rows.append(
                {
                    "hash": md5_value,
                    "count": len(group),
                    "classes": " | ".join(classes),
                    "paths": " | ".join(group["path"]),
                    "label_conflict": is_conflict,
                    "kept_path": kept_path,
                }
            )

    duplicates = pd.DataFrame(duplicate_rows, columns=DUPLICATE_COLUMNS)
    flags = pd.DataFrame(flag_rows, columns=FLAG_COLUMNS)
    return duplicates, flags


def print_conflicts(duplicates):
    """Print cross-class hash conflicts loudly, as required by AGENTS.md."""
    conflicts = duplicates[duplicates["label_conflict"]]
    if conflicts.empty:
        print("Label conflicts (same hash, different class): none")
        return

    print("\n" + "!" * 70)
    print("LABEL CONFLICTS FOUND: THE SAME FILE BYTES APPEAR IN DIFFERENT CLASSES!")
    print("All copies in these groups are marked drop=True and will be removed in Step 4.")
    for _, conflict_row in conflicts.iterrows():
        print(f"  hash: {conflict_row['hash']}")
        print(f"  classes: {conflict_row['classes']}")
        print(f"  paths: {conflict_row['paths']}")
    print("!" * 70 + "\n")


def main():
    """Run Step 2 and write the duplicate and flag files."""
    parser = argparse.ArgumentParser(description="Find exact duplicate images.")
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Check only the first N images (for a quick test run).",
    )
    args = parser.parse_args()

    print("Step 2/8: finding exact duplicate files...")
    set_seed(42)
    ensure_directories(SPLITS_DIR, REPORTS_DIR)

    images = load_original_images(args.max_images)
    print(f"Loaded {len(images)} original image(s) from data/splits/all_images.csv")

    hashed_images = hash_all_images(images)
    duplicates, flags = build_reports(hashed_images)

    duplicates_path = REPORTS_DIR / "exact_duplicates.csv"
    flags_path = SPLITS_DIR / "dedup_flags.csv"
    save_dataframe(duplicates, duplicates_path)
    save_dataframe(flags, flags_path)

    print_conflicts(duplicates)

    conflict_count = int(duplicates["label_conflict"].sum())
    duplicate_group_count = len(duplicates)
    extra_copies = int((flags["drop_reason"] == "exact_duplicate").sum())
    conflict_images = int(flags["label_conflict"].sum())
    dropped_images = int(flags["drop"].sum())

    print("\nSummary:")
    print(f"  images checked: {len(hashed_images)}")
    print(f"  unique MD5 hashes: {hashed_images['md5'].nunique()}")
    print(f"  duplicate hash groups: {duplicate_group_count}")
    print(f"  extra copies to drop (same class): {extra_copies}")
    print(f"  conflicting hash groups: {conflict_count}")
    print(f"  conflicting images to drop: {conflict_images}")
    print(f"  total images marked drop=True: {dropped_images}")
    print(f"  images remaining after drops: {len(flags) - dropped_images}")

    print("\nStep 2 finished.")


if __name__ == "__main__":
    main()
