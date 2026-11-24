"""
Step 4 of 8: make train/validation/test splits.

Input:  `all_images.csv`, `dedup_flags.csv`, `groups.csv`,
        `lookalike_strict_a_pairs.csv`.
Output: `data/splits/train.csv`, `val.csv`, `test.csv`,
        `data/splits/label_mapping.csv`,
        `outputs/reports/split_summary.csv`,
        `outputs/reports/split_leakage_checks.csv`.

Only images with drop=False are used. Groups stay together so no look-alike
can appear in two splits. The test set is only for the final report.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Allow running this file directly from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    PROJECT_ROOT,
    REPORTS_DIR,
    SEED,
    SPLITS_DIR,
    TEST_FRACTION,
    TRAIN_FRACTION,
    VAL_FRACTION,
)
from src.utils import (  # noqa: E402
    ensure_directories,
    set_seed,
    to_dataset_relative_string,
)

MIN_IMAGES_PER_CLASS_IN_VAL_TEST = 2
NUMBER_OF_CANDIDATE_SPLITS = 500


def load_usable_images():
    """Load the 4,351 images that are kept after Steps 2 and 3."""
    all_images = pd.read_csv(SPLITS_DIR / "all_images.csv")
    flags = pd.read_csv(SPLITS_DIR / "dedup_flags.csv")
    groups = pd.read_csv(SPLITS_DIR / "groups.csv")

    # Normalise stored paths so old project-relative CSVs and new portable CSVs
    # can be joined. Split membership is not affected by this string change.
    for frame in [all_images, flags, groups]:
        frame["path"] = frame["path"].map(to_dataset_relative_string)

    kept_flags = flags.loc[~flags["drop"], ["path", "md5"]]
    images = all_images.merge(kept_flags, on="path", how="inner")
    images = images.merge(groups[["path", "group_id"]], on="path", how="inner")
    images = images.reset_index(drop=True)

    print(f"Loaded {len(images)} usable image(s) with drop=False")
    if len(images) != 4351:
        print("WARNING: expected 4,351 usable images; check dedup_flags.csv and groups.csv")
    return images


def add_numeric_labels(images):
    """Add label numbers 0..9 using sorted folder/class names."""
    class_names = sorted(images["class"].unique())
    class_to_label = {class_name: index for index, class_name in enumerate(class_names)}
    images = images.copy()
    images["label"] = images["class"].map(class_to_label)

    mapping = pd.DataFrame(
        [{"label": index, "class": class_name} for class_name, index in class_to_label.items()]
    )
    return images, mapping, class_names


def choose_balanced_group_split(labels, groups, class_names, test_fraction, title):
    """
    Try many GroupShuffleSplit candidates and keep the most class-balanced one.

    GroupShuffleSplit guarantees that all images with the same group_id land in
    the same piece. We score each candidate by how close its class proportions
    are to the full dataset, and we reject pieces where a class has fewer than
    2 images.
    """
    number_of_images = len(labels)
    X = np.arange(number_of_images)
    overall_counts = np.bincount(labels, minlength=len(class_names))
    overall_proportions = overall_counts / number_of_images
    target_size = test_fraction * number_of_images

    splitter = GroupShuffleSplit(
        n_splits=NUMBER_OF_CANDIDATE_SPLITS,
        test_size=test_fraction,
        random_state=SEED,
    )

    best = None
    for keep_indices, test_indices in splitter.split(X, labels, groups):
        test_counts = np.bincount(labels[test_indices], minlength=len(class_names))
        if np.any(test_counts < MIN_IMAGES_PER_CLASS_IN_VAL_TEST):
            continue

        test_proportions = test_counts / len(test_indices)
        balance_score = float(np.abs(test_proportions - overall_proportions).sum())
        size_score = 0.5 * abs(len(test_indices) - target_size) / target_size
        score = balance_score + size_score

        if best is None or score < best["score"]:
            best = {
                "score": score,
                "keep": keep_indices,
                "test": test_indices,
                "max_class_difference": float(np.max(np.abs(test_proportions - overall_proportions))),
            }

    if best is None:
        raise RuntimeError(
            f"No valid split found for {title}. A class had fewer than "
            f"{MIN_IMAGES_PER_CLASS_IN_VAL_TEST} images in the candidate piece."
        )

    print(
        f"  {title}: {len(best['test'])} images, "
        f"largest class proportion difference {best['max_class_difference']:.3f}"
    )
    return best


def make_splits(images, class_names):
    """Create 70/15/15 train/val/test pieces without breaking groups."""
    labels = images["label"].to_numpy()
    groups = images["group_id"].to_numpy()

    print("Choosing test split (15%)...")
    trainval_choice = choose_balanced_group_split(
        labels, groups, class_names, TEST_FRACTION, "test"
    )

    trainval_indices = trainval_choice["keep"]
    test_indices = trainval_choice["test"]

    print("Choosing validation split (15% of all images)...")
    val_fraction_of_remainder = VAL_FRACTION / (TRAIN_FRACTION + VAL_FRACTION)
    val_choice = choose_balanced_group_split(
        labels[trainval_indices],
        groups[trainval_indices],
        class_names,
        val_fraction_of_remainder,
        "val",
    )

    val_indices = trainval_indices[val_choice["test"]]
    train_indices = trainval_indices[val_choice["keep"]]

    splits = {
        "train": images.iloc[train_indices].reset_index(drop=True),
        "val": images.iloc[val_indices].reset_index(drop=True),
        "test": images.iloc[test_indices].reset_index(drop=True),
    }
    return splits


def run_leakage_checks(splits, images, class_names):
    """Check paths, MD5 hashes, groups, and setting-A pairs across splits."""
    checks = []

    for column, check_name in [
        ("path", "No file path appears in two splits"),
        ("md5", "No MD5 hash appears in two splits"),
        ("group_id", "No group_id appears in two splits"),
    ]:
        sets = {name: set(frame[column]) for name, frame in splits.items()}
        overlaps = [
            sets["train"] & sets["val"],
            sets["train"] & sets["test"],
            sets["val"] & sets["test"],
        ]
        overlap_count = sum(len(group) for group in overlaps)
        checks.append(
            {
                "check": check_name,
                "result": "PASS" if overlap_count == 0 else "FAIL",
                "detail": f"{overlap_count} overlap(s)",
            }
        )

    path_to_split = {}
    for split_name, frame in splits.items():
        for path in frame["path"]:
            path_to_split[path] = split_name

    setting_a_pairs = pd.read_csv(REPORTS_DIR / "lookalike_strict_a_pairs.csv")
    for column in ["path_a", "path_b"]:
        setting_a_pairs[column] = setting_a_pairs[column].map(to_dataset_relative_string)

    cross_split_pairs = 0
    for _, pair in setting_a_pairs.iterrows():
        first_split = path_to_split.get(pair["path_a"])
        second_split = path_to_split.get(pair["path_b"])
        if first_split and second_split and first_split != second_split:
            cross_split_pairs += 1

    checks.append(
        {
            "check": "No setting-A look-alike pair crosses splits",
            "result": "PASS" if cross_split_pairs == 0 else "FAIL",
            "detail": f"{cross_split_pairs} cross-split pair(s)",
        }
    )

    return pd.DataFrame(checks)


def load_existing_splits():
    """Read the saved split files so we can re-check leakage without resplitting."""
    splits = {}
    for name in ["train", "val", "test"]:
        frame = pd.read_csv(SPLITS_DIR / f"{name}.csv")
        frame["path"] = frame["path"].map(to_dataset_relative_string)
        splits[name] = frame

    flags = pd.read_csv(SPLITS_DIR / "dedup_flags.csv")
    flags["path"] = flags["path"].map(to_dataset_relative_string)
    md5_by_path = flags.set_index("path")["md5"].to_dict()

    for frame in splits.values():
        frame["md5"] = frame["path"].map(md5_by_path)

    print(f"Loaded saved splits: " + ", ".join(f"{n}={len(f)}" for n, f in splits.items()))
    return splits


def print_class_counts(splits, class_names):
    """Print a class-by-split count table."""
    table = pd.DataFrame({"class": class_names})
    for split_name, frame in splits.items():
        counts = frame["class"].value_counts().reindex(class_names, fill_value=0)
        table[split_name] = counts.to_numpy()

    print("\nClass counts per split")
    print("-" * 72)
    print(table.to_string(index=False))
    print("-" * 72)
    print(
        "TOTAL "
        + "  ".join(f"{name}={len(frame)}" for name, frame in splits.items())
    )
    return table


def main():
    """Run Step 4 and write the split files only when every check passes."""
    parser = argparse.ArgumentParser(description="Create train/val/test splits.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print checks without writing split CSV files.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Re-run leakage checks on the saved split files only.",
    )
    args = parser.parse_args()

    print("Step 4/8: creating train/val/test splits...")
    set_seed(SEED)
    ensure_directories(SPLITS_DIR, REPORTS_DIR)

    if args.check_only:
        print("Re-checking leakage in the existing split files (no resplitting)...")
        splits = load_existing_splits()
        checks = run_leakage_checks(splits, None, None)
        print("\nLeakage checks")
        print("-" * 72)
        print(checks.to_string(index=False))
        checks.to_csv(REPORTS_DIR / "split_leakage_checks.csv", index=False)
        print(f"Saved: {REPORTS_DIR / 'split_leakage_checks.csv'}")
        if (checks["result"] == "FAIL").any():
            print("\nFAIL: a leakage check failed. Waiting for your instruction.")
        else:
            print("\nPASS: all four leakage checks still pass.")
        return

    images = load_usable_images()
    images, label_mapping, class_names = add_numeric_labels(images)
    splits = make_splits(images, class_names)

    print("\nSplit sizes:")
    for name, frame in splits.items():
        print(f"  {name}: {len(frame)} images ({100 * len(frame) / len(images):.1f}%)")

    count_table = print_class_counts(splits, class_names)

    checks = run_leakage_checks(splits, images, class_names)
    print("\nLeakage checks")
    print("-" * 72)
    print(checks.to_string(index=False))

    small_classes = []
    for split_name in ["val", "test"]:
        low = count_table[count_table[split_name] < MIN_IMAGES_PER_CLASS_IN_VAL_TEST]
        for _, row in low.iterrows():
            small_classes.append((split_name, row["class"], int(row[split_name])))

    failed = (checks["result"] == "FAIL").any()
    if failed or small_classes:
        print("\nSTOP: problems were found. No split files were written.")
        if failed:
            print("  At least one leakage check FAILED.")
        for split_name, class_name, count in small_classes:
            print(f"  {split_name} has only {count} image(s) for class: {class_name}")
        print("Waiting for your instruction.")
        return

    print("\nAll leakage checks PASSED and every class has >= 2 images in val and test.")

    summary_rows = [
        {
            "split": name,
            "images": len(frame),
            "percent": 100 * len(frame) / len(images),
        }
        for name, frame in splits.items()
    ]
    summary = pd.DataFrame(summary_rows)

    if args.dry_run:
        print("\nDRY RUN: no split files were written.")
        return

    for name, frame in splits.items():
        output_path = SPLITS_DIR / f"{name}.csv"
        frame[["path", "label", "group_id"]].to_csv(output_path, index=False)
        print(f"Saved: {output_path}")

    label_mapping.to_csv(SPLITS_DIR / "label_mapping.csv", index=False)
    print(f"Saved: {SPLITS_DIR / 'label_mapping.csv'}")
    summary.to_csv(REPORTS_DIR / "split_summary.csv", index=False)
    print(f"Saved: {REPORTS_DIR / 'split_summary.csv'}")
    checks.to_csv(REPORTS_DIR / "split_leakage_checks.csv", index=False)
    print(f"Saved: {REPORTS_DIR / 'split_leakage_checks.csv'}")

    print("\nStep 4 finished.")


if __name__ == "__main__":
    main()
