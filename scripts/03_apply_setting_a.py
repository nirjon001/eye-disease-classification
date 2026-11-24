"""
Step 3 follow-up: apply chosen setting A to groups and conflict drops.

Input:  `data/splits/groups_strict_a.csv`
        `outputs/reports/lookalike_strict_a_pairs.csv`
        `data/splits/dedup_flags.csv`
Output: `data/splits/groups.csv` (copy of the chosen candidate)
        `outputs/reports/lookalike_a_conflicts.csv`
        updated `data/splits/dedup_flags.csv`

No images are deleted. Same-class look-alike pairs stay grouped and kept.
"""

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

# Allow running this file directly from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROJECT_ROOT, REPORTS_DIR, SPLITS_DIR  # noqa: E402


def load_setting_a_pairs():
    """Load the 233 setting-A pairs (cosine similarity >= 0.98)."""
    pairs_path = REPORTS_DIR / "lookalike_strict_a_pairs.csv"
    pairs = pd.read_csv(pairs_path)
    required = {"path_a", "path_b", "class_a", "class_b", "cosine_similarity"}
    missing = required - set(pairs.columns)
    if missing:
        raise RuntimeError(f"{pairs_path} is missing columns: {sorted(missing)}")
    return pairs


def copy_chosen_groups(dry_run):
    """Copy the chosen candidate groups into the active groups.csv."""
    source = SPLITS_DIR / "groups_strict_a.csv"
    target = SPLITS_DIR / "groups.csv"
    if dry_run:
        print(f"DRY RUN: would copy {source} -> {target}")
    else:
        shutil.copyfile(source, target)
        print(f"Copied: {source} -> {target}")
    return pd.read_csv(source)


def print_conflicts(conflicts):
    """Print and save the different-class setting-A pairs."""
    pair_counts = (
        conflicts.assign(
            class_pair=conflicts.apply(
                lambda row: " | ".join(sorted([row["class_a"], row["class_b"]])),
                axis=1,
            )
        )
        .groupby("class_pair")
        .size()
        .sort_values(ascending=False)
    )

    print(f"\nSetting-A pairs with different classes: {len(conflicts)}")
    print("Count per class pair:")
    print(pair_counts.to_string())


def mark_conflict_drops(flags, conflict_paths):
    """Mark every image in a cross-class setting-A pair as drop=True."""
    updated = flags.copy()
    mask = updated["path"].isin(conflict_paths)

    old_reasons = updated.loc[mask, "drop_reason"].fillna("").astype(str)
    new_reasons = old_reasons.apply(
        lambda reason: f"{reason};lookalike_label_conflict" if reason else "lookalike_label_conflict"
    )
    updated.loc[mask, "drop_reason"] = new_reasons
    updated.loc[mask, "drop"] = True
    updated.loc[mask, "keep"] = False
    updated.loc[mask, "label_conflict"] = True

    return updated, int(mask.sum())


def print_class_counts(flags, title):
    """Print per-class kept counts and how many were dropped."""
    summary = flags.groupby("class").agg(
        before=("path", "size"),
        dropped=("drop", "sum"),
    )
    summary["after"] = summary["before"] - summary["dropped"]
    summary["percent_removed"] = 100.0 * summary["dropped"] / summary["before"]
    summary = summary.reset_index()

    print(f"\n{title}")
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
    print("-" * 72)
    print(
        f"TOTAL before={total_before}  dropped={total_dropped}  "
        f"after={total_after}  percent_removed={100.0 * total_dropped / total_before:.1f}%"
    )


def print_group_distribution(groups, kept_paths, title):
    """Print group sizes, optionally after removing newly dropped images."""
    if kept_paths is not None:
        groups = groups[groups["path"].isin(kept_paths)]

    sizes = groups.groupby("group_id").size()
    print(f"\n{title}")
    print("-" * 52)
    print(f"  number_of_groups: {len(sizes)}")
    print(f"  largest_group: {int(sizes.max())}")
    print(f"  size_1: {int((sizes == 1).sum())}")
    print(f"  size_2: {int((sizes == 2).sum())}")
    print(f"  size_3_5: {int(sizes.between(3, 5).sum())}")
    print(f"  size_over_5: {int((sizes > 5).sum())}")
    print(f"  images_in_groups: {int(sizes.sum())}")


def main():
    """Apply setting A, update conflict drops, and print all summaries."""
    parser = argparse.ArgumentParser(description="Apply chosen setting A.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print results without changing any CSV file.",
    )
    args = parser.parse_args()

    print("Applying chosen setting A (cosine >= 0.98)...")
    if args.dry_run:
        print("DRY RUN: no CSV files will be changed.")

    groups = copy_chosen_groups(args.dry_run)
    pairs = load_setting_a_pairs()

    conflicts = pairs[~pairs["same_class"]].copy()
    same_class_pairs = pairs[pairs["same_class"]].copy()
    print_conflicts(conflicts)
    print(f"\nSame-class pairs kept: {len(same_class_pairs)}")

    conflict_paths = sorted(set(conflicts["path_a"]) | set(conflicts["path_b"]))
    print(f"Unique images in different-class pairs: {len(conflict_paths)}")

    flags = pd.read_csv(SPLITS_DIR / "dedup_flags.csv")
    print_class_counts(flags, "Class counts before setting-A conflict drops")

    updated_flags, newly_dropped = mark_conflict_drops(flags, conflict_paths)
    print_class_counts(updated_flags, "Class counts after setting-A conflict drops")
    print(f"Newly marked drop=True: {newly_dropped}")

    kept_paths = set(updated_flags.loc[~updated_flags["drop"], "path"])
    print_group_distribution(groups, None, "Full groups.csv distribution (all images)")
    print_group_distribution(groups, kept_paths, "FINAL group distribution (drop=False images only)")

    if not args.dry_run:
        conflicts.to_csv(REPORTS_DIR / "lookalike_a_conflicts.csv", index=False)
        print(f"Saved: {REPORTS_DIR / 'lookalike_a_conflicts.csv'}")
        updated_flags.to_csv(SPLITS_DIR / "dedup_flags.csv", index=False)
        print(f"Updated: {SPLITS_DIR / 'dedup_flags.csv'}")
        print(f"Updated: {SPLITS_DIR / 'groups.csv'}")
    else:
        print("DRY RUN complete. No CSV files were changed.")

    print("\nDone. No images were deleted.")


if __name__ == "__main__":
    main()
