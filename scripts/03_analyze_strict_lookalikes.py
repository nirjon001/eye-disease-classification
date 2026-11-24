"""
Helper after Step 3: study stricter look-alike settings.

Input:  `outputs/reports/lookalike_pairs.csv`
        `data/splits/all_images.csv`
        `data/splits/dedup_flags.csv`
        `data/splits/groups.csv` (current groups, read only)
Output: strict pair reports and 3 random-pair figures in `outputs/`,
        candidate group files in `data/splits/groups_strict_*.csv`.

This helper does NOT change `data/splits/groups.csv` and never deletes images.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# Allow running this file directly from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    FIGURES_DIR,
    PIXEL_CHECK_SIZE,
    PIXEL_CORRELATION_THRESHOLD,
    PROJECT_ROOT,
    REPORTS_DIR,
    SPLITS_DIR,
)
from src.utils import ensure_directories  # noqa: E402

RANDOM_SEED = 42
NUMBER_FOR_FIGURE = 10


def load_remaining_images():
    """Load the same 4,387 images used by Step 3."""
    all_images = pd.read_csv(SPLITS_DIR / "all_images.csv")
    flags = pd.read_csv(SPLITS_DIR / "dedup_flags.csv")
    kept_paths = flags.loc[~flags["drop"], ["path"]]
    return all_images.merge(kept_paths, on="path", how="inner").reset_index(drop=True)


def group_size_distribution(groups):
    """Count group sizes in the bins requested by the user."""
    sizes = groups.groupby("group_id").size()
    distribution = {
        "number_of_groups": len(sizes),
        "largest_group": int(sizes.max()),
        "size_1": int((sizes == 1).sum()),
        "size_2": int((sizes == 2).sum()),
        "size_3_5": int(sizes.between(3, 5).sum()),
        "size_over_5": int((sizes > 5).sum()),
        "images_in_groups": int(sizes.sum()),
    }
    return sizes, distribution


def print_distribution(title, distribution):
    """Print one group-size distribution as a small table."""
    print(f"\n{title}")
    print("-" * 52)
    for name, value in distribution.items():
        print(f"  {name}: {value}")


def load_canonical_pairs(max_pairs=None):
    """Load pHash and cosine pairs with paths sorted into a consistent order."""
    step3_pairs = pd.read_csv(REPORTS_DIR / "lookalike_pairs.csv")
    path_to_class = (
        pd.read_csv(SPLITS_DIR / "all_images.csv")
        .set_index("path")["class"]
        .to_dict()
    )

    rows = []
    for _, pair in step3_pairs.iterrows():
        first_path, second_path = sorted([pair["path_a"], pair["path_b"]])
        rows.append(
            {
                "path_a": first_path,
                "path_b": second_path,
                "method": pair["method"],
                "score": float(pair["score"]),
            }
        )

    canonical = pd.DataFrame(rows)
    phash_pairs = canonical[canonical["method"] == "pHash"].drop_duplicates(
        ["path_a", "path_b"]
    )
    cosine_pairs = canonical[canonical["method"] == "cosine"].drop_duplicates(
        ["path_a", "path_b"]
    )

    if max_pairs is not None:
        phash_pairs = phash_pairs.head(max_pairs)
        cosine_pairs = cosine_pairs.head(max_pairs)

    return phash_pairs, cosine_pairs, path_to_class


def add_classes(pairs, path_to_class):
    """Add class names and a same_class flag to a pair table."""
    result = pairs.copy()
    result["class_a"] = result["path_a"].map(path_to_class)
    result["class_b"] = result["path_b"].map(path_to_class)
    result["same_class"] = result["class_a"] == result["class_b"]
    return result


def load_pixel_vector(relative_path, cache):
    """Load one image as a small 64x64 grayscale array and cache it."""
    if relative_path not in cache:
        image_path = PROJECT_ROOT / relative_path
        with Image.open(image_path) as image:
            gray = image.convert("L").resize(
                (PIXEL_CHECK_SIZE, PIXEL_CHECK_SIZE), Image.Resampling.LANCZOS
            )
            cache[relative_path] = np.asarray(gray, dtype=np.float32)
    return cache[relative_path]


def pixel_correlation(path_a, path_b, cache):
    """
    Compare the ACTUAL picture content at the same pixel locations.

    pHash and ResNet18 can both say "this is another fundus photo" even when
    the disease is different. Pearson correlation on small grayscale images
    asks a stricter question: are the bright/dark patterns arranged the same
    way? A constant image has no usable correlation, so we return 0.0.
    """
    vector_a = load_pixel_vector(path_a, cache)
    vector_b = load_pixel_vector(path_b, cache)

    if vector_a.std() == 0 or vector_b.std() == 0:
        return 0.0

    return float(np.corrcoef(vector_a.ravel(), vector_b.ravel())[0, 1])


def make_strict_a(cosine_pairs, path_to_class):
    """Setting A: cosine similarity >= 0.98."""
    result = cosine_pairs[cosine_pairs["score"] >= 0.98].copy()
    result = result.rename(columns={"score": "cosine_similarity"})
    result["score_text"] = result["cosine_similarity"].map(lambda value: f"cos={value:.3f}")
    result["method"] = "cosine>=0.98"
    return add_classes(result, path_to_class)


def make_strict_b(phash_pairs, cosine_pairs, path_to_class):
    """Setting B: pHash distance <= 4 AND cosine similarity >= 0.95."""
    merged = phash_pairs.merge(
        cosine_pairs,
        on=["path_a", "path_b"],
        suffixes=("_phash", "_cosine"),
    )
    result = merged[
        (merged["score_phash"] <= 4) & (merged["score_cosine"] >= 0.95)
    ].copy()
    result = result.rename(
        columns={"score_phash": "phash_distance", "score_cosine": "cosine_similarity"}
    )
    result["score_text"] = result.apply(
        lambda row: f"pH={int(row['phash_distance'])}, cos={row['cosine_similarity']:.3f}",
        axis=1,
    )
    result["method"] = "pHash<=4 AND cosine>=0.95"
    return add_classes(result, path_to_class)


def make_strict_c(phash_pairs, path_to_class, suffix):
    """Setting C: pHash distance <= 4 AND pixel correlation >= 0.95."""
    candidates = phash_pairs[phash_pairs["score"] <= 4].copy()
    candidates = candidates.rename(columns={"score": "phash_distance"})

    pixel_cache = {}
    correlations = []
    for index, (_, pair) in enumerate(candidates.iterrows(), start=1):
        correlations.append(
            pixel_correlation(pair["path_a"], pair["path_b"], pixel_cache)
        )
        if index % 1000 == 0 or index == len(candidates):
            print(f"  pixel correlations computed: {index}/{len(candidates)}")

    candidates["pixel_correlation"] = correlations
    save_dataframe_local(
        candidates.drop(columns=["method"]),
        REPORTS_DIR / f"phash4_pixel_correlations{suffix}.csv",
    )

    result = candidates[
        candidates["pixel_correlation"] >= PIXEL_CORRELATION_THRESHOLD
    ].copy()
    result["score_text"] = result.apply(
        lambda row: f"pH={int(row['phash_distance'])}, corr={row['pixel_correlation']:.3f}",
        axis=1,
    )
    result["method"] = "pHash<=4 AND pixel>=0.95"
    return add_classes(result, path_to_class)


def save_dataframe_local(dataframe, output_path):
    """Write a CSV and create its parent folder if needed."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


def plot_random_pairs(pairs, output_path, title):
    """Save 10 randomly chosen flagged pairs with their scores in the titles."""
    if pairs.empty:
        figure, axis = plt.subplots(figsize=(8, 3))
        axis.text(0.5, 0.5, "No pairs flagged by this setting", ha="center")
        axis.axis("off")
        figure.savefig(output_path, dpi=150)
        plt.close(figure)
        print(f"Saved: {output_path}")
        return

    chosen = pairs.sample(n=min(NUMBER_FOR_FIGURE, len(pairs)), random_state=RANDOM_SEED)
    figure, axes = plt.subplots(
        nrows=len(chosen), ncols=2, figsize=(10, 2.9 * len(chosen))
    )
    axes = np.atleast_2d(axes)

    for row_number, (_, pair) in enumerate(chosen.iterrows()):
        for column_number, path_column in enumerate(["path_a", "path_b"]):
            axis = axes[row_number, column_number]
            with Image.open(PROJECT_ROOT / pair[path_column]) as image:
                image.thumbnail((320, 220))
                axis.imshow(image)
            class_name = pair["class_a" if column_number == 0 else "class_b"]
            axis.set_title(f"{class_name}\n{pair['score_text']}", fontsize=7)
            axis.axis("off")

    figure.suptitle(f"{title}: {len(chosen)} random flagged pairs", fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"Saved: {output_path}")


def find_parent(parent, item):
    """Find the group representative using path compression."""
    while parent[item] != item:
        parent[item] = parent[parent[item]]
        item = parent[item]
    return item


def union_two(parent, first, second):
    """Join two images into the same candidate group."""
    root_first = find_parent(parent, first)
    root_second = find_parent(parent, second)
    if root_first != root_second:
        parent[root_second] = root_first


def build_candidate_groups(images, pairs):
    """Build union-find groups for one strict setting."""
    parent = {path: path for path in images["path"]}
    for _, pair in pairs.iterrows():
        union_two(parent, pair["path_a"], pair["path_b"])

    paths_by_group = {}
    for path in images["path"]:
        root = find_parent(parent, path)
        paths_by_group.setdefault(root, []).append(path)

    rows = []
    for group_number, (_, paths) in enumerate(sorted(paths_by_group.items()), start=1):
        group_id = f"group_{group_number:05d}"
        for path in paths:
            rows.append({"path": path, "group_id": group_id})

    return pd.DataFrame(rows, columns=["path", "group_id"])


def main():
    """Run the strict-setting comparison without changing the active groups."""
    parser = argparse.ArgumentParser(description="Study strict look-alike settings.")
    parser.add_argument("--max-pairs", type=int, default=None, help="Quick test mode.")
    parser.add_argument("--suffix", type=str, default="", help="Suffix for test outputs.")
    args = parser.parse_args()

    print("Analyzing stricter look-alike settings...")
    ensure_directories(FIGURES_DIR, REPORTS_DIR)

    current_groups = pd.read_csv(SPLITS_DIR / "groups.csv")
    _, current_distribution = group_size_distribution(current_groups)
    print_distribution("CURRENT groups.csv (Step 3 default settings)", current_distribution)
    print(
        "\nThe 427 groups include images that had NO flagged look-alike. "
        "Those images are singletons (size 1), so every remaining image "
        "has a group_id for the later leakage check."
    )

    phash_pairs, cosine_pairs, path_to_class = load_canonical_pairs(args.max_pairs)
    if args.max_pairs is not None:
        print(f"TEST MODE: using at most {args.max_pairs} pairs per method")

    strict_a = make_strict_a(cosine_pairs, path_to_class)
    strict_b = make_strict_b(phash_pairs, cosine_pairs, path_to_class)
    strict_c = make_strict_c(phash_pairs, path_to_class, args.suffix)

    count_rows = [
        {"setting": "a) cosine >= 0.98", "pairs": len(strict_a)},
        {"setting": "b) pHash <= 4 AND cosine >= 0.95", "pairs": len(strict_b)},
        {"setting": "c) pHash <= 4 AND pixel corr >= 0.95", "pairs": len(strict_c)},
    ]
    counts = pd.DataFrame(count_rows)
    print("\nStrict setting pair counts")
    print("-" * 52)
    print(counts.to_string(index=False))
    save_dataframe_local(counts, REPORTS_DIR / f"strict_setting_counts{args.suffix}.csv")

    settings = {
        "a": strict_a,
        "b": strict_b,
        "c": strict_c,
    }
    titles = {
        "a": "Setting A",
        "b": "Setting B",
        "c": "Setting C",
    }

    images = load_remaining_images()
    for name, strict_pairs in settings.items():
        save_dataframe_local(
            strict_pairs,
            REPORTS_DIR / f"lookalike_strict_{name}_pairs{args.suffix}.csv",
        )
        plot_random_pairs(
            strict_pairs,
            FIGURES_DIR / f"lookalike_strict_{name}{args.suffix}.png",
            titles[name],
        )

        candidate_groups = build_candidate_groups(images, strict_pairs)
        save_dataframe_local(
            candidate_groups,
            SPLITS_DIR / f"groups_strict_{name}{args.suffix}.csv",
        )
        _, distribution = group_size_distribution(candidate_groups)
        print_distribution(f"CANDIDATE groups for setting {name}", distribution)

    print("\nIMPORTANT: data/splits/groups.csv was NOT changed.")
    print("Choose setting a, b, or c, and I will copy the matching candidate groups.")
    print("Analysis finished. No images were deleted.")


if __name__ == "__main__":
    main()
