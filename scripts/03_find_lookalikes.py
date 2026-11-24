"""
Step 3 of 8: find look-alike images using two independent methods.

Input:  `data/splits/all_images.csv` and `data/splits/dedup_flags.csv`.
        Only images with drop=False are used (4,387 images in the full run).
Output: `outputs/reports/lookalike_pairs.csv`
        `outputs/reports/lookalike_conflicts.csv` (different-class pairs)
        `data/splits/groups.csv`
        `outputs/figures/lookalike_pairs.png`
        `outputs/reports/phash_threshold_counts.csv`
        `outputs/reports/cosine_distribution/cosine_threshold_counts.csv`
        `outputs/reports/cosine_distribution/cosine_similarity_buckets.csv`
        `outputs/figures/cosine_similarity_buckets.png`

Method A (pHash): a small visual fingerprint. Similar pictures have a small
Hamming distance between their hashes. Distance is lower-is-better.

Method B (ResNet18): a frozen ImageNet model gives a 512-number embedding per
image. Similar pictures have a high cosine similarity. Score is higher-is-better.

We only GROUP look-alikes here. No image is deleted or changed.
"""

import argparse
import sys
from pathlib import Path

import imagehash
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.spatial.distance import pdist, squareform
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

# Allow running this file directly from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    BATCH_SIZE,
    COSINE_BUCKET_EDGES,
    COSINE_DISTRIBUTION_DIR,
    COSINE_THRESHOLDS,
    DEEP_SIMILARITY_THRESHOLD,
    DEEP_EMBEDDING_SIZE,
    FIGURES_DIR,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NEIGHBORS_K,
    PHASH_DISTANCE_THRESHOLD,
    PHASH_HASH_SIZE,
    PHASH_THRESHOLDS,
    PROJECT_ROOT,
    REPORTS_DIR,
    SPLITS_DIR,
)
from src.dataset import EyeImageDataset  # noqa: E402
from src.utils import ensure_directories, get_device, save_dataframe, set_seed  # noqa: E402

PAIR_COLUMNS = [
    "path_a",
    "path_b",
    "method",
    "score",
    "same_class",
    "class_a",
    "class_b",
]


def load_remaining_images(max_images=None):
    """Load Step 1 images after removing the Step 2 drop=True rows."""
    all_images = pd.read_csv(SPLITS_DIR / "all_images.csv")
    flags = pd.read_csv(SPLITS_DIR / "dedup_flags.csv")

    kept_paths = flags.loc[~flags["drop"], ["path"]]
    remaining = all_images.merge(kept_paths, on="path", how="inner")
    remaining = remaining.reset_index(drop=True)

    if max_images is not None:
        remaining = remaining.head(max_images)

    print(f"Loaded {len(remaining)} image(s) after Step 2 drops")
    return remaining


def compute_phash_values(images):
    """Compute one 64-bit perceptual hash per image with the imagehash library."""
    hash_rows = []
    total = len(images)

    for index, image_row in images.iterrows():
        image_path = PROJECT_ROOT / image_row["path"]
        with Image.open(image_path) as image:
            phash = imagehash.phash(image, hash_size=PHASH_HASH_SIZE)
        hash_rows.append(phash.hash.flatten())  # A row of True/False bits.

        if (index + 1) % 500 == 0 or index + 1 == total:
            print(f"  pHash computed for {index + 1}/{total} images")

    return np.array(hash_rows, dtype=bool)


def phash_threshold_counts(hash_values, thresholds):
    """Count all unique image pairs whose pHash Hamming distance is <= each threshold."""
    if len(hash_values) < 2:
        empty = pd.DataFrame(columns=["threshold", "pair_count"])
        return empty, np.array([], dtype=np.uint8)

    bit_count = hash_values.shape[1]
    distances = pdist(hash_values, metric="hamming") * bit_count
    distances = np.rint(distances).astype(np.uint8)

    rows = [
        {"threshold": threshold, "pair_count": int(np.count_nonzero(distances <= threshold))}
        for threshold in thresholds
    ]
    return pd.DataFrame(rows), distances


def phash_pairs_at_threshold(distances, threshold):
    """Return (row, column, distance) for pHash pairs at or below the threshold."""
    if len(distances) == 0:
        return []

    distance_matrix = squareform(distances)
    close_pairs = np.triu(distance_matrix <= threshold, k=1)
    rows, columns = np.nonzero(close_pairs)
    return [(int(row), int(column), int(distance_matrix[row, column])) for row, column in zip(rows, columns)]


def build_resnet_embeddings(images):
    """Use frozen ResNet18 (without its classifier) to get a 512-number vector per image."""
    device = get_device()
    print(f"Device used for ResNet18 embeddings: {device}")

    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()  # Remove the 1,000-class head.
    model.to(device)
    model.eval()

    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    dataset = EyeImageDataset(images, transform=preprocess, project_root=PROJECT_ROOT)
    image_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    embedding_parts = []
    with torch.no_grad():
        for batch_number, (image_batch, _, _) in enumerate(image_loader, start=1):
            features = model(image_batch.to(device))
            embedding_parts.append(features.cpu().numpy())
            if batch_number % 10 == 0:
                print(f"  embedding batch {batch_number}")

    embeddings = np.vstack(embedding_parts).astype(np.float32)
    if embeddings.shape[1] != DEEP_EMBEDDING_SIZE:
        raise RuntimeError(f"Expected {DEEP_EMBEDDING_SIZE} embedding values, got {embeddings.shape[1]}")

    # Cosine similarity is easier once every vector has length 1.
    lengths = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / lengths
    print(f"Embeddings ready: {embeddings.shape}")
    return embeddings


def bucket_name(low, high):
    """Make a short label for one cosine-similarity interval."""
    if low is None:
        return f"<{high:.2f}"
    if high is None:
        return f">={low:.2f}"
    return f"{low:.2f}-{high:.2f}"


def cosine_counts_and_pairs(embeddings, default_threshold):
    """
    Count cosine results for every unique pair, in row batches to save memory.

    We also collect the pairs above the default threshold for the pair report.
    Bucket counts are exact over all n*(n-1)/2 pairs, not just nearest neighbours.
    """
    number_of_images = len(embeddings)
    total_pairs = number_of_images * (number_of_images - 1) // 2
    threshold_counts = {threshold: 0 for threshold in COSINE_THRESHOLDS}
    bucket_counts = {
        bucket_name(low, high): 0 for low, high in COSINE_BUCKET_EDGES
    }
    flagged_pairs = []

    batch_size = 256
    for start in range(0, number_of_images, batch_size):
        end = min(start + batch_size, number_of_images)
        scores = embeddings[start:end] @ embeddings.T  # (batch, all images)
        np.clip(scores, -1.0, 1.0, out=scores)

        row_numbers = np.arange(start, end)[:, None]
        column_numbers = np.arange(number_of_images)[None, :]
        upper_half = column_numbers > row_numbers  # Count each pair only once.

        valid_scores = scores[upper_half]
        for threshold in COSINE_THRESHOLDS:
            threshold_counts[threshold] += int(np.count_nonzero(valid_scores >= threshold))

        for low, high in COSINE_BUCKET_EDGES:
            name = bucket_name(low, high)
            if low is None:
                in_bucket = valid_scores < high
            elif high is None:
                in_bucket = valid_scores >= low
            else:
                in_bucket = (valid_scores >= low) & (valid_scores < high)
            bucket_counts[name] += int(np.count_nonzero(in_bucket))

        local_rows, columns = np.nonzero((scores >= default_threshold) & upper_half)
        for local_row, column in zip(local_rows, columns):
            flagged_pairs.append(
                (int(start + local_row), int(column), float(scores[local_row, column]))
            )

    threshold_rows = [
        {
            "threshold": threshold,
            "pair_count": count,
            "percent_of_all_pairs": 100.0 * count / total_pairs if total_pairs else 0.0,
        }
        for threshold, count in threshold_counts.items()
    ]
    bucket_rows = [
        {
            "bucket": name,
            "pair_count": count,
            "percent_of_all_pairs": 100.0 * count / total_pairs if total_pairs else 0.0,
        }
        for name, count in bucket_counts.items()
    ]

    return pd.DataFrame(threshold_rows), pd.DataFrame(bucket_rows), flagged_pairs


def make_pair_dataframe(index_pairs, images, method):
    """Turn (row_a, row_b, score) tuples into the required CSV columns."""
    rows = []
    for row_a, row_b, score in index_pairs:
        image_a = images.iloc[row_a]
        image_b = images.iloc[row_b]
        rows.append(
            {
                "path_a": image_a["path"],
                "path_b": image_b["path"],
                "method": method,
                "score": score,
                "same_class": image_a["class"] == image_b["class"],
                "class_a": image_a["class"],
                "class_b": image_b["class"],
            }
        )
    return pd.DataFrame(rows, columns=PAIR_COLUMNS)


def find_parent(parent, item):
    """Find the representative of an image using path compression."""
    while parent[item] != item:
        parent[item] = parent[parent[item]]
        item = parent[item]
    return item


def union_two(parent, first, second):
    """Join two images into one look-alike group."""
    root_first = find_parent(parent, first)
    root_second = find_parent(parent, second)
    if root_first != root_second:
        parent[root_second] = root_first


def build_groups(images, all_pairs):
    """
    Build connected components with a tiny union-find.

    If A~B and B~C, union(A,B) and union(B,C) put A, B, C under the same root,
    so they receive the same group_id. Images with no flagged pair get their
    own one-image group.
    """
    parent = {path: path for path in images["path"]}

    for _, pair in all_pairs.iterrows():
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


def plot_flagged_pairs(all_pairs, output_path, max_pairs=10):
    """Save up to 10 flagged pairs, preferring different-class examples."""
    if all_pairs.empty:
        figure, axis = plt.subplots(figsize=(8, 3))
        axis.text(0.5, 0.5, "No look-alike pairs were flagged", ha="center")
        axis.axis("off")
        figure.savefig(output_path, dpi=150)
        plt.close(figure)
        print(f"Saved: {output_path}")
        return

    chosen = all_pairs.copy()
    chosen["different_class"] = ~chosen["same_class"]
    chosen = chosen.sort_values(["different_class", "method"], ascending=[False, True])
    chosen = chosen.head(max_pairs)

    figure, axes = plt.subplots(nrows=len(chosen), ncols=2, figsize=(10, 2.7 * len(chosen)))
    axes = np.atleast_2d(axes)

    for row_number, (_, pair) in enumerate(chosen.iterrows()):
        for column_number, path_column in enumerate(["path_a", "path_b"]):
            axis = axes[row_number, column_number]
            image_path = PROJECT_ROOT / pair[path_column]
            with Image.open(image_path) as image:
                image.thumbnail((320, 220))
                axis.imshow(image)
            class_name = pair["class_a" if column_number == 0 else "class_b"]
            axis.set_title(f"{class_name}\n{pair['method']} {pair['score']:.3f}", fontsize=7)
            axis.axis("off")

    figure.suptitle("Examples of flagged look-alike pairs", fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"Saved: {output_path}")


def plot_bucket_counts(bucket_counts, output_path):
    """Save a bar chart of the cosine-similarity bucket counts."""
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(bucket_counts["bucket"], bucket_counts["pair_count"], color="#4C78A8")
    axis.set_xlabel("Cosine similarity interval")
    axis.set_ylabel("Number of image pairs")
    axis.set_title("Distribution of cosine similarity over all image pairs")
    for index, value in enumerate(bucket_counts["pair_count"]):
        axis.text(index, value, str(value), ha="center", va="bottom", fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"Saved: {output_path}")


def print_different_class_pairs(all_pairs, max_to_print=20):
    """Print a short list and save the full list of different-class look-alikes."""
    conflicts = all_pairs[~all_pairs["same_class"]].copy()
    print(f"\nLook-alike pairs with different classes: {len(conflicts)}")

    if conflicts.empty:
        return

    conflicts = conflicts.sort_values(["method", "score"], ascending=[True, False])
    print("First 20 examples (full list is in outputs/reports/lookalike_conflicts.csv):")
    print(
        conflicts.head(max_to_print)[
            ["path_a", "path_b", "method", "score", "class_a", "class_b"]
        ].to_string(index=False)
    )


def main():
    """Run Step 3 and write all look-alike reports, groups, and figures."""
    parser = argparse.ArgumentParser(description="Find look-alike eye images.")
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Use only the first N remaining images (for a quick test run).",
    )
    args = parser.parse_args()

    print("Step 3/8: finding look-alike images...")
    set_seed(42)
    ensure_directories(SPLITS_DIR, REPORTS_DIR, FIGURES_DIR, COSINE_DISTRIBUTION_DIR)

    images = load_remaining_images(args.max_images)
    if len(images) < 2:
        raise RuntimeError("At least two images are needed for look-alike detection.")

    print("\nMethod A: perceptual hashing (pHash)")
    phash_values = compute_phash_values(images)
    phash_counts, phash_distances = phash_threshold_counts(phash_values, PHASH_THRESHOLDS)
    save_dataframe(phash_counts, REPORTS_DIR / "phash_threshold_counts.csv")
    print("\nPairs flagged by pHash distance (lower distance = more similar):")
    print(phash_counts.to_string(index=False))

    phash_index_pairs = phash_pairs_at_threshold(phash_distances, PHASH_DISTANCE_THRESHOLD)
    print(f"Default pHash threshold {PHASH_DISTANCE_THRESHOLD} flags {len(phash_index_pairs)} pair(s)")

    print("\nMethod B: pretrained ResNet18 embeddings and cosine similarity")
    embeddings = build_resnet_embeddings(images)
    cosine_counts, cosine_buckets, cosine_index_pairs = cosine_counts_and_pairs(
        embeddings, DEEP_SIMILARITY_THRESHOLD
    )
    save_dataframe(cosine_counts, COSINE_DISTRIBUTION_DIR / "cosine_threshold_counts.csv")
    save_dataframe(cosine_buckets, COSINE_DISTRIBUTION_DIR / "cosine_similarity_buckets.csv")
    print("\nPairs flagged by cosine similarity (higher score = more similar):")
    print(cosine_counts.to_string(index=False))
    print("\nCosine similarity distribution for ALL pairs (separate report):")
    print(cosine_buckets.to_string(index=False))
    print(
        f"Default cosine threshold {DEEP_SIMILARITY_THRESHOLD} flags "
        f"{len(cosine_index_pairs)} pair(s)"
    )

    phash_pairs = make_pair_dataframe(phash_index_pairs, images, "pHash")
    cosine_pairs = make_pair_dataframe(cosine_index_pairs, images, "cosine")
    all_pairs = pd.concat([phash_pairs, cosine_pairs], ignore_index=True)

    save_dataframe(all_pairs, REPORTS_DIR / "lookalike_pairs.csv")
    conflicts = all_pairs[~all_pairs["same_class"]].copy()
    save_dataframe(conflicts, REPORTS_DIR / "lookalike_conflicts.csv")

    groups = build_groups(images, all_pairs)
    save_dataframe(groups, SPLITS_DIR / "groups.csv")

    plot_flagged_pairs(all_pairs, FIGURES_DIR / "lookalike_pairs.png")
    plot_bucket_counts(cosine_buckets, FIGURES_DIR / "cosine_similarity_buckets.png")
    print_different_class_pairs(all_pairs)

    print("\nSummary:")
    print(f"  images used: {len(images)}")
    print(f"  pHash pairs at distance <= {PHASH_DISTANCE_THRESHOLD}: {len(phash_pairs)}")
    print(f"  cosine pairs at similarity >= {DEEP_SIMILARITY_THRESHOLD}: {len(cosine_pairs)}")
    print(f"  total flagged rows: {len(all_pairs)}")
    print(f"  different-class flagged rows: {len(conflicts)}")
    print(f"  groups created: {groups['group_id'].nunique()}")
    print(f"  nearest-neighbour setting kept in config for reference: k={NEIGHBORS_K}")
    print("\nStep 3 finished. No images were deleted.")


if __name__ == "__main__":
    main()
