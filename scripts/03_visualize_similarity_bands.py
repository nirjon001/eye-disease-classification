"""
Visual threshold helper after Step 3: look at random pairs by score band.

Input:  the same 4,387 images used in Step 3, plus Step 3 pHash pairs.
Output: `outputs/figures/cosine_bands.png`
        `outputs/figures/phash_bands.png`
        `outputs/reports/band_pair_counts.csv`

This helper does NOT change `groups.csv`, drop flags, or any image.
"""

import argparse
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

# Allow running this file directly from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    BANDS_FOR_FIGURE,
    BATCH_SIZE,
    BAND_SAMPLE_POOL_SIZE,
    COSINE_BANDS,
    FIGURES_DIR,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MODELS_DIR,
    PHASH_BANDS,
    PROJECT_ROOT,
    REPORTS_DIR,
    SPLITS_DIR,
)
from src.dataset import EyeImageDataset  # noqa: E402
from src.utils import get_device, set_seed  # noqa: E402

RANDOM_SEED = 42


def load_remaining_images(max_images=None):
    """Load the images that remain after Step 2 drops."""
    all_images = pd.read_csv(SPLITS_DIR / "all_images.csv")
    flags = pd.read_csv(SPLITS_DIR / "dedup_flags.csv")
    kept_paths = flags.loc[~flags["drop"], ["path"]]
    remaining = all_images.merge(kept_paths, on="path", how="inner").reset_index(drop=True)
    if max_images is not None:
        remaining = remaining.head(max_images)
    return remaining


def compute_or_load_embeddings(images, use_cache):
    """Compute frozen ResNet18 embeddings, loading a matching cache when possible."""
    embedding_path = MODELS_DIR / "lookalike_embeddings.npy"
    embedding_paths_path = MODELS_DIR / "lookalike_embedding_paths.csv"

    if use_cache and embedding_path.exists() and embedding_paths_path.exists():
        cached_paths = pd.read_csv(embedding_paths_path)["path"].tolist()
        if cached_paths == images["path"].tolist():
            print(f"Loaded cached embeddings: {embedding_path}")
            return np.load(embedding_path)

    print("Computing ResNet18 embeddings for band sampling...")
    device = get_device()
    print(f"Device used: {device}")

    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
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
    dataset = EyeImageDataset(images, transform=preprocess)
    image_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    parts = []
    with torch.no_grad():
        for batch_number, (image_batch, _, _) in enumerate(image_loader, start=1):
            features = model(image_batch.to(device))
            parts.append(features.cpu().numpy())
            if batch_number % 20 == 0:
                print(f"  embedding batch {batch_number}")

    embeddings = np.vstack(parts).astype(np.float32)
    lengths = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / lengths

    if use_cache:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        np.save(embedding_path, embeddings)
        pd.DataFrame({"path": images["path"]}).to_csv(embedding_paths_path, index=False)
        print(f"Saved embedding cache: {embedding_path}")

    return embeddings


def diverse_choice(pool, number_wanted):
    """
    Pick random pairs that cover as many different classes as possible.

    We first take a random pool from the band. Then each pick prefers pairs that
    add new class names and a new class combination. This stops one common
    class pair (for example Glaucoma vs Healthy) from filling the whole figure.
    """
    if pool.empty:
        return pool

    remaining = pool.reset_index(drop=True)
    chosen_rows = []
    covered_classes = set()
    used_combinations = set()
    generator = random.Random(RANDOM_SEED)

    while len(chosen_rows) < number_wanted and not remaining.empty:
        best_position = None
        best_key = None

        for position, row in remaining.iterrows():
            classes = {row["class_a"], row["class_b"]}
            new_classes = len(classes - covered_classes)
            combination = tuple(sorted(classes))
            new_combination = 1 if combination not in used_combinations else 0
            # Random tie-break keeps the choice fair when scores are equal.
            key = (new_classes, new_combination, generator.random())
            if best_key is None or key > best_key:
                best_key = key
                best_position = position

        row = remaining.loc[best_position]
        chosen_rows.append(row)
        covered_classes.update({row["class_a"], row["class_b"]})
        used_combinations.add(tuple(sorted({row["class_a"], row["class_b"]})))
        remaining = remaining.drop(index=best_position)

    return pd.DataFrame(chosen_rows).reset_index(drop=True)


def make_pool_from_indices(indices, rows, columns, images, score_values, rng):
    """Turn selected condensed-pair indices into a class-labelled pair table."""
    pool_size = min(BAND_SAMPLE_POOL_SIZE, len(indices))
    chosen_indices = rng.choice(indices, size=pool_size, replace=False)

    records = []
    for pair_index in chosen_indices:
        row_number = int(rows[pair_index])
        column_number = int(columns[pair_index])
        image_a = images.iloc[row_number]
        image_b = images.iloc[column_number]
        records.append(
            {
                "path_a": image_a["path"],
                "path_b": image_b["path"],
                "class_a": image_a["class"],
                "class_b": image_b["class"],
                "score": float(score_values[pair_index]),
            }
        )
    return pd.DataFrame(records)


def sample_cosine_bands(embeddings, images):
    """Count all cosine band pairs and choose random, class-diverse examples."""
    number_of_images = len(embeddings)
    rows, columns = np.triu_indices(number_of_images, k=1)

    scores = embeddings @ embeddings.T
    np.clip(scores, -1.0, 1.0, out=scores)
    condensed_scores = scores[rows, columns]
    del scores

    rng = np.random.default_rng(RANDOM_SEED)
    band_results = {}
    count_records = []

    for band_name, low, high in COSINE_BANDS:
        in_band = np.flatnonzero((condensed_scores >= low) & (condensed_scores < high))
        count_records.append(
            {
                "metric": "cosine",
                "band": band_name,
                "pair_count": int(len(in_band)),
            }
        )
        pool = make_pool_from_indices(in_band, rows, columns, images, condensed_scores, rng)
        band_results[band_name] = diverse_choice(pool, BANDS_FOR_FIGURE)

    return band_results, pd.DataFrame(count_records)


def sample_phash_bands(images):
    """Count pHash band pairs from Step 3 and choose class-diverse examples."""
    step3_pairs = pd.read_csv(REPORTS_DIR / "lookalike_pairs.csv")
    phash_pairs = step3_pairs[step3_pairs["method"] == "pHash"].copy()

    # In quick test mode, keep only pairs whose images are in the test subset.
    allowed_paths = set(images["path"])
    phash_pairs = phash_pairs[
        phash_pairs["path_a"].isin(allowed_paths)
        & phash_pairs["path_b"].isin(allowed_paths)
    ].copy()

    path_to_class = images.set_index("path")["class"].to_dict()
    phash_pairs["class_a"] = phash_pairs["path_a"].map(path_to_class)
    phash_pairs["class_b"] = phash_pairs["path_b"].map(path_to_class)

    rng = np.random.default_rng(RANDOM_SEED)
    band_results = {}
    count_records = []

    for band_name, low, high in PHASH_BANDS:
        in_band = phash_pairs[(phash_pairs["score"] >= low) & (phash_pairs["score"] <= high)]
        count_records.append(
            {
                "metric": "pHash",
                "band": band_name,
                "pair_count": int(len(in_band)),
            }
        )
        if in_band.empty:
            band_results[band_name] = pd.DataFrame()
            continue

        pool_size = min(BAND_SAMPLE_POOL_SIZE, len(in_band))
        pool = in_band.sample(n=pool_size, random_state=RANDOM_SEED)
        band_results[band_name] = diverse_choice(pool, BANDS_FOR_FIGURE)

    return band_results, pd.DataFrame(count_records)


def plot_band_grid(band_results, output_path, title, score_label, integer_score=False):
    """Save one figure with one row per band and six pairs per row."""
    band_names = list(band_results.keys())
    number_of_columns = BANDS_FOR_FIGURE * 2

    figure, axes = plt.subplots(
        nrows=len(band_names),
        ncols=number_of_columns,
        figsize=(22, 2.8 * len(band_names)),
    )
    axes = np.atleast_2d(axes)

    for row_number, band_name in enumerate(band_names):
        chosen = band_results[band_name]
        axes[row_number, 0].text(
            -0.18,
            0.5,
            band_name,
            transform=axes[row_number, 0].transAxes,
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

        for column_number in range(number_of_columns):
            axis = axes[row_number, column_number]
            pair_number = column_number // 2
            side = column_number % 2

            if pair_number >= len(chosen):
                axis.axis("off")
                continue

            pair = chosen.iloc[pair_number]
            path_column = "path_a" if side == 0 else "path_b"
            class_column = "class_a" if side == 0 else "class_b"

            with Image.open(PROJECT_ROOT / pair[path_column]) as image:
                image.thumbnail((170, 130))
                axis.imshow(image)

            if integer_score:
                score_text = f"{score_label}={int(pair['score'])}"
            else:
                score_text = f"{score_label}={pair['score']:.3f}"
            axis.set_title(f"{pair[class_column]}\n{score_text}", fontsize=6)
            axis.axis("off")

    figure.suptitle(title, fontsize=12)
    figure.tight_layout(rect=(0.03, 0, 1, 0.96))
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"Saved: {output_path}")


def main():
    """Create both band figures and print the exact pair counts."""
    parser = argparse.ArgumentParser(description="Visualize similarity score bands.")
    parser.add_argument("--max-images", type=int, default=None, help="Quick test mode.")
    parser.add_argument("--suffix", type=str, default="", help="Suffix for test outputs.")
    args = parser.parse_args()

    print("Visualizing similarity bands...")
    set_seed(RANDOM_SEED)
    ensure_dirs = (FIGURES_DIR, REPORTS_DIR, MODELS_DIR)
    for folder in ensure_dirs:
        folder.mkdir(parents=True, exist_ok=True)

    images = load_remaining_images(args.max_images)
    print(f"Using {len(images)} remaining images")

    cosine_bands, cosine_counts = sample_cosine_bands(
        compute_or_load_embeddings(images, use_cache=args.max_images is None), images
    )
    phash_bands, phash_counts = sample_phash_bands(images)

    counts = pd.concat([cosine_counts, phash_counts], ignore_index=True)
    print("\nPair counts in each band")
    print("-" * 46)
    print(counts.to_string(index=False))
    counts.to_csv(REPORTS_DIR / f"band_pair_counts{args.suffix}.csv", index=False)
    print(f"Saved: {REPORTS_DIR / f'band_pair_counts{args.suffix}.csv'}")

    plot_band_grid(
        cosine_bands,
        FIGURES_DIR / f"cosine_bands{args.suffix}.png",
        "Cosine similarity bands — 6 random pairs per band",
        "cos",
    )
    plot_band_grid(
        phash_bands,
        FIGURES_DIR / f"phash_bands{args.suffix}.png",
        "pHash distance bands — 6 random pairs per band",
        "pHash",
        integer_score=True,
    )

    print("\nDone. groups.csv was NOT changed.")


if __name__ == "__main__":
    main()
