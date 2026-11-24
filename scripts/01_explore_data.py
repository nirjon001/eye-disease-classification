"""
Step 1 of 8: explore the eye disease images.

Input:  images in `config.RAW_DATA_DIR` (one folder per class).
Output: `data/splits/all_images.csv` (originals only),
        `outputs/reports/image_inventory_all.csv`,
        `outputs/reports/corrupted_files.csv`,
        `outputs/figures/class_counts.png`,
        `outputs/figures/samples.png`.

This script only reads images. It never changes or deletes original images.
"""

import argparse
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

from src.config import (  # noqa: E402
    EXPECTED_ORIGINAL_COUNT,
    FIGURES_DIR,
    IMAGE_EXTENSIONS,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    REPORTS_DIR,
    SPLITS_DIR,
)
from src.utils import ensure_directories, save_dataframe, set_seed, to_project_relative_string  # noqa: E402


def looks_augmented(path):
    """
    Mark whether an image is a pre-made augmented copy.

    Only original images were downloaded, so this is always False.
    We keep the function so the CSV column has a clear meaning.
    """
    return False


def inspect_one_image(path):
    """
    Open one image with Pillow and return basic information about it.

    `img.load()` forces Pillow to really read the pixels, so truncated files
    are caught here instead of later during training.
    """
    try:
        with Image.open(path) as image:
            image.load()
            return {
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "corrupted": False,
                "error": "",
            }
    except Exception as error:
        return {
            "width": None,
            "height": None,
            "mode": "",
            "corrupted": True,
            "error": str(error),
        }


def build_inventory(max_images_per_class=None):
    """Walk every class folder and build one row per image file."""
    class_folders = sorted(
        folder for folder in RAW_DATA_DIR.iterdir() if folder.is_dir()
    )
    if not class_folders:
        raise RuntimeError(f"No class folders found in: {RAW_DATA_DIR}")

    rows = []
    for class_folder in class_folders:
        class_name = class_folder.name  # Real class name comes from the folder.
        image_paths = sorted(
            path
            for path in class_folder.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if max_images_per_class is not None:
            image_paths = image_paths[:max_images_per_class]

        print(f"  {class_name}: {len(image_paths)} image(s)")

        for image_path in image_paths:
            image_info = inspect_one_image(image_path)
            rows.append(
                {
                    "path": to_project_relative_string(image_path, PROJECT_ROOT),
                    "class": class_name,
                    "width": image_info["width"],
                    "height": image_info["height"],
                    "mode": image_info["mode"],
                    "file_size_bytes": image_path.stat().st_size,
                    "is_augmented": looks_augmented(image_path),
                    "corrupted": image_info["corrupted"],
                    "error": image_info["error"],
                }
            )

    return pd.DataFrame(rows)


def plot_class_counts(originals, output_path):
    """Save a horizontal bar chart showing how many originals each class has."""
    counts = originals.groupby("class").size().sort_values()
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.barh(counts.index, counts.values, color="#4C78A8")
    axis.set_xlabel("Number of original images")
    axis.set_title("Original images per class")
    for index, value in enumerate(counts.values):
        axis.text(value + 5, index, str(value), va="center", fontsize=8)
    # Long class names need extra space on the left, so we set margins by hand.
    figure.subplots_adjust(left=0.55, right=0.95, top=0.92, bottom=0.10)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"Saved: {output_path}")


def plot_sample_grid(originals, output_path, images_per_class=3):
    """Save a small grid with a few example images from every class."""
    classes = sorted(originals["class"].unique())
    figure, axes = plt.subplots(
        nrows=images_per_class,
        ncols=len(classes),
        figsize=(2.1 * len(classes), 2.2 * images_per_class),
    )

    for column, class_name in enumerate(classes):
        class_images = originals[originals["class"] == class_name].head(images_per_class)
        for row in range(images_per_class):
            axis = axes[row, column]
            if row >= len(class_images):
                axis.axis("off")
                continue

            image_path = PROJECT_ROOT / class_images.iloc[row]["path"]
            try:
                with Image.open(image_path) as image:
                    image.thumbnail((180, 150))
                    axis.imshow(image)
            except Exception:
                axis.text(0.5, 0.5, "unreadable", ha="center", va="center", fontsize=7)
            axis.axis("off")
            if row == 0:
                axis.set_title(class_name, fontsize=7)

    figure.suptitle("Example original images (3 per class)", fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"Saved: {output_path}")


def main():
    """Run the full Step 1 exploration and write all output files."""
    parser = argparse.ArgumentParser(description="Explore the eye disease dataset.")
    parser.add_argument(
        "--max-images-per-class",
        type=int,
        default=None,
        help="Use only the first N images per class (for a quick test run).",
    )
    args = parser.parse_args()

    print("Step 1/8: exploring the dataset...")
    set_seed(42)
    ensure_directories(SPLITS_DIR, FIGURES_DIR, REPORTS_DIR)

    inventory = build_inventory(args.max_images_per_class)
    if inventory.empty:
        raise RuntimeError("No images were found. Check RAW_DATA_DIR in src/config.py.")

    inventory_path = REPORTS_DIR / "image_inventory_all.csv"
    save_dataframe(inventory, inventory_path)

    corrupted = inventory[inventory["corrupted"]].copy()
    corrupted_path = REPORTS_DIR / "corrupted_files.csv"
    save_dataframe(corrupted, corrupted_path)
    if corrupted.empty:
        print("Corrupted files: none")
    else:
        print(f"WARNING: {len(corrupted)} corrupted file(s). See {corrupted_path}")

    augmented_count = int(inventory["is_augmented"].sum())
    originals = inventory[
        (~inventory["is_augmented"]) & (~inventory["corrupted"])
    ].copy()

    original_columns = ["path", "class", "width", "height", "file_size_bytes"]
    originals_path = SPLITS_DIR / "all_images.csv"
    save_dataframe(originals[original_columns], originals_path)

    print(f"Images found: {len(inventory)}")
    print(f"Pre-made augmented files ignored: {augmented_count}")
    print(f"Original images kept: {len(originals)}")

    if args.max_images_per_class is None and len(originals) != EXPECTED_ORIGINAL_COUNT:
        print(
            f"WARNING: expected {EXPECTED_ORIGINAL_COUNT} original images, "
            f"found {len(originals)}. Please check the dataset folder."
        )

    print("\nOriginal images per class:")
    print(originals.groupby("class").size().to_string())

    valid_sizes = originals.dropna(subset=["width", "height"])
    if not valid_sizes.empty:
        print("\nImage size summary (width x height):")
        print(
            f"  width:  min={int(valid_sizes['width'].min())}, "
            f"max={int(valid_sizes['width'].max())}"
        )
        print(
            f"  height: min={int(valid_sizes['height'].min())}, "
            f"max={int(valid_sizes['height'].max())}"
        )
        size_counts = valid_sizes.groupby(["width", "height"]).size().sort_values(
            ascending=False
        )
        print("  most common sizes:")
        for (width, height), count in size_counts.head(10).items():
            print(f"    {int(width)}x{int(height)}: {count} image(s)")

    plot_class_counts(originals, FIGURES_DIR / "class_counts.png")
    plot_sample_grid(originals, FIGURES_DIR / "samples.png")

    print("\nStep 1 finished.")


if __name__ == "__main__":
    main()
