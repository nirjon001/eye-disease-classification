"""
Step 9 (helper): build a portable Kaggle image package.

Input:  `train.csv`, `val.csv`, `test.csv` and the images in `DATA_ROOT`.
Output: `kaggle_package/eye_data/<class>/<file name>` for every listed image,
        `kaggle_package/eye_data/splits/{train,val,test}.csv`,
        `kaggle_package/eye_data.zip`.

Only images named in the three split CSVs are copied. The original dataset is
read only: nothing there is deleted or moved.
"""

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

# Allow running this file directly from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_ROOT, IMAGE_EXTENSIONS, PROJECT_ROOT, SPLITS_DIR  # noqa: E402
from src.utils import to_dataset_relative_string  # noqa: E402

PACKAGE_DIR = PROJECT_ROOT / "kaggle_package"
EYE_DATA_DIR = PACKAGE_DIR / "eye_data"
SPLIT_NAMES = ["train", "val", "test"]


def load_split_paths(max_images=None):
    """Read the three split CSVs and return their unique image paths."""
    relative_paths = []
    row_counts = {}

    for split_name in SPLIT_NAMES:
        frame = pd.read_csv(SPLITS_DIR / f"{split_name}.csv")
        if max_images is not None:
            frame = frame.head(max_images)
        row_counts[split_name] = len(frame)
        for value in frame["path"]:
            relative_paths.append(to_dataset_relative_string(value))

    unique_paths = sorted(set(relative_paths))
    print("Images listed in each split:")
    for split_name, count in row_counts.items():
        print(f"  {split_name}: {count}")

    if len(unique_paths) != len(relative_paths):
        print(
            f"WARNING: {len(relative_paths) - len(unique_paths)} row(s) repeated "
            "a path; each image is copied only once."
        )
    print(f"Unique images to copy: {len(unique_paths)}")
    return unique_paths


def check_source_images(relative_paths):
    """Stop early if any CSV path does not exist under DATA_ROOT."""
    missing = []
    for relative_path in relative_paths:
        if not (DATA_ROOT / relative_path).is_file():
            missing.append(relative_path)

    if missing:
        preview = "\n".join(f"  {path}" for path in missing[:10])
        raise RuntimeError(
            f"{len(missing)} image(s) not found in {DATA_ROOT}. First few:\n{preview}"
        )
    print(f"All {len(relative_paths)} source images were found.")


def prepare_destination():
    """Create a fresh package folder so only the listed images are present."""
    if EYE_DATA_DIR.exists():
        print(f"Clearing old generated folder: {EYE_DATA_DIR}")
        shutil.rmtree(EYE_DATA_DIR)

    EYE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (EYE_DATA_DIR / "splits").mkdir(parents=True, exist_ok=True)


def destination_path(relative_path):
    """
    Build the package path from the CSV path.

    The first part of a relative path is the class folder (read from the real
    dataset layout, never hardcoded). The remaining parts are the file name.
    """
    parts = Path(relative_path).parts
    if len(parts) < 2:
        raise RuntimeError(f"Expected '<class>/<file>' in CSV, got: {relative_path}")
    return EYE_DATA_DIR.joinpath(*parts)


def copy_images(relative_paths):
    """Copy each listed image, keeping its class folder and file name."""
    total = len(relative_paths)
    for image_number, relative_path in enumerate(relative_paths, start=1):
        source = DATA_ROOT / relative_path
        destination = destination_path(relative_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

        if image_number % 500 == 0 or image_number == total:
            print(f"  copied {image_number}/{total} images")


def copy_split_files():
    """Copy the split CSVs into the package's `splits` folder."""
    for split_name in SPLIT_NAMES:
        source = SPLITS_DIR / f"{split_name}.csv"
        destination = EYE_DATA_DIR / "splits" / f"{split_name}.csv"
        shutil.copy2(source, destination)
        print(f"  copied {destination}")


def zip_package():
    """Zip the `eye_data` folder into `kaggle_package/eye_data.zip`."""
    zip_path = PACKAGE_DIR / "eye_data.zip"
    shutil.make_archive(
        base_name=str(PACKAGE_DIR / "eye_data"),
        format="zip",
        root_dir=str(PACKAGE_DIR),
        base_dir="eye_data",
    )
    print(f"Created: {zip_path}")
    return zip_path


def main():
    """Prepare the Kaggle image package and print its size."""
    parser = argparse.ArgumentParser(description="Prepare a Kaggle image package.")
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Use only the first N rows of each split (for a quick test run).",
    )
    args = parser.parse_args()

    print("Step 9/8 (helper): preparing Kaggle package...")
    if args.max_images is not None:
        print(f"TEST MODE: at most {args.max_images} image(s) per split.")

    relative_paths = load_split_paths(args.max_images)
    check_source_images(relative_paths)
    prepare_destination()

    print("\nCopying images...")
    copy_images(relative_paths)

    print("\nCopying split CSV files...")
    copy_split_files()

    print("\nZipping the package...")
    zip_path = zip_package()

    copied_count = sum(
        1
        for path in EYE_DATA_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    print("\nKaggle package ready:")
    print(f"  images copied: {copied_count}")
    print(f"  zip file: {zip_path}")
    print(f"  zip size: {zip_size_mb:.1f} MB")
    print("\nDone. The original dataset was not changed.")


if __name__ == "__main__":
    main()
