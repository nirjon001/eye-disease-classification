# Methods Used in This Project

This file records **which method we use in each step** of the eye disease
classification project. It is updated as the scripts are completed.

## Dataset inspected on 2026-11-24

- Dataset folder: `Original Dataset`
- Class folders found: 10 (names are read from the folders, never hardcoded)
- Total images found: **5,335**
- This matches the documented number of original images.
- No augmented-image folders or augmented-style filenames were found.
  Therefore, this copy is treated as **originals only**. The 16,242 pre-made
  augmented images are not present and are ignored.
- augmented images were never downloaded
- `RAW_DATA_DIR` in `src/config.py` is the single place for the raw image path.

| Step | Script | Method used | Main output | Status |
|---|---|---|---|---|
| 1 | `scripts/01_explore_data.py` | Walk all class folders, open each image with Pillow, read size/file size, mark augmented files from folder/file names, keep originals only, plot class counts and samples | `data/splits/all_images.csv`, figures | Done — 5,335 originals, 0 augmented, 0 corrupted, all 2004x1690 |
| 2 | `scripts/02_find_exact_duplicates.py` | MD5 hash of file bytes; same hash means identical file; report cross-class hash conflicts and mark extra copies to drop | `outputs/reports/exact_duplicates.csv`, `data/splits/dedup_flags.csv` | Done — 470 duplicate groups (464 label conflicts, 6 same-class), 948 images marked drop=True |
| 3 | `scripts/03_find_lookalikes.py` | Method A: perceptual hash (pHash) with Hamming distance ≤ 8. Method B: frozen pretrained ResNet18 embeddings with cosine similarity ≥ 0.95. Build groups with connected components/union-find | `outputs/reports/lookalike_pairs.csv`, `data/splits/groups.csv` | Done — final choice is setting A (cosine ≥ 0.98); active groups now from `groups_strict_a.csv` |
| 4 | `scripts/04_make_splits.py` | Remove duplicates/conflicts, then use group-aware splitting (70/15/15) with class-balanced candidate selection and seed 42; run four leakage checks | `train.csv`, `val.csv`, `test.csv` | Done — 3,038/656/657, all 4 leakage checks PASS |
| 5 | `scripts/05_train_classical_ml.py` | Cache frozen ResNet18 features (512 values/image), train RBF SVM and Random Forest with balanced class weights | models + reports | Planned |
| 6 | `scripts/06_train_cnn.py` | Fine-tune torchvision models (simple_cnn, resnet18, resnet50, mobilenet_v2, alexnet, efficientnet_b0) with weighted loss, optional warm-up, early stopping, train-only augmentation, CUDA mixed precision, and per-epoch saving | best `.pth` + history plots | Script ready; local `--quick` test passed. History has train/val loss, top-1/3/5 accuracy, and macro F1 |
| 7 | `scripts/07_evaluate.py` | Load best weights, evaluate once on test: accuracy, macro F1, classification report, confusion matrix, wrong-prediction grid | final test report | Planned |
| 8 | `scripts/08_compare_models.py` | Collect all model results into one table and make a comparison bar chart | `model_comparison.csv` | Planned |

## Kaggle package helper

`scripts/09_prepare_kaggle_package.py` copies only the images listed in the
three split CSVs into `kaggle_package/eye_data/<class>/`, copies the split CSVs
into `kaggle_package/eye_data/splits/`, and zips the folder to
`kaggle_package/eye_data.zip`. Class folder names come from the CSV paths, not
from a hardcoded list. The original dataset is never modified.

## Portable paths for Kaggle

`src/config.py` now defines `DATA_ROOT` (from `EYE_DATA_ROOT`, otherwise the
local `Original Dataset` folder) and `OUTPUT_DIR` (from `EYE_OUTPUT_DIR`,
otherwise the local `outputs` folder). `src/dataset.py` opens images as
`DATA_ROOT / relative_path`. The three split CSVs store dataset-relative paths
with forward slashes, for example `Glaucoma/Glaucoma131.jpg`. Only the path
strings changed; split membership is identical. Re-running
`python scripts/04_make_splits.py --check-only` still gives PASS on all four
leakage checks.

## Step 3 band figures

`scripts/03_visualize_similarity_bands.py` samples random pairs from each
cosine band (0.90-0.95, 0.95-0.98, >=0.98) and each pHash band (0-2, 3-4,
5-6, 7-8). Sampling prefers pairs that add new class names so one common class
pair does not fill a whole figure. Outputs: `cosine_bands.png`,
`phash_bands.png`, and `band_pair_counts.csv`. `groups.csv` was not changed.

## Step 3 strict-setting analysis

`scripts/03_analyze_strict_lookalikes.py` added a pixel-level check: both images
are resized to 64x64 grayscale and Pearson correlation is computed, so a pair
must match in actual brightness pattern rather than only looking like a fundus
photo. Candidate groups were saved as `groups_strict_a.csv`, `_b.csv`, and
`_c.csv`.

Final user choice: **setting A (cosine >= 0.98)**. `groups_strict_a.csv` was
copied to the active `groups.csv`. Setting A had 233 pairs: 215 same-class pairs
were kept and grouped, and 18 different-class pairs were treated as label
conflicts. All 36 images in those 18 pairs were marked `drop=True` in
`dedup_flags.csv`. No images were deleted.

| Setting | Rule | Pairs | Largest candidate group |
|---|---|---:|---:|
| A | cosine >= 0.98 | 233 | 14 |
| B | pHash <= 4 AND cosine >= 0.95 | 1,216 | 164 |
| C | pHash <= 4 AND pixel correlation >= 0.95 | 6,763 | 321 |

## Step 3 threshold study

All threshold counts were computed over the 4,387 images that remain after
Step 2 drops. Extra cosine results are saved separately in
`outputs/reports/cosine_distribution/` so they are not mixed with the main
Step 3 reports.

| Method | Threshold | Pairs flagged |
|---|---:|---:|
| pHash distance | 4 | 6,814 |
| pHash distance | 6 | 29,179 |
| pHash distance | 8 | 85,829 |
| pHash distance | 10 | 196,400 |
| Cosine similarity | 0.90 | 1,073,728 |
| Cosine similarity | 0.95 | 42,068 |
| Cosine similarity | 0.98 | 233 |

Cosine buckets over all 9,621,141 unique pairs: `<0.70` = 814,213;
`0.70-0.80` = 3,014,364; `0.80-0.90` = 4,718,386; `>=0.90` = 1,073,728.

## Step 2 verification

- `scripts/02_verify_conflicts.py` compared 5 conflict groups with
  `filecmp.cmp(a, b, shallow=False)`. All 5 pairs were byte-for-byte identical.
- `outputs/figures/conflict_examples.png` shows 6 conflict pairs with class
  names as titles.
- Drop flags were not changed by this verification.

## Important rules followed

- Only original images are used.
- Class names always come from folder names.
- One seed (`SEED = 42`) is used everywhere.
- Device is chosen automatically: CUDA if available, otherwise CPU.
- Original images are never deleted or changed.
