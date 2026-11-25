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
| 5 | full Kaggle notebook | Cache frozen ResNet18 features (512 values/image), train RBF SVM and Random Forest with balanced class weights | SVM/RF test scores | Done in `notebooks/eyediseaseclassification1 (2).ipynb`; standalone `05_...` script not written |
| 6 | `scripts/06_train_cnn.py` | Fine-tune nine models (simple_cnn, resnet18, resnet50, mobilenet_v2, alexnet, efficientnet_b0, densenet121, convnext_tiny, swin_t) with weighted loss, optional warm-up, early stopping, train-only augmentation, CUDA mixed precision, and per-epoch saving. Default 50 epochs | best `.pth` + history plots | Done on Kaggle — 9 models, 50 epochs, no failures. Weights remain in Kaggle Output |
| 7 | full Kaggle notebook | Evaluate once on test: accuracy, macro F1, top-1/3/5, classification report, confusion matrix | per-model test reports and matrices | Done in the same notebook; final test set touched once |
| 8 | full Kaggle notebook | Collect all model results into one table and make a comparison bar chart | `summary_table.csv`, `model_comparison.png` | Done in the same notebook |

## Final result

The full Kaggle run is summarized in `outputs/reports/FINAL_REPORT.md`.
The model was chosen by validation macro F1. `swin_t` won with validation macro F1
0.799; its test accuracy was 0.849, test macro F1 0.818, top-3 0.945, and top-5
0.953. `resnet50` had slightly higher accuracy (0.857) but lower macro F1 (0.815).

Extracted result files are in `outputs/reports/kaggle_run/` and
`outputs/figures/kaggle_run/`. The original downloaded bundle is
`notebooks/results_bundle_seed42.zip`.

## Kaggle training batches

The notebook runs all nine models sequentially in two groups so one GPU is
never overloaded: light (`simple_cnn`, `mobilenet_v2`, `resnet18`,
`efficientnet_b0`) first, then heavy (`alexnet`, `densenet121`, `resnet50`,
`convnext_tiny`, `swin_t`). Both flags can be toggled to split the work into
two Commits.

## Kaggle package helper

`scripts/09_prepare_kaggle_package.py` copies only the images listed in the
three split CSVs into `kaggle_package/eye_data/<class>/`, copies the split CSVs
and `label_mapping.csv` into `kaggle_package/eye_data/splits/`, and zips the folder to
`kaggle_package/eye_data.zip`. Class folder names come from the CSV paths, not
from a hardcoded list. The original dataset is never modified.

Kaggle rejects `[` and `]` in uploaded names, so the one affected class folder
is written as `Central Serous Chorioretinopathy Color Fundus` in the package.
The packaged split CSVs use the same safe name. `src/dataset.py` also tries this
safe name automatically, so the repo CSVs still work unchanged on Kaggle.

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
