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
| 3 | `scripts/03_find_lookalikes.py` | Method A: perceptual hash (pHash) with Hamming distance ≤ 8. Method B: frozen pretrained ResNet18 embeddings with cosine similarity ≥ 0.95. Build groups with connected components/union-find | `outputs/reports/lookalike_pairs.csv`, `data/splits/groups.csv` | Done — 85,829 pHash pairs, 42,068 cosine pairs, 79,211 different-class rows, 427 groups |
| 4 | `scripts/04_make_splits.py` | Remove duplicates/conflicts, then use group-aware stratified splitting (70/15/15) so same-image groups stay together; run four leakage checks | `train.csv`, `val.csv`, `test.csv` | Planned |
| 5 | `scripts/05_train_classical_ml.py` | Cache frozen ResNet18 features (512 values/image), train RBF SVM and Random Forest with balanced class weights | models + reports | Planned |
| 6 | `scripts/06_train_cnn.py` | Fine-tune torchvision models (simple_cnn, resnet18, resnet50, mobilenet_v2, alexnet, efficientnet_b0) with weighted loss, optional warm-up, early stopping, augmentation on train only | best `.pth` + history plots | Planned |
| 7 | `scripts/07_evaluate.py` | Load best weights, evaluate once on test: accuracy, macro F1, classification report, confusion matrix, wrong-prediction grid | final test report | Planned |
| 8 | `scripts/08_compare_models.py` | Collect all model results into one table and make a comparison bar chart | `model_comparison.csv` | Planned |

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
