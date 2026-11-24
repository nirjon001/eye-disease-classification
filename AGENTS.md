# Eye Disease Image Classification — Project Instructions for opencode

Put this file in the project root (opencode reads `AGENTS.md` automatically).

## 1. Goal (read this first)

Build a **classification-only** pipeline (NO segmentation, NO detection) on the
Mendeley "Eye Disease Image Dataset":
https://data.mendeley.com/datasets/s9bfhswzjb/1

- Task: given one eye image, predict the disease class.
- Classes (from the dataset page): Retinitis Pigmentosa, Retinal Detachment,
  Pterygium, Myopia, Macular Scar, Glaucoma, Disc Edema, Diabetic Retinopathy,
  Central Serous Chorioretinopathy, Healthy.
  **Always read the real class names from the folder names. Do not hardcode.**
- Purpose: learning exercise before Capstone 1. The code is read by
  undergraduate students, so clarity beats cleverness.

## 2. Coding rules (very important)

1. Write simple Python. No fancy one-liners, no metaclasses, no deep abstractions.
2. Every file starts with a short comment block: what it does, input, output.
3. Every function has a docstring in plain English and comments on non-obvious lines.
4. Use clear variable names (`train_loader`, not `tl`).
5. One script = one job. Scripts run in order `01_...` to `08_...`.
6. Use **PyTorch + torchvision** for deep learning, **scikit-learn** for SVM / Random Forest.
7. Put all settings (paths, image size, batch size, epochs, seed) in `src/config.py`.
8. Set one random seed everywhere (`SEED = 42`) so results are repeatable.
9. Print friendly progress messages (e.g. `"Step 2/8: finding duplicates..."`).
10. After finishing each script, run it on a tiny subset first (e.g. 50 images) to check it works.
11. Never delete original images. Scripts only read them and write CSV/model/plot files.

## 3. Hardware notes

Local PC: Ryzen 5 5600, RX 580 (AMD, no CUDA), 16 GB RAM.
- PyTorch GPU does not work well on RX 580 on Windows. So:
  - Code must auto-pick device: `cuda` if available, else `cpu`.
  - Heavy training (Step 6) is meant to run on **free Google Colab or Kaggle GPU**.
  - On CPU, use small image size (128 or 160) and small models (MobileNet, ResNet18).
- Use `num_workers=2` at most. Keep batch size 16–32.

## 4. Folder structure

```
eye-disease/
├── AGENTS.md
├── requirements.txt
├── Original Dataset/        # original images only; RAW_DATA_DIR in config.py points here
├── data/
│   └── splits/              # CSV files made by our scripts
├── src/
│   ├── config.py            # all settings
│   ├── utils.py             # small helpers (seed, device, plotting)
│   └── dataset.py           # PyTorch Dataset that reads a split CSV
├── scripts/
│   ├── 01_explore_data.py
│   ├── 02_find_exact_duplicates.py
│   ├── 03_find_lookalikes.py
│   ├── 04_make_splits.py
│   ├── 05_train_classical_ml.py
│   ├── 06_train_cnn.py
│   ├── 07_evaluate.py
│   └── 08_compare_models.py
├── outputs/
│   ├── figures/
│   ├── models/
│   └── reports/
└── notebooks/               # optional, for Colab
```

## 5. Original images only (important decision)

The dataset has **5,335 original images** and **16,242 pre-made augmented images**.
**We use ONLY the original images.** Ignore every augmented image.
augmented images were never downloaded

Why: augmented images are rotated/shifted/flipped copies of originals. Mixing
them into a random split would leak the answer from train into test.

- Step 1 must find which files are originals and which are augmented, and keep
  only the originals. If the folder layout does not make this clear, STOP and
  ask me. Do not guess.
- Leakage can still happen with originals: the same patient may have several
  very similar photos, and the dataset has no patient ID. So we still check
  exact duplicates and look-alikes (Steps 2 and 3) and split by **group**.
- Light augmentation *during training* (random flip, small rotation) is allowed,
  controlled by `USE_TRAIN_AUGMENTATION = True` in `config.py`. It only touches
  the train set, creates no files, and cannot leak. Set it to `False` to
  turn it off completely.

## 6. Workflow (do the steps in order)

### Step 1 — `01_explore_data.py`
- Walk through `Original Dataset` (`RAW_DATA_DIR` in `src/config.py`), list every image with: path, class (folder name),
  width, height, file size, and `is_augmented` (always `False` because augmented images were never downloaded).
- **Keep only original images** (all images, because none are augmented).
  Save `data/splits/all_images.csv` with originals only.
- Print: number of images per class (originals only; also print how many augmented files were ignored), image size
  summary, any corrupted files (try opening with PIL, record failures).
- Save bar chart of class counts to `outputs/figures/class_counts.png`.
- Save a grid of 3 sample images per class to `outputs/figures/samples.png`.

### Step 2 — `02_find_exact_duplicates.py` (exact copies)
- Compute an **MD5 hash** of each file's bytes. Same hash = identical file.
- Save `outputs/reports/exact_duplicates.csv` (hash, list of paths).
- Special check: same hash but **different class folders** = label conflict.
  Print these loudly; they must be removed from the dataset.
- Keep one copy of each exact duplicate (mark others `drop=True`).

### Step 3 — `03_find_lookalikes.py` (near-duplicates / look-alikes)
Exact hashing misses resized, re-saved, flipped, or lightly edited copies.
Use **two methods** and explain each in comments:

**Method A: perceptual hash (pHash)** using the `imagehash` library.
- Similar-looking images get hashes with small Hamming distance.
- Flag pairs with distance <= 8 (put threshold in config; user can tune).
- Fast, good first pass.

**Method B: deep-feature similarity.**
- Use pretrained ResNet18 (ImageNet weights), remove the last layer, get a
  512-number vector per image (call it an "embedding").
- Compute cosine similarity between images. Flag pairs with similarity >= 0.95
  (configurable).
- To avoid comparing 21,000 x 21,000 images in memory, use `sklearn.neighbors.NearestNeighbors`
  (cosine metric, k=5) or process in batches.

**Output:**
- `outputs/reports/lookalike_pairs.csv` (path_a, path_b, method, score, same_class)
- Build **groups**: if A~B and B~C then A, B, C are one group (use a simple
  union-find / connected components; explain it in comments).
- Save `data/splits/groups.csv` with columns: `path, group_id`.
- Save a figure with 10 example flagged pairs side by side so the human can
  eyeball whether the threshold is sensible.
- Report pairs that are look-alikes but have **different classes** (possible
  label noise). Do not auto-delete; just list them.

### Step 4 — `04_make_splits.py` (train / val / test)
- Remove exact duplicates and label-conflict images.
- Use `StratifiedGroupKFold` or `GroupShuffleSplit` from scikit-learn so that:
  - all images in a group stay together,
  - class proportions stay about the same in each split.
- Target split: **70% train / 15% val / 15% test**.
- Save `train.csv`, `val.csv`, `test.csv` in `data/splits/` (columns: path, label).
- **Leakage check (must run and print PASS/FAIL):**
  1. No file path appears in two splits.
  2. No MD5 hash appears in two splits.
  3. No group_id appears in two splits.
  4. No lookalike pair crosses splits.
- Print the class counts for each split.

**What each split is for (put this in comments):**
- Train: the model learns from this.
- Val: we tune settings and pick the best epoch/model using this.
- Test: touched ONCE at the very end for the final honest score.

### Step 5 — `05_train_classical_ml.py` (baseline: SVM and Random Forest)
- Extract features with a pretrained ResNet18 (frozen) -> 512 numbers per image.
  Cache features in `outputs/models/features_*.npy` so reruns are fast.
- Train `SVC(kernel="rbf", class_weight="balanced")` and
  `RandomForestClassifier(n_estimators=300, class_weight="balanced")`.
- Evaluate on val. Save models with `joblib`.
- Purpose: a cheap baseline to compare against deep models.

### Step 6 — `06_train_cnn.py` (deep learning models)
Make the model name a command-line argument: `--model resnet18` etc.

Models to support (all via torchvision, ImageNet pretrained, replace last layer):

| Model | Why we include it |
|---|---|
| `simple_cnn` | tiny CNN written by hand (3 conv blocks). Baseline so students see the basics |
| `resnet18` | small, fast, **recommended first choice** |
| `resnet50` | stronger, heavier (use on Colab/Kaggle GPU) |
| `mobilenet_v2` | very light, good for CPU / low memory |
| `alexnet` | old, simple, useful for comparison in the report |
| `efficientnet_b0` | (optional) good accuracy for its size |

Training details:
- Image size 224 (128/160 if CPU only). Normalize with ImageNet mean/std.
- **Train augmentation (on the fly, train only, if `USE_TRAIN_AUGMENTATION`):**
  random horizontal flip, small rotation (<=15 deg), small color jitter.
  Val/test: only resize + normalize.
- Loss: `CrossEntropyLoss` with **class weights** (classes are imbalanced).
- Optimizer: Adam, lr=1e-4 for pretrained models (1e-3 for simple_cnn).
- Epochs: 15–25. Use **early stopping** on val loss (patience 4) and save the
  best weights to `outputs/models/<model>_best.pth`.
- Two-phase option: first train only the new last layer for 3 epochs, then
  unfreeze everything. Explain why in comments.
- Log train/val loss and accuracy each epoch to
  `outputs/reports/<model>_history.csv` and plot curves
  (`outputs/figures/<model>_curves.png`).
- Always set the seed. Always print device used.

### Step 7 — `07_evaluate.py` (final test, run once per model)
- Load best weights, run on **test.csv**.
- Report: accuracy, **macro F1**, per-class precision/recall/F1
  (`sklearn.metrics.classification_report`), confusion matrix plot.
- Why macro F1: classes are imbalanced, so plain accuracy can hide weak classes.
- Save the 20 most confident WRONG predictions as an image grid
  (helps us understand failures).
- Optional: Grad-CAM heatmap for 6 test images (helps explain *where* the model looks;
  this is still classification, not segmentation).

### Step 8 — `08_compare_models.py`
- Read every model's test results and make one table
  (model, params count, train time, val F1, test accuracy, test macro F1).
- Save to `outputs/reports/model_comparison.csv` and a bar chart.

## 7. Which model should we pick? (guidance for the final report)

- Start with **ResNet18** (fast, reliable). Then try **MobileNetV2** and **ResNet50**.
- Keep **SVM / Random Forest on pretrained features** and **simple_cnn** as baselines.
- Pick the final model by **validation macro F1**, not by test score.
  Test is only for the final report.
- If two models are close, choose the smaller/faster one.

## 8. Rules for the final report

- Say clearly how duplicates and look-alikes were handled and how many were found.
- Say that only the 5,335 original images were used (pre-made augmented images ignored),
  and that no look-alikes of train images are in val/test.
- Report macro F1 and the confusion matrix, not only accuracy.
- Mention limits: images come from only two hospitals in one district, so results
  may not generalize to other cameras/populations. This is a learning
  project, NOT a medical tool.

## 9. requirements.txt

```
torch
torchvision
numpy
pandas
scikit-learn
matplotlib
seaborn
pillow
imagehash
tqdm
joblib
```

## 10. How opencode should work with me

- Do ONE step at a time. After each script: show how to run it, what output to
  expect, and wait for me to confirm before the next step.
- If the dataset folder layout is different from what is assumed here, inspect
  it first and tell me, then adapt Step 1.
- If something is ambiguous, ask a short question instead of guessing.
- Keep explanations short and in simple English.
- To save tokens: do not re-print whole files after small edits; show only what changed.
