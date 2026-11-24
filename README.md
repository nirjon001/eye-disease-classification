# Eye Disease Image Classification

A classification-only learning project on the Mendeley Eye Disease Image Dataset.

## Important notes

- Only the 5,335 original images are used.
- augmented images were never downloaded.
- The raw image path is set once in `src/config.py` as `RAW_DATA_DIR`.
- Class names are read from folder names; they are never hardcoded.
- Original images are never deleted or edited.

## Current progress

| Step | Script | Status |
|---|---|---|
| 1 | `scripts/01_explore_data.py` | Done — 5,335 originals, 0 corrupted |
| 2 | `scripts/02_find_exact_duplicates.py` | Done — 948 images marked `drop=True` |
| 2 check | `scripts/02_verify_conflicts.py` | Done — 5 sampled conflicts verified byte-for-byte |
| 3 | `scripts/03_find_lookalikes.py` | Done — pHash + ResNet18 embeddings |
| 3 choice | `scripts/03_apply_setting_a.py` | Done — cosine >= 0.98 groups; 36 conflict images dropped |
| 4 | `scripts/04_make_splits.py` | Done — 3,038 train / 656 val / 657 test; all leakage checks PASS |
| 5 | — | Not started yet |
| 6 | `scripts/06_train_cnn.py` | Ready — local `--quick` test passed; full runs on Kaggle |
| 7–8 | — | Not started yet |
| 9 | `scripts/09_prepare_kaggle_package.py` | Helper — packages only split images for Kaggle |

Current kept images after Step 2 and Step 3 conflict drops: **4,351**.

## How to run

Run from the project root:

```bash
python scripts/01_explore_data.py
python scripts/02_find_exact_duplicates.py
python scripts/02_verify_conflicts.py
python scripts/03_find_lookalikes.py
```

Install dependencies with:

```bash
pip install -r requirements.txt
```

For Kaggle, set the two environment variables before running training:

```bash
export EYE_DATA_ROOT=/kaggle/input/eye-data/eye_data
export EYE_OUTPUT_DIR=/kaggle/working/outputs
```

Split CSVs store paths relative to `EYE_DATA_ROOT` (for example
`Glaucoma/Glaucoma131.jpg`), so no path editing is needed on Kaggle.

## How to run on Kaggle

1. Verify your phone number in **Settings -> Phone verification** (unlocks GPU and internet).
2. Open `notebooks/kaggle_train.ipynb` in Kaggle. Use **Add Data** to attach the
   `eye-data` dataset, choose **GPU T4/P100**, and turn **Internet: On**.
3. Run the first cells to clone this repo and set the paths. Check the real dataset
   folder with `!ls /kaggle/input/eye-data` and change `EYE_DATA_ROOT` if needed.
4. Smoke-test first (no torch install needed):

   ```bash
   python scripts/06_train_cnn.py --model resnet18 --quick
   ```

5. Run one model at a time, then use **Save Version -> Save & Run All (Commit)**:

   ```bash
   python scripts/06_train_cnn.py --model resnet18 --epochs 20
   python scripts/06_train_cnn.py --model mobilenet_v2 --epochs 20
   python scripts/06_train_cnn.py --model simple_cnn --epochs 20
   python scripts/06_train_cnn.py --model alexnet --epochs 20
   python scripts/06_train_cnn.py --model resnet50 --epochs 20
   python scripts/06_train_cnn.py --model efficientnet_b0 --epochs 20
   ```

   After every epoch the script prints train/val loss, top-1/3/5 accuracy, and
   macro F1, and saves the same values in `*_history.csv`. Weights, history
   CSVs, and curves are written every epoch, so a stopped session does not lose
   finished work. Download them from the notebook's **Output** tab into your
   local `outputs/` folder before running Steps 7 and 8.

## Main outputs

- `data/splits/all_images.csv` — original images from Step 1
- `data/splits/dedup_flags.csv` — keep/drop flags from Step 2
- `data/splits/groups.csv` — look-alike groups from Step 3
- `outputs/reports/` — duplicate, look-alike, and threshold reports
- `outputs/figures/` — class counts, samples, conflict examples, and pair grids
- `METHODS.md` — detailed record of which method was used in each step

The original images and large model files are excluded by `.gitignore`.
