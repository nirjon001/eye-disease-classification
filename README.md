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
| 4–8 | — | Not started yet |

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

## Main outputs

- `data/splits/all_images.csv` — original images from Step 1
- `data/splits/dedup_flags.csv` — keep/drop flags from Step 2
- `data/splits/groups.csv` — look-alike groups from Step 3
- `outputs/reports/` — duplicate, look-alike, and threshold reports
- `outputs/figures/` — class counts, samples, conflict examples, and pair grids
- `METHODS.md` — detailed record of which method was used in each step

The original images and large model files are excluded by `.gitignore`.
