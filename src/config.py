"""
Configuration settings for the eye disease classification project.

Input: none.
Output: settings imported by the scripts in `scripts/`.
All paths, sizes, seeds, and training choices are kept here on purpose.
"""

from pathlib import Path

# Project folders.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# This is the ONLY place where the raw dataset folder is named.
# Pathlib handles the space in "Original Dataset" safely.
RAW_DATA_DIR = PROJECT_ROOT / "Original Dataset"
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
# Separate folder for the extra cosine-distribution study in Step 3.
COSINE_DISTRIBUTION_DIR = REPORTS_DIR / "cosine_distribution"

# Image files we are willing to read.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# The published dataset has this many original images. We print a warning if a
# full run (no subset flag) does not match this number.
EXPECTED_ORIGINAL_COUNT = 5335

# One seed for repeatable results.
SEED = 42

# Image and training settings.
IMAGE_SIZE = 224
CPU_IMAGE_SIZE = 160
BATCH_SIZE = 32
NUM_WORKERS = 2
EPOCHS = 20
LEARNING_RATE_PRETRAINED = 1e-4
LEARNING_RATE_SCRATCH = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 4
WARMUP_EPOCHS = 3
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
USE_TRAIN_AUGMENTATION = True
ROTATION_DEGREES = 15
COLOR_JITTER_BRIGHTNESS = 0.1
COLOR_JITTER_CONTRAST = 0.1

# Step 3 settings: look-alike detection.
# hash_size=8 gives a 64-bit pHash, which is the usual choice with a
# Hamming-distance threshold of 8.
PHASH_HASH_SIZE = 8
PHASH_DISTANCE_THRESHOLD = 8
DEEP_EMBEDDING_SIZE = 512
DEEP_SIMILARITY_THRESHOLD = 0.95
NEIGHBORS_K = 5

# Extra thresholds we print so we can compare them before fixing the defaults.
PHASH_THRESHOLDS = [4, 6, 8, 10]
COSINE_THRESHOLDS = [0.90, 0.95, 0.98]

# Cosine similarity buckets for the separate distribution report.
COSINE_BUCKET_EDGES = [
    (None, 0.70),
    (0.70, 0.80),
    (0.80, 0.90),
    (0.90, None),
]

# Step 4 settings: 70% train, 15% validation, 15% test.
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
GROUP_SPLIT_SEED = SEED

# Step 5 settings.
CLASSICAL_FEATURES_FILE = MODELS_DIR / "resnet18_features.npy"
CLASSICAL_LABELS_FILE = MODELS_DIR / "resnet18_labels.npy"
SVM_MODEL_FILE = MODELS_DIR / "svm.joblib"
RANDOM_FOREST_MODEL_FILE = MODELS_DIR / "random_forest.joblib"

# Step 6 model choices.
SUPPORTED_MODELS = [
    "simple_cnn",
    "resnet18",
    "resnet50",
    "mobilenet_v2",
    "alexnet",
    "efficientnet_b0",
]
DEFAULT_MODEL = "resnet18"
