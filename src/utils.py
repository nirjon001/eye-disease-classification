"""
Small helper functions shared by the project scripts.

Input: paths, seeds, and data frames from the scripts.
Output: created folders, a chosen torch device, and simple plotting helpers.
"""

import random
from pathlib import Path

import numpy as np


def ensure_directories(*paths):
    """Create each folder in `paths` if it does not exist yet."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def set_seed(seed):
    """Set the random seed in Python, NumPy, and PyTorch when available."""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # These two lines make CPU convolution results more repeatable.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def get_device():
    """Return 'cuda' when a Nvidia GPU is available, otherwise 'cpu'."""
    try:
        import torch
    except ImportError:
        return "cpu"

    return "cuda" if torch.cuda.is_available() else "cpu"


def to_project_relative_string(path, project_root):
    """Make a short path string that is easy to store in CSV files."""
    path = Path(path)
    try:
        return path.resolve().relative_to(Path(project_root).resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def to_dataset_relative_string(path_string):
    """
    Turn a stored image path into a portable dataset-relative string.

    Older CSVs stored "Original Dataset/Glaucoma/x.jpg" relative to the project
    folder. Training on Kaggle needs just "Glaucoma/x.jpg" relative to the
    dataset root. This function accepts either form and always returns forward
    slashes, so Windows and Linux agree.
    """
    text = str(path_string).replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]

    prefix = "Original Dataset/"
    if text.startswith(prefix):
        text = text[len(prefix) :]

    return text


def save_dataframe(dataframe, output_path):
    """Write a pandas DataFrame to CSV and create its parent folder if needed."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
