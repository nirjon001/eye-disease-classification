"""
Small PyTorch Dataset that reads image paths from a CSV file.

Input:  a CSV with a `path` column (and optionally a `label` column).
Output: one transformed image tensor, its label index, and its CSV path.
Used by Steps 3, 5, 6, and 7.
"""

from PIL import Image
from torch.utils.data import Dataset

from src.config import DATA_ROOT
from src.utils import to_kaggle_safe_relative_string


class EyeImageDataset(Dataset):
    """Read one eye image at a time from disk using a path stored in a CSV."""

    def __init__(self, dataframe, transform=None):
        """Store the CSV rows and the transform; no images are loaded yet."""
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        """Return how many images are in this dataset."""
        return len(self.dataframe)

    def __getitem__(self, index):
        """Load and transform one image; return (image, label, path)."""
        row = self.dataframe.iloc[index]
        # CSV paths are relative to the dataset root, so they work on Kaggle.
        # Kaggle forbids square brackets, so its copy uses a cleaned folder name.
        image_path = DATA_ROOT / row["path"]
        if not image_path.is_file():
            image_path = DATA_ROOT / to_kaggle_safe_relative_string(row["path"])

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)

        # Steps that only need images (like Step 3) can omit the label column.
        label = int(row["label"]) if "label" in self.dataframe.columns else -1
        return image, label, row["path"]
