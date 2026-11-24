"""
Small PyTorch Dataset that reads image paths from a CSV file.

Input:  a CSV with a `path` column (and optionally a `label` column).
Output: one transformed image tensor, its label index, and its CSV path.
Used by Steps 3, 5, 6, and 7.
"""

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class EyeImageDataset(Dataset):
    """Read one eye image at a time from disk using a path stored in a CSV."""

    def __init__(self, dataframe, transform=None, project_root=None):
        """Store the CSV rows and the transform; no images are loaded yet."""
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        if project_root is None:
            project_root = Path(__file__).resolve().parents[1]
        self.project_root = Path(project_root)

    def __len__(self):
        """Return how many images are in this dataset."""
        return len(self.dataframe)

    def __getitem__(self, index):
        """Load and transform one image; return (image, label, path)."""
        row = self.dataframe.iloc[index]
        image_path = self.project_root / row["path"]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)

        # Steps that only need images (like Step 3) can omit the label column.
        label = int(row["label"]) if "label" in self.dataframe.columns else -1
        return image, label, row["path"]
