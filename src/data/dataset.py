from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class LeafDiseaseDataset(Dataset):
    """CSV manifest columns: path, label, class_name.

    ``path`` may be absolute or relative to ``project_root``. Each item returns
    ``(image, label)`` where image is produced by ``transform`` (RGB) and label
    is an integer class index.
    """

    REQUIRED_COLUMNS = ("path", "label", "class_name")

    def __init__(self, manifest_csv, project_root=".", transform=None):
        manifest_csv = Path(manifest_csv)
        if not manifest_csv.is_file():
            raise FileNotFoundError(f"manifest csv not found: {manifest_csv}")

        self.items = pd.read_csv(manifest_csv)
        missing = [c for c in self.REQUIRED_COLUMNS if c not in self.items.columns]
        if missing:
            raise ValueError(f"CSV must contain columns: {sorted(self.REQUIRED_COLUMNS)}")

        self.root = Path(project_root)
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def _image_path(self, row):
        path = Path(row["path"])
        if not path.is_absolute():
            path = self.root / path
        return path

    def __getitem__(self, index):
        row = self.items.iloc[index]
        path = self._image_path(row)
        if not path.is_file():
            raise FileNotFoundError(f"image not found: {path}")
        with Image.open(path) as im:
            image = im.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(row["label"])
