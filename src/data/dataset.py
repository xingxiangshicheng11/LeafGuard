from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class LeafDiseaseDataset(Dataset):
    """CSV manifest columns: path, label, class_name."""
    def __init__(self, manifest_csv, project_root=".", transform=None):
        self.items = pd.read_csv(manifest_csv)
        required = {"path", "label", "class_name"}
        if not required.issubset(self.items.columns):
            raise ValueError(f"CSV must contain columns: {sorted(required)}")
        self.root = Path(project_root)
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        row = self.items.iloc[index]
        path = Path(row["path"])
        if not path.is_absolute():
            path = self.root / path
        with Image.open(path) as im:
            image = im.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(row["label"])
