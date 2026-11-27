import csv
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")

from src.data.dataset import LeafDiseaseDataset
from src.data.transforms import build_transforms


def make_image(path, size=(64, 64), color=(20, 120, 40)):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


def write_manifest(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "class_name"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_dataset_returns_image_and_label(tmp_path):
    make_image(tmp_path / "data/plantvillage/class_a/a.png")
    manifest = write_manifest(
        tmp_path / "train.csv",
        [
            {
                "path": "data/plantvillage/class_a/a.png",
                "label": 0,
                "class_name": "class_a",
            }
        ],
    )

    dataset = LeafDiseaseDataset(
        manifest, project_root=tmp_path, transform=build_transforms(train=False)
    )
    image, label = dataset[0]

    assert tuple(image.shape) == (3, 224, 224)
    assert image.dtype == torch.float32
    assert label == 0
    assert isinstance(label, int)


def test_dataset_accepts_absolute_path(tmp_path):
    image_path = make_image(tmp_path / "abs.png")
    manifest = write_manifest(
        tmp_path / "train.csv",
        [{"path": str(image_path), "label": 1, "class_name": "class_b"}],
    )
    dataset = LeafDiseaseDataset(manifest, project_root=tmp_path)
    _, label = dataset[0]
    assert label == 1


def test_dataset_rejects_missing_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("path,label\nx,0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        LeafDiseaseDataset(bad)


def test_dataset_reports_missing_image(tmp_path):
    manifest = write_manifest(
        tmp_path / "train.csv",
        [{"path": "missing.png", "label": 0, "class_name": "class_a"}],
    )
    dataset = LeafDiseaseDataset(manifest, project_root=tmp_path)
    with pytest.raises(FileNotFoundError):
        _ = dataset[0]
