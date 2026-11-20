import csv
import json
from pathlib import Path

import pytest

Image = pytest.importorskip("PIL.Image")

from src.data.external import external_summary, map_external_items
from src.data.scan import scan_dataset
from src.data.splits import write_manifest


def make_image(path, color=(20, 120, 40)):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color).save(path)
    return path


def test_map_external_items_maps_and_skips():
    items = [
        {"path": "a.png", "class_name": "Apple leaf"},
        {"path": "b.png", "class_name": "Apple Scab Leaf"},
        {"path": "c.png", "class_name": "Unknown leaf"},
    ]
    class_map = {
        "Apple leaf": "Apple___healthy",
        "Apple Scab Leaf": "Apple___Apple_scab",
    }
    class_to_idx = {"Apple___healthy": 0, "Apple___Apple_scab": 1}

    mapped, skipped = map_external_items(items, class_map, class_to_idx)
    by_path = {item["path"]: item for item in mapped}
    assert set(by_path) == {"a.png", "b.png"}
    assert by_path["a.png"]["class_name"] == "Apple___healthy"
    assert by_path["b.png"]["class_name"] == "Apple___Apple_scab"
    assert skipped == {"Unknown leaf": 1}


def test_map_external_items_rejects_missing_target():
    items = [{"path": "a.png", "class_name": "Apple leaf"}]
    with pytest.raises(ValueError):
        map_external_items(
            items, {"Apple leaf": "NotInPlantVillage"}, {"Apple___healthy": 0}
        )


def test_external_summary_counts():
    mapped = [
        {"path": "a", "class_name": "A", "source_class": "a"},
        {"path": "b", "class_name": "A", "source_class": "b"},
    ]
    summary = external_summary(mapped, {"x": 3}, damaged_count=2, duplicate_count=1)
    assert summary["num_items"] == 2
    assert summary["num_target_classes"] == 1
    assert summary["per_target_class"] == {"A": 2}
    assert summary["skipped_unmapped"] == {"x": 3}
    assert summary["damaged_excluded"] == 2
    assert summary["duplicate_copies_excluded"] == 1


def test_plantdoc_manifest_end_to_end(tmp_path):
    root = tmp_path / "data/plantdoc"
    make_image(root / "Apple leaf" / "healthy1.png", (10, 100, 10))
    make_image(root / "Apple leaf" / "healthy2.png", (20, 100, 10))
    make_image(root / "Apple Scab Leaf" / "scab1.png", (100, 50, 10))
    make_image(root / "Apple Scab Leaf" / "scab2.png", (110, 50, 10))
    make_image(root / "Unknown leaf" / "unknown.png", (50, 50, 50))
    (root / "Apple leaf" / "broken.png").write_bytes(b"not an image")

    class_map = {
        "Apple leaf": "Apple___healthy",
        "Apple Scab Leaf": "Apple___Apple_scab",
    }
    class_to_idx = {"Apple___healthy": 0, "Apple___Apple_scab": 1}

    report = scan_dataset(root, project_root=tmp_path, include_images=True)
    assert report["num_damaged"] == 1
    mapped, skipped = map_external_items(report["images"], class_map, class_to_idx)
    assert skipped == {"Unknown leaf": 1}

    out = tmp_path / "splits/plantdoc_eval.csv"
    write_manifest(mapped, out, class_to_idx)

    with open(out, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["path", "label", "class_name"]
        rows = list(reader)

    assert len(rows) == 4
    for row in rows:
        assert not Path(row["path"]).is_absolute()
        assert row["class_name"] in class_to_idx
        assert int(row["label"]) == class_to_idx[row["class_name"]]

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_committed_plantdoc_manifest_contract():
    manifest = REPO_ROOT / "splits/plantdoc_eval.csv"
    mapping_path = REPO_ROOT / "splits/plantdoc_class_map.json"
    classes_path = REPO_ROOT / "splits/class_to_idx.json"
    if not manifest.is_file() or not mapping_path.is_file():
        pytest.skip("PlantDoc manifest not generated yet")

    class_to_idx = json.loads(classes_path.read_text(encoding="utf-8"))
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))["mapping"]
    assert set(mapping.values()).issubset(class_to_idx)

    with open(manifest, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["path", "label", "class_name"]
        rows = list(reader)
    assert rows
    for row in rows:
        assert not Path(row["path"]).is_absolute()
        assert row["path"].startswith("data/plantdoc/")
        assert row["class_name"] in class_to_idx
        assert int(row["label"]) == class_to_idx[row["class_name"]]
