import csv
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = REPO_ROOT / "splits_combined"


def load_rows(name):
    path = SPLIT_DIR / f"{name}.csv"
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["path", "label", "class_name"]
        return list(reader)


def test_combined_splits_contract():
    mapping_path = SPLIT_DIR / "class_to_idx.json"
    if not mapping_path.is_file():
        pytest.skip("combined splits not generated yet")

    class_to_idx = json.loads(mapping_path.read_text(encoding="utf-8"))
    assert len(class_to_idx) == 132
    assert class_to_idx == dict(sorted(class_to_idx.items()))
    assert len(set(class_to_idx.values())) == len(class_to_idx)

    assert not (SPLIT_DIR / "test.csv").exists()

    train = load_rows("train")
    val = load_rows("val")
    assert len(train) == 55625
    assert len(val) == 9878

    seen = set()
    for rows in (train, val):
        for row in rows:
            assert not Path(row["path"]).is_absolute()
            assert row["path"].startswith(("data/plantvillage/", "data/plantwild/"))
            assert row["class_name"] in class_to_idx
            assert int(row["label"]) == class_to_idx[row["class_name"]]
            assert row["path"] not in seen
            seen.add(row["path"])

    assert len(seen) == 55625 + 9878
    assert {row["class_name"] for row in train} == set(class_to_idx)
    assert {row["class_name"] for row in val} == set(class_to_idx)
