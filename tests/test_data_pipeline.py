import csv
import json
from pathlib import Path

import pytest

Image = pytest.importorskip("PIL.Image")

from src.data.scan import scan_dataset
from src.data.splits import (
    _allocate,
    build_stratified_splits,
    check_disjoint,
    remove_duplicate_copies,
    split_summary,
    write_splits,
)


def make_image(path, size=(32, 32), color=(20, 120, 40)):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


def make_items(classes, per_class):
    return [
        {"path": f"data/plantvillage/{name}/{i:03d}.png", "class_name": name}
        for name in classes
        for i in range(per_class)
    ]


def test_allocate_respects_small_classes():
    assert _allocate(0, {"train": 0.7, "val": 0.15, "test": 0.15}) == (0, 0, 0)
    assert _allocate(1, {"train": 0.7, "val": 0.15, "test": 0.15}) == (1, 0, 0)
    assert _allocate(2, {"train": 0.7, "val": 0.15, "test": 0.15}) == (1, 1, 0)
    train, val, test = _allocate(10, {"train": 0.7, "val": 0.15, "test": 0.15})
    assert train + val + test == 10
    assert train >= 1 and val >= 1 and test >= 1


def test_stratified_split_is_reproducible_and_disjoint():
    items = make_items(["c0", "c1"], 20)
    first = build_stratified_splits(items, seed=42)
    second = build_stratified_splits(list(reversed(items)), seed=42)
    assert first == second
    assert check_disjoint(first)

    all_paths = [item["path"] for split in first.values() for item in split]
    assert len(all_paths) == len(set(all_paths)) == len(items)


def test_stratified_split_ratio_close_to_target():
    items = make_items(["c0", "c1", "c2"], 100)
    splits = build_stratified_splits(items, seed=42)
    summary = split_summary(splits)
    assert summary["total"] == len(items)
    assert abs(summary["train"]["ratio"] - 0.70) < 0.01
    assert abs(summary["val"]["ratio"] - 0.15) < 0.01
    assert abs(summary["test"]["ratio"] - 0.15) < 0.01


def test_split_is_stratified_per_class():
    items = make_items(["c0", "c1"], 100)
    splits = build_stratified_splits(items, seed=42)
    for split_name in ("train", "val", "test"):
        classes = {item["class_name"] for item in splits[split_name]}
        assert classes == {"c0", "c1"}


def test_remove_duplicate_copies_keeps_one():
    items = [
        {"path": "b.png", "class_name": "c0"},
        {"path": "a.png", "class_name": "c0"},
        {"path": "c.png", "class_name": "c1"},
    ]
    groups = [["b.png", "a.png"]]
    kept, removed = remove_duplicate_copies(items, groups)
    assert [item["path"] for item in kept] == ["a.png", "c.png"]
    assert removed == ["b.png"]


def test_scan_separates_within_and_cross_class_duplicates(tmp_path):
    data_root = tmp_path / "data/plantvillage"
    make_image(data_root / "c0" / "x.png", color=(1, 1, 2))
    make_image(data_root / "c0" / "y.png", color=(1, 1, 2))
    make_image(data_root / "c0" / "w.png", color=(1, 1, 1))
    make_image(data_root / "c1" / "z.png", color=(1, 1, 1))

    report = scan_dataset(
        data_root, project_root=tmp_path, hash_duplicates=True, include_images=True
    )
    assert report["duplicates"] == [
        ["data/plantvillage/c0/x.png", "data/plantvillage/c0/y.png"]
    ]
    assert report["cross_class_duplicates"] == [
        ["data/plantvillage/c0/w.png", "data/plantvillage/c1/z.png"]
    ]


def test_scan_and_split_end_to_end(tmp_path):
    data_root = tmp_path / "data/plantvillage"
    make_image(data_root / "Apple___scab" / "000.png", color=(20, 120, 40))
    make_image(data_root / "Apple___scab" / "001.png", color=(30, 120, 40))
    for i in range(10):
        make_image(data_root / "Apple___rot" / f"{i:03d}.png", color=(i, i, i))
    make_image(data_root / "Apple___scab" / "010.png", color=(40, 120, 40))
    (data_root / "Apple___scab" / "broken.png").write_bytes(b"not an image")

    report = scan_dataset(
        data_root, project_root=tmp_path, hash_duplicates=True, include_images=True
    )
    assert report["num_classes"] == 2
    assert report["num_damaged"] == 1
    assert report["num_valid_images"] == 13
    assert report["classes"] == {"Apple___rot": 10, "Apple___scab": 3}
    assert "data/plantvillage/Apple___scab/broken.png" in report["damaged"]
    assert report["duplicates"] == []

    splits = build_stratified_splits(report["images"], seed=42)
    check_disjoint(splits)
    out_dir = tmp_path / "splits"
    class_to_idx = write_splits(
        splits, out_dir, [item["class_name"] for item in report["images"]]
    )

    assert class_to_idx == {"Apple___rot": 0, "Apple___scab": 1}
    assert json.loads((out_dir / "class_to_idx.json").read_text(encoding="utf-8")) == class_to_idx

    for split_name in ("train", "val", "test"):
        with open(out_dir / f"{split_name}.csv", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == ["path", "label", "class_name"]
            rows = list(reader)
        for row in rows:
            assert not Path(row["path"]).is_absolute()
            assert row["class_name"] in class_to_idx
            assert int(row["label"]) == class_to_idx[row["class_name"]]

    total = sum(len(splits[name]) for name in ("train", "val", "test"))
    assert total == 13

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_committed_split_manifests_follow_contract():
    split_dir = REPO_ROOT / "splits"
    mapping_path = split_dir / "class_to_idx.json"
    if not mapping_path.is_file():
        pytest.skip("splits not generated yet")

    class_to_idx = json.loads(mapping_path.read_text(encoding="utf-8"))
    assert class_to_idx == dict(sorted(class_to_idx.items()))
    assert len(set(class_to_idx.values())) == len(class_to_idx)

    all_paths = set()
    counts = {}
    classes_seen = {}
    for name in ("train", "val", "test"):
        manifest = split_dir / f"{name}.csv"
        assert manifest.is_file()
        with open(manifest, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == ["path", "label", "class_name"]
            rows = list(reader)
        counts[name] = len(rows)
        assert rows
        for row in rows:
            assert not Path(row["path"]).is_absolute()
            assert row["class_name"] in class_to_idx
            assert int(row["label"]) == class_to_idx[row["class_name"]]
            assert row["path"] not in all_paths, "same image crosses splits"
            all_paths.add(row["path"])
            classes_seen.setdefault(name, set()).add(row["class_name"])

    total = sum(counts.values())
    assert abs(counts["train"] / total - 0.70) < 0.01
    assert abs(counts["val"] / total - 0.15) < 0.01
    assert abs(counts["test"] / total - 0.15) < 0.01
    assert classes_seen["train"] == set(class_to_idx)
    assert classes_seen["val"] == set(class_to_idx)
    assert classes_seen["test"] == set(class_to_idx)
