from pathlib import Path

import pytest

Image = pytest.importorskip("PIL.Image")

from src.data.grouped_splits import (
    assign_groups,
    partition_groups_by_class,
    split_groups_by_class,
)
from src.data.near_duplicates import dhash
from src.data.splits import check_disjoint


def make_item(path, class_name):
    return {"path": path, "class_name": class_name}


def test_assign_groups_keeps_groups_intact():
    groups = [
        [make_item("a1", "A"), make_item("a2", "A")],
        [make_item("a3", "A")],
        [make_item("a4", "A")],
        [make_item("a5", "A"), make_item("a6", "A"), make_item("a7", "A")],
        [make_item("a8", "A")],
        [make_item("a9", "A")],
        [make_item("a10", "A")],
    ]
    assignment = assign_groups(groups)
    seen = {}
    for split_name, split_groups in assignment.items():
        for group in split_groups:
            paths = {item["path"] for item in group}
            for path in paths:
                assert path not in seen
                seen[path] = split_name
    assert len(seen) == 10
    for group in groups:
        assert len({seen[item["path"]] for item in group}) == 1


def test_split_groups_by_class_is_deterministic_and_disjoint():
    groups_by_class = {
        "A": [[make_item(f"A{i}", "A")] for i in range(10)],
        "B": [[make_item(f"B{i}", "B")] for i in range(10)],
    }
    first = split_groups_by_class(groups_by_class, seed=42)
    second = split_groups_by_class(groups_by_class, seed=42)
    assert first == second
    assert check_disjoint(first)
    total = sum(len(rows) for rows in first.values())
    assert total == 20


def test_partition_groups_by_class_reports_cross_class():
    items = [
        make_item("a1", "A"),
        make_item("a2", "A"),
        make_item("b1", "B"),
    ]
    groups = [[0, 1, 2]]
    groups_by_class, cross = partition_groups_by_class(groups, items)
    assert sorted(len(group) for group in groups_by_class["A"]) == [2]
    assert sorted(len(group) for group in groups_by_class["B"]) == [1]
    assert len(cross) == 1
    assert cross[0]["classes"] == ["A", "B"]


def test_dhash_identical_and_different(tmp_path):
    identical_a = tmp_path / "a.png"
    identical_b = tmp_path / "b.png"
    different = tmp_path / "c.png"

    Image.new("RGB", (64, 64), (20, 120, 40)).save(identical_a)
    Image.new("RGB", (64, 64), (20, 120, 40)).save(identical_b)

    image = Image.new("RGB", (64, 64), (0, 0, 0))
    for x in range(32, 64):
        for y in range(64):
            image.putpixel((x, y), (255, 255, 255))
    image.save(different)

    hash_a, bits = dhash(identical_a)
    hash_b, _ = dhash(identical_b)
    hash_c, _ = dhash(different)

    assert bits == 64
    assert hash_a == hash_b
    assert hash_a != hash_c
