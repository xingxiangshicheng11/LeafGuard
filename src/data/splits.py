from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

DEFAULT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def _allocate(n, ratios):
    """Split n samples into (train, val, test) deterministically.

    Small classes fall back to keeping at least one sample in train.
    """
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 1, 0

    n_val = max(1, int(n * ratios["val"] + 0.5))
    n_test = max(1, int(n * ratios["test"] + 0.5))
    n_train = n - n_val - n_test
    while n_train < 1:
        if n_val >= n_test and n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            break
        n_train = n - n_val - n_test
    return n_train, n_val, n_test


def build_stratified_splits(items, seed=42, ratios=None):
    ratios = ratios or DEFAULT_RATIOS
    by_class = {}
    for item in items:
        by_class.setdefault(item["class_name"], []).append(item)

    rng = np.random.default_rng(seed)
    splits = {"train": [], "val": [], "test": []}

    for class_name in sorted(by_class):
        class_items = sorted(by_class[class_name], key=lambda x: x["path"])
        order = rng.permutation(len(class_items))
        shuffled = [class_items[i] for i in order]
        n_train, n_val, _ = _allocate(len(shuffled), ratios)
        splits["train"].extend(shuffled[:n_train])
        splits["val"].extend(shuffled[n_train:n_train + n_val])
        splits["test"].extend(shuffled[n_train + n_val:])

    for split_name in splits:
        splits[split_name].sort(key=lambda x: (x["class_name"], x["path"]))
    return splits


def remove_duplicate_copies(items, duplicate_groups):
    """Keep the first path in each duplicate group, drop the rest."""
    drop = set()
    for group in duplicate_groups:
        ordered = sorted(group)
        drop.update(ordered[1:])
    kept = [item for item in items if item["path"] not in drop]
    return kept, sorted(drop)


def write_manifest(items, manifest_path, class_to_idx):
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "class_name"])
        for item in items:
            writer.writerow(
                [item["path"], class_to_idx[item["class_name"]], item["class_name"]]
            )


def write_splits(splits, out_dir, class_names):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    class_names = sorted(set(class_names))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    for split_name in ("train", "val", "test"):
        write_manifest(splits[split_name], out_dir / f"{split_name}.csv", class_to_idx)

    with open(out_dir / "class_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return class_to_idx


def check_disjoint(splits):
    seen = {}
    for split_name, items in splits.items():
        for item in items:
            path = item["path"]
            if path in seen:
                raise ValueError(
                    f"image appears in both {seen[path]} and {split_name}: {path}"
                )
            seen[path] = split_name
    return True


def split_summary(splits):
    total = sum(len(items) for items in splits.values())
    summary = {"total": total}
    for split_name in ("train", "val", "test"):
        count = len(splits[split_name])
        summary[split_name] = {
            "count": count,
            "ratio": (count / total) if total else 0.0,
        }
    return summary
