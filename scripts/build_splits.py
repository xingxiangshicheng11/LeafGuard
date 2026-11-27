import argparse
import csv
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.grouped_splits import partition_groups_by_class, split_groups_by_class
from src.data.near_duplicates import dhash, group_near_duplicates
from src.data.scan import scan_dataset
from src.data.splits import (
    DEFAULT_RATIOS,
    build_stratified_splits,
    check_disjoint,
    remove_duplicate_copies,
    split_summary,
    write_splits,
)


def load_split_map(split_dir):
    split_dir = Path(split_dir)
    mapping = {}
    for split_name in ("train", "val", "test"):
        path = split_dir / f"{split_name}.csv"
        if not path.is_file():
            continue
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                mapping[row["path"]] = split_name
    return mapping


def group_leakage(groups_by_class, path_to_split):
    leaked_groups = 0
    leaked_images = 0
    for groups in groups_by_class.values():
        for group in groups:
            if len(group) < 2:
                continue
            splits = {
                path_to_split[item["path"]]
                for item in group
                if item["path"] in path_to_split
            }
            if len(splits) > 1:
                leaked_groups += 1
                leaked_images += len(group)
    return {"groups": leaked_groups, "images": leaked_images}


def find_groups(items, project_root, hash_size, max_distance):
    records = []
    for index, item in enumerate(items):
        value, bits = dhash(project_root / item["path"], hash_size=hash_size)
        records.append({"hash": value, "bits": bits})
        if (index + 1) % 5000 == 0:
            print(f"hashed {index + 1}/{len(items)}", flush=True)
    groups, stats = group_near_duplicates(records, max_distance=max_distance)
    return partition_groups_by_class(groups, items), stats


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create stratified 70/15/15 splits. By default, perceptual "
            "near-duplicate groups are kept within a single split."
        )
    )
    parser.add_argument("--data-root", default="data/plantvillage")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--out-dir", default="splits")
    parser.add_argument("--compare-dir", default="splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=float, default=DEFAULT_RATIOS["train"])
    parser.add_argument("--val", type=float, default=DEFAULT_RATIOS["val"])
    parser.add_argument("--test", type=float, default=DEFAULT_RATIOS["test"])
    parser.add_argument("--max-distance", type=int, default=4)
    parser.add_argument("--hash-size", type=int, default=8)
    parser.add_argument(
        "--no-near-duplicates",
        action="store_true",
        help="Use plain image-level stratified splitting (not recommended).",
    )
    parser.add_argument("--json", dest="json_path", default="results/grouped_splits_report.json")
    args = parser.parse_args()

    ratios = {"train": args.train, "val": args.val, "test": args.test}
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise SystemExit(f"split ratios must sum to 1.0, got {sum(ratios.values())}")

    project_root = Path(args.project_root).resolve()
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = project_root / data_root
    if not data_root.is_dir():
        print(f"[ERROR] data root not found: {data_root}")
        return 1

    scan = scan_dataset(
        data_root,
        project_root=project_root,
        hash_duplicates=True,
        include_images=True,
    )
    items = scan["images"]
    duplicate_copies = 0
    if scan["duplicates"]:
        items, removed = remove_duplicate_copies(items, scan["duplicates"])
        duplicate_copies = len(removed)
    if not items:
        print("[ERROR] no usable images")
        return 1

    old_map = load_split_map(
        Path(args.compare_dir)
        if Path(args.compare_dir).is_absolute()
        else project_root / args.compare_dir
    )

    started = time.time()
    group_stats = {}
    cross_class_groups = []
    groups_by_class = {}
    if args.no_near_duplicates:
        splits = build_stratified_splits(items, seed=args.seed, ratios=ratios)
        old_leakage = None
        new_leakage = None
    else:
        (groups_by_class, cross_class_groups), group_stats = find_groups(
            items, project_root, args.hash_size, args.max_distance
        )
        splits = split_groups_by_class(groups_by_class, ratios, seed=args.seed)
        old_leakage = group_leakage(groups_by_class, old_map) if old_map else None
        new_map = {
            item["path"]: name for name, rows in splits.items() for item in rows
        }
        new_leakage = group_leakage(groups_by_class, new_map)

    check_disjoint(splits)
    class_names = sorted({item["class_name"] for item in items})
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    class_to_idx = write_splits(splits, out_dir, class_names)
    try:
        out_dir_display = str(out_dir.relative_to(project_root))
    except ValueError:
        out_dir_display = str(out_dir)

    summary = split_summary(splits)
    per_class = {}
    for split_name, rows in splits.items():
        for item in rows:
            entry = per_class.setdefault(
                item["class_name"], {"train": 0, "val": 0, "test": 0}
            )
            entry[split_name] += 1

    report = {
        "data_root": scan["data_root"],
        "seed": args.seed,
        "near_duplicates_enabled": not args.no_near_duplicates,
        "max_distance": None if args.no_near_duplicates else args.max_distance,
        "hash_size": None if args.no_near_duplicates else args.hash_size,
        "num_valid_images_before_dedupe": scan["num_valid_images"],
        "exact_duplicate_copies_removed": duplicate_copies,
        "num_images": len(items),
        "num_classes": len(class_names),
        **group_stats,
        "num_cross_class_groups": len(cross_class_groups),
        "cross_class_group_images": sum(entry["size"] for entry in cross_class_groups),
        "split_summary": summary,
        "per_class": dict(sorted(per_class.items())),
        "old_split_group_leakage": old_leakage,
        "new_split_group_leakage": new_leakage,
        "out_dir": out_dir_display,
        "seconds": round(time.time() - started, 3),
    }

    print(f"images: {len(items)} (removed exact copies: {duplicate_copies})")
    if args.no_near_duplicates:
        print("near-duplicate grouping: disabled")
    else:
        print(f"near-duplicate groups: {group_stats['num_groups']}")
        print(f"multi-image groups: {group_stats['num_multi_groups']}")
        print(f"images in multi-image groups: {group_stats['num_grouped_images']}")
        print(f"cross-class groups (reported only): {len(cross_class_groups)}")
        if old_leakage is not None:
            print(
                "old split near-duplicate leakage: "
                f"{old_leakage['groups']} groups / {old_leakage['images']} images"
            )
        print(
            "new split near-duplicate leakage: "
            f"{new_leakage['groups']} groups / {new_leakage['images']} images"
        )
    print(
        "splits: "
        + ", ".join(
            f"{name}={summary[name]['count']} ({summary[name]['ratio'] * 100:.1f}%)"
            for name in ("train", "val", "test")
        )
    )
    print(f"output written to {out_dir}")

    json_path = Path(args.json_path)
    if not json_path.is_absolute():
        json_path = project_root / json_path
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"report written to {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
