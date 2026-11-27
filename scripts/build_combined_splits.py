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
    check_disjoint,
    remove_duplicate_copies,
    split_summary,
    write_splits,
)


def load_pv_split(split_dir, split_name):
    path = Path(split_dir) / f"{split_name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"missing PlantVillage split: {path}")
    with open(path, newline="", encoding="utf-8") as f:
        return [
            {"path": row["path"], "class_name": row["class_name"]}
            for row in csv.DictReader(f)
        ]


def main():
    parser = argparse.ArgumentParser(
        description="Build combined PlantVillage + PlantWild splits for Plan B."
    )
    parser.add_argument("--plantvillage-splits", default="splits")
    parser.add_argument("--plantwild-root", default="data/plantwild")
    parser.add_argument("--class-map", default="splits/plantwild_class_map.json")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--out-dir", default="splits_combined")
    parser.add_argument("--test-to-train", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-distance", type=int, default=4)
    parser.add_argument("--hash-size", type=int, default=8)
    parser.add_argument("--json", dest="json_path", default="results/combined_splits_report.json")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()

    def resolve(path_value):
        path = Path(path_value)
        return path if path.is_absolute() else project_root / path

    split_dir = resolve(args.plantvillage_splits)
    plantwild_root = resolve(args.plantwild_root)
    class_map_path = resolve(args.class_map)

    if not plantwild_root.is_dir():
        print(f"[ERROR] PlantWild root not found: {plantwild_root}")
        return 1

    class_map_doc = json.loads(class_map_path.read_text(encoding="utf-8"))
    class_map = class_map_doc["mapping"]

    pv_splits = {
        name: load_pv_split(split_dir, name) for name in ("train", "val", "test")
    }

    started = time.time()
    scan = scan_dataset(
        plantwild_root,
        project_root=project_root,
        hash_duplicates=True,
        include_images=True,
        deep=False,
    )
    items = scan["images"]
    duplicate_copies = 0
    if scan["duplicates"]:
        items, removed = remove_duplicate_copies(items, scan["duplicates"])
        duplicate_copies = len(removed)

    missing = sorted({item["class_name"] for item in items} - set(class_map))
    if missing:
        raise SystemExit(f"PlantWild classes missing from mapping: {missing}")

    records = []
    for index, item in enumerate(items):
        source_class = item["class_name"]
        item["source_class"] = source_class
        item["class_name"] = class_map[source_class]
        value, bits = dhash(project_root / item["path"], hash_size=args.hash_size)
        records.append({"hash": value, "bits": bits})
        if (index + 1) % 2000 == 0:
            print(f"hashed {index + 1}/{len(items)}", flush=True)

    groups, group_stats = group_near_duplicates(records, max_distance=args.max_distance)
    groups_by_class, cross_class_groups = partition_groups_by_class(groups, items)
    pw_splits = split_groups_by_class(groups_by_class, DEFAULT_RATIOS, seed=args.seed)

    combined = {name: [] for name in ("train", "val", "test")}
    for name in combined:
        combined[name].extend(pv_splits[name])
        combined[name].extend(pw_splits[name])
        for item in combined[name]:
            item.setdefault("source", "plantvillage" if item["path"].startswith("data/plantvillage/") else "plantwild")
    if args.test_to_train:
        combined["train"].extend(combined["test"])
        combined["test"] = []
    check_disjoint(combined)

    class_names = sorted({item["class_name"] for rows in combined.values() for item in rows})
    out_dir = resolve(args.out_dir)
    class_to_idx = write_splits(combined, out_dir, class_names)

    summary = split_summary(combined)
    per_source = {}
    per_class = {}
    for split_name, rows in combined.items():
        for item in rows:
            source_counts = per_source.setdefault(item["source"], {"train": 0, "val": 0, "test": 0})
            source_counts[split_name] += 1
            class_counts = per_class.setdefault(item["class_name"], {"train": 0, "val": 0, "test": 0})
            class_counts[split_name] += 1

    report = {
        "plantvillage_splits": str(split_dir.relative_to(project_root)),
        "plantwild_root": scan["data_root"],
        "seed": args.seed,
        "test_to_train": args.test_to_train,
        "max_distance": args.max_distance,
        "hash_size": args.hash_size,
        "num_classes": len(class_names),
        "num_images": summary["total"],
        "plantwild_exact_duplicate_copies_removed": duplicate_copies,
        "plantwild_near_duplicate_groups": group_stats,
        "plantwild_cross_class_groups": len(cross_class_groups),
        "split_summary": summary,
        "per_source": per_source,
        "per_class": dict(sorted(per_class.items())),
        "class_map": str(class_map_path.relative_to(project_root)),
        "out_dir": str(out_dir.relative_to(project_root)),
        "seconds": round(time.time() - started, 3),
        "license_note": class_map_doc.get("notes", []),
    }

    print(f"classes: {len(class_names)}")
    print(f"images: {summary['total']} (PlantWild duplicate copies removed: {duplicate_copies})")
    print(f"PlantWild near-duplicate groups: {group_stats['num_multi_groups']}")
    for name in ("train", "val", "test"):
        info = summary[name]
        print(f"  {name}: {info['count']} ({info['ratio'] * 100:.1f}%)")
    for source, counts in sorted(per_source.items()):
        print(f"  {source}: {counts}")
    print(f"class_to_idx written to {out_dir / 'class_to_idx.json'}")

    json_path = resolve(args.json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"report written to {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
