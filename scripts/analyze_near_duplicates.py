import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.near_duplicates import dhash, group_near_duplicates
from src.data.scan import scan_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Analyze perceptual near-duplicates in PlantVillage."
    )
    parser.add_argument("--data-root", default="data/plantvillage")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--max-distance", type=int, default=4)
    parser.add_argument("--hash-size", type=int, default=8)
    parser.add_argument("--json", dest="json_path", default="results/near_duplicates.json")
    parser.add_argument("--show-top", type=int, default=10)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = project_root / data_root
    if not data_root.is_dir():
        print(f"[ERROR] data root not found: {data_root}")
        return 1

    report = scan_dataset(
        data_root,
        project_root=project_root,
        hash_duplicates=True,
        include_images=True,
    )
    items = report["images"]
    if not items:
        print("[ERROR] no valid images")
        return 1

    started = time.time()
    records = []
    for index, item in enumerate(items):
        value, bits = dhash(project_root / item["path"], hash_size=args.hash_size)
        records.append({"hash": value, "bits": bits})
        if (index + 1) % 5000 == 0:
            print(f"hashed {index + 1}/{len(items)}", flush=True)
    hash_seconds = time.time() - started

    groups, stats = group_near_duplicates(
        records, max_distance=args.max_distance
    )

    size_histogram = Counter(len(group) for group in groups)
    multi_groups = []
    cross_class_groups = []
    per_class_grouped = Counter()
    for group in groups:
        if len(group) < 2:
            continue
        paths = [items[index]["path"] for index in group]
        classes = sorted({items[index]["class_name"] for index in group})
        entry = {"size": len(group), "classes": classes, "paths": paths}
        multi_groups.append(entry)
        if len(classes) > 1:
            cross_class_groups.append(entry)
        for class_name in classes:
            per_class_grouped[class_name] += sum(
                1 for index in group if items[index]["class_name"] == class_name
            )

    summary = {
        "data_root": report["data_root"],
        "num_images": len(items),
        "num_damaged": report["num_damaged"],
        "exact_duplicate_groups": len(report["duplicates"]),
        "exact_duplicate_extra_copies": sum(
            len(group) - 1 for group in report["duplicates"]
        ),
        "cross_class_exact_duplicate_groups": len(report["cross_class_duplicates"]),
        "max_distance": args.max_distance,
        "hash_size": args.hash_size,
        "hash_seconds": round(hash_seconds, 3),
        **stats,
        "num_cross_class_groups": len(cross_class_groups),
        "cross_class_grouped_images": sum(
            entry["size"] for entry in cross_class_groups
        ),
        "size_histogram": {str(size): count for size, count in sorted(size_histogram.items())},
        "per_class_grouped_images": dict(sorted(per_class_grouped.items())),
        "multi_groups": multi_groups,
        "cross_class_groups": cross_class_groups,
    }

    print(f"images: {len(items)}")
    print(f"dHash: {args.hash_size * args.hash_size}-bit, max_distance={args.max_distance}")
    print(f"groups total: {summary['num_groups']}")
    print(f"multi-image groups: {summary['num_multi_groups']}")
    print(f"images in multi-image groups: {summary['num_grouped_images']}")
    print(f"largest group: {summary['largest_group']}")
    print(f"cross-class groups: {len(cross_class_groups)}")
    print(f"hash seconds: {summary['hash_seconds']}")
    print("size histogram:", summary["size_histogram"])

    top = sorted(multi_groups, key=lambda entry: -entry["size"])[: args.show_top]
    print("\nlargest groups:")
    for entry in top:
        print(f"  size={entry['size']} classes={entry['classes']}")
        for path in entry["paths"][:3]:
            print(f"    {path}")

    json_path = Path(args.json_path)
    if not json_path.is_absolute():
        json_path = project_root / json_path
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nreport written to {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
