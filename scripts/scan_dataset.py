import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.scan import scan_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Scan a PlantVillage-style dataset (one folder per class)."
    )
    parser.add_argument("--data-root", default="data/plantvillage")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--json", dest="json_path", default="results/data_scan.json")
    parser.add_argument(
        "--hash-duplicates",
        action="store_true",
        help="Compute SHA-256 hashes to find exact duplicate images (slower).",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = project_root / data_root
    if not data_root.is_dir():
        print(f"[ERROR] data root not found: {data_root}")
        print("Place PlantVillage under data/plantvillage/<class_name>/<image>.")
        return 1

    report = scan_dataset(
        data_root,
        project_root=project_root,
        hash_duplicates=args.hash_duplicates,
    )

    print(f"data root : {report['data_root']}")
    print(f"classes   : {report['num_classes']}")
    print(f"files     : {report['num_files']}")
    print(f"valid     : {report['num_valid_images']}")
    print(f"damaged   : {report['num_damaged']}")
    if report["duplicates"]:
        print(f"duplicates: {sum(len(g) - 1 for g in report['duplicates'])} extra copies")
    if report["cross_class_duplicates"]:
        print(
            "cross-class duplicates: "
            f"{len(report['cross_class_duplicates'])} group(s), review labels manually"
        )
    print("\nper-class counts:")
    for name, count in report["classes"].items():
        print(f"  {name}: {count}")

    if report["damaged"]:
        print("\ndamaged images (first 20):")
        for path in report["damaged"][:20]:
            print(f"  {path}")

    json_path = Path(args.json_path)
    if not json_path.is_absolute():
        json_path = project_root / json_path
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nreport written to {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
