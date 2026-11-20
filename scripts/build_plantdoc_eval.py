import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.external import external_summary, map_external_items
from src.data.scan import scan_dataset
from src.data.splits import remove_duplicate_copies, write_manifest


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Build a PlantDoc external-evaluation manifest in the PlantVillage label space."
    )
    parser.add_argument("--plantdoc-root", default="data/plantdoc")
    parser.add_argument("--class-map", default="splits/plantdoc_class_map.json")
    parser.add_argument("--class-to-idx", default="splits/class_to_idx.json")
    parser.add_argument("--out", default="splits/plantdoc_eval.csv")
    parser.add_argument("--report", default="results/plantdoc_eval_report.json")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    plantdoc_root = Path(args.plantdoc_root)
    if not plantdoc_root.is_absolute():
        plantdoc_root = project_root / plantdoc_root
    if not plantdoc_root.is_dir():
        print(f"[ERROR] PlantDoc root not found: {plantdoc_root}")
        return 1

    class_map_path = Path(args.class_map)
    if not class_map_path.is_absolute():
        class_map_path = project_root / class_map_path
    class_to_idx_path = Path(args.class_to_idx)
    if not class_to_idx_path.is_absolute():
        class_to_idx_path = project_root / class_to_idx_path

    class_map_doc = load_json(class_map_path)
    class_map = class_map_doc.get("mapping", class_map_doc)
    class_to_idx = load_json(class_to_idx_path)

    report = scan_dataset(
        plantdoc_root,
        project_root=project_root,
        hash_duplicates=True,
        include_images=True,
    )
    items = report["images"]
    duplicate_copies = 0
    if report["duplicates"]:
        items, removed = remove_duplicate_copies(items, report["duplicates"])
        duplicate_copies = len(removed)

    mapped, skipped = map_external_items(items, class_map, class_to_idx)
    if not mapped:
        print("[ERROR] no PlantDoc images matched the mapping")
        return 1

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = project_root / out_path
    write_manifest(mapped, out_path, class_to_idx)

    summary = external_summary(
        mapped,
        skipped,
        damaged_count=report["num_damaged"],
        duplicate_count=duplicate_copies,
    )
    summary["plantdoc_root"] = report["data_root"]
    summary["class_map"] = str(class_map_path.relative_to(project_root))
    summary["manifest"] = str(out_path.relative_to(project_root))
    summary["notes"] = class_map_doc.get("notes", [])

    print(f"PlantDoc classes scanned: {report['num_classes']}")
    print(f"PlantDoc images: {report['num_files']} (valid {report['num_valid_images']})")
    print(f"mapped images: {summary['num_items']}")
    print(f"target PlantVillage classes: {summary['num_target_classes']}")
    print(f"unmapped classes skipped: {len(skipped)}")
    print(f"damaged excluded: {report['num_damaged']}")
    print(f"duplicate copies excluded: {duplicate_copies}")
    print(f"manifest written to {out_path}")

    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = project_root / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"report written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
