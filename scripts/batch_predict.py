import argparse
import csv
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from scripts.predict import (
    build_model,
    load_checkpoint,
    load_class_to_idx,
    resolve_device,
    strip_module_prefix,
)
from src.data.transforms import build_transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def display_path(path):
    path = Path(path)
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


class ImageListDataset(Dataset):
    def __init__(self, paths, transform, image_size):
        self.paths = [Path(path) for path in paths]
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        try:
            with Image.open(path) as image:
                tensor = self.transform(image.convert("RGB"))
            return tensor, index, ""
        except Exception as exc:
            return torch.zeros(3, self.image_size, self.image_size), index, str(exc)


def collect_images(input_path, recursive=True, extensions=None):
    input_path = Path(input_path)
    extensions = {ext.lower() for ext in (extensions or IMAGE_EXTENSIONS)}
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"input not found: {input_path}")
    pattern = "**/*" if recursive else "*"
    return sorted(
        path
        for path in input_path.glob(pattern)
        if path.is_file() and path.suffix.lower() in extensions
    )


def predict_batch(
    image_paths,
    checkpoint_path,
    classes_path="splits/class_to_idx.json",
    device_arg="auto",
    batch_size=32,
    num_workers=4,
    top_k=3,
):
    if not image_paths:
        raise ValueError("no images to predict")

    device = resolve_device(device_arg)
    checkpoint = load_checkpoint(checkpoint_path, device)
    if "model_name" not in checkpoint:
        raise ValueError("checkpoint missing required field: model_name")

    class_to_idx, mapping_source = load_class_to_idx(checkpoint, classes_path)
    num_classes = int(checkpoint.get("num_classes", len(class_to_idx)))
    if num_classes != len(class_to_idx):
        raise ValueError(
            f"num_classes={num_classes} does not match class mapping size={len(class_to_idx)}"
        )
    idx_to_class = {index: name for name, index in class_to_idx.items()}

    model = build_model(checkpoint["model_name"], num_classes)
    state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
    if state_dict is None:
        raise ValueError("checkpoint missing required field: model_state_dict")
    model.load_state_dict(strip_module_prefix(state_dict))
    model.to(device).eval()

    input_size = int(checkpoint.get("input_size", 224))
    transform = build_transforms(image_size=input_size, train=False, augment=False)
    dataset = ImageListDataset(image_paths, transform, input_size)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    k = min(max(1, top_k), num_classes)
    rows = []
    started = time.perf_counter()

    with torch.no_grad():
        for images, indices, errors in loader:
            valid_positions = [
                position for position, error in enumerate(errors) if not error
            ]
            if valid_positions:
                valid_images = images[valid_positions].to(device)
                logits = model(valid_images)
                probabilities = torch.softmax(logits, dim=1)
                scores, predictions = probabilities.topk(k, dim=1)
            else:
                scores = predictions = None

            for position, index in enumerate(indices.tolist()):
                path = display_path(dataset.paths[index])
                error = errors[position]
                if error:
                    rows.append(
                        {
                            "path": path,
                            "pred_class": "",
                            "pred_label": "",
                            "confidence": "",
                            "top_k_classes": "",
                            "top_k_scores": "",
                            "error": error,
                        }
                    )
                    continue
                row_position = valid_positions.index(position)
                row_scores = scores[row_position].tolist()
                row_predictions = predictions[row_position].tolist()
                rows.append(
                    {
                        "path": path,
                        "pred_class": idx_to_class[int(row_predictions[0])],
                        "pred_label": int(row_predictions[0]),
                        "confidence": float(row_scores[0]),
                        "top_k_classes": "|".join(
                            idx_to_class[int(label)] for label in row_predictions
                        ),
                        "top_k_scores": "|".join(
                            f"{float(score):.6f}" for score in row_scores
                        ),
                        "error": "",
                    }
                )

    elapsed = time.perf_counter() - started
    ok_rows = [row for row in rows if not row["error"]]
    failed_rows = [row for row in rows if row["error"]]
    counts = {}
    for row in ok_rows:
        counts[row["pred_class"]] = counts.get(row["pred_class"], 0) + 1

    summary = {
        "checkpoint": str(checkpoint_path),
        "model_name": checkpoint.get("model_name"),
        "input_size": input_size,
        "device": str(device),
        "class_mapping_source": mapping_source,
        "num_images": len(rows),
        "num_ok": len(ok_rows),
        "num_failed": len(failed_rows),
        "predicted_class_counts": dict(sorted(counts.items())),
        "seconds": round(elapsed, 3),
        "failures": [
            {"path": row["path"], "error": row["error"]} for row in failed_rows
        ],
        "note": "Scores are softmax values, not calibrated probabilities.",
    }
    return rows, summary


def write_outputs(rows, summary, output_csv, summary_json):
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "path",
        "pred_class",
        "pred_label",
        "confidence",
        "top_k_classes",
        "top_k_scores",
        "error",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if summary_json:
        summary_json = Path(summary_json)
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def main():
    parser = argparse.ArgumentParser(
        description="Batch prediction for a folder of leaf images."
    )
    parser.add_argument("--input", required=True, help="image file or directory")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--classes", default="splits/class_to_idx.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument(
        "--output-csv", default="results/predictions/batch_predictions.csv"
    )
    parser.add_argument(
        "--summary-json", default="results/predictions/batch_summary.json"
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    image_paths = collect_images(
        args.input, recursive=not args.no_recursive
    )
    if args.limit > 0:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise SystemExit(f"no images found under {args.input}")
    if not Path(args.checkpoint).is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    rows, summary = predict_batch(
        image_paths,
        args.checkpoint,
        classes_path=args.classes,
        device_arg=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        top_k=args.top_k,
    )
    write_outputs(rows, summary, args.output_csv, args.summary_json)

    print(f"images: {summary['num_images']}")
    print(f"ok: {summary['num_ok']}, failed: {summary['num_failed']}")
    if summary["predicted_class_counts"]:
        print("predicted classes:")
        for name, count in summary["predicted_class_counts"].items():
            print(f"  {name}: {count}")
    print(f"predictions written to {args.output_csv}")
    if args.summary_json:
        print(f"summary written to {args.summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
