import argparse
import csv
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from scripts.predict import (
    build_model,
    load_checkpoint,
    load_class_to_idx,
    resolve_device,
    strip_module_prefix,
)
from src.data.dataset import LeafDiseaseDataset
from src.data.evaluation import compute_external_metrics
from src.data.transforms import build_transforms


def evaluate_manifest(
    checkpoint_path,
    manifest_csv,
    classes_path="splits/class_to_idx.json",
    project_root=".",
    device_arg="auto",
    batch_size=32,
    num_workers=4,
):
    device = resolve_device(device_arg)
    checkpoint = load_checkpoint(checkpoint_path, device)
    if "model_name" not in checkpoint:
        raise ValueError("checkpoint missing required field: model_name")
    if "model_state_dict" not in checkpoint and "state_dict" not in checkpoint:
        raise ValueError("checkpoint missing required field: model_state_dict")

    class_to_idx, mapping_source = load_class_to_idx(checkpoint, classes_path)
    num_classes = int(checkpoint.get("num_classes", len(class_to_idx)))
    if num_classes != len(class_to_idx):
        raise ValueError(
            f"num_classes={num_classes} does not match class mapping size={len(class_to_idx)}"
        )
    idx_to_class = {index: name for name, index in class_to_idx.items()}

    model = build_model(checkpoint["model_name"], num_classes)
    state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
    model.load_state_dict(strip_module_prefix(state_dict))
    model.to(device).eval()

    input_size = int(checkpoint.get("input_size", 224))
    dataset = LeafDiseaseDataset(
        manifest_csv,
        project_root=project_root,
        transform=build_transforms(image_size=input_size, train=False, augment=False),
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    y_true = []
    y_pred = []
    rows = []
    offset = 0
    started = time.perf_counter()

    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            probabilities = torch.softmax(logits, dim=1)
            confidences, predictions = probabilities.max(dim=1)
            top_k = min(3, num_classes)
            top_scores, top_indices = probabilities.topk(top_k, dim=1)

            for row_index in range(len(labels)):
                item = dataset.items.iloc[offset + row_index]
                true_label = int(labels[row_index])
                pred_label = int(predictions[row_index])
                y_true.append(true_label)
                y_pred.append(pred_label)
                rows.append(
                    {
                        "path": item["path"],
                        "true_label": true_label,
                        "true_class": idx_to_class.get(true_label, str(true_label)),
                        "pred_label": pred_label,
                        "pred_class": idx_to_class.get(pred_label, str(pred_label)),
                        "confidence": float(confidences[row_index]),
                        "correct": int(pred_label == true_label),
                        "top3_classes": "|".join(
                            idx_to_class.get(int(index), str(int(index)))
                            for index in top_indices[row_index].tolist()
                        ),
                        "top3_scores": "|".join(
                            f"{float(score):.6f}"
                            for score in top_scores[row_index].tolist()
                        ),
                    }
                )
            offset += len(labels)

    elapsed = time.perf_counter() - started
    metrics = compute_external_metrics(y_true, y_pred)
    evaluated_names = [
        idx_to_class.get(int(label), str(label)) for label in metrics["labels"]
    ]
    per_class_named = {
        idx_to_class.get(int(label), str(label)): values
        for label, values in metrics["per_class"].items()
    }

    result = {
        "dataset": "PlantDoc external evaluation",
        "manifest": str(manifest_csv),
        "checkpoint": str(checkpoint_path),
        "model_name": checkpoint.get("model_name"),
        "input_size": input_size,
        "device": str(device),
        "num_samples": metrics["num_samples"],
        "num_model_classes": num_classes,
        "num_evaluated_classes": len(metrics["labels"]),
        "class_names": evaluated_names,
        "labels": [int(label) for label in metrics["labels"]],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "macro_recall": metrics["macro_recall"],
        "confusion_matrix": metrics["confusion_matrix"],
        "per_class": per_class_named,
        "out_of_label_predictions": metrics["out_of_label_predictions"],
        "class_mapping_source": mapping_source,
        "evaluation_seconds": round(elapsed, 3),
        "note": (
            "External evaluation only; the external data is not used for training "
            "or tuning. Scores are softmax values, not calibrated probabilities."
        ),
    }
    return result, rows


def write_outputs(result, rows, metrics_path, predictions_path=None):
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if predictions_path:
        predictions_path = Path(predictions_path)
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "path",
            "true_label",
            "true_class",
            "pred_label",
            "pred_class",
            "confidence",
            "correct",
            "top3_classes",
            "top3_scores",
        ]
        with open(predictions_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint on PlantDoc (external evaluation)."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="splits/plantdoc_eval.csv")
    parser.add_argument("--classes", default="splits/class_to_idx.json")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--metrics-out", default="results/metrics/external_metrics.json"
    )
    parser.add_argument(
        "--predictions-out", default="results/metrics/external_predictions.csv"
    )
    parser.add_argument("--no-predictions", action="store_true")
    args = parser.parse_args()

    if not Path(args.checkpoint).is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    if not Path(args.manifest).is_file():
        raise SystemExit(f"manifest not found: {args.manifest}")

    result, rows = evaluate_manifest(
        args.checkpoint,
        args.manifest,
        classes_path=args.classes,
        project_root=args.project_root,
        device_arg=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    write_outputs(
        result,
        rows,
        args.metrics_out,
        None if args.no_predictions else args.predictions_out,
    )

    print(f"samples: {result['num_samples']}")
    print(f"evaluated classes: {result['num_evaluated_classes']}")
    print(f"accuracy: {result['accuracy']:.4f}")
    print(f"macro_f1: {result['macro_f1']:.4f}")
    print(f"macro_recall: {result['macro_recall']:.4f}")
    print(f"metrics written to {args.metrics_out}")
    if not args.no_predictions:
        print(f"predictions written to {args.predictions_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
