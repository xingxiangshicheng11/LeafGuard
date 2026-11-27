import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import LeafDiseaseDataset
from src.data.transforms import build_transforms
from src.models.cnn import SmallCNN
from src.models.resnet import build_resnet18
from src.training.checkpoint import load_checkpoint
from src.training.metrics import compute_metrics


def build_model(model_name, num_classes):
    if model_name == "cnn":
        return SmallCNN(num_classes)
    return build_resnet18(num_classes, pretrained=False)


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on a split (default: test)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="splits/test.csv")
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    model_name = checkpoint["model_name"]
    num_classes = checkpoint["num_classes"]
    image_size = checkpoint.get("input_size", 224)

    class_names = [None] * num_classes
    for name, idx in checkpoint["class_to_idx"].items():
        class_names[int(idx)] = name

    model = build_model(model_name, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    dataset = LeafDiseaseDataset(
        args.split,
        project_root=args.data_root,
        transform=build_transforms(image_size, train=False, augment=False),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)

    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
            y_true.extend(labels.tolist())

    metrics = compute_metrics(y_true, y_pred, class_names)
    result = {
        "model_name": model_name,
        "num_classes": num_classes,
        "split": args.split,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "macro_recall": metrics["macro_recall"],
        "class_names": class_names,
        "confusion_matrix": metrics["confusion_matrix"],
    }

    out = args.out
    if out is None:
        out = str(Path(args.checkpoint).parent.parent / "metrics" / f"{model_name}_test_metrics.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"device: {device}")
    print(f"accuracy={result['accuracy']:.4f} macro_f1={result['macro_f1']:.4f} "
          f"macro_recall={result['macro_recall']:.4f}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
