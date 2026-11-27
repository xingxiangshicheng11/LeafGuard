import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image

from src.data.transforms import build_transforms
from src.models.cnn import SmallCNN
from src.models.resnet import build_resnet18

MODEL_ALIASES = {
    "cnn": "cnn",
    "smallcnn": "cnn",
    "small_cnn": "cnn",
    "resnet": "resnet18",
    "resnet18": "resnet18",
}


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def build_model(model_name, num_classes):
    key = str(model_name).strip().lower()
    key = MODEL_ALIASES.get(key, key)
    if key == "cnn":
        return SmallCNN(num_classes)
    if key == "resnet18":
        return build_resnet18(num_classes, pretrained=False, freeze_backbone=False)
    raise ValueError(
        f"unsupported model_name: {model_name!r}; expected cnn or resnet18"
    )


def strip_module_prefix(state_dict):
    if any(key.startswith("module.") for key in state_dict):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_class_to_idx(checkpoint, classes_path):
    if "class_to_idx" in checkpoint and checkpoint["class_to_idx"]:
        return dict(checkpoint["class_to_idx"]), "checkpoint"
    classes_path = Path(classes_path)
    if not classes_path.is_file():
        raise FileNotFoundError(
            f"class mapping not found in checkpoint or file: {classes_path}"
        )
    with open(classes_path, encoding="utf-8") as f:
        return json.load(f), str(classes_path)


def main():
    parser = argparse.ArgumentParser(description="LeafGuard single-image prediction")
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--classes", default="splits/class_to_idx.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    image_path = Path(args.image)
    checkpoint_path = Path(args.checkpoint)
    if not image_path.is_file():
        raise SystemExit(f"image not found: {image_path}")
    if not checkpoint_path.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint_path}")

    device = resolve_device(args.device)
    checkpoint = load_checkpoint(checkpoint_path, device)

    if "model_name" not in checkpoint:
        raise SystemExit("checkpoint missing required field: model_name")
    if "num_classes" not in checkpoint and "class_to_idx" not in checkpoint:
        raise SystemExit("checkpoint missing required field: num_classes/class_to_idx")

    class_to_idx, mapping_source = load_class_to_idx(checkpoint, args.classes)
    num_classes = int(checkpoint.get("num_classes", len(class_to_idx)))
    if num_classes != len(class_to_idx):
        raise SystemExit(
            f"num_classes={num_classes} does not match class mapping size={len(class_to_idx)}"
        )

    model = build_model(checkpoint["model_name"], num_classes)
    state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
    if state_dict is None:
        raise SystemExit("checkpoint missing required field: model_state_dict")
    model.load_state_dict(strip_module_prefix(state_dict))
    model.to(device).eval()

    input_size = int(checkpoint.get("input_size", 224))
    transform = build_transforms(image_size=input_size, train=False, augment=False)
    with Image.open(image_path) as im:
        tensor = transform(im.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    top_k = min(max(1, args.top_k), num_classes)
    scores, indices = probabilities.topk(top_k)
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    predictions = [
        {
            "class_name": idx_to_class[int(idx)],
            "label": int(idx),
            "score": float(score),
        }
        for score, idx in zip(scores.tolist(), indices.tolist())
    ]

    advice_map = {}
    advice_path = PROJECT_ROOT / "assets" / "disease_advice_zh.json"
    if advice_path.is_file():
        with open(advice_path, encoding="utf-8") as f:
            advice_map = json.load(f)

    result = {
        "image": str(image_path),
        "checkpoint": str(checkpoint_path),
        "model_name": checkpoint["model_name"],
        "class_mapping_source": mapping_source,
        "prediction": predictions[0]["class_name"],
        "top_k": predictions,
        "advice": advice_map.get(predictions[0]["class_name"]),
        "note": "scores are softmax values, not calibrated diagnostic probabilities",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
