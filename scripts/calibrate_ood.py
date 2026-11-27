import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
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
from src.data.transforms import build_transforms


def main():
    parser = argparse.ArgumentParser(description="Calibrate OOD rejection thresholds on a split")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="splits/val.csv")
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--classes", default="splits/class_to_idx.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--msp-quantile", type=float, default=0.01)
    parser.add_argument("--energy-quantile", type=float, default=0.99)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    ckpt = load_checkpoint(args.checkpoint, device)
    if "model_name" not in ckpt:
        raise SystemExit("checkpoint missing required field: model_name")
    class_to_idx, _ = load_class_to_idx(ckpt, args.classes)
    num_classes = int(ckpt.get("num_classes", len(class_to_idx)))
    model = build_model(ckpt["model_name"], num_classes)
    state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict")
    if state_dict is None:
        raise SystemExit("checkpoint missing required field: model_state_dict")
    model.load_state_dict(strip_module_prefix(state_dict))
    model.to(device).eval()

    input_size = int(ckpt.get("input_size", 224))
    dataset = LeafDiseaseDataset(
        args.split,
        project_root=args.data_root,
        transform=build_transforms(input_size, train=False, augment=False),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    msps = []
    energies = []
    with torch.no_grad():
        for images, _labels in loader:
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1)
            msps.append(probs.max(dim=1).values.cpu().numpy())
            energies.append((-torch.logsumexp(logits, dim=1)).cpu().numpy())

    msp_arr = np.concatenate(msps)
    energy_arr = np.concatenate(energies)
    msp_threshold = float(np.quantile(msp_arr, args.msp_quantile))
    energy_threshold = float(np.quantile(energy_arr, args.energy_quantile))

    result = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "num_samples": int(len(msp_arr)),
        "msp_quantile": args.msp_quantile,
        "msp_threshold": msp_threshold,
        "energy_quantile": args.energy_quantile,
        "energy_threshold": energy_threshold,
        "note": "Reject when msp < msp_threshold OR energy > energy_threshold.",
    }

    out = args.out
    if out is None:
        out = str(Path(args.checkpoint).parent.parent / "metrics" / "ood_thresholds.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"msp_threshold={msp_threshold:.4f}  energy_threshold={energy_threshold:.4f}  n={len(msp_arr)}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
