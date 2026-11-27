import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from src.data.dataset import LeafDiseaseDataset
from src.data.transforms import build_transforms
from src.models.cnn import SmallCNN
from src.models.resnet import build_resnet18
from src.training.config import load_config
from src.training.trainer import Trainer
from src.utils.reproducibility import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train a leaf disease classification model")
    parser.add_argument("--model", choices=["cnn", "resnet18"], required=True)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--freeze", action="store_true", help="freeze ResNet18 backbone")
    parser.add_argument("--finetune-layer4", action="store_true", help="unfreeze ResNet18 layer4")
    parser.add_argument("--pretrained", action="store_true", help="use ImageNet pretrained ResNet18")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--augment", dest="augment", action="store_true", default=None,
                        help="force-enable training augmentation")
    parser.add_argument("--no-augment", dest="augment", action="store_false",
                        help="disable training augmentation")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-classes", type=int, default=38)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--split-dir", default="splits")
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--smoke-size", type=int, default=0,
                        help="if >0, train on random tensors instead of real data")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_model(model_name, num_classes, freeze, finetune_layer4, pretrained):
    if model_name == "cnn":
        return SmallCNN(num_classes)
    return build_resnet18(
        num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze,
        finetune_layer4=finetune_layer4,
    )


def build_smoke_loaders(num_classes, image_size, batch_size, smoke_size, num_workers, generator):
    total = max(smoke_size, 2)
    images = torch.randn(total, 3, image_size, image_size, generator=generator)
    labels = torch.randint(0, num_classes, (total,), generator=generator)
    dataset = TensorDataset(images, labels)
    val_size = max(1, int(total * 0.2))
    train_ds, val_ds = random_split(dataset, [total - val_size, val_size], generator=generator)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def build_real_loaders(config, split_dir, data_root, image_size, batch_size, num_workers,
                       augment_override=None):
    split_dir = Path(split_dir)
    class_to_idx_path = split_dir / "class_to_idx.json"
    if not class_to_idx_path.exists():
        sys.exit(f"Missing {class_to_idx_path}; run data pipeline first or use --smoke-size")
    with open(class_to_idx_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)

    train_csv = split_dir / "train.csv"
    val_csv = split_dir / "val.csv"
    if not train_csv.exists() or not val_csv.exists():
        sys.exit(f"Missing train.csv/val.csv in {split_dir}; use --smoke-size instead")

    if augment_override is None:
        augment = bool(config["augmentation"]["enabled"])
    else:
        augment = augment_override
    train_ds = LeafDiseaseDataset(
        train_csv,
        project_root=data_root,
        transform=build_transforms(image_size, train=True, augment=augment),
    )
    val_ds = LeafDiseaseDataset(
        val_csv,
        project_root=data_root,
        transform=build_transforms(image_size, train=False, augment=False),
    )
    persistent = num_workers > 0
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              persistent_workers=persistent)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            persistent_workers=persistent)
    return train_loader, val_loader, class_to_idx


def main():
    args = parse_args()
    config = load_config(args.config)

    seed = args.seed if args.seed is not None else config["seed"]
    set_seed(seed)

    device = resolve_device(args.device)
    image_size = config["data"]["image_size"]
    batch_size = args.batch_size or config["training"]["batch_size"]
    num_workers = args.num_workers if args.num_workers is not None else config["data"]["num_workers"]
    epochs = args.epochs if args.epochs is not None else config["training"]["epochs"]

    generator = torch.Generator().manual_seed(seed)

    if args.lr is not None:
        lr = args.lr
    elif args.model == "cnn":
        lr = config["training"]["cnn_learning_rate"]
    elif args.finetune_layer4:
        lr = config["training"]["finetune_learning_rate"]
    else:
        lr = config["training"]["classifier_learning_rate"]

    if args.augment is None:
        augment = bool(config["augmentation"]["enabled"])
    else:
        augment = args.augment

    if args.smoke_size and args.smoke_size > 0:
        num_classes = args.num_classes
        class_names = [str(i) for i in range(num_classes)]
        class_to_idx = {name: i for i, name in enumerate(class_names)}
        train_loader, val_loader = build_smoke_loaders(
            num_classes, image_size, batch_size, args.smoke_size, 0, generator
        )
        mode = "smoke"
    else:
        train_loader, val_loader, class_to_idx = build_real_loaders(
            config, args.split_dir, args.data_root, image_size, batch_size, num_workers, augment
        )
        idx_to_name = {v: k for k, v in class_to_idx.items()}
        num_classes = len(class_to_idx)
        class_names = [idx_to_name[i] for i in range(num_classes)]
        mode = "real"

    model = build_model(args.model, num_classes, args.freeze, args.finetune_layer4, args.pretrained)

    print("=== Training configuration ===")
    print(f"mode: {mode}")
    print(f"model: {args.model}")
    print(f"device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"num_classes: {num_classes}")
    print(f"train samples: {len(train_loader.dataset)}")
    print(f"val samples: {len(val_loader.dataset)}")
    print(f"batch_size: {batch_size}, epochs: {epochs}, lr: {lr}")
    print(f"augmentation: {augment}")
    if args.model == "resnet18":
        print(f"freeze: {args.freeze}, finetune_layer4: {args.finetune_layer4}, pretrained: {args.pretrained}")

    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs

    trainer = Trainer(
        model=model,
        device=device,
        config=config,
        class_names=class_names,
        class_to_idx=class_to_idx,
        model_name=args.model,
        output_dir=args.output_dir,
    )
    best_macro_f1, _ = trainer.fit(train_loader, val_loader, lr=lr)
    print(f"Training finished. best_val_macro_f1={best_macro_f1:.4f}")


if __name__ == "__main__":
    main()
