import json

import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader, TensorDataset

from src.models.cnn import SmallCNN
from src.training.checkpoint import load_checkpoint
from src.training.trainer import Trainer


def _make_loaders(num_samples=16, num_classes=4, image_size=32, batch_size=8, seed=42):
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(num_samples, 3, image_size, image_size, generator=generator)
    labels = torch.randint(0, num_classes, (num_samples,), generator=generator)
    dataset = TensorDataset(images, labels)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def test_trainer_smoke_saves_checkpoint_and_metrics(tmp_path):
    image_size = 32
    num_classes = 4
    config = {
        "data": {"image_size": image_size},
        "training": {"epochs": 2, "early_stopping_patience": 5, "weight_decay": 0.0001, "amp": False},
    }
    model = SmallCNN(num_classes)
    class_names = [str(i) for i in range(num_classes)]
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    train_loader, val_loader = _make_loaders(
        num_samples=16, num_classes=num_classes, image_size=image_size
    )

    trainer = Trainer(
        model=model,
        device=torch.device("cpu"),
        config=config,
        class_names=class_names,
        class_to_idx=class_to_idx,
        model_name="cnn",
        output_dir=tmp_path,
    )
    best_macro_f1, history = trainer.fit(train_loader, val_loader, lr=1e-3)

    assert len(history["val_macro_f1"]) == 2
    assert best_macro_f1 >= 0.0

    checkpoint = tmp_path / "checkpoints" / "cnn_best.pt"
    assert checkpoint.exists()

    ckpt = load_checkpoint(checkpoint)
    for key in ["model_state_dict", "num_classes", "class_to_idx", "model_name", "input_size", "config", "best_val_macro_f1"]:
        assert key in ckpt
    assert ckpt["num_classes"] == num_classes
    assert ckpt["model_name"] == "cnn"

    metrics_path = tmp_path / "metrics" / "cnn_metrics.json"
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    for key in ["accuracy", "macro_f1", "macro_recall", "training_seconds", "class_names"]:
        assert key in metrics
    assert metrics["class_names"] == class_names
