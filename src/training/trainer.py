import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from src.training.checkpoint import save_checkpoint
from src.training.metrics import compute_metrics


class Trainer:
    def __init__(self, model, device, config, class_names, class_to_idx, model_name, output_dir):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.class_names = list(class_names)
        self.class_to_idx = dict(class_to_idx)
        self.model_name = model_name
        self.output_dir = Path(output_dir)

        training = config["training"]
        self.epochs = int(training.get("epochs", 25))
        self.patience = int(training.get("early_stopping_patience", 5))
        self.weight_decay = float(training.get("weight_decay", 0.0001))
        self.use_amp = bool(training.get("amp", True)) and device.type == "cuda"

        self.criterion = nn.CrossEntropyLoss()
        self.best_val_macro_f1 = -1.0
        self.best_state = None
        self.training_seconds = None
        self.history = {"train_loss": [], "val_loss": [], "val_macro_f1": []}

    def _run_epoch(self, loader, optimizer, scaler, train):
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total = 0
        all_preds = []
        all_labels = []

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            if train:
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                with torch.no_grad():
                    with torch.amp.autocast("cuda", enabled=self.use_amp):
                        logits = self.model(images)
                        loss = self.criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            total += images.size(0)
            all_preds.append(logits.argmax(dim=1).detach().cpu())
            all_labels.append(labels.detach().cpu())

        avg_loss = total_loss / total
        y_true = torch.cat(all_labels).numpy()
        y_pred = torch.cat(all_preds).numpy()
        metrics = compute_metrics(y_true, y_pred, self.class_names)
        return avg_loss, metrics

    def fit(self, train_loader, val_loader, lr):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=float(lr), weight_decay=self.weight_decay
        )
        scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        start = time.time()
        patience_counter = 0
        for epoch in range(1, self.epochs + 1):
            train_loss, _ = self._run_epoch(train_loader, optimizer, scaler, train=True)
            val_loss, val_metrics = self._run_epoch(val_loader, optimizer, scaler, train=False)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_macro_f1"].append(val_metrics["macro_f1"])

            print(
                f"Epoch {epoch}/{self.epochs} | "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_macro_f1={val_metrics['macro_f1']:.4f}"
            )

            if val_metrics["macro_f1"] > self.best_val_macro_f1:
                self.best_val_macro_f1 = val_metrics["macro_f1"]
                self.best_state = {
                    name: param.detach().cpu().clone()
                    for name, param in self.model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        self.training_seconds = time.time() - start

        if self.best_state is None:
            raise RuntimeError("No validation epoch completed; cannot save a checkpoint")

        self.model.load_state_dict(self.best_state)

        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"{self.model_name}_best.pt"
        save_checkpoint(
            checkpoint_path,
            self.model,
            num_classes=len(self.class_names),
            class_to_idx=self.class_to_idx,
            model_name=self.model_name,
            input_size=self.config["data"]["image_size"],
            config=self.config,
            best_val_macro_f1=self.best_val_macro_f1,
            epoch=len(self.history["val_macro_f1"]),
        )

        final_loss, final_metrics = self._run_epoch(val_loader, optimizer, scaler, train=False)
        metrics_json = {
            "accuracy": final_metrics["accuracy"],
            "macro_f1": final_metrics["macro_f1"],
            "macro_recall": final_metrics["macro_recall"],
            "training_seconds": self.training_seconds,
            "class_names": self.class_names,
            "confusion_matrix": final_metrics["confusion_matrix"],
            "best_val_macro_f1": self.best_val_macro_f1,
        }
        metrics_dir = self.output_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = metrics_dir / f"{self.model_name}_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_json, f, ensure_ascii=False, indent=2)

        history_path = metrics_dir / f"{self.model_name}_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

        return self.best_val_macro_f1, self.history
