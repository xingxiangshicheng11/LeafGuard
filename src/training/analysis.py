import json
from pathlib import Path

import numpy as np


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect(results_root, experiments):
    rows = []
    for exp in experiments:
        metrics_dir = Path(results_root) / exp / "metrics"
        if not metrics_dir.is_dir():
            continue
        for mf in sorted(metrics_dir.glob("*_metrics.json")):
            if mf.name.endswith("_test_metrics.json"):
                continue
            m = load_json(mf)
            rows.append({
                "experiment": exp,
                "model": mf.stem.replace("_metrics", ""),
                "accuracy": m.get("accuracy"),
                "macro_f1": m.get("macro_f1"),
                "macro_recall": m.get("macro_recall"),
                "training_seconds": m.get("training_seconds"),
                "best_val_macro_f1": m.get("best_val_macro_f1"),
            })
    return rows


def top_confusions(confusion, class_names, k=10):
    cm = np.asarray(confusion)
    pairs = []
    n = cm.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                pairs.append((int(cm[i, j]), class_names[i], class_names[j]))
    pairs.sort(key=lambda x: x[0], reverse=True)
    return pairs[:k]


def plot_confusion(confusion, class_names, out_path, title="Confusion matrix"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(confusion)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if class_names and len(class_names) == cm.shape[0]:
        ticks = list(range(cm.shape[0]))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(class_names, rotation=90, fontsize=6)
        ax.set_yticklabels(class_names, fontsize=6)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_history(history, out_path, title="Training curves"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(history.get("train_loss", [])) + 1))
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, history.get("train_loss", []), "b-", label="train_loss")
    ax1.plot(epochs, history.get("val_loss", []), "r-", label="val_loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend(loc="upper right")
    if history.get("val_macro_f1"):
        ax2 = ax1.twinx()
        ax2.plot(epochs, history["val_macro_f1"], "g-", label="val_macro_f1")
        ax2.set_ylabel("val_macro_f1")
        ax2.legend(loc="lower right")
    ax1.set_title(title)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
