from __future__ import annotations

import numpy as np


def compute_external_metrics(y_true, y_pred, labels=None):
    """Compute external-evaluation metrics over the classes present in y_true.

    Predictions outside ``labels`` still count as errors for accuracy, but the
    macro metrics and confusion matrix cover only the evaluated classes.
    """
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        recall_score,
    )

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if labels is None:
        labels = sorted({int(value) for value in y_true})
    else:
        labels = sorted({int(value) for value in labels})
    if not labels:
        raise ValueError("cannot compute metrics without labels")

    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    per_class = {}
    for label in labels:
        entry = report.get(label) or report.get(str(label)) or {}
        per_class[label] = {
            "precision": float(entry.get("precision", 0.0)),
            "recall": float(entry.get("recall", 0.0)),
            "f1": float(entry.get("f1-score", 0.0)),
            "support": int(entry.get("support", 0)),
        }

    label_set = set(labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "labels": labels,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "num_samples": int(len(y_true)),
        "out_of_label_predictions": int(
            sum(1 for value in y_pred if int(value) not in label_set)
        ),
    }
