import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score


def compute_metrics(y_true, y_pred, class_names=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if class_names is not None:
        labels = list(range(len(class_names)))
    else:
        labels = None
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    return metrics
