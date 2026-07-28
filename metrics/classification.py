"""Fixed image-level metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


def classification_metrics(target: np.ndarray, prediction: np.ndarray, labels: list[int]) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(target, prediction, labels=labels, zero_division=0)
    return {"accuracy": float(accuracy_score(target, prediction)), "balanced_accuracy": float(balanced_accuracy_score(target, prediction)), "macro_f1": float(f1_score(target, prediction, labels=labels, average="macro", zero_division=0)), "weighted_f1": float(f1_score(target, prediction, labels=labels, average="weighted", zero_division=0)), "per_class_precision": precision.tolist(), "per_class_recall": recall.tolist(), "per_class_f1": f1.tolist(), "per_class_support": support.tolist(), "confusion_matrix": confusion_matrix(target, prediction, labels=labels).tolist()}
