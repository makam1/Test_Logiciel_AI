"""Calcul des métriques et choix du seuil de décision."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(y_true, y_pred, y_proba):
    """Calcule toutes les métriques demandées et la matrice de confusion."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float((tp + tn) / (tn + fp + fn + tp)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "fnr": float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def select_threshold(y_true, y_proba, min_recall=0.80):
    """Choisit, sur la VALIDATION, le seuil qui maximise la précision
    sous la contrainte rappel >= min_recall."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    precision, recall = precision[:-1], recall[:-1]
    acceptable = recall >= min_recall
    if not acceptable.any():
        raise ValueError("Aucun seuil n'atteint le rappel minimal demandé.")
    best = int(np.argmax(np.where(acceptable, precision, -1.0)))
    return float(thresholds[best])
