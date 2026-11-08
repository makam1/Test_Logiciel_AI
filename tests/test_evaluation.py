import numpy as np
import pytest

from app.evaluation import evaluate_model, select_threshold


def test_perfect_predictions():
    y_true = [0, 0, 1, 1]
    metrics = evaluate_model(y_true, [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["fpr"] == 0.0
    assert metrics["roc_auc"] == 1.0


def test_confusion_counts_and_rates():
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    y_pred = [0, 0, 1, 0, 1, 1, 0, 0]
    y_proba = [0.1, 0.2, 0.6, 0.3, 0.9, 0.8, 0.4, 0.45]
    m = evaluate_model(y_true, y_pred, y_proba)

    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (3, 1, 2, 2)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(0.5)
    assert m["fpr"] == pytest.approx(0.25)
    assert m["fnr"] == pytest.approx(0.5)


def test_accuracy_is_misleading_on_imbalanced_data():
    """Un modèle qui ne prédit jamais de fraude a 99 % d'accuracy mais détecte 0 fraude."""
    y_true = [0] * 990 + [1] * 10
    y_pred = [0] * 1000
    m = evaluate_model(y_true, y_pred, [0.0] * 1000)

    assert m["accuracy"] == pytest.approx(0.99)
    assert m["recall"] == 0.0
    assert m["fnr"] == 1.0


def test_select_threshold_respects_min_recall():
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.35, 0.5, 0.7, 0.8, 0.9])

    threshold = select_threshold(y_true, y_proba, min_recall=0.8)
    y_pred = (y_proba >= threshold).astype(int)
    m = evaluate_model(y_true, y_pred, y_proba)

    assert m["recall"] >= 0.8
    assert threshold == pytest.approx(0.5)
