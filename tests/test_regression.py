from pathlib import Path

import pytest

from app.regression import (
    BASELINE_PATH,
    DATA_PATH,
    MAX_FPR,
    MIN_RECALL,
    compare_to_baseline,
    compute_validation_metrics,
    decide,
    load_baseline,
)
from app.training import load_data

REQUIRED_BASELINE_KEYS = {
    "code_version",
    "model",
    "random_state",
    "threshold",
    "metrics_validation",
    "library_versions",
    "dataset",
}

BASE = {"recall": 0.85, "precision": 0.70, "pr_auc": 0.69, "fpr": 0.0006}

requires_data = pytest.mark.skipif(
    not Path(DATA_PATH).is_file(),
    reason="creditcard.csv absent (données non versionnées)",
)


# T06 : résultats reproductibles
def test_reproducibility(synthetic_df):
    threshold_1, metrics_1 = compute_validation_metrics(synthetic_df)
    threshold_2, metrics_2 = compute_validation_metrics(synthetic_df)

    assert threshold_1 == pytest.approx(threshold_2, abs=1e-12)
    for name, value in metrics_1.items():
        assert value == pytest.approx(metrics_2[name], abs=1e-9)


# T13 : le seuil choisi sur la validation respecte le rappel minimal
def test_recall_threshold(synthetic_df):
    _, metrics = compute_validation_metrics(synthetic_df)
    assert metrics["recall"] >= MIN_RECALL


def test_baseline_file_is_complete():
    baseline = load_baseline(BASELINE_PATH)
    assert REQUIRED_BASELINE_KEYS.issubset(baseline)
    assert {"recall", "precision", "pr_auc", "fpr"}.issubset(baseline["metrics_validation"])


def test_baseline_meets_acceptance_criteria():
    metrics = load_baseline(BASELINE_PATH)["metrics_validation"]
    assert metrics["recall"] >= MIN_RECALL
    assert metrics["fpr"] <= MAX_FPR


def test_load_baseline_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="introuvable"):
        load_baseline(tmp_path / "absente.json")


def test_fpr_increase_is_a_degradation():
    degradations = compare_to_baseline({**BASE, "fpr": 0.0016}, BASE)
    assert degradations["fpr"] == pytest.approx(0.001)
    assert degradations["recall"] == pytest.approx(0.0)


DECISION_CASES = [
    pytest.param({}, "ACCEPTER", id="identique"),
    pytest.param({"recall": 0.83}, "ACCEPTER", id="degradation_dans_la_tolerance"),
    pytest.param({"pr_auc": 0.64}, "REJETER", id="pr_auc_degrade"),
    pytest.param({"recall": 0.79}, "REJETER", id="rappel_sous_le_seuil"),
    pytest.param({"precision": 0.60}, "INVESTIGUER", id="precision_degradee"),
    pytest.param({"fpr": 0.0016}, "INVESTIGUER", id="fpr_degrade"),
    pytest.param({"fpr": 0.02}, "REJETER", id="fpr_hors_critere"),
    pytest.param({"pr_auc": 0.79}, "INVESTIGUER", id="gain_suspect"),
]


@pytest.mark.parametrize(("changes", "expected"), DECISION_CASES)
def test_decision(changes, expected):
    decision, _ = decide({**BASE, **changes}, BASE)
    assert decision == expected


@requires_data
def test_no_regression_against_baseline():
    """Réentraîne sur les vraies données et compare à la baseline figée."""
    baseline = load_baseline(BASELINE_PATH)
    df = load_data(DATA_PATH)
    _, current = compute_validation_metrics(df)

    decision, reasons = decide(current, baseline["metrics_validation"])
    assert decision == "ACCEPTER", reasons
