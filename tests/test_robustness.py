import time

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from app.prediction import predict_fraud, predict_from_dataframe, predict_proba_fraud

# Scénarios d'entrées invalides : le système doit REJETER avec un message clair.
INVALID_INPUTS = [
    pytest.param(lambda df: df.assign(V1=np.nan), "NaN", id="valeurs_manquantes"),
    pytest.param(lambda df: df.assign(V1=np.inf), "infinies", id="valeurs_infinies"),
    pytest.param(lambda df: df.assign(Amount="abc"), "Type incorrect", id="type_incorrect"),
    pytest.param(lambda df: df.drop(columns="V3"), "manquantes", id="colonne_absente"),
    pytest.param(
        lambda df: df.rename(columns={"V2": "V1"}), "dupliquées", id="colonne_dupliquee"
    ),
    pytest.param(lambda df: df.iloc[0:0], "vides", id="donnees_vides"),
]


# T12 : entrée invalide rejetée avec une erreur explicite
@pytest.mark.parametrize(("mutate", "message"), INVALID_INPUTS)
def test_invalid_input_rejected(trained_model, synthetic_df, mutate, message):
    with pytest.raises(ValueError, match=message):
        predict_from_dataframe(trained_model, mutate(synthetic_df))


def test_extra_columns_ignored(trained_model, synthetic_df):
    extended = synthetic_df.assign(unexpected_field="x")
    expected = predict_from_dataframe(trained_model, synthetic_df)
    result = predict_from_dataframe(trained_model, extended)
    np.testing.assert_array_equal(result, expected)


# Valeurs extrêmes de Amount : le système TRAITE et la sortie reste valide.
# Hypothesis génère 30 montants au hasard entre 0 et 1 milliard.
@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    amount=st.floats(min_value=0, max_value=1e9, allow_nan=False, allow_infinity=False)
)
def test_probability_valid_for_any_amount(trained_model, synthetic_xy, amount):
    X, _ = synthetic_xy
    row = X.iloc[[0]].copy()
    row["Amount"] = amount

    proba = predict_proba_fraud(trained_model, row)
    assert np.isfinite(proba).all()
    assert np.all(proba >= 0)
    assert np.all(proba <= 1)


# Perturbation limitée : on MESURE la sensibilité, on n'impose pas l'invariance.
def test_small_perturbation_sensitivity(trained_model, synthetic_xy):
    """Bruit gaussien de 1 % de l'écart-type, sur V1-V28 uniquement."""
    X, _ = synthetic_xy
    rng = np.random.default_rng(123)
    v_cols = [c for c in X.columns if c.startswith("V")]

    noisy = X.copy()
    scale = 0.01 * X[v_cols].std().to_numpy()
    noisy[v_cols] = X[v_cols].to_numpy() + rng.normal(size=(len(X), len(v_cols))) * scale

    ref = predict_proba_fraud(trained_model, X)
    new = predict_proba_fraud(trained_model, noisy)
    class_change = (
        predict_fraud(trained_model, X) != predict_fraud(trained_model, noisy)
    ).mean()

    assert np.abs(new - ref).mean() < 0.02
    assert class_change < 0.05


# Décalage de distribution : le système TRAITE, et le décalage est mesurable.
def test_distribution_shift_is_processed_and_measurable(trained_model, synthetic_xy):
    X, _ = synthetic_xy
    shifted = X.copy()
    shifted["V1"] += 3.0
    shifted["V2"] -= 3.0

    rate_ref = predict_fraud(trained_model, X).mean()
    proba_shifted = predict_proba_fraud(trained_model, shifted)
    rate_shifted = predict_fraud(trained_model, shifted).mean()

    assert np.isfinite(proba_shifted).all()
    assert rate_shifted > rate_ref


# T14 : latence P95 d'une prédiction isolée
def test_latency_p95(trained_model, synthetic_xy):
    X, _ = synthetic_xy
    row = X.iloc[[0]]
    predict_proba_fraud(trained_model, row)  # échauffement

    latencies = []
    for _ in range(200):
        start = time.perf_counter()
        predict_proba_fraud(trained_model, row)
        latencies.append(time.perf_counter() - start)

    assert np.percentile(latencies, 95) < 0.050
