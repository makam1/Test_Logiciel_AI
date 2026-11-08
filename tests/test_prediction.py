import numpy as np
import pytest

from app.prediction import predict_fraud, predict_proba_fraud


# T04 : probabilité valide
def test_probabilities_valid(trained_model, synthetic_xy):
    X, _ = synthetic_xy
    proba = predict_proba_fraud(trained_model, X)

    assert proba.shape == (len(X),)
    assert np.isfinite(proba).all()
    assert np.all(proba >= 0)
    assert np.all(proba <= 1)


def test_predictions_are_binary(trained_model, synthetic_xy):
    X, _ = synthetic_xy
    pred = predict_fraud(trained_model, X)
    assert set(np.unique(pred)) <= {0, 1}


def test_higher_threshold_flags_fewer(trained_model, synthetic_xy):
    X, _ = synthetic_xy
    low = predict_fraud(trained_model, X, threshold=0.2)
    high = predict_fraud(trained_model, X, threshold=0.8)
    assert high.sum() <= low.sum()


@pytest.mark.parametrize("bad_threshold", [-0.1, 1.5])
def test_invalid_threshold_rejected(trained_model, synthetic_xy, bad_threshold):
    X, _ = synthetic_xy
    with pytest.raises(ValueError, match="seuil"):
        predict_fraud(trained_model, X, threshold=bad_threshold)


def test_model_scores_frauds_higher(trained_model, synthetic_xy):
    X, y = synthetic_xy
    proba = predict_proba_fraud(trained_model, X)
    is_fraud = y.to_numpy() == 1
    assert proba[is_fraud].mean() > proba[~is_fraud].mean()
