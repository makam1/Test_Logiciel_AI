import numpy as np
import pytest

from app.evaluation import evaluate_model, select_threshold
from app.prediction import predict_fraud, predict_proba_fraud
from app.preprocessing import FEATURE_COLUMNS, preprocess_data
from app.training import (
    load_data,
    load_pipeline,
    save_pipeline,
    split_data,
    train_model,
)


# T05 : jeux indépendants
def test_split_no_overlap(synthetic_df):
    X, y = preprocess_data(synthetic_df)
    X_train, X_val, X_test, *_ = split_data(X, y)

    assert set(X_train.index).isdisjoint(X_val.index)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert set(X_val.index).isdisjoint(X_test.index)
    assert len(X_train) + len(X_val) + len(X_test) == len(X)


# T10 : proportion de fraudes conservée
def test_stratified_split(synthetic_df):
    X, y = preprocess_data(synthetic_df)
    _, _, _, y_train, y_val, y_test = split_data(X, y)

    for part in (y_train, y_val, y_test):
        assert abs(part.mean() - y.mean()) < 0.001


# T08 : le scaler n'est ajusté que sur le train
def test_scaler_fit_on_train_only(synthetic_df):
    X, y = preprocess_data(synthetic_df)
    X_train, _, _, y_train, _, _ = split_data(X, y)
    scaler = train_model(X_train, y_train).named_steps["scaler"]

    assert np.allclose(scaler.mean_, X_train.mean().to_numpy())
    assert not np.allclose(scaler.mean_, X.mean().to_numpy())


# T11 : sauvegarde puis rechargement
def test_save_reload_same_predictions(synthetic_df, tmp_path):
    X, y = preprocess_data(synthetic_df)
    X_train, _, X_test, y_train, _, _ = split_data(X, y)
    model = train_model(X_train, y_train)

    path = tmp_path / "pipeline.joblib"
    save_pipeline(model, path)
    reloaded = load_pipeline(path)

    before = predict_proba_fraud(model, X_test)
    after = predict_proba_fraud(reloaded, X_test)
    np.testing.assert_allclose(before, after, atol=1e-9)


def test_load_pipeline_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="introuvable"):
        load_pipeline(tmp_path / "absent.joblib")


def test_dimensions_consistent(synthetic_df):
    X, y = preprocess_data(synthetic_df)
    model = train_model(X, y)

    assert model.named_steps["scaler"].n_features_in_ == len(FEATURE_COLUMNS)
    assert model.predict_proba(X).shape == (len(X), 2)
    assert predict_proba_fraud(model, X).shape == (len(X),)


def test_no_non_finite_after_scaling(synthetic_df):
    X, y = preprocess_data(synthetic_df)
    X_train, _, X_test, y_train, _, _ = split_data(X, y)
    scaler = train_model(X_train, y_train).named_steps["scaler"]

    transformed = scaler.transform(X_test)
    assert np.isfinite(transformed).all()
    assert transformed.shape == (len(X_test), len(FEATURE_COLUMNS))


def test_end_to_end_pipeline(synthetic_df, tmp_path):
    """Chargement -> validation -> séparation -> entraînement -> seuil sur la
    validation -> sauvegarde -> rechargement -> prédiction sur le test -> métriques."""
    csv_path = tmp_path / "transactions.csv"
    synthetic_df.to_csv(csv_path, index=False)

    df = load_data(csv_path)
    X, y = preprocess_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    model = train_model(X_train, y_train)
    threshold = select_threshold(y_val, predict_proba_fraud(model, X_val))

    save_pipeline(model, tmp_path / "pipeline.joblib")
    reloaded = load_pipeline(tmp_path / "pipeline.joblib")

    proba = predict_proba_fraud(reloaded, X_test)
    pred = predict_fraud(reloaded, X_test, threshold)
    report = evaluate_model(y_test, pred, proba)

    assert len(pred) == len(y_test)
    assert all(np.isfinite(value) for value in report.values())
    assert report["recall"] > 0.5
