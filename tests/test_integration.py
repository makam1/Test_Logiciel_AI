import numpy as np

from app.preprocessing import preprocess_data
from app.training import split_data, train_model


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
