import pytest

from app.training import load_data, train_model


def test_unknown_model_rejected(synthetic_xy):
    X, y = synthetic_xy
    with pytest.raises(ValueError, match="inconnu"):
        train_model(X, y, "foo")


def test_load_data_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="introuvable"):
        load_data(tmp_path / "absent.csv")
