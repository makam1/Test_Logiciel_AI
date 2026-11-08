import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from app.preprocessing import FEATURE_COLUMNS, preprocess_data, validate_data


# T01 : colonnes requises présentes
def test_required_columns(valid_df):
    result = validate_data(valid_df)
    assert set(FEATURE_COLUMNS + ["Class"]).issubset(result.columns)


@pytest.mark.parametrize("column", ["Time", "V1", "V28", "Amount", "Class"])
def test_missing_column_rejected(valid_df, column):
    with pytest.raises(ValueError, match="manquantes"):
        validate_data(valid_df.drop(columns=column))


def test_duplicate_column_rejected(valid_df):
    df = pd.concat([valid_df, valid_df[["V1"]]], axis=1)
    with pytest.raises(ValueError, match="dupliquées"):
        validate_data(df)


def test_empty_data_rejected(valid_df):
    with pytest.raises(ValueError, match="vides"):
        validate_data(valid_df.iloc[0:0])

def test_not_a_dataframe_rejected():
    with pytest.raises(TypeError, match="DataFrame"):
        validate_data([1, 2, 3])

# T02 : types attendus
def test_column_types(valid_df):
    assert validate_data(valid_df) is valid_df  # types conformes : accepté

    bad = valid_df.copy()
    bad["Amount"] = bad["Amount"].astype(str)
    with pytest.raises(ValueError, match="Type incorrect"):
        validate_data(bad)


def test_nan_rejected(valid_df):
    valid_df.loc[0, "V1"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        validate_data(valid_df)


def test_infinite_rejected(valid_df):
    valid_df.loc[0, "V1"] = np.inf
    with pytest.raises(ValueError, match="infinies"):
        validate_data(valid_df)


# T09 : cible binaire
def test_target_binary(valid_df):
    assert set(valid_df["Class"].unique()) <= {0, 1}
    assert validate_data(valid_df) is valid_df


@pytest.mark.parametrize("bad_value", [2, -1])
def test_target_invalid_rejected(valid_df, bad_value):
    valid_df.loc[0, "Class"] = bad_value
    with pytest.raises(ValueError, match="0 ou 1"):
        validate_data(valid_df)


# T03 : aucune valeur non finie en sortie
def test_no_non_finite(valid_df):
    X, _y = preprocess_data(valid_df)
    assert np.isfinite(X.to_numpy()).all()

    scaled = StandardScaler().fit_transform(X)
    assert np.isfinite(scaled).all()

def test_single_observation(valid_df):
    X, y = preprocess_data(valid_df.iloc[[0]])
    assert X.shape == (1, len(FEATURE_COLUMNS))
    assert len(y) == 1


def test_dimensions_consistent(valid_df):
    X, y = preprocess_data(valid_df)
    assert X.shape == (len(valid_df), len(FEATURE_COLUMNS))
    assert len(y) == len(valid_df)
    assert list(X.columns) == FEATURE_COLUMNS
    assert "Class" not in X.columns


def test_prediction_mode_without_target(valid_df):
    no_target = valid_df.drop(columns="Class")
    X, y = preprocess_data(no_target, require_target=False)
    assert y is None
    assert X.shape[1] == len(FEATURE_COLUMNS)

    with pytest.raises(ValueError, match="manquantes"):
        preprocess_data(no_target)  # par défaut, la cible est exigée


def test_standard_scaler():
    data = pd.DataFrame({"Amount": [100.0, 200.0, 300.0]})
    scaled = StandardScaler().fit_transform(data)

    assert np.isfinite(scaled).all()
    assert abs(scaled.mean()) < 1e-10
    assert np.isclose(scaled.std(), 1.0)
