import numpy as np
import pandas as pd
import pytest

from app.preprocessing import FEATURE_COLUMNS, TARGET_COLUMN, preprocess_data
from app.training import train_model


@pytest.fixture
def valid_df():
    """10 transactions synthétiques valides (8 normales, 2 frauduleuses)."""
    rng = np.random.default_rng(42)
    n = 10
    df = pd.DataFrame(
        rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS
    )
    df["Time"] = np.arange(n, dtype=float)
    df["Amount"] = rng.uniform(1, 500, size=n)
    df[TARGET_COLUMN] = [0] * 8 + [1] * 2
    return df

@pytest.fixture
def synthetic_df():
    """1000 transactions synthétiques, 5 % de fraudes séparables sur V1 et V2."""
    rng = np.random.default_rng(0)
    n = 1000
    df = pd.DataFrame(
        rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS
    )
    df["Time"] = np.arange(n, dtype=float)
    df["Amount"] = rng.uniform(1, 500, size=n)
    df[TARGET_COLUMN] = 0
    fraud_idx = rng.choice(n, size=50, replace=False)
    df.loc[fraud_idx, TARGET_COLUMN] = 1
    df.loc[fraud_idx, "V1"] += 3.0
    df.loc[fraud_idx, "V2"] -= 3.0
    return df


@pytest.fixture
def synthetic_xy(synthetic_df):
    return preprocess_data(synthetic_df)


@pytest.fixture
def trained_model(synthetic_xy):
    X, y = synthetic_xy
    return train_model(X, y)
