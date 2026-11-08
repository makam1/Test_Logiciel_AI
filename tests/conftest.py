import numpy as np
import pandas as pd
import pytest

from app.preprocessing import FEATURE_COLUMNS, TARGET_COLUMN


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
