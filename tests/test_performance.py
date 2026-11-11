import time

import numpy as np

from app.prediction import predict_proba_fraud
from app.training import save_pipeline

MAX_MODEL_SIZE_MB = 10
MIN_THROUGHPUT = 1000  # prédictions par seconde en lot


def test_saved_model_size(trained_model, tmp_path):
    path = tmp_path / "pipeline.joblib"
    save_pipeline(trained_model, path)
    assert path.stat().st_size / 1_000_000 < MAX_MODEL_SIZE_MB


def test_batch_throughput(trained_model, synthetic_xy):
    X, _ = synthetic_xy
    predict_proba_fraud(trained_model, X)  # échauffement

    durations = []
    for _ in range(20):
        start = time.perf_counter()
        predict_proba_fraud(trained_model, X)
        durations.append(time.perf_counter() - start)

    assert len(X) / np.median(durations) >= MIN_THROUGHPUT
