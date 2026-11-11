"""Mesure la mémoire d'un processus minimal : imports, chargement, une prédiction."""

import pandas as pd
import psutil

from app.preprocessing import FEATURE_COLUMNS
from app.training import load_pipeline

process = psutil.Process()
print(f"Mémoire après imports : {process.memory_info().rss / 1e6:.1f} Mo")

model = load_pipeline("models/pipeline.joblib")
row = pd.DataFrame([[0.0] * len(FEATURE_COLUMNS)], columns=FEATURE_COLUMNS)
model.predict_proba(row)
print(f"Mémoire après chargement et prédiction : {process.memory_info().rss / 1e6:.1f} Mo")
