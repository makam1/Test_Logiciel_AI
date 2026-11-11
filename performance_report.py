"""Mesure le coût d'exécution du modèle : latence, débit, taille et mémoire."""

import gc
import json
import os
import platform
import time
from functools import partial
from pathlib import Path

import numpy as np
import psutil
import sklearn

from app.prediction import predict_from_dataframe, predict_proba_fraud
from app.preprocessing import preprocess_data
from app.training import (
    load_data,
    load_pipeline,
    save_pipeline,
    split_data,
    train_model,
)

DATA_PATH = "data/creditcard.csv"
MODEL_PATH = Path("models/pipeline.joblib")
REPORT_PATH = Path("reports/performance.json")
SINGLE_REPEATS = 500
BATCH_REPEATS = {1: 500, 100: 200, 1000: 50, 10000: 20}

# Seuils d'acceptation : hypothèses de travail, à valider avec la banque.
MAX_P95_MS = 50.0  # budget d'une prédiction dans un flux d'autorisation de paiement
MIN_THROUGHPUT = 1000.0  # prédictions par seconde, lots de 100 (marge sur un pic supposé)
MAX_MODEL_MB = 10.0  # déploiement léger, chargement rapide
MAX_MEMORY_MB = 200.0  # surcoût mémoire du modèle chargé


def time_calls(func, repeats):
    """Exécute func() plusieurs fois (échauffement exclu) et retourne les durées."""
    func()
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        durations.append(time.perf_counter() - start)
    return np.array(durations)


def latency_stats(durations):
    ms = durations * 1000
    return {
        "moyenne_ms": float(ms.mean()),
        "mediane_ms": float(np.median(ms)),
        "p95_ms": float(np.percentile(ms, 95)),
        "p99_ms": float(np.percentile(ms, 99)),
    }


def main():
    df = load_data(DATA_PATH)
    X, y = preprocess_data(df)
    X_train, X_val, _, y_train, _, _ = split_data(X, y)

    model = train_model(X_train, y_train, "logreg")
    save_pipeline(model, MODEL_PATH)
    size_mb = MODEL_PATH.stat().st_size / 1_000_000

    # Mémoire : on supprime le modèle d'entraînement, puis on recharge depuis le disque.
    del model
    gc.collect()
    process = psutil.Process()
    batch = X_val.iloc[:10000]
    rss_before = process.memory_info().rss
    model = load_pipeline(MODEL_PATH)
    predict_proba_fraud(model, batch)
    rss_after = process.memory_info().rss
    memory_mb = (rss_after - rss_before) / 1_000_000

    # Latence d'une prédiction isolée : brute, puis avec validation des entrées.
    row = X_val.iloc[[0]]
    raw_row = df.loc[row.index]
    latency_raw = latency_stats(
        time_calls(partial(predict_proba_fraud, model, row), SINGLE_REPEATS)
    )
    latency_validated = latency_stats(
        time_calls(partial(predict_from_dataframe, model, raw_row), SINGLE_REPEATS)
    )

    # Débit par taille de lot.
    throughput = {}
    for size, repeats in BATCH_REPEATS.items():
        durations = time_calls(partial(predict_proba_fraud, model, X_val.iloc[:size]), repeats)
        throughput[size] = float(size / np.median(durations))

    verdicts = {
        "latence_p95_brute": latency_raw["p95_ms"] < MAX_P95_MS,
        "latence_p95_avec_validation": latency_validated["p95_ms"] < MAX_P95_MS,
        "debit_lot_100": throughput[100] >= MIN_THROUGHPUT,
        "taille_modele": size_mb < MAX_MODEL_MB,
        "memoire": memory_mb < MAX_MEMORY_MB,
    }

    result = {
        "environnement": {
            "plateforme": platform.platform(),
            "processeur": platform.processor() or "non renseigné",
            "coeurs_logiques": os.cpu_count(),
            "ram_totale_go": round(psutil.virtual_memory().total / 1e9, 1),
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
        },
        "latence_isolee_brute": latency_raw,
        "latence_isolee_avec_validation": latency_validated,
        "debit_predictions_par_s": throughput,
        "taille_modele_mo": size_mb,
        "surcout_memoire_mo": memory_mb,
        "memoire_totale_processus_mo": rss_after / 1_000_000,
        "seuils": {
            "p95_ms": MAX_P95_MS,
            "debit_min": MIN_THROUGHPUT,
            "taille_max_mo": MAX_MODEL_MB,
            "memoire_max_mo": MAX_MEMORY_MB,
        },
        "verdicts": verdicts,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("\nVerdicts :")
    for name, ok in verdicts.items():
        print(f"  {name:<30} {'OK' if ok else 'ÉCHEC'}")


if __name__ == "__main__":
    main()
