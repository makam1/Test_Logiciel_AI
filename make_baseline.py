"""Crée la baseline figée de non-régression.

À lancer UNE FOIS, après avoir commité le code de référence. La baseline n'est
jamais recalculée automatiquement : l'écraser exige --force.
"""

import argparse
import hashlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from app.regression import (
    BASELINE_PATH,
    DATA_PATH,
    compute_validation_metrics,
    git_commit,
)
from app.training import RANDOM_STATE, build_pipeline, load_data


def sha256_of(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Écrase une baseline existante")
    args = parser.parse_args()

    if BASELINE_PATH.exists() and not args.force:
        raise SystemExit(
            "La baseline existe déjà : elle ne doit pas être recalculée "
            "automatiquement. Utilisez --force pour la remplacer volontairement."
        )

    df = load_data(DATA_PATH)
    threshold, metrics = compute_validation_metrics(df)
    n_frauds = int(df["Class"].sum())

    baseline = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "code_version": git_commit(),
        "model": {
            "name": "logreg",
            "params": build_pipeline("logreg").named_steps["model"].get_params(),
        },
        "random_state": RANDOM_STATE,
        "split": {"train": 0.6, "validation": 0.2, "test": 0.2, "stratified": True},
        "threshold": threshold,
        "metrics_validation": metrics,
        "library_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "dataset": {
            "file": DATA_PATH,
            "rows": len(df),
            "frauds": n_frauds,
            "fraud_rate": n_frauds / len(df),
            "sha256": sha256_of(DATA_PATH),
        },
    }

    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BASELINE_PATH.open("w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2, ensure_ascii=False, default=str)

    print(f"Baseline enregistrée dans {BASELINE_PATH}")
    print(f"Seuil : {threshold:.4f}")
    for name in ("precision", "recall", "f1", "pr_auc", "fpr"):
        print(f"{name:>10} : {metrics[name]:.4f}")


if __name__ == "__main__":
    main()
