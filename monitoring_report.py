"""Simule la surveillance après déploiement : alertes sur des données perturbées."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.evaluation import select_threshold
from app.monitoring import (
    build_reference,
    collect_alerts,
    drift_report,
    out_of_range_rate,
)
from app.prediction import predict_fraud, predict_proba_fraud
from app.preprocessing import preprocess_data
from app.training import load_data, split_data, train_model
from robustness_report import RNG_SEED, SCENARIOS

DATA_PATH = "data/creditcard.csv"
REFERENCE_PATH = Path("reports/reference_monitoring.json")
MAX_ALERTS_SHOWN = 5


def main():
    df = load_data(DATA_PATH)
    X, y = preprocess_data(df)
    X_train, X_val, _, y_train, y_val, _ = split_data(X, y)

    model = train_model(X_train, y_train, "logreg")
    threshold = select_threshold(y_val, predict_proba_fraud(model, X_val), 0.80)
    reference = build_reference(X_train)
    reference_flag_rate = float(predict_fraud(model, X_val, threshold).mean())

    REFERENCE_PATH.write_text(
        json.dumps(
            {
                "threshold": threshold,
                "flag_rate": reference_flag_rate,
                "features": reference,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    rng = np.random.default_rng(RNG_SEED)
    rows, details = [], {}
    for name, perturb in SCENARIOS.items():
        X_new = perturb(X_val, rng)
        flag_rate = float(predict_fraud(model, X_new, threshold).mean())
        report = drift_report(X_new, reference)
        alerts = collect_alerts(X_new, reference, flag_rate, reference_flag_rate)
        rows.append(
            {
                "scenario": name,
                "variables_en_alerte": int((report["niveau"] == "ALERTE").sum()),
                "variables_en_avertissement": int(
                    (report["niveau"] == "AVERTISSEMENT").sum()
                ),
                "lignes_hors_plage_pct": out_of_range_rate(X_new, reference) * 100,
                "taux_signale_pct": flag_rate * 100,
                "nb_alertes": len(alerts),
            }
        )
        details[name] = alerts

    table = pd.DataFrame(rows)
    print(f"Taux de référence signalé (validation) : {reference_flag_rate:.3%}\n")
    print(table.round(3).to_string(index=False))
    for name, alerts in details.items():
        print(f"\n--- {name} : {len(alerts)} alerte(s)")
        for alert in alerts[:MAX_ALERTS_SHOWN]:
            print(f"  {alert}")
        if len(alerts) > MAX_ALERTS_SHOWN:
            print(f"  ... ({len(alerts) - MAX_ALERTS_SHOWN} autres)")
    table.to_csv("reports/monitoring_simulation.csv", index=False)


if __name__ == "__main__":
    main()
