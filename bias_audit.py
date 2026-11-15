"""Disparités de performance par tranche de montant (validation).

ATTENTION : le montant n'est pas une caractéristique de personne. Cette analyse
ne dit rien de l'équité du système envers des groupes (âge, sexe, localisation),
car le jeu de données ne contient aucune de ces informations.
"""

import numpy as np
import pandas as pd

from app.evaluation import select_threshold
from app.prediction import predict_fraud, predict_proba_fraud
from app.preprocessing import preprocess_data
from app.training import load_data, split_data, train_model

DATA_PATH = "data/creditcard.csv"
LABELS = ["T1 (plus faibles)", "T2", "T3", "T4 (plus élevés)"]


def wilson_interval(successes, total, z=1.96):
    """Intervalle de confiance à 95 % d'une proportion."""
    p = successes / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return center - half, center + half


def rate_with_ci(successes, total):
    if total == 0:
        return "n/a"
    low, high = wilson_interval(successes, total)
    return f"{successes / total:.4f} [{low:.4f} ; {high:.4f}]"


def main():
    df = load_data(DATA_PATH)
    X, y = preprocess_data(df)
    X_train, X_val, _, y_train, y_val, _ = split_data(X, y)

    model = train_model(X_train, y_train, "logreg")
    threshold = select_threshold(y_val, predict_proba_fraud(model, X_val), 0.80)
    pred = predict_fraud(model, X_val, threshold)
    y_true = y_val.to_numpy()

    # Tranches de montant : quartiles calculés sur l'entraînement uniquement.
    edges = np.quantile(X_train["Amount"], [0.25, 0.5, 0.75])
    segment = np.searchsorted(edges, X_val["Amount"].to_numpy(), side="right")

    rows = []
    for index, label in enumerate(LABELS):
        in_segment = segment == index
        tp = int(((pred == 1) & (y_true == 1) & in_segment).sum())
        fn = int(((pred == 0) & (y_true == 1) & in_segment).sum())
        fp = int(((pred == 1) & (y_true == 0) & in_segment).sum())
        tn = int(((pred == 0) & (y_true == 0) & in_segment).sum())
        rows.append(
            {
                "tranche": label,
                "transactions": int(in_segment.sum()),
                "fraudes": tp + fn,
                "rappel [IC 95 %]": rate_with_ci(tp, tp + fn),
                "fpr [IC 95 %]": rate_with_ci(fp, fp + tn),
                "precision [IC 95 %]": rate_with_ci(tp, tp + fp),
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "tn": tn,
            }
        )

    table = pd.DataFrame(rows)
    bounds = ", ".join(f"{edge:.2f}" for edge in edges)
    print(f"Seuil : {threshold:.4f} | bornes des tranches de montant : {bounds}\n")
    print(table.to_string(index=False))
    table.to_csv("reports/audit_tranches_montant.csv", index=False)


if __name__ == "__main__":
    main()
