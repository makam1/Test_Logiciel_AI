"""Mesure la sensibilité du modèle à des perturbations (sur la VALIDATION)."""

import numpy as np
import pandas as pd

from app.evaluation import evaluate_model, select_threshold
from app.prediction import predict_fraud, predict_proba_fraud
from app.preprocessing import preprocess_data
from app.training import load_data, split_data, train_model

DATA_PATH = "data/creditcard.csv"
V_COLUMNS = [f"V{i}" for i in range(1, 29)]
RNG_SEED = 123


def noise_on_v(ratio):
    """Bruit gaussien d'amplitude ratio x écart-type, sur V1-V28."""

    def perturb(X, rng):
        X = X.copy()
        scale = ratio * X[V_COLUMNS].std().to_numpy()
        noise = rng.normal(size=(len(X), len(V_COLUMNS))) * scale
        X[V_COLUMNS] = X[V_COLUMNS].to_numpy() + noise
        return X

    return perturb


def scale_amount(factor):
    def perturb(X, rng):
        X = X.copy()
        X["Amount"] = X["Amount"] * factor
        return X

    return perturb


def set_amount(value):
    def perturb(X, rng):
        X = X.copy()
        X["Amount"] = value
        return X

    return perturb


def shift_v(ratio):
    """Décale toutes les variables V1-V28 de ratio x écart-type."""

    def perturb(X, rng):
        X = X.copy()
        X[V_COLUMNS] = X[V_COLUMNS] + ratio * X[V_COLUMNS].std()
        return X

    return perturb


SCENARIOS = {
    "Référence": lambda X, rng: X,
    "Bruit 1 % sur V1-V28": noise_on_v(0.01),
    "Bruit 10 % sur V1-V28": noise_on_v(0.10),
    "Montant +10 %": scale_amount(1.10),
    "Montant extrême (1 000 000)": set_amount(1_000_000.0),
    "Décalage +0,5 écart-type sur V1-V28": shift_v(0.5),
}


def main():
    df = load_data(DATA_PATH)
    X, y = preprocess_data(df)
    X_train, X_val, _, y_train, y_val, _ = split_data(X, y)

    model = train_model(X_train, y_train, "logreg")
    ref_proba = predict_proba_fraud(model, X_val)
    threshold = select_threshold(y_val, ref_proba, min_recall=0.80)
    ref_pred = predict_fraud(model, X_val, threshold)

    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for name, perturb in SCENARIOS.items():
        X_new = perturb(X_val, rng)
        proba = predict_proba_fraud(model, X_new)
        pred = predict_fraud(model, X_new, threshold)
        m = evaluate_model(y_val, pred, proba)
        rows.append(
            {
                "scenario": name,
                "variation_proba_moyenne": float(np.abs(proba - ref_proba).mean()),
                "changement_classe_pct": float((pred != ref_pred).mean() * 100),
                "rappel": m["recall"],
                "precision": m["precision"],
                "fpr": m["fpr"],
                "fp": m["fp"],
                "fn": m["fn"],
                "lignes_invalides": 0,
            }
        )

    # Valeurs manquantes : le pipeline REJETTE ; on compte les lignes concernées.
    missing_mask = rng.random(X_val.shape) < 0.01
    rows.append(
        {
            "scenario": "1 % de valeurs manquantes (entrée rejetée)",
            "lignes_invalides": int(missing_mask.any(axis=1).sum()),
        }
    )

    table = pd.DataFrame(rows)
    print(f"Seuil utilisé : {threshold:.4f}\n")
    print(table.round(4).to_string(index=False))
    table.to_csv("reports/robustesse.csv", index=False)


if __name__ == "__main__":
    main()
