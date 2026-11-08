"""Compare régression logistique et arbre de décision, puis évalue sur le test."""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

from app.evaluation import evaluate_model, select_threshold
from app.prediction import predict_fraud, predict_proba_fraud
from app.preprocessing import preprocess_data
from app.training import load_data, split_data, train_model

DATA_PATH = "data/creditcard.csv"
MODELS = ["logreg", "tree"]
MIN_RECALL = 0.80


def main():
    df = load_data(DATA_PATH)
    X, y = preprocess_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    rows, fitted, probas_val, thresholds = [], {}, {}, {}
    for name in MODELS:
        model = train_model(X_train, y_train, name)
        proba_val = predict_proba_fraud(model, X_val)
        thresholds[name] = select_threshold(y_val, proba_val, MIN_RECALL)
        fitted[name], probas_val[name] = model, proba_val

        for label, threshold in [("0.5", 0.5), ("choisi", thresholds[name])]:
            pred = predict_fraud(model, X_val, threshold)
            metrics = evaluate_model(y_val, pred, proba_val)
            rows.append({"modele": name, "seuil_type": label,
                         "seuil": threshold, **metrics})

    table = pd.DataFrame(rows)
    print("\n=== Comparaison sur la VALIDATION ===")
    print(table.round(4).to_string(index=False))
    table.to_csv("reports/comparaison_modeles.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in MODELS:
        RocCurveDisplay.from_predictions(y_val, probas_val[name], name=name, ax=axes[0])
        PrecisionRecallDisplay.from_predictions(
            y_val, probas_val[name], name=name, ax=axes[1]
        )
    axes[0].set_title("Courbe ROC (validation)")
    axes[1].set_title("Courbe précision-rappel (validation)")
    fig.tight_layout()
    fig.savefig("reports/courbes_roc_pr.png", dpi=150)

    best = max(MODELS, key=lambda m: table[table.modele == m].pr_auc.iloc[0])
    proba_test = predict_proba_fraud(fitted[best], X_test)
    pred_test = predict_fraud(fitted[best], X_test, thresholds[best])
    final = evaluate_model(y_test, pred_test, proba_test)

    print(f"\n=== Évaluation FINALE sur le TEST : {best} (seuil {thresholds[best]:.4f}) ===")
    for key, value in final.items():
        print(f"{key:>10} : {value:.4f}" if isinstance(value, float) else f"{key:>10} : {value}")
    pd.DataFrame([{"modele": best, "seuil": thresholds[best], **final}]).to_csv(
        "reports/evaluation_test_finale.csv", index=False
    )


if __name__ == "__main__":
    main()
