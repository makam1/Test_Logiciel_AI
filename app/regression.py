"""Non-régression : comparaison d'une version du modèle à la baseline figée."""

import json
import subprocess
from pathlib import Path

from app.evaluation import evaluate_model, select_threshold
from app.prediction import predict_fraud, predict_proba_fraud
from app.preprocessing import preprocess_data
from app.training import RANDOM_STATE, split_data, train_model

DATA_PATH = "data/creditcard.csv"
BASELINE_PATH = Path("reports/baseline.json")

# Critères d'acceptation (plan de test, section 5)
MIN_RECALL = 0.80
MAX_FPR = 0.01

# Dégradation maximale tolérée par métrique, fixée une fois pour toutes.
# Avec environ 98 fraudes en validation, 1 fraude = 0,01 de rappel :
# 0,03 correspond à 3 fraudes ; 0,0003 de FPR à environ 17 fausses alertes.
TOLERANCES = {"recall": 0.03, "precision": 0.05, "pr_auc": 0.02, "fpr": 0.0003}
CRITICAL_METRICS = ("recall", "pr_auc")
LOWER_IS_BETTER = ("fpr",)
SUSPICIOUS_GAIN = 0.05


def git_commit():
    """Retourne le hash du commit courant, en signalant les modifications non commitées."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "inconnu"
    return f"{commit} (modifications non commitées)" if dirty else commit


def load_baseline(path):
    """Charge la baseline, avec une erreur explicite si elle n'existe pas."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"Baseline introuvable : {path}")
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def compute_validation_metrics(
    df, model_name="logreg", min_recall=MIN_RECALL, random_state=RANDOM_STATE
):
    """Entraîne le pipeline, choisit le seuil sur la VALIDATION, retourne
    (seuil, métriques sur la validation)."""
    X, y = preprocess_data(df)
    X_train, X_val, _, y_train, y_val, _ = split_data(X, y, random_state)
    model = train_model(X_train, y_train, model_name, random_state)

    proba = predict_proba_fraud(model, X_val)
    threshold = select_threshold(y_val, proba, min_recall)
    pred = predict_fraud(model, X_val, threshold)
    return threshold, evaluate_model(y_val, pred, proba)


def compare_to_baseline(current, baseline_metrics):
    """Dégradation de chaque métrique suivie (positive = la version est moins bonne)."""
    degradations = {}
    for name in TOLERANCES:
        delta = current[name] - baseline_metrics[name]
        degradations[name] = delta if name in LOWER_IS_BETTER else -delta
    return degradations


def decide(current, baseline_metrics):
    """Retourne (décision, raisons) : ACCEPTER, INVESTIGUER ou REJETER."""
    degradations = compare_to_baseline(current, baseline_metrics)
    decision, reasons = "ACCEPTER", []

    if current["recall"] < MIN_RECALL or current["fpr"] > MAX_FPR:
        decision = "REJETER"
        reasons.append("critère d'acceptation violé (rappel < 0,80 ou FPR > 1 %)")

    for name, degradation in degradations.items():
        if degradation > TOLERANCES[name]:
            reasons.append(
                f"{name} dégradé de {degradation:.4f} (tolérance {TOLERANCES[name]})"
            )
            if name in CRITICAL_METRICS:
                decision = "REJETER"
            elif decision == "ACCEPTER":
                decision = "INVESTIGUER"
        elif degradation < -SUSPICIOUS_GAIN and decision == "ACCEPTER":
            decision = "INVESTIGUER"
            reasons.append(f"{name} amélioré de {-degradation:.4f} : gain suspect (fuite ?)")

    if not reasons:
        reasons.append("aucune dégradation au-delà des tolérances")
    return decision, reasons
