"""Production des prédictions de fraude."""

import logging

from app.preprocessing import preprocess_data

logger = logging.getLogger(__name__)


def predict_proba_fraud(model, X):
    """Retourne la probabilité de fraude (classe 1) pour chaque transaction."""
    return model.predict_proba(X)[:, 1]


def predict_fraud(model, X, threshold=0.5):
    """Retourne 1 (fraude) si la probabilité atteint le seuil, sinon 0."""
    if not 0 <= threshold <= 1:
        raise ValueError("Le seuil doit être compris entre 0 et 1.")
    return (predict_proba_fraud(model, X) >= threshold).astype(int)


def predict_from_dataframe(model, df, threshold=0.5):
    """Valide un DataFrame brut (rejette toute entrée invalide), puis prédit.

    Les colonnes supplémentaires sont ignorées ; la colonne Class est facultative.
    Les journaux ne contiennent que des comptages, jamais de valeurs de transactions.
    """
    try:
        X, _ = preprocess_data(df, require_target=False)
    except (TypeError, ValueError) as exc:
        logger.warning("Entrée rejetée : %s", exc)
        raise
    logger.info("Prédiction sur %d transactions (seuil %.4f)", len(X), threshold)
    return predict_fraud(model, X, threshold)
