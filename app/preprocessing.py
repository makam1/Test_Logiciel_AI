"""Validation et préparation des données de transactions bancaires."""

import numpy as np
import pandas as pd

FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COLUMN = "Class"


def validate_data(df, require_target=True):
    """Vérifie le contrat des données. Rejette explicitement toute donnée invalide.

    Retourne le DataFrame inchangé s'il est valide, sinon lève ValueError.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("L'entrée doit être un DataFrame pandas.")
    if df.empty:
        raise ValueError("Les données sont vides.")
    if df.columns.duplicated().any():
        raise ValueError("Colonnes dupliquées détectées.")

    required = FEATURE_COLUMNS + ([TARGET_COLUMN] if require_target else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")

    for col in required:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Type incorrect pour la colonne {col} : numérique attendu.")

    values = df[required].to_numpy(dtype=float)
    if np.isnan(values).any():
        raise ValueError("Valeurs manquantes (NaN) détectées.")
    if np.isinf(values).any():
        raise ValueError("Valeurs infinies détectées.")

    if require_target and not df[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError("La cible Class doit contenir uniquement 0 ou 1.")

    return df


def preprocess_data(df, require_target=True):
    """Valide les données puis sépare les variables explicatives (X) et la cible (y).
    """
    validate_data(df, require_target=require_target)
    X = df[FEATURE_COLUMNS].astype(float).copy()
    y = df[TARGET_COLUMN].astype(int).copy() if require_target else None
    return X, y
