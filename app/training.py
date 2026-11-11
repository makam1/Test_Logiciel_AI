"""Chargement, séparation et entraînement du pipeline."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42


def load_data(path):
    """Charge le CSV, avec des erreurs explicites si le fichier est absent ou illisible."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"Fichier de données introuvable : {path}")
    try:
        return pd.read_csv(path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise ValueError(f"Fichier de données illisible : {path}") from exc

def split_data(X, y, random_state=RANDOM_STATE):
    """Sépare en train (60 %), validation (20 %) et test (20 %), avec stratification."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_pipeline(model_name="logreg", random_state=RANDOM_STATE):
    """Construit un Pipeline : mise à l'échelle puis modèle."""
    if model_name == "logreg":
        model = LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=random_state
        )
    elif model_name == "tree":
        model = DecisionTreeClassifier(
            max_depth=5, class_weight="balanced", random_state=random_state
        )
    else:
        raise ValueError(f"Modèle inconnu : {model_name}")
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def train_model(X, y, model_name="logreg", random_state=RANDOM_STATE):
    """Entraîne le pipeline. Le scaler est ajusté uniquement sur les données reçues."""
    pipeline = build_pipeline(model_name, random_state)
    pipeline.fit(X, y)
    return pipeline

def save_pipeline(model, path):
    """Sauvegarde le pipeline complet (scaler + modèle) sur disque."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_pipeline(path):
    """Recharge un pipeline sauvegardé, avec une erreur explicite si absent."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"Pipeline introuvable : {path}")
    return joblib.load(path)
