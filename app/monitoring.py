"""Surveillance après déploiement : dérive des données, valeurs hors plage,
taux de transactions signalées."""

import numpy as np
import pandas as pd

from app.preprocessing import FEATURE_COLUMNS

# Time est un compteur relatif au premier enregistrement du jeu : en production il
# sort toujours de la plage d'entraînement, donc il n'est pas surveillé ici.
MONITORED_COLUMNS = [c for c in FEATURE_COLUMNS if c != "Time"]

N_BINS = 10
EPSILON = 1e-4
PSI_WARNING = 0.10  # seuils conventionnels du PSI
PSI_ALERT = 0.25
MAX_OUT_OF_RANGE = 0.01  # part tolérée de transactions hors plage d'entraînement
FLAG_RATE_FACTOR = 2.0  # le taux signalé ne doit pas être multiplié ni divisé par plus de 2


def _inner_edges(values, n_bins=N_BINS):
    """Bornes intérieures des déciles de la distribution de référence."""
    return np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1]))


def _proportions(values, inner_edges):
    bins = np.searchsorted(inner_edges, values, side="right")
    counts = np.bincount(bins, minlength=len(inner_edges) + 1)
    return counts / counts.sum()


def build_reference(X_train):
    """Statistiques de référence, calculées sur les données d'entraînement."""
    reference = {}
    for column in MONITORED_COLUMNS:
        values = X_train[column].to_numpy(dtype=float)
        edges = _inner_edges(values)
        reference[column] = {
            "edges": edges.tolist(),
            "proportions": _proportions(values, edges).tolist(),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return reference


def population_stability_index(reference_props, new_props):
    """PSI : mesure l'écart entre deux distributions découpées en classes."""
    ref = np.clip(np.asarray(reference_props, dtype=float), EPSILON, None)
    new = np.clip(np.asarray(new_props, dtype=float), EPSILON, None)
    return float(np.sum((new - ref) * np.log(new / ref)))


def _level(psi):
    if psi >= PSI_ALERT:
        return "ALERTE"
    if psi >= PSI_WARNING:
        return "AVERTISSEMENT"
    return "OK"


def drift_report(X_new, reference):
    """PSI, part hors plage et niveau d'alerte pour chaque variable surveillée."""
    rows = []
    for column in MONITORED_COLUMNS:
        ref = reference[column]
        values = X_new[column].to_numpy(dtype=float)
        psi = population_stability_index(
            ref["proportions"], _proportions(values, np.array(ref["edges"]))
        )
        outside = (values < ref["min"]) | (values > ref["max"])
        rows.append(
            {
                "variable": column,
                "psi": psi,
                "hors_plage_pct": float(outside.mean() * 100),
                "niveau": _level(psi),
            }
        )
    return pd.DataFrame(rows)


def out_of_range_rate(X_new, reference):
    """Part des transactions ayant au moins une variable hors de la plage d'entraînement."""
    mask = np.zeros(len(X_new), dtype=bool)
    for column in MONITORED_COLUMNS:
        values = X_new[column].to_numpy(dtype=float)
        mask |= (values < reference[column]["min"]) | (values > reference[column]["max"])
    return float(mask.mean())


def collect_alerts(X_new, reference, flag_rate=None, reference_flag_rate=None):
    """Liste les alertes à envoyer pour un lot de transactions."""
    alerts = []

    report = drift_report(X_new, reference)
    for row in report[report["niveau"] != "OK"].itertuples():
        alerts.append(f"{row.niveau} dérive : {row.variable} (PSI = {row.psi:.3f})")

    out_rate = out_of_range_rate(X_new, reference)
    if out_rate > MAX_OUT_OF_RANGE:
        alerts.append(
            f"ALERTE hors plage : {out_rate:.1%} des transactions ont au moins une "
            "variable hors de la plage d'entraînement"
        )

    if flag_rate is not None and reference_flag_rate:
        ratio = flag_rate / reference_flag_rate
        if ratio > FLAG_RATE_FACTOR or ratio < 1 / FLAG_RATE_FACTOR:
            alerts.append(
                f"ALERTE taux de transactions signalées : {flag_rate:.2%} "
                f"(référence {reference_flag_rate:.2%})"
            )
    return alerts
