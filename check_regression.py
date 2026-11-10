"""Compare la version courante du modèle à la baseline figée (non-régression)."""

import argparse
import sys
from pathlib import Path

import sklearn

from app.regression import (
    BASELINE_PATH,
    DATA_PATH,
    TOLERANCES,
    compare_to_baseline,
    compute_validation_metrics,
    decide,
    git_commit,
    load_baseline,
)
from app.training import load_data

EXIT_CODES = {"ACCEPTER": 0, "REJETER": 1, "INVESTIGUER": 2}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="logreg", choices=["logreg", "tree"])
    parser.add_argument("--output", default="reports/rapport_non_regression.md")
    args = parser.parse_args()

    baseline = load_baseline(BASELINE_PATH)
    reference = baseline["metrics_validation"]

    df = load_data(DATA_PATH)
    threshold, current = compute_validation_metrics(df, args.model)
    degradations = compare_to_baseline(current, reference)
    decision, reasons = decide(current, reference)

    lines = [
        "# Rapport de non-régression",
        "",
        f"- Version de référence : {baseline['code_version']}",
        f"- Version évaluée : {git_commit()}",
        f"- Modèle évalué : {args.model}",
        f"- Seuil : {threshold:.4f} (baseline : {baseline['threshold']:.4f})",
    ]
    if baseline["library_versions"]["scikit-learn"] != sklearn.__version__:
        lines.append("- ATTENTION : la version de scikit-learn a changé depuis la baseline.")
    lines += [
        "",
        "| Métrique | Baseline | Version évaluée | Dégradation | Tolérance | Verdict |",
        "|---|---|---|---|---|---|",
    ]
    for name, tolerance in TOLERANCES.items():
        verdict = "hors tolérance" if degradations[name] > tolerance else "OK"
        lines.append(
            f"| {name} | {reference[name]:.4f} | {current[name]:.4f} "
            f"| {degradations[name]:+.4f} | {tolerance} | {verdict} |"
        )
    lines += ["", f"**Décision : {decision}**", ""]
    lines += [f"- {reason}" for reason in reasons]

    report = "\n".join(lines)
    print(report)
    Path(args.output).write_text(report + "\n", encoding="utf-8")
    sys.exit(EXIT_CODES[decision])


if __name__ == "__main__":
    main()
