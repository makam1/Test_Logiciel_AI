import logging
import re
import subprocess
from pathlib import Path

import pytest

from app.prediction import predict_from_dataframe
from app.preprocessing import preprocess_data
from app.training import load_data

ROOT = Path(__file__).resolve().parents[1]
OWN_FILE = "tests/test_security.py"
BINARY_SUFFIXES = (".png", ".joblib", ".pkl")
FORBIDDEN_SUFFIXES = (".joblib", ".pkl", ".parquet")
MAX_TRACKED_SIZE = 5 * 1024 * 1024

SECRET_PATTERNS = {
    "clé d'accès AWS": re.compile(r"AKIA[0-9A-Z]{16}"),
    "clé privée": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "jeton GitHub": re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"),
    "secret en clair": re.compile(
        r"(?i)(password|passwd|secret|api[_-]?key|token)\s*[:=]\s*[\"'][^\"']{8,}[\"']"
    ),
}


def tracked_files():
    """Liste les fichiers suivis par Git (le test est ignoré hors dépôt Git)."""
    try:
        output = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("dépôt Git indisponible")
    return [line for line in output.splitlines() if line]


def test_no_secrets_in_repository():
    findings = []
    for name in tracked_files():
        path = ROOT / name
        if name == OWN_FILE or name.endswith(BINARY_SUFFIXES) or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        findings += [
            f"{name} : {label}"
            for label, pattern in SECRET_PATTERNS.items()
            if pattern.search(text)
        ]
    assert not findings, f"Secrets potentiels détectés : {findings}"


def test_no_data_or_model_files_tracked():
    files = tracked_files()

    data_files = [f for f in files if f.startswith("data/") and f != "data/README.md"]
    model_files = [f for f in files if f.endswith(FORBIDDEN_SUFFIXES)]
    too_big = [
        f for f in files if (ROOT / f).is_file() and (ROOT / f).stat().st_size > MAX_TRACKED_SIZE
    ]

    assert not data_files, f"Données versionnées : {data_files}"
    assert not model_files, f"Modèles versionnés : {model_files}"
    assert not too_big, f"Fichiers trop volumineux : {too_big}"


def test_no_sensitive_values_in_errors_or_logs(trained_model, synthetic_df, caplog):
    sentinel = "SENTINEL-VALUE-12345"
    invalid = synthetic_df.assign(Amount=sentinel)

    with caplog.at_level(logging.INFO), pytest.raises(ValueError, match="Type incorrect") as error:
        predict_from_dataframe(trained_model, invalid)

    assert sentinel not in str(error.value)
    assert sentinel not in caplog.text
    assert "rejetée" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.INFO):
        predict_from_dataframe(trained_model, synthetic_df)
    assert "transactions" in caplog.text
    assert str(synthetic_df["Amount"].iloc[0]) not in caplog.text


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(b"", id="fichier_vide"),
        pytest.param(b"\x80\x81\xff\xfe", id="octets_invalides"),
        pytest.param(b"<html><body>Not a CSV</body></html>", id="contenu_html"),
        pytest.param(b"a,b\n1,2\n", id="mauvaises_colonnes"),
    ],
)
def test_invalid_files_rejected(tmp_path, content):
    path = tmp_path / "input.csv"
    path.write_bytes(content)
    with pytest.raises(ValueError, match="illisible|vides|manquantes"):
        preprocess_data(load_data(path))
