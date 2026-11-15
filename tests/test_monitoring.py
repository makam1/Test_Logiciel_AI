import pytest

from app.monitoring import (
    MONITORED_COLUMNS,
    build_reference,
    collect_alerts,
    drift_report,
    out_of_range_rate,
    population_stability_index,
)


def test_identical_data_raises_no_alert(synthetic_xy):
    X, _ = synthetic_xy
    reference = build_reference(X)

    report = drift_report(X, reference)
    assert report["psi"].max() == pytest.approx(0.0, abs=1e-12)
    assert set(report["niveau"]) == {"OK"}
    assert out_of_range_rate(X, reference) == 0.0
    assert collect_alerts(X, reference) == []


def test_psi_is_zero_for_identical_distributions():
    assert population_stability_index([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)


def test_psi_known_value():
    psi = population_stability_index([0.5, 0.5], [0.8, 0.2])
    assert psi == pytest.approx(0.4159, abs=1e-3)


def test_shifted_variable_raises_drift_alert(synthetic_xy):
    X, _ = synthetic_xy
    reference = build_reference(X)
    shifted = X.copy()
    shifted["V1"] += 3.0

    report = drift_report(shifted, reference).set_index("variable")
    assert report.loc["V1", "niveau"] == "ALERTE"
    assert report.loc["V2", "niveau"] == "OK"
    alerts = collect_alerts(shifted, reference)
    assert any("dérive : V1 (" in alert for alert in alerts)


def test_extreme_amount_is_out_of_range(synthetic_xy):
    X, _ = synthetic_xy
    reference = build_reference(X)
    extreme = X.assign(Amount=1_000_000.0)

    assert out_of_range_rate(extreme, reference) == 1.0
    alerts = collect_alerts(extreme, reference)
    assert any("hors plage" in alert for alert in alerts)


def test_flag_rate_alert(synthetic_xy):
    X, _ = synthetic_xy
    reference = build_reference(X)

    abnormal = collect_alerts(X, reference, flag_rate=0.05, reference_flag_rate=0.01)
    normal = collect_alerts(X, reference, flag_rate=0.011, reference_flag_rate=0.01)

    assert any("taux de transactions signalées" in alert for alert in abnormal)
    assert normal == []


def test_time_is_not_monitored():
    assert "Time" not in MONITORED_COLUMNS
