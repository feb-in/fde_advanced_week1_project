"""Tier 3 — behavioral sanity of the served model.

Not hard thresholds (those live in Tier 2) — these catch gross regressions: scores
out of range, the flag rule breaking, /health drifting, or risk ordering inverting.
"""
from _serving_helpers import HIGH_RISK, LOW_RISK, SAMPLE, THRESHOLD


def _predict(client, rec):
    r = client.post("/predict", json=rec)
    assert r.status_code == 200, r.text
    return r.json()


def test_score_in_unit_interval(client):
    for rec in (SAMPLE, HIGH_RISK, LOW_RISK):
        p = _predict(client, rec)["readmission_probability"]
        assert 0.0 <= p <= 1.0


def test_flag_matches_threshold_both_sides(client):
    # The flag rule must hold for records spanning the threshold.
    for rec in (SAMPLE, HIGH_RISK, LOW_RISK):
        body = _predict(client, rec)
        assert body["threshold"] == THRESHOLD
        assert body["flag"] == (body["readmission_probability"] >= THRESHOLD)
    # And cover both sides explicitly: high-risk flagged, low-risk not.
    assert _predict(client, HIGH_RISK)["flag"] is True
    assert _predict(client, LOW_RISK)["flag"] is False


def test_health_reports_model_identity(client):
    h = client.get("/health").json()
    assert h["status"] == "ok"
    assert h["model_version"] == "1"
    assert h["model_alias"] == "staging"
    assert h["threshold"] == THRESHOLD


def test_high_risk_outranks_low_risk(client):
    hp = _predict(client, HIGH_RISK)["readmission_probability"]
    lp = _predict(client, LOW_RISK)["readmission_probability"]
    assert hp > lp, f"monotonic sanity broke: high={hp} !> low={lp}"
