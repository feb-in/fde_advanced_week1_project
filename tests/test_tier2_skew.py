"""Tier 2 — the train/serve skew invariant. THE must-not-break test.

A record pushed through /predict must produce the SAME calibrated score the training
pipeline produced for that record — exact — for ≥3 known encounters, including the
golden 12522 → 0.074595.
"""
from _serving_helpers import (
    GOLDEN_ENCOUNTER,
    GOLDEN_SCORE,
    SKEW_ENCOUNTERS,
    raw_record,
    training_score,
)


def test_skew_exact_for_known_encounters(client, predictor):
    assert GOLDEN_ENCOUNTER in SKEW_ENCOUNTERS
    assert len(SKEW_ENCOUNTERS) >= 3
    for enc in SKEW_ENCOUNTERS:
        resp = client.post("/predict", json=raw_record(enc))
        assert resp.status_code == 200, resp.text
        api_score = resp.json()["readmission_probability"]
        train = round(training_score(predictor, enc), 6)
        assert abs(api_score - train) < 1e-6, (
            f"SKEW on encounter {enc}: api={api_score} train={train}")


def test_golden_number_exact(client):
    resp = client.post("/predict", json=raw_record(GOLDEN_ENCOUNTER))
    assert resp.status_code == 200, resp.text
    assert resp.json()["readmission_probability"] == GOLDEN_SCORE
