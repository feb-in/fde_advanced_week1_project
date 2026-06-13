"""Model-dependent smoke tests run AGAINST the built container over HTTP.

This is how CI verifies the BUILT ARTIFACT on a stateless runner: the model is BAKED
into the image, so no registry / mlflow.db / data files are needed on the runner.
Skipped unless API_BASE_URL is set (CI points it at the running container); uses only
the committed sample_request.json + synthetic records, so it needs no local data.
"""
import os

import pytest

from _serving_helpers import GOLDEN_SCORE, HIGH_RISK, LOW_RISK, SAMPLE, THRESHOLD

BASE = os.environ.get("API_BASE_URL")
pytestmark = pytest.mark.skipif(
    not BASE, reason="API_BASE_URL not set (no running container to test against)")


@pytest.fixture(scope="module")
def http():
    import httpx
    with httpx.Client(base_url=BASE, timeout=30.0) as c:
        yield c


def test_health(http):
    h = http.get("/health").json()
    assert h["status"] == "ok"
    assert h["model_version"] == "1"
    assert h["model_alias"] == "staging"
    assert h["threshold"] == THRESHOLD
    assert h["load_source"] == "baked-bundle"   # proves the bake-in path


def test_golden_score(http):
    r = http.post("/predict", json=SAMPLE)   # SAMPLE == encounter 12522
    assert r.status_code == 200, r.text
    assert r.json()["readmission_probability"] == GOLDEN_SCORE


def test_flag_both_sides_and_monotonic(http):
    hi = http.post("/predict", json=HIGH_RISK).json()
    lo = http.post("/predict", json=LOW_RISK).json()
    assert hi["flag"] is True and lo["flag"] is False
    assert hi["readmission_probability"] > lo["readmission_probability"]
    for body in (hi, lo):
        assert body["flag"] == (body["readmission_probability"] >= THRESHOLD)


def test_rejection_policy(http):
    assert http.post("/predict", json={**SAMPLE, "gender": "Unknown/Invalid"}).status_code == 422
