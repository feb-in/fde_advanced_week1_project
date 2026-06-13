"""api_client.py — the ONLY thing the UI does: talk HTTP to the model API.

This is a thin, pure client. It does NOT featurize, score, threshold, or load a model —
it POSTs a raw record to the running API's /predict and returns the parsed JSON. Keeping
this separate (and import-safe, no Streamlit) means it is unit-testable and makes the
"the UI computes nothing" rule structurally obvious: all intelligence is server-side.
"""
from __future__ import annotations

import os

import requests

DEFAULT_BASE_URL = os.environ.get("READMISSION_API_URL", "http://localhost:8000")


class APIError(Exception):
    """Generic non-200 from the API."""


class APIValidationError(APIError):
    """422 — the API rejected the record (bad/missing/extra fields)."""

    def __init__(self, detail):
        self.detail = detail or []
        super().__init__("validation error")


class APIConnectionError(APIError):
    """The API could not be reached at all."""


def predict(payload: dict, base_url: str = DEFAULT_BASE_URL, timeout: float = 30.0) -> dict:
    """POST one raw patient record to /predict and return the response JSON unchanged."""
    url = base_url.rstrip("/") + "/predict"
    try:
        r = requests.post(url, json=payload, timeout=timeout)
    except requests.exceptions.RequestException as exc:
        raise APIConnectionError(f"{base_url} — {exc}") from exc
    if r.status_code == 422:
        try:
            detail = r.json().get("detail", [])
        except ValueError:
            detail = r.text
        raise APIValidationError(detail)
    if r.status_code != 200:
        raise APIError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.json()


def health(base_url: str = DEFAULT_BASE_URL, timeout: float = 10.0) -> dict:
    """GET /health — model identity for the demo footer. Returns {} if unreachable."""
    try:
        r = requests.get(base_url.rstrip("/") + "/health", timeout=timeout)
        return r.json() if r.status_code == 200 else {}
    except requests.exceptions.RequestException:
        return {}
