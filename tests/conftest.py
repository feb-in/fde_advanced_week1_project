"""Session-scoped fixtures for the tiered serving tests.

Importing _serving_helpers first triggers sys.path + MODEL_BUNDLE_DIR setup, so the
predictor/client load the baked bundle (offline). The model is a process-wide
singleton (app.model.get_predictor), so it loads once for the whole suite.
"""
import pytest

import _serving_helpers  # noqa: F401  (side effect: sys.path + env setup)


@pytest.fixture(scope="session")
def predictor():
    from app.model import get_predictor
    return get_predictor()


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from app.app import app
    with TestClient(app) as c:   # lifespan loads the model once
        yield c
