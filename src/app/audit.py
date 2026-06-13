"""audit.py — append-only audit trail for every scored request.

GOVERNANCE REQUIREMENT: every prediction must be reconstructable later — what went in,
what came out, which model produced it, and when. One JSON object per line (JSONL),
append-only, so the log is greppable and replayable (e.g. to recompute a score, or to
feed Stage-6 drift / a fairness re-audit on real traffic).

Each entry records:
  * request_id   — server-generated UUID (correlate logs ↔ a single scoring)
  * timestamp    — UTC ISO-8601, when scoring completed
  * model        — name / version / alias / threshold / calibration (the LINEAGE: a
                   registry coordinate that pins the exact artifact, src/app/model.py)
  * inputs       — the 44 validated raw discharge fields (the model's actual input)
  * output       — calibrated probability, flag, and the top SHAP factors returned
  * latency_ms   — server-side scoring latency

PRIVACY: entries contain clinical inputs — treat the log as PHI-adjacent. In production
it must go to an access-controlled, append-only, retention-governed sink (not a flat
file on the container's ephemeral disk); a *gap* in the audit trail should itself alert.

Best-effort by design: an I/O failure here is logged to stderr but NEVER raises, so a
logging problem can never drop a patient's prediction.
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

AUDIT_LOG_PATH = os.environ.get("AUDIT_LOG_PATH", "logs/audit/predictions.jsonl")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_entry(inputs: dict, result: dict, latency_ms: float, predictor) -> dict:
    """Assemble one audit record (pure — no I/O, so it's unit-testable)."""
    return {
        "request_id": str(uuid.uuid4()),
        "timestamp": _now_iso(),
        "model": {
            "name": result.get("model_name"),
            "version": str(result.get("model_version")),
            "alias": result.get("model_alias"),
            "threshold": result.get("threshold"),
            "calibration_method": getattr(predictor, "calibration_method", None),
            "load_source": getattr(predictor, "load_source", None),
        },
        "inputs": inputs,
        "output": {
            "readmission_probability": result.get("readmission_probability"),
            "flag": result.get("flag"),
            "top_factors": result.get("top_factors"),
        },
        "latency_ms": round(float(latency_ms), 2),
    }


def log_prediction(inputs: dict, result: dict, latency_ms: float, predictor) -> str | None:
    """Append one audit entry as a JSON line. Returns the request_id, or None on failure.

    Best-effort: never raises — a broken audit sink must not break scoring (but it is
    surfaced to stderr so the failure is visible to ops / an alert)."""
    entry = build_entry(inputs, result, latency_ms, predictor)
    try:
        path = Path(AUDIT_LOG_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
        return entry["request_id"]
    except Exception as exc:  # noqa: BLE001 — audit must not break the request path
        print(f"[audit] WARNING: failed to write audit entry: {exc}", file=sys.stderr)
        return None
