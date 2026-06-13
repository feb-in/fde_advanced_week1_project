"""retrain_trigger.py — Stage 6 observability: the concrete retrain decision rule.

This is the numeric policy that turns the drift signals (src/monitoring/drift.py →
reports/monitoring/drift_summary.json) into a yes/no "retrain now" decision. It is
deliberately a pure function over a drift summary so it is deterministic and testable;
it is the same rule a scheduled job (or a Grafana/Prometheus alert) would evaluate
against real scored-traffic batches once they exist.

THE RULE — retrain if ANY of:
  1. dataset-drift share  > 0.10   (Evidently: >10% of monitored columns drifted), OR
  2. PSI > 0.20 on ANY top-SHAP feature (a key driver's distribution shifted
     significantly — the industry-standard 0.2 PSI line), OR
  3. PR-AUC on freshly-labelled data < 0.15 (only evaluable once outcomes are
     observed; ~75% of the model's 0.207 test AUPRC — a real performance regression).

Rationale for the thresholds:
  * 0.10 share mirrors the documented drift_share already used by the detector
    (docs/MONITORING.md) — one consistent dataset-drift line, no second arbitrary number.
  * 0.20 PSI is the standard "significant population shift" threshold (<0.1 stable,
    0.1-0.2 moderate, >0.2 act). Applied to the model's OWN top drivers, not all 54
    features, so it catches shifts the model is actually sensitive to.
  * The PR-AUC floor is the ground-truth backstop: drift is a leading indicator, but a
    measured performance drop on labelled feedback is the definitive retrain signal.
These map to the project's three day-one thresholds — keep / retrain / retire.

Run (evaluates the committed control + shifted summaries):
    uv run python src/monitoring/retrain_trigger.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SUMMARY = Path("reports/monitoring/drift_summary.json")

DATASET_DRIFT_SHARE = 0.10   # rule 1
PSI_THRESHOLD = 0.20         # rule 2
PRAUC_FLOOR = 0.15           # rule 3 (when labels available)


def evaluate(batch: dict) -> dict:
    """Apply the retrain rule to ONE drift-summary batch (control or shifted).

    `batch` is the per-batch object from drift_summary.json: it has `overall.share`,
    `top_feature_psi` {feature: psi}, and optionally `new_data_prauc` (present only when
    labelled feedback exists). Returns {trigger: bool, reasons: [...]}.
    """
    reasons = []

    share = float(batch.get("overall", {}).get("share", 0.0))
    if share > DATASET_DRIFT_SHARE:
        reasons.append(
            f"dataset-drift share {share:.3f} > {DATASET_DRIFT_SHARE}")

    psis = batch.get("top_feature_psi", {})
    hot = {f: v for f, v in psis.items() if v > PSI_THRESHOLD}
    if hot:
        top = ", ".join(f"{f}={v:.2f}" for f, v in sorted(hot.items(), key=lambda x: -x[1]))
        reasons.append(f"PSI>{PSI_THRESHOLD} on top-SHAP feature(s): {top}")

    prauc = batch.get("new_data_prauc")  # None in the offline simulation (no labels)
    if prauc is not None and prauc < PRAUC_FLOOR:
        reasons.append(f"new-data PR-AUC {prauc:.3f} < {PRAUC_FLOOR}")

    return {"trigger": bool(reasons), "reasons": reasons}


def _report(name: str, batch: dict, expected: bool) -> bool:
    r = evaluate(batch)
    verdict = "RETRAIN" if r["trigger"] else "keep model"
    ok = r["trigger"] == expected
    print(f"\n── {name} → {verdict}  [{'OK' if ok else 'UNEXPECTED'}]")
    if r["reasons"]:
        for reason in r["reasons"]:
            print(f"     • {reason}")
    else:
        print("     • no rule tripped")
    return ok


def main():
    if not SUMMARY.exists():
        sys.exit(f"{SUMMARY} missing — run src/monitoring/drift.py first")
    summary = json.loads(SUMMARY.read_text())

    print("=" * 70)
    print("RETRAIN TRIGGER — evaluating drift_summary.json")
    print(f"rule: share>{DATASET_DRIFT_SHARE}  OR  PSI>{PSI_THRESHOLD} on a top-SHAP "
          f"feature  OR  new-data PR-AUC<{PRAUC_FLOOR}")
    print("=" * 70)

    ok_control = _report("CONTROL (unshifted)", summary["control"], expected=False)
    ok_shift = _report("SHIFTED (intentional)", summary["shifted"], expected=True)

    passed = ok_control and ok_shift
    print(f"\n[trigger validation] silent on control AND fires on shifted: "
          f"{'PASS' if passed else 'FAIL'}")
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
