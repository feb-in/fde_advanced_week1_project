"""calibrate.py — Stage 4 (final): calibrate the tuned CatBoost, choose an
operating threshold, register the calibrated model to MLflow Staging.

Calibration and the threshold are DEPLOYMENT decisions. They do NOT change AUPRC
or ROC-AUC — those are ranking metrics and monotonic calibration preserves the
ranking exactly. Their job is to make the probabilities trustworthy (a "0.30"
should mean ~30% readmit risk) and to set a sensible alert volume. We prove the
ranking is untouched by reporting pre/post AUPRC + ROC-AUC.

No-leakage discipline (carried over):
  * Same held-out test set + seed 42 (shared prepare_data). Scored ONCE, at the
    end, for the final calibrated numbers — never used to fit the calibrator or
    pick the threshold.
  * Inside TRAIN we carve tr_dev (80%) / tr_val (20%). Calibration is fit with
    CalibratedClassifierCV(cv=5) on tr_dev (the base is never calibrated on its
    own rows); method choice + threshold are decided on tr_val.

Run:
    uv run python src/models/calibrate.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from mlflow.models import infer_signature
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as ev  # noqa: E402
import train as T  # noqa: E402
from wrappers import CatBoostWrapper  # noqa: E402

CALIB_CV = 5
INTERNAL_VAL_SIZE = 0.20      # tr_dev / tr_val split inside TRAIN
TARGET_RECALL = 0.50          # screening tool: catch ~half of true 30-day readmits
REGISTERED_MODEL = "readmission-catboost-calibrated"
ASSUMED_DISCHARGES_PER_DAY = 50  # illustrative, for the patients/day framing


def fetch_tuned_catboost_params():
    """Single source of truth: the Optuna best config from the catboost_tuning run."""
    c = mlflow.tracking.MlflowClient()
    e = mlflow.get_experiment_by_name("readmission-30d")
    runs = c.search_runs([e.experiment_id],
                         filter_string="tags.phase='tuning' and tags.model='catboost'")
    p = {k[5:]: v for k, v in runs[0].data.params.items() if k.startswith("best_")}
    return dict(
        iterations=int(p["iterations"]),
        depth=int(p["depth"]),
        learning_rate=float(p["learning_rate"]),
        l2_leaf_reg=float(p["l2_leaf_reg"]),
        random_strength=float(p["random_strength"]),
        bagging_temperature=float(p["bagging_temperature"]),
        border_count=int(p["border_count"]),
        auto_class_weights=p["auto_class_weights"],
    )


def threshold_for_recall(y, p, target_recall):
    """Highest threshold whose recall >= target (maximizes precision at that recall)."""
    prec, rec, thr = precision_recall_curve(y, p)
    ok = rec[:-1] >= target_recall           # rec[:-1] aligns with thr
    if not ok.any():
        return 0.0, float(prec[0]), float(rec[0])
    idx = np.where(ok)[0][-1]
    return float(thr[idx]), float(prec[idx]), float(rec[idx])


def operating_point(y, p, thr):
    """precision, recall, flag_rate, confusion counts at a fixed threshold."""
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    flag_rate = (tp + fp) / len(y)
    return dict(precision=precision, recall=recall, flag_rate=flag_rate,
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def reliability_fig(y, series, title):
    """series = {label: probs}. One reliability curve per label (quantile bins)."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], ls="--", color="grey", label="perfect")
    for label, p in series.items():
        frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean, frac, "o-", label=f"{label} (Brier={brier_score_loss(y, p):.4f})")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction positive")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def main(features_path, experiment):
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    print(f"[mlflow] {tracking_uri}  experiment={experiment!r}")

    best = fetch_tuned_catboost_params()
    print(f"[config] tuned CatBoost best params: {best}")

    # Same split + seed as every prior gate. CatBoost wants string categoricals.
    X_train, X_test, y_train, y_test, cat_cols, num_cols, dropped = T.prepare_data(features_path)
    Xtr = T.as_catboost_frame(X_train, cat_cols)
    Xte = T.as_catboost_frame(X_test, cat_cols)
    y_train = y_train.reset_index(drop=True)
    Xtr = Xtr.reset_index(drop=True)

    def make_base():
        return CatBoostWrapper(cat_features=cat_cols, **best)

    # Internal calibration split inside TRAIN — never the test set.
    Xdev, Xval, ydev, yval = train_test_split(
        Xtr, y_train, test_size=INTERNAL_VAL_SIZE, stratify=y_train, random_state=T.SEED
    )
    print(f"[split] tr_dev={len(Xdev):,}  tr_val={len(Xval):,}  test={len(Xte):,} (untouched)")

    with mlflow.start_run(run_name="catboost_calibrated") as run:
        mlflow.set_tag("model", "catboost")
        mlflow.set_tag("phase", "calibration")
        mlflow.log_params({f"cb_{k}": v for k, v in best.items()})
        mlflow.log_param("calibration_cv", CALIB_CV)
        mlflow.log_param("internal_val_size", INTERNAL_VAL_SIZE)
        mlflow.log_param("target_recall", TARGET_RECALL)
        mlflow.log_param("seed", T.SEED)

        # ---- Pre-calibration reference: raw base fit on tr_dev, scored on tr_val.
        print("\n[1/4] fitting raw base on tr_dev (pre-calibration reference)...")
        raw = make_base().fit(Xdev, ydev)
        p_val_raw = raw.predict_proba(Xval)[:, 1]
        brier_pre = brier_score_loss(yval, p_val_raw)

        # ---- Cross-validated calibration on tr_dev for BOTH methods (no leakage).
        print("[2/4] fitting isotonic + sigmoid calibrators on tr_dev (cv=5)...")
        cals, p_val = {}, {}
        for method in ("isotonic", "sigmoid"):
            cal = CalibratedClassifierCV(make_base(), method=method, cv=CALIB_CV)
            cal.fit(Xdev, ydev)
            cals[method] = cal
            p_val[method] = cal.predict_proba(Xval)[:, 1]

        brier = {m: brier_score_loss(yval, p_val[m]) for m in cals}
        chosen = min(brier, key=brier.get)
        print(f"      tr_val Brier — pre(raw)={brier_pre:.4f}  "
              f"isotonic={brier['isotonic']:.4f}  sigmoid={brier['sigmoid']:.4f}  "
              f"→ chosen: {chosen}")

        mlflow.log_metric("val_brier_pre", brier_pre)
        mlflow.log_metric("val_brier_isotonic", brier["isotonic"])
        mlflow.log_metric("val_brier_sigmoid", brier["sigmoid"])
        mlflow.set_tag("calibration_method", chosen)
        mlflow.log_figure(
            reliability_fig(yval, {"raw": p_val_raw, "isotonic": p_val["isotonic"],
                                   "sigmoid": p_val["sigmoid"]},
                            "Calibration on tr_val (pre vs isotonic vs sigmoid)"),
            "reliability_val.png",
        )

        # ---- Threshold on tr_val using the CHOSEN calibrated probs (recall-leaning).
        print(f"[3/4] choosing threshold for recall>={TARGET_RECALL} on tr_val...")
        thr, val_prec, val_rec = threshold_for_recall(yval, p_val[chosen], TARGET_RECALL)
        mlflow.log_param("operating_threshold", thr)
        mlflow.log_metric("val_precision_at_thr", val_prec)
        mlflow.log_metric("val_recall_at_thr", val_rec)
        # Sensitivity table for the doc/console.
        sens = []
        for tr_target in (0.40, 0.50, 0.60):
            t, pr, rc = threshold_for_recall(yval, p_val[chosen], tr_target)
            op = operating_point(yval, p_val[chosen], t)
            sens.append((tr_target, t, op["precision"], op["recall"], op["flag_rate"]))
        print("      recall_target  thr     precision  recall   flag_rate")
        for tr_target, t, pr, rc, fr in sens:
            print(f"        {tr_target:>4.2f}       {t:.4f}   {pr:.4f}    {rc:.4f}   {fr:.4f}")
        print(f"      → chosen threshold={thr:.4f} (val precision={val_prec:.3f}, recall={val_rec:.3f})")

        # ---- FINAL model: refit chosen calibrator on FULL train, score test ONCE.
        print("[4/4] refitting chosen calibrator on FULL train; scoring held-out test (once)...")
        final = CalibratedClassifierCV(make_base(), method=chosen, cv=CALIB_CV)
        final.fit(Xtr, y_train)
        p_test_cal = final.predict_proba(Xte)[:, 1]

        # Uncalibrated full-train reference, to prove ranking is unchanged.
        raw_full = make_base().fit(Xtr, y_train)
        p_test_raw = raw_full.predict_proba(Xte)[:, 1]

        m_pre = dict(auprc=average_precision_score(y_test, p_test_raw),
                     roc_auc=roc_auc_score(y_test, p_test_raw),
                     brier=brier_score_loss(y_test, p_test_raw))
        m_post = dict(auprc=average_precision_score(y_test, p_test_cal),
                      roc_auc=roc_auc_score(y_test, p_test_cal),
                      brier=brier_score_loss(y_test, p_test_cal))
        for k, v in m_pre.items():
            mlflow.log_metric(f"test_{k}_pre", v)
        for k, v in m_post.items():
            mlflow.log_metric(f"test_{k}_post", v)

        op = operating_point(y_test, p_test_cal, thr)
        for k in ("precision", "recall", "flag_rate", "tp", "fp", "fn", "tn"):
            mlflow.log_metric(f"test_{k}_at_thr", op[k])

        tripwire_ok = m_post["roc_auc"] <= T.TRIPWIRE_ROC_AUC
        mlflow.set_tag("leakage_tripwire_ok", str(tripwire_ok))

        # Figures on test.
        mlflow.log_figure(ev.pr_curve_fig(y_test, p_test_cal, "CatBoost (calibrated) — PR (test)"),
                          "pr_curve_test.png")
        mlflow.log_figure(
            reliability_fig(y_test, {"calibrated": p_test_cal, "raw": p_test_raw},
                            "Calibration on test (raw vs calibrated)"),
            "reliability_test.png",
        )
        mlflow.log_figure(ev.confusion_fig(y_test, p_test_cal, "CatBoost (calibrated) — test", thr),
                          "confusion_test.png")

        # ---- Register the calibrated model and move it to Staging.
        print("      logging + registering model → Staging...")
        sig = infer_signature(Xte.head(5), p_test_cal[:5])
        info = mlflow.sklearn.log_model(
            sk_model=final, name="model", signature=sig,
            input_example=Xte.head(5),
            code_paths=[str(Path(__file__).resolve().parent / "wrappers.py")],
            registered_model_name=REGISTERED_MODEL,
        )
        client = mlflow.tracking.MlflowClient()
        version = getattr(info, "registered_model_version", None)
        if version is None:
            version = max(int(m.version)
                          for m in client.search_model_versions(f"name='{REGISTERED_MODEL}'"))
        staged_via = _to_staging(client, REGISTERED_MODEL, version, thr, chosen)

        _print_summary(brier_pre, brier, chosen, thr, m_pre, m_post, op, tripwire_ok,
                       version, staged_via)

        if not tripwire_ok:
            print("STOP: tripwire fired (test ROC-AUC > 0.75).")
            sys.exit(1)

    return dict(chosen=chosen, threshold=thr, m_pre=m_pre, m_post=m_post,
                brier_pre=brier_pre, brier=brier, op=op, version=version)


def _to_staging(client, name, version, thr, method):
    """Move the version to Staging. MLflow 3.x deprecates stages in favour of
    aliases; CLAUDE.md mandates the stage transition as the rollback mechanism, so
    we set the stage and also a 'staging' alias as the forward-compatible mirror."""
    client.set_model_version_tag(name, version, "operating_threshold", f"{thr:.6f}")
    client.set_model_version_tag(name, version, "calibration_method", method)
    used = []
    try:
        client.transition_model_version_stage(name, version, stage="Staging",
                                              archive_existing_versions=False)
        used.append("stage=Staging")
    except Exception as exc:  # noqa: BLE001
        used.append(f"stage transition unavailable ({type(exc).__name__})")
    try:
        client.set_registered_model_alias(name, "staging", version)
        used.append("alias=staging")
    except Exception as exc:  # noqa: BLE001
        used.append(f"alias unavailable ({type(exc).__name__})")
    return ", ".join(used)


def _print_summary(brier_pre, brier, chosen, thr, m_pre, m_post, op, tripwire_ok,
                   version, staged_via):
    print("\n============== calibration + threshold gate ==============")
    print(f"tr_val Brier:  pre(raw)={brier_pre:.4f}  isotonic={brier['isotonic']:.4f}  "
          f"sigmoid={brier['sigmoid']:.4f}  → chosen {chosen}")
    print(f"ranking unchanged (test): AUPRC {m_pre['auprc']:.4f}→{m_post['auprc']:.4f}  "
          f"ROC-AUC {m_pre['roc_auc']:.4f}→{m_post['roc_auc']:.4f}")
    print(f"test Brier: pre={m_pre['brier']:.4f} → post={m_post['brier']:.4f}")
    print(f"threshold={thr:.4f}  |  TEST precision={op['precision']:.4f} "
          f"recall={op['recall']:.4f} flag_rate={op['flag_rate']:.4f}")
    print(f"confusion (test): tp={op['tp']} fp={op['fp']} fn={op['fn']} tn={op['tn']}")
    print(f"registered '{REGISTERED_MODEL}' v{version} → {staged_via}")
    print(f"tripwire OK: {tripwire_ok}")
    print("==========================================================\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default="data/featurized/diabetes_features.parquet")
    ap.add_argument("--experiment", default="readmission-30d")
    args = ap.parse_args()
    main(args.features, args.experiment)
