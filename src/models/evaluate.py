"""evaluate.py — metric + plot helpers for the readmission models.

Pure, importable functions. No training, no I/O, no MLflow here — train.py owns
orchestration and logging; this module just turns (y_true, y_prob) into the
numbers and figures we report.

Headline metrics for this imbalanced problem (positive rate ~0.09):
  * AUPRC (average precision)  — area under the precision/recall curve. The
    no-skill baseline is the prevalence (~0.09), NOT 0.5.
  * ROC-AUC                    — ranking quality; reported but secondary because
    it is optimistic under heavy imbalance.
  * recall @ fixed precision   — "if we commit to precision >= P, how many of the
    true 30-day readmits do we catch?" The operational lever for a care team
    with limited follow-up capacity.
  * Brier score                — calibration/sharpness of the raw probabilities
    (lower is better). Expect this to be POOR for class-weighted models — they
    inflate probabilities; calibration is a later stage. We report it honestly.

Accuracy is deliberately absent: "always predict no" scores ~0.91 and is useless.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: render figures to file/buffer, never to a window
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.calibration import calibration_curve

# Fixed reporting precision for recall@precision. 0.30 is ~3.3x the ~0.09
# prevalence — a defensible "worth a follow-up call" bar. This is a *reporting*
# point only; the real operating threshold is chosen later in THRESHOLD_DECISION.
TARGET_PRECISION = 0.30


def recall_at_precision(y_true, y_prob, target_precision: float = TARGET_PRECISION) -> float:
    """Best recall achievable while holding precision >= target_precision.

    Reads the precision/recall curve and returns the max recall among all
    thresholds whose precision clears the bar. Returns 0.0 if the bar is never
    reachable (model cannot hit that precision at any threshold).
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ok = precision >= target_precision
    return float(recall[ok].max()) if ok.any() else 0.0


def compute_metrics(y_true, y_prob, target_precision: float = TARGET_PRECISION) -> dict:
    """Return the headline metric dict for a set of predicted probabilities."""
    return {
        "auprc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "recall_at_precision": recall_at_precision(y_true, y_prob, target_precision),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }


# ---------------------------------------------------------------------------
# Figures — each returns a matplotlib Figure for mlflow.log_figure(...).
# ---------------------------------------------------------------------------

def pr_curve_fig(y_true, y_prob, title: str):
    """Precision/recall curve with the no-skill (prevalence) reference line."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    prevalence = float(y_true.mean())
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, label=f"AP = {ap:.3f}")
    ax.axhline(prevalence, ls="--", color="grey",
               label=f"no-skill = {prevalence:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def roc_curve_fig(y_true, y_prob, title: str):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], ls="--", color="grey", label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def calibration_fig(y_true, y_prob, title: str, n_bins: int = 10):
    """Reliability curve. A class-weighted model will sit well ABOVE the diagonal
    (systematically over-predicting) — that gap is what calibration fixes later."""
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(mean_pred, frac_pos, "o-", label="model")
    ax.plot([0, 1], [0, 1], ls="--", color="grey", label="perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction positive")
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def confusion_fig(y_true, y_prob, title: str, threshold: float = 0.5):
    """Confusion matrix at a FIXED 0.5 cut — illustrative only.

    The chosen operating threshold is decided later (THRESHOLD_DECISION.md); 0.5
    on raw class-weighted scores is just a stake in the ground so the run has a
    confusion-matrix artifact.
    """
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in [((i, j), cm[i, j]) for i in range(2) for j in range(2)]:
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["pred 0", "pred 1"])
    ax.set_yticks([0, 1], ["true 0", "true 1"])
    ax.set_title(f"{title}  (threshold={threshold})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig
