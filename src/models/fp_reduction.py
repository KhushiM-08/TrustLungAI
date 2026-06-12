"""
TrustLung AI — Module 5: False Positive Reduction
==================================================
Medical AI systems often over-predict cancer (false positives),
causing unnecessary patient anxiety, biopsies, and costs.

This module implements:
  1. Confidence Thresholding  — only predict Malignant if confidence > threshold
  2. Ensemble Averaging       — average predictions from multiple models
  3. Isotonic / Platt Calibration — align raw probabilities with true rates
  4. Probability Smoothing    — temperature scaling to soften over-confident outputs

Each technique is evaluated by comparing:
  - Before vs After false positive rates
  - Precision, Recall, F1 changes
"""

import logging
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score,
)

logger = logging.getLogger("TrustLungAI.fp_reduction")

# Class index for Malignant (adjust if class order differs)
MALIGNANT_IDX = 2


# ============================================================
# 1. CONFIDENCE THRESHOLDING
# ============================================================

def apply_confidence_threshold(
    probs: np.ndarray,
    threshold: float = 0.75,
    malignant_idx: int = MALIGNANT_IDX,
    abstain_label: int = -1,
) -> np.ndarray:
    """
    Only predict 'Malignant' if confidence ≥ threshold.
    Low-confidence Malignant predictions are abstained (set to -1)
    or reassigned to the next most likely class.

    Args:
        probs:         (N, n_classes) probability array.
        threshold:     Minimum confidence to predict Malignant.
        malignant_idx: Index of the Malignant class.
        abstain_label: Label to use for abstained predictions (-1 = flag for review).

    Returns:
        (N,) array of predicted class indices.
    """
    preds = np.argmax(probs, axis=1).copy()
    confidences = probs.max(axis=1)

    # Flag malignant predictions below threshold for human review
    flagged_mask = (preds == malignant_idx) & (confidences < threshold)
    if abstain_label == -1:
        # Reassign to second-best class (conservative approach)
        for i in np.where(flagged_mask)[0]:
            sorted_classes = np.argsort(probs[i])[::-1]
            preds[i] = sorted_classes[1]  # Take second best
    else:
        preds[flagged_mask] = abstain_label

    n_flagged = flagged_mask.sum()
    logger.info(f"Confidence threshold ({threshold}): {n_flagged} malignant predictions flagged/reassigned")
    return preds


# ============================================================
# 2. ENSEMBLE AVERAGING
# ============================================================

def ensemble_predict(
    prob_list: List[np.ndarray],
    weights: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Average probability predictions from multiple models.
    Ensemble reduces variance and smooths out individual model errors.

    Args:
        prob_list: List of (N, n_classes) probability arrays, one per model.
        weights:   Optional per-model weights (e.g. based on val accuracy).
                   If None, uniform averaging is used.

    Returns:
        (N, n_classes) averaged probability array.
    """
    if weights is None:
        weights = [1.0 / len(prob_list)] * len(prob_list)

    # Normalize weights
    total = sum(weights)
    weights = [w / total for w in weights]

    ensemble_probs = sum(w * p for w, p in zip(weights, prob_list))
    logger.info(f"Ensemble of {len(prob_list)} models averaged with weights: {[f'{w:.2f}' for w in weights]}")
    return ensemble_probs


# ============================================================
# 3. PROBABILITY CALIBRATION
# ============================================================

class ProbabilityCalibrator:
    """
    Calibrates model probability outputs so they better reflect
    true class frequencies (i.e., P(Malignant) = 0.8 means
    80% of such cases are actually malignant).

    Methods:
      - Isotonic Regression: non-parametric, more flexible
      - Platt Scaling:       logistic regression on raw scores
      - Temperature Scaling: single scalar divides logits
    """

    def __init__(self, method: str = "isotonic"):
        """
        Args:
            method: "isotonic" | "platt" | "temperature"
        """
        self.method     = method
        self.calibrators = {}  # one per class
        self.temperature = 1.0
        self.fitted      = False

    def fit(self, probs: np.ndarray, y_true: np.ndarray) -> None:
        """
        Fit calibration on validation set.

        Args:
            probs:  (N, n_classes) probability array.
            y_true: (N,) true integer class labels.
        """
        n_classes = probs.shape[1]

        if self.method == "isotonic":
            for c in range(n_classes):
                ir = IsotonicRegression(out_of_bounds="clip")
                binary_labels = (y_true == c).astype(int)
                ir.fit(probs[:, c], binary_labels)
                self.calibrators[c] = ir

        elif self.method == "platt":
            for c in range(n_classes):
                lr = LogisticRegression(C=1.0)
                binary_labels = (y_true == c).astype(int)
                lr.fit(probs[:, c].reshape(-1, 1), binary_labels)
                self.calibrators[c] = lr

        elif self.method == "temperature":
            # Optimize temperature via NLL on validation set
            from scipy.optimize import minimize_scalar
            import tensorflow as tf

            logits = np.log(probs + 1e-8)

            def nll(T):
                scaled = logits / T
                # Softmax
                exp_s = np.exp(scaled - scaled.max(axis=1, keepdims=True))
                cal_probs = exp_s / exp_s.sum(axis=1, keepdims=True)
                n = len(y_true)
                return -sum(np.log(cal_probs[i, y_true[i]] + 1e-8) for i in range(n)) / n

            result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
            self.temperature = result.x
            logger.info(f"Optimal temperature: {self.temperature:.3f}")

        self.fitted = True
        logger.info(f"Calibration fitted using method: {self.method}")

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """
        Apply fitted calibration to new probability predictions.

        Args:
            probs: (N, n_classes) raw probability array.

        Returns:
            (N, n_classes) calibrated probability array.
        """
        if not self.fitted:
            raise RuntimeError("Call fit() before calibrate().")

        if self.method in ("isotonic", "platt"):
            n_classes = probs.shape[1]
            cal_probs = np.zeros_like(probs)
            for c in range(n_classes):
                if self.method == "isotonic":
                    cal_probs[:, c] = self.calibrators[c].predict(probs[:, c])
                else:
                    cal_probs[:, c] = self.calibrators[c].predict_proba(probs[:, c].reshape(-1, 1))[:, 1]
            # Re-normalize rows
            row_sums = cal_probs.sum(axis=1, keepdims=True)
            cal_probs = cal_probs / (row_sums + 1e-8)
            return cal_probs

        elif self.method == "temperature":
            logits = np.log(probs + 1e-8) / self.temperature
            exp_l = np.exp(logits - logits.max(axis=1, keepdims=True))
            return exp_l / exp_l.sum(axis=1, keepdims=True)

        return probs


# ============================================================
# 4. FALSE POSITIVE ANALYSIS
# ============================================================

def false_positive_analysis(
    y_true: np.ndarray,
    y_pred_before: np.ndarray,
    y_pred_after: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
) -> Dict:
    """
    Compare false positive rates before and after reduction.

    Args:
        y_true:         (N,) true class labels.
        y_pred_before:  (N,) raw model predictions.
        y_pred_after:   (N,) predictions after FP reduction.
        class_names:    Class name strings.
        save_path:      Path to save comparison figure.

    Returns:
        dict with before/after metrics.
    """
    mal_idx = MALIGNANT_IDX

    def fp_rate(y_true, y_pred, pos_class=mal_idx):
        """False positive rate: FP / (FP + TN)"""
        tn = ((y_pred != pos_class) & (y_true != pos_class)).sum()
        fp = ((y_pred == pos_class) & (y_true != pos_class)).sum()
        return fp / (fp + tn + 1e-8)

    metrics_before = {
        "fp_rate":   fp_rate(y_true, y_pred_before),
        "precision": precision_score(y_true, y_pred_before, average="macro", zero_division=0),
        "recall":    recall_score(y_true, y_pred_before, average="macro", zero_division=0),
        "f1":        f1_score(y_true, y_pred_before, average="macro", zero_division=0),
    }
    metrics_after = {
        "fp_rate":   fp_rate(y_true, y_pred_after),
        "precision": precision_score(y_true, y_pred_after, average="macro", zero_division=0),
        "recall":    recall_score(y_true, y_pred_after, average="macro", zero_division=0),
        "f1":        f1_score(y_true, y_pred_after, average="macro", zero_division=0),
    }

    logger.info("=== False Positive Analysis ===")
    for k in metrics_before:
        delta = metrics_after[k] - metrics_before[k]
        logger.info(f"  {k:<12} Before={metrics_before[k]:.4f} | After={metrics_after[k]:.4f} | Δ={delta:+.4f}")

    # ── Visualization ──────────────────────────────────────────
    if save_path:
        metric_names = list(metrics_before.keys())
        vals_before  = [metrics_before[k] for k in metric_names]
        vals_after   = [metrics_after[k]  for k in metric_names]

        x = np.arange(len(metric_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#1e293b")

        bars1 = ax.bar(x - width/2, vals_before, width, label="Before FP Reduction", color="#f87171", alpha=0.8)
        bars2 = ax.bar(x + width/2, vals_after,  width, label="After FP Reduction",  color="#4ade80", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_", "\n") for m in metric_names], color="white", fontsize=11)
        ax.set_ylabel("Score", color="white")
        ax.set_title("False Positive Reduction — Before vs After", color="white", fontsize=13, fontweight="bold")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#475569")
        ax.legend(facecolor="#1e293b", labelcolor="white")
        ax.set_ylim(0, 1.1)

        # Value labels
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", color="white", fontsize=9)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", color="white", fontsize=9)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=150)
        logger.info(f"FP analysis plot saved → {save_path}")
        plt.close(fig)

    return {"before": metrics_before, "after": metrics_after}
