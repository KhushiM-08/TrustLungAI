"""
TrustLung AI — Module 10: Error Analysis
=========================================
Analyzes model mistakes to understand failure modes:
  - False Positive cases (model predicted cancer, but no cancer)
  - False Negative cases (model missed actual cancer — most dangerous)
  - Difficult / ambiguous samples (high uncertainty + wrong prediction)
  - Error pattern analysis by confidence level

Key insight: False Negatives are clinically more dangerous than
False Positives in cancer screening. We track both separately.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logger = logging.getLogger("TrustLungAI.error_analysis")

BG_DARK  = "#0f172a"
BG_PANEL = "#1e293b"
BORDER   = "#475569"
TEXT     = "white"


# ============================================================
# ERROR CASE FINDER
# ============================================================

def find_errors(
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    confidences: np.ndarray,
    uncertainties: Optional[np.ndarray] = None,
    paths:       Optional[np.ndarray]   = None,
    class_names: Optional[List[str]]    = None,
    malignant_idx: int = 2,
) -> pd.DataFrame:
    """
    Build a DataFrame of all misclassified samples with metadata.

    Args:
        y_true:        (N,) true class indices.
        y_pred:        (N,) predicted class indices.
        confidences:   (N,) confidence scores.
        uncertainties: (N,) uncertainty scores (optional).
        paths:         (N,) image paths (optional).
        class_names:   Class name list.
        malignant_idx: Index of Malignant class.

    Returns:
        pd.DataFrame with columns: index, true_label, pred_label,
        error_type, confidence, uncertainty, path
    """
    class_names = class_names or ["Normal", "Benign", "Malignant"]
    errors = []

    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            continue  # Correct prediction

        true_cls = int(y_true[i])
        pred_cls = int(y_pred[i])

        # Determine error type
        if true_cls == malignant_idx and pred_cls != malignant_idx:
            error_type = "False Negative"   # Missed cancer
        elif pred_cls == malignant_idx and true_cls != malignant_idx:
            error_type = "False Positive"   # Unnecessary alarm
        else:
            error_type = "Class Confusion"  # Wrong class, not cancer-related

        errors.append({
            "index":          i,
            "true_label":     class_names[true_cls],
            "pred_label":     class_names[pred_cls],
            "error_type":     error_type,
            "confidence":     float(confidences[i]),
            "uncertainty":    float(uncertainties[i]) if uncertainties is not None else None,
            "path":           str(paths[i]) if paths is not None else f"sample_{i}",
        })

    df = pd.DataFrame(errors)
    if len(df) > 0:
        logger.info(f"Total errors: {len(df)}")
        logger.info(f"  False Negatives: {(df['error_type'] == 'False Negative').sum()}")
        logger.info(f"  False Positives: {(df['error_type'] == 'False Positive').sum()}")
        logger.info(f"  Class Confusions: {(df['error_type'] == 'Class Confusion').sum()}")
    else:
        logger.info("No errors found (perfect predictions).")

    return df


# ============================================================
# DIFFICULT SAMPLE IDENTIFICATION
# ============================================================

def find_difficult_samples(
    y_true:       np.ndarray,
    y_pred:       np.ndarray,
    confidences:  np.ndarray,
    uncertainties: np.ndarray,
    top_k:        int = 10,
) -> pd.DataFrame:
    """
    Find samples that are both wrong AND highly uncertain.
    These are the most challenging cases for the model.

    Also finds "overconfident errors": wrong + high confidence.
    These are the most dangerous predictions.

    Args:
        y_true:        (N,) true labels.
        y_pred:        (N,) predicted labels.
        confidences:   (N,) confidence scores.
        uncertainties: (N,) uncertainty scores.
        top_k:         Return top-K difficult samples.

    Returns:
        pd.DataFrame sorted by danger score (wrong + high conf + low unc).
    """
    wrong_mask = y_true != y_pred

    records = []
    for i in np.where(wrong_mask)[0]:
        # "Danger score" = high confidence wrong prediction
        danger = float(confidences[i])   # High confidence wrong = most dangerous
        records.append({
            "index":       i,
            "true":        int(y_true[i]),
            "pred":        int(y_pred[i]),
            "confidence":  float(confidences[i]),
            "uncertainty": float(uncertainties[i]),
            "danger_score": danger,
        })

    df = pd.DataFrame(records)
    if len(df) > 0:
        df = df.sort_values("danger_score", ascending=False).head(top_k)
    return df


# ============================================================
# ERROR ANALYSIS PLOTS
# ============================================================

def plot_error_analysis(
    y_true:        np.ndarray,
    y_pred:        np.ndarray,
    confidences:   np.ndarray,
    class_names:   List[str],
    uncertainties: Optional[np.ndarray] = None,
    save_path:     Optional[str] = None,
) -> None:
    """
    Comprehensive error analysis visualization with 4 panels:
      1. Error type breakdown (bar)
      2. Confidence of correct vs incorrect predictions (box)
      3. Error distribution by true class (stacked bar)
      4. Confidence bins vs error rate

    Args:
        y_true:        (N,) true labels.
        y_pred:        (N,) predicted labels.
        confidences:   (N,) confidence scores.
        class_names:   Class label strings.
        uncertainties: (N,) uncertainty scores (optional).
        save_path:     Output file path.
    """
    error_df = find_errors(y_true, y_pred, confidences, uncertainties)
    correct_mask = y_true == y_pred

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(BG_DARK)

    # ── Panel 1: Error type breakdown ─────────────────────────
    ax = axes[0, 0]
    ax.set_facecolor(BG_PANEL)
    if len(error_df) > 0:
        counts = error_df["error_type"].value_counts()
        colors_err = {"False Negative": "#ef4444", "False Positive": "#f59e0b", "Class Confusion": "#60a5fa"}
        c_list = [colors_err.get(k, "#94a3b8") for k in counts.index]
        bars = ax.bar(counts.index, counts.values, color=c_list, alpha=0.85)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    str(int(bar.get_height())), ha="center", va="bottom", color=TEXT, fontsize=10)
    ax.set_title("Error Types", color=TEXT, fontsize=11, fontweight="bold")
    ax.set_ylabel("Count", color=TEXT)
    ax.tick_params(colors=TEXT)
    ax.spines[:].set_color(BORDER)
    for tick in ax.get_xticklabels():
        tick.set_color(TEXT)
        tick.set_fontsize(9)

    # ── Panel 2: Confidence: correct vs incorrect ──────────────
    ax = axes[0, 1]
    ax.set_facecolor(BG_PANEL)
    data_groups = [confidences[correct_mask], confidences[~correct_mask]]
    labels      = ["Correct", "Incorrect"]
    bp = ax.boxplot(data_groups, patch_artist=True, labels=labels, notch=False)
    colors_bp = ["#4ade80", "#f87171"]
    for patch, color in zip(bp["boxes"], colors_bp):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title("Confidence: Correct vs Incorrect", color=TEXT, fontsize=11, fontweight="bold")
    ax.set_ylabel("Confidence", color=TEXT)
    ax.tick_params(colors=TEXT)
    ax.spines[:].set_color(BORDER)
    for tick in ax.get_xticklabels():
        tick.set_color(TEXT)

    # ── Panel 3: Errors per true class ────────────────────────
    ax = axes[1, 0]
    ax.set_facecolor(BG_PANEL)
    n_classes = len(class_names)
    error_counts = np.zeros((n_classes, n_classes))  # [true, pred]
    for _, row in error_df.iterrows():
        t = class_names.index(row["true_label"]) if row["true_label"] in class_names else 0
        p = class_names.index(row["pred_label"]) if row["pred_label"] in class_names else 0
        error_counts[t, p] += 1

    x = np.arange(n_classes)
    for j in range(n_classes):
        ax.bar(x, error_counts[:, j], bottom=error_counts[:, :j].sum(axis=1),
               label=f"→ {class_names[j]}", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, color=TEXT, fontsize=9)
    ax.set_title("Errors by True Class", color=TEXT, fontsize=11, fontweight="bold")
    ax.set_ylabel("Error Count", color=TEXT)
    ax.tick_params(colors=TEXT)
    ax.spines[:].set_color(BORDER)
    ax.legend(facecolor=BG_PANEL, labelcolor=TEXT, fontsize=8)

    # ── Panel 4: Confidence bins vs error rate ─────────────────
    ax = axes[1, 1]
    ax.set_facecolor(BG_PANEL)
    bins = np.linspace(0, 1, 11)
    bin_labels, bin_error_rates = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() > 0:
            err_rate = (y_true[mask] != y_pred[mask]).mean()
            bin_labels.append(f"{lo:.1f}-{hi:.1f}")
            bin_error_rates.append(err_rate)

    ax.bar(range(len(bin_error_rates)), bin_error_rates, color="#f59e0b", alpha=0.85)
    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels, rotation=40, ha="right", fontsize=8, color=TEXT)
    ax.set_title("Error Rate by Confidence Bin", color=TEXT, fontsize=11, fontweight="bold")
    ax.set_ylabel("Error Rate", color=TEXT)
    ax.tick_params(colors=TEXT)
    ax.spines[:].set_color(BORDER)

    plt.suptitle("TrustLung AI — Error Analysis", color=TEXT, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", facecolor=BG_DARK, dpi=150)
        logger.info(f"Error analysis plot saved → {save_path}")
    plt.show()
    plt.close(fig)
