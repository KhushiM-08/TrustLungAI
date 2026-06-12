"""
TrustLung AI — Module 8: Model Evaluation
==========================================
Comprehensive evaluation pipeline generating:
  - Accuracy, Precision, Recall, F1 (per-class + macro/weighted)
  - ROC-AUC (one-vs-rest, per class)
  - Precision-Recall curves
  - Confusion matrices
  - Calibration curves
  - Model comparison tables

All plots are publication-quality with dark theme.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report,
)
from sklearn.preprocessing import label_binarize

logger = logging.getLogger("TrustLungAI.evaluation")

# ── Color palette for plots ────────────────────────────────
COLORS = ["#60a5fa", "#f59e0b", "#4ade80", "#c084fc", "#f87171"]


# ============================================================
# METRICS COMPUTATION
# ============================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
) -> Dict:
    """
    Compute the full suite of evaluation metrics.

    Args:
        y_true:      (N,) true integer class labels.
        y_pred:      (N,) predicted integer class labels.
        y_prob:      (N, n_classes) probability array.
        class_names: List of class name strings.

    Returns:
        dict with all metric values.
    """
    n_classes = len(class_names)

    # ── Basic metrics ──────────────────────────────────────────
    acc       = accuracy_score(y_true, y_pred)
    prec_mac  = precision_score(y_true, y_pred, average="macro",    zero_division=0)
    rec_mac   = recall_score(y_true, y_pred,    average="macro",    zero_division=0)
    f1_mac    = f1_score(y_true, y_pred,         average="macro",    zero_division=0)
    prec_wt   = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_wt    = recall_score(y_true, y_pred,    average="weighted", zero_division=0)
    f1_wt     = f1_score(y_true, y_pred,         average="weighted", zero_division=0)

    # ── Per-class metrics ──────────────────────────────────────
    prec_per  = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_per   = recall_score(y_true, y_pred,    average=None, zero_division=0)
    f1_per    = f1_score(y_true, y_pred,         average=None, zero_division=0)

    # ── ROC-AUC (macro OvR) ────────────────────────────────────
    try:
        roc_auc_mac = roc_auc_score(
            label_binarize(y_true, classes=range(n_classes)),
            y_prob, average="macro", multi_class="ovr",
        )
    except Exception:
        roc_auc_mac = 0.0

    # ── Per-class AUC ──────────────────────────────────────────
    y_bin = label_binarize(y_true, classes=range(n_classes))
    per_class_auc = {}
    for i, name in enumerate(class_names):
        try:
            per_class_auc[name] = float(roc_auc_score(y_bin[:, i], y_prob[:, i]))
        except Exception:
            per_class_auc[name] = 0.0

    metrics = {
        "accuracy":           round(float(acc), 4),
        "precision_macro":    round(float(prec_mac), 4),
        "recall_macro":       round(float(rec_mac), 4),
        "f1_macro":           round(float(f1_mac), 4),
        "precision_weighted": round(float(prec_wt), 4),
        "recall_weighted":    round(float(rec_wt), 4),
        "f1_weighted":        round(float(f1_wt), 4),
        "roc_auc_macro":      round(float(roc_auc_mac), 4),
        "per_class": {
            name: {
                "precision": round(float(prec_per[i]), 4),
                "recall":    round(float(rec_per[i]),  4),
                "f1":        round(float(f1_per[i]),   4),
                "auc":       per_class_auc[name],
            }
            for i, name in enumerate(class_names)
        },
    }

    logger.info(
        f"Accuracy={acc:.4f} | Prec={prec_mac:.4f} | Recall={rec_mac:.4f} | "
        f"F1={f1_mac:.4f} | AUC={roc_auc_mac:.4f}"
    )
    return metrics


# ============================================================
# CONFUSION MATRIX PLOT
# ============================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    normalize: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot a styled confusion matrix.

    Args:
        y_true:      True labels.
        y_pred:      Predicted labels.
        class_names: Class label strings.
        title:       Figure title.
        normalize:   Show percentages (True) or raw counts (False).
        save_path:   Output file path.
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    if normalize:
        cm_display = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        fmt = ".2%"
    else:
        cm_display = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    sns.heatmap(
        cm_display, annot=True, fmt=fmt,
        xticklabels=class_names, yticklabels=class_names,
        cmap="Blues", ax=ax,
        linewidths=0.5, linecolor="#475569",
    )
    ax.set_xlabel("Predicted Label", color="white", fontsize=12)
    ax.set_ylabel("True Label",      color="white", fontsize=12)
    ax.set_title(title,              color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=150)
        logger.info(f"Confusion matrix saved → {save_path}")
    plt.show()
    plt.close(fig)


# ============================================================
# ROC CURVES
# ============================================================

def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    title: str = "ROC Curves",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot one-vs-rest ROC curves for each class.

    Args:
        y_true:      (N,) true labels.
        y_prob:      (N, n_classes) probabilities.
        class_names: Class label strings.
        title:       Figure title.
        save_path:   Output file path.
    """
    n_classes = len(class_names)
    y_bin     = label_binarize(y_true, classes=range(n_classes))

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    # Per-class ROC
    for i, (name, color) in enumerate(zip(class_names, COLORS)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2.5,
                label=f"{name} (AUC = {roc_auc_val:.3f})")

    # Macro average
    all_fpr = np.unique(np.concatenate([
        roc_curve(y_bin[:, i], y_prob[:, i])[0] for i in range(n_classes)
    ]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)
    ax.plot(all_fpr, mean_tpr, color="white", linewidth=2.5, linestyle="--",
            label=f"Macro Average (AUC = {macro_auc:.3f})")

    # Diagonal (random classifier)
    ax.plot([0, 1], [0, 1], color="#475569", linestyle=":", linewidth=1.5, label="Random")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", color="white", fontsize=12)
    ax.set_ylabel("True Positive Rate",  color="white", fontsize=12)
    ax.set_title(title, color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#475569")
    ax.legend(facecolor="#1e293b", labelcolor="white", fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=150)
        logger.info(f"ROC curves saved → {save_path}")
    plt.show()
    plt.close(fig)


# ============================================================
# PRECISION-RECALL CURVES
# ============================================================

def plot_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    title: str = "Precision-Recall Curves",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot precision-recall curves for each class.

    Args:
        y_true:      (N,) true labels.
        y_prob:      (N, n_classes) probabilities.
        class_names: Class label strings.
        title:       Figure title.
        save_path:   Output file path.
    """
    n_classes = len(class_names)
    y_bin     = label_binarize(y_true, classes=range(n_classes))

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    for i, (name, color) in enumerate(zip(class_names, COLORS)):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        ap = average_precision_score(y_bin[:, i], y_prob[:, i])
        ax.plot(recall, precision, color=color, linewidth=2.5,
                label=f"{name} (AP = {ap:.3f})")

    ax.set_xlabel("Recall",    color="white", fontsize=12)
    ax.set_ylabel("Precision", color="white", fontsize=12)
    ax.set_title(title,        color="white", fontsize=13, fontweight="bold")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#475569")
    ax.legend(facecolor="#1e293b", labelcolor="white", fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=150)
        logger.info(f"PR curves saved → {save_path}")
    plt.show()
    plt.close(fig)


# ============================================================
# MODEL COMPARISON TABLE
# ============================================================

def build_comparison_table(
    model_results: Dict[str, Dict],
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build a Markdown/CSV comparison table across multiple models.

    Args:
        model_results: {model_name → metrics_dict} from compute_metrics().
        save_path:     If provided, saves CSV to this path.

    Returns:
        pd.DataFrame comparison table.
    """
    rows = []
    for model_name, metrics in model_results.items():
        rows.append({
            "Model":              model_name,
            "Accuracy":           metrics.get("accuracy",           "—"),
            "Precision (Macro)":  metrics.get("precision_macro",    "—"),
            "Recall (Macro)":     metrics.get("recall_macro",       "—"),
            "F1 (Macro)":         metrics.get("f1_macro",           "—"),
            "ROC-AUC (Macro)":    metrics.get("roc_auc_macro",      "—"),
            "F1 (Weighted)":      metrics.get("f1_weighted",        "—"),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("F1 (Macro)", ascending=False)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        logger.info(f"Comparison table saved → {save_path}")

    return df
