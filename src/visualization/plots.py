"""
TrustLung AI — Module 9: Visualization & Analytics
===================================================
Generates all publication-quality plots:
  - Training/validation accuracy and loss curves
  - Prediction probability distributions
  - Class distribution bar charts
  - Combined evaluation dashboard
  - Error analysis plots

All plots use a consistent dark medical-AI theme.
"""

import os
import json
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

logger = logging.getLogger("TrustLungAI.visualization")

# ── Consistent dark theme ──────────────────────────────────
BG_DARK   = "#0f172a"
BG_PANEL  = "#1e293b"
BORDER    = "#475569"
TEXT      = "white"
COLORS    = ["#60a5fa", "#f59e0b", "#4ade80", "#c084fc", "#f87171"]

def _style_ax(ax):
    ax.set_facecolor(BG_PANEL)
    ax.tick_params(colors=TEXT)
    ax.spines[:].set_color(BORDER)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(TEXT)


# ============================================================
# TRAINING CURVES
# ============================================================

def plot_training_curves(
    history: Dict,
    model_name: str = "Model",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot training and validation accuracy + loss curves.
    Handles both Phase 1 and Phase 2 histories.

    Args:
        history:    Dict with 'phase1' and/or 'phase2' keys,
                    each containing Keras history dicts.
        model_name: Label for the figure title.
        save_path:  Output file path.
    """
    # Merge phases if both exist
    merged = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}
    phase_boundaries = []

    for phase_key in ["phase1", "phase2"]:
        if phase_key in history:
            h = history[phase_key]
            for k in merged:
                if k in h:
                    merged[k].extend(h[k])
            if merged["loss"]:
                phase_boundaries.append(len(merged["loss"]))
    # Single-phase fallback
    if not any(merged.values()):
        merged = history

    epochs = range(1, len(merged.get("loss", [])) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(BG_DARK)

    # ── Accuracy ────────────────────────────────────────────────
    _style_ax(ax1)
    if "accuracy" in merged and merged["accuracy"]:
        ax1.plot(epochs, merged["accuracy"],     color=COLORS[0], linewidth=2, label="Train Accuracy")
        ax1.plot(epochs, merged["val_accuracy"], color=COLORS[2], linewidth=2, linestyle="--", label="Val Accuracy")
        for b in phase_boundaries[:-1]:
            ax1.axvline(b, color="#94a3b8", linestyle=":", linewidth=1.5, label="Phase boundary")
        ax1.set_xlabel("Epoch", color=TEXT, fontsize=11)
        ax1.set_ylabel("Accuracy", color=TEXT, fontsize=11)
        ax1.set_title(f"{model_name} — Accuracy", color=TEXT, fontsize=12, fontweight="bold")
        ax1.legend(facecolor=BG_PANEL, labelcolor=TEXT)
        ax1.set_ylim(0, 1.05)

    # ── Loss ────────────────────────────────────────────────────
    _style_ax(ax2)
    if "loss" in merged and merged["loss"]:
        ax2.plot(epochs, merged["loss"],     color=COLORS[4], linewidth=2, label="Train Loss")
        ax2.plot(epochs, merged["val_loss"], color=COLORS[1], linewidth=2, linestyle="--", label="Val Loss")
        for b in phase_boundaries[:-1]:
            ax2.axvline(b, color="#94a3b8", linestyle=":", linewidth=1.5)
        ax2.set_xlabel("Epoch", color=TEXT, fontsize=11)
        ax2.set_ylabel("Loss",   color=TEXT, fontsize=11)
        ax2.set_title(f"{model_name} — Loss",  color=TEXT, fontsize=12, fontweight="bold")
        ax2.legend(facecolor=BG_PANEL, labelcolor=TEXT)

    plt.suptitle(f"{model_name} — Training Curves", color=TEXT, fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", facecolor=BG_DARK, dpi=150)
        logger.info(f"Training curves saved → {save_path}")
    plt.show()
    plt.close(fig)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

def plot_class_distribution(
    labels: np.ndarray,
    class_names: List[str],
    title: str = "Class Distribution",
    save_path: Optional[str] = None,
) -> None:
    """
    Bar chart showing number of samples per class.
    Highlights class imbalance visually.

    Args:
        labels:      (N,) integer class labels.
        class_names: Class label strings.
        title:       Figure title.
        save_path:   Output file path.
    """
    counts = [np.sum(labels == i) for i in range(len(class_names))]
    total  = sum(counts)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(BG_DARK)
    _style_ax(ax)

    bars = ax.bar(class_names, counts, color=COLORS[:len(class_names)], alpha=0.85, edgecolor="none")

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{count}\n({count/total:.1%})",
            ha="center", va="bottom", color=TEXT, fontsize=11, fontweight="bold",
        )

    ax.set_title(title, color=TEXT, fontsize=13, fontweight="bold")
    ax.set_ylabel("Sample Count", color=TEXT, fontsize=11)
    ax.set_xlabel("Class",        color=TEXT, fontsize=11)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", facecolor=BG_DARK, dpi=150)
        logger.info(f"Class distribution saved → {save_path}")
    plt.show()
    plt.close(fig)


# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

def plot_prediction_distribution(
    y_prob:      np.ndarray,
    y_true:      np.ndarray,
    class_names: List[str],
    title:       str = "Prediction Probability Distribution",
    save_path:   Optional[str] = None,
) -> None:
    """
    Violin / KDE plot of predicted probabilities per class.
    Shows model confidence profile.

    Args:
        y_prob:      (N, n_classes) probability array.
        y_true:      (N,) true labels.
        class_names: Class label strings.
        title:       Figure title.
        save_path:   Output file path.
    """
    n_classes = len(class_names)
    fig, axes = plt.subplots(1, n_classes, figsize=(n_classes * 5, 5), sharey=True)
    fig.patch.set_facecolor(BG_DARK)

    if n_classes == 1:
        axes = [axes]

    for i, (ax, name, color) in enumerate(zip(axes, class_names, COLORS)):
        _style_ax(ax)
        # Separate correctly classified vs misclassified
        correct_mask   = (y_true == i)
        incorrect_mask = (y_true != i)

        probs_correct   = y_prob[correct_mask, i]
        probs_incorrect = y_prob[incorrect_mask, i]

        if len(probs_correct) > 0:
            ax.hist(probs_correct,   bins=20, alpha=0.7, color=color,   label="Correct",   density=True)
        if len(probs_incorrect) > 0:
            ax.hist(probs_incorrect, bins=20, alpha=0.5, color="#f87171", label="Incorrect", density=True)

        ax.set_title(name, color=TEXT, fontsize=11, fontweight="bold")
        ax.set_xlabel("P(class)", color=TEXT, fontsize=10)
        ax.legend(facecolor=BG_PANEL, labelcolor=TEXT, fontsize=8)

    axes[0].set_ylabel("Density", color=TEXT, fontsize=11)
    plt.suptitle(title, color=TEXT, fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", facecolor=BG_DARK, dpi=150)
        logger.info(f"Prediction distribution saved → {save_path}")
    plt.show()
    plt.close(fig)


# ============================================================
# EVALUATION DASHBOARD (All-in-One)
# ============================================================

def plot_evaluation_dashboard(
    metrics:     Dict,
    model_name:  str = "Model",
    save_path:   Optional[str] = None,
) -> None:
    """
    Single-figure dashboard summarizing key evaluation metrics.

    Args:
        metrics:    Output of compute_metrics().
        model_name: Label for title.
        save_path:  Output file path.
    """
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor(BG_DARK)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    ax_summary = fig.add_subplot(gs[0, 0])
    ax_perclass = fig.add_subplot(gs[0, 1:])
    ax_text     = fig.add_subplot(gs[1, :])

    # ── Overall metrics bar ────────────────────────────────────
    _style_ax(ax_summary)
    summary_keys  = ["accuracy", "f1_macro", "roc_auc_macro", "precision_macro", "recall_macro"]
    summary_vals  = [metrics.get(k, 0) for k in summary_keys]
    summary_labels = ["Accuracy", "F1 (Macro)", "AUC", "Precision", "Recall"]
    bars = ax_summary.barh(summary_labels, summary_vals, color=COLORS[:5], alpha=0.85)
    for bar in bars:
        ax_summary.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                        f"{bar.get_width():.3f}", va="center", color=TEXT, fontsize=9)
    ax_summary.set_xlim(0, 1.1)
    ax_summary.set_title("Overall Metrics", color=TEXT, fontsize=11, fontweight="bold")

    # ── Per-class bar chart ────────────────────────────────────
    _style_ax(ax_perclass)
    per = metrics.get("per_class", {})
    class_names  = list(per.keys())
    per_metrics  = ["precision", "recall", "f1", "auc"]
    x = np.arange(len(class_names))
    width = 0.2

    for m_idx, (pm, color) in enumerate(zip(per_metrics, COLORS)):
        vals = [per[c].get(pm, 0) for c in class_names]
        ax_perclass.bar(x + m_idx * width, vals, width, label=pm.capitalize(), color=color, alpha=0.85)

    ax_perclass.set_xticks(x + width * 1.5)
    ax_perclass.set_xticklabels(class_names, color=TEXT, fontsize=10)
    ax_perclass.set_ylim(0, 1.15)
    ax_perclass.set_title("Per-Class Metrics", color=TEXT, fontsize=11, fontweight="bold")
    ax_perclass.legend(facecolor=BG_PANEL, labelcolor=TEXT, fontsize=8)

    # ── Metrics text summary ───────────────────────────────────
    ax_text.set_facecolor(BG_PANEL)
    ax_text.axis("off")
    summary_lines = [f"{'Metric':<30} {'Value':>10}"]
    summary_lines.append("-" * 42)
    for k in summary_keys:
        summary_lines.append(f"{k.replace('_', ' ').title():<30} {metrics.get(k, 0):>10.4f}")
    ax_text.text(
        0.02, 0.95, "\n".join(summary_lines),
        transform=ax_text.transAxes, fontsize=10,
        color=TEXT, va="top", fontfamily="monospace",
    )

    fig.suptitle(f"{model_name} — Evaluation Dashboard", color=TEXT, fontsize=14, fontweight="bold")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", facecolor=BG_DARK, dpi=150)
        logger.info(f"Evaluation dashboard saved → {save_path}")
    plt.show()
    plt.close(fig)
