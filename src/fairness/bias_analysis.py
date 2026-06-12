"""
TrustLung AI — Module 7: Fairness & Bias Analysis
==================================================
Evaluates whether the model performs equally well across
demographic subgroups — a critical requirement for ethical medical AI.

Why fairness analysis matters:
  - AI trained on biased data can miss cancer in underrepresented groups
  - Lower recall for elderly / rural / minority patients = delayed diagnosis
  - Regulatory frameworks (EU AI Act, FDA guidance) require fairness audits
  - Builds trust with hospitals and patients

Evaluated subgroups:
  - Age groups    : <40 | 40-55 | 55-70 | 70+
  - Gender        : Male | Female
  - Smoking status: Smoker | Non-smoker

Metrics per subgroup:
  - Accuracy, Precision, Recall, F1-score
  - Confusion Matrix
  - Equalized Odds Gap (fairness metric)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

logger = logging.getLogger("TrustLungAI.fairness")


# ============================================================
# SUBGROUP EVALUATOR
# ============================================================

class FairnessAnalyzer:
    """
    Analyzes model performance across demographic subgroups.

    Usage:
        analyzer = FairnessAnalyzer(class_names=["Normal", "Benign", "Malignant"])
        report   = analyzer.analyze(y_true, y_pred, demographic_df)
        analyzer.plot_fairness_dashboard(report, save_path="outputs/plots/fairness.png")
    """

    def __init__(
        self,
        class_names: List[str] = None,
        demographic_columns: Optional[List[str]] = None,
    ):
        self.class_names          = class_names or ["Normal", "Benign", "Malignant"]
        self.demographic_columns  = demographic_columns or ["age_group", "gender", "smoking_status"]

    def analyze(
        self,
        y_true:          np.ndarray,
        y_pred:          np.ndarray,
        demographics_df: pd.DataFrame,
    ) -> Dict:
        """
        Compute per-subgroup performance metrics.

        Args:
            y_true:          (N,) true class labels.
            y_pred:          (N,) predicted class labels.
            demographics_df: DataFrame with demographic columns (same N rows).

        Returns:
            dict: {column → {group_value → metrics_dict}}
        """
        results = {}

        for col in self.demographic_columns:
            if col not in demographics_df.columns:
                logger.warning(f"Demographic column '{col}' not found — skipping.")
                continue

            col_results = {}
            groups = demographics_df[col].unique()

            for group in sorted(groups):
                mask = (demographics_df[col] == group).values
                if mask.sum() < 5:  # skip tiny groups
                    continue

                gt = y_true[mask]
                gp = y_pred[mask]

                col_results[str(group)] = {
                    "n_samples": int(mask.sum()),
                    "accuracy":  float(accuracy_score(gt, gp)),
                    "precision": float(precision_score(gt, gp, average="macro", zero_division=0)),
                    "recall":    float(recall_score(gt, gp, average="macro", zero_division=0)),
                    "f1":        float(f1_score(gt, gp, average="macro", zero_division=0)),
                    "confusion_matrix": confusion_matrix(gt, gp, labels=range(len(self.class_names))).tolist(),
                }

            results[col] = col_results
            logger.info(f"Fairness analysis for '{col}': {list(col_results.keys())}")

        # Compute overall metrics
        results["overall"] = {
            "accuracy":  float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall":    float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1":        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }

        return results

    def equalized_odds_gap(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        demographics_df: pd.DataFrame,
        col: str,
        positive_class: int = 2,
    ) -> Dict:
        """
        Compute Equalized Odds Gap (EOG) for a binary-style fairness check.

        Equalized odds requires equal TPR and FPR across groups.
        A large gap indicates the model is unfair for a subgroup.

        Args:
            y_true:         (N,) true labels.
            y_pred:         (N,) predicted labels.
            demographics_df: DataFrame with demographic columns.
            col:            Column to split on.
            positive_class: Class treated as "positive" (Malignant=2).

        Returns:
            dict: group → {tpr, fpr, tpr_gap, fpr_gap}
        """
        groups = demographics_df[col].unique()
        eog = {}

        for group in sorted(groups):
            mask = (demographics_df[col] == group).values
            if mask.sum() < 5:
                continue

            gt = (y_true[mask] == positive_class).astype(int)
            gp = (y_pred[mask] == positive_class).astype(int)

            tp = ((gp == 1) & (gt == 1)).sum()
            fn = ((gp == 0) & (gt == 1)).sum()
            fp = ((gp == 1) & (gt == 0)).sum()
            tn = ((gp == 0) & (gt == 0)).sum()

            tpr = tp / (tp + fn + 1e-8)
            fpr = fp / (fp + tn + 1e-8)
            eog[str(group)] = {"tpr": float(tpr), "fpr": float(fpr)}

        # Compute gaps from the group with highest TPR
        if eog:
            tprs = [v["tpr"] for v in eog.values()]
            fprs = [v["fpr"] for v in eog.values()]
            max_tpr, min_tpr = max(tprs), min(tprs)
            max_fpr, min_fpr = max(fprs), min(fprs)

            for group in eog:
                eog[group]["tpr_gap"] = float(max_tpr - min_tpr)
                eog[group]["fpr_gap"] = float(max_fpr - min_fpr)

        logger.info(f"EOG for '{col}': {eog}")
        return eog

    def fairness_report_dataframe(self, results: Dict) -> pd.DataFrame:
        """
        Convert analysis results to a tidy DataFrame for reporting.

        Args:
            results: Output of analyze().

        Returns:
            pd.DataFrame with columns: attribute, group, n_samples, accuracy, precision, recall, f1
        """
        rows = []
        for col, groups in results.items():
            if col == "overall":
                continue
            for group, metrics in groups.items():
                rows.append({
                    "attribute": col,
                    "group":     group,
                    "n_samples": metrics.get("n_samples", "—"),
                    "accuracy":  round(metrics.get("accuracy", 0), 4),
                    "precision": round(metrics.get("precision", 0), 4),
                    "recall":    round(metrics.get("recall", 0), 4),
                    "f1":        round(metrics.get("f1", 0), 4),
                })
        return pd.DataFrame(rows)

    # ── Visualization ──────────────────────────────────────────

    def plot_fairness_dashboard(
        self,
        results: Dict,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Multi-panel fairness dashboard showing per-subgroup metrics.

        Args:
            results:   Output of analyze().
            save_path: Path to save figure.
        """
        demographic_cols = [k for k in results if k != "overall"]
        n_cols_plot = len(demographic_cols)
        if n_cols_plot == 0:
            logger.warning("No demographic columns to plot.")
            return

        fig, axes = plt.subplots(2, n_cols_plot, figsize=(n_cols_plot * 6, 10))
        fig.patch.set_facecolor("#0f172a")

        if n_cols_plot == 1:
            axes = axes.reshape(2, 1)

        metric_colors = {
            "accuracy":  "#60a5fa",
            "precision": "#f59e0b",
            "recall":    "#4ade80",
            "f1":        "#c084fc",
        }
        overall = results.get("overall", {})

        for col_idx, col in enumerate(demographic_cols):
            groups_data = results[col]
            groups      = list(groups_data.keys())
            metrics     = ["accuracy", "precision", "recall", "f1"]

            # ── Top row: grouped bar chart ─────────────────────
            ax = axes[0, col_idx]
            ax.set_facecolor("#1e293b")
            x = np.arange(len(groups))
            width = 0.2

            for m_idx, (metric, color) in enumerate(metric_colors.items()):
                vals = [groups_data[g].get(metric, 0) for g in groups]
                bars = ax.bar(x + m_idx * width, vals, width, label=metric, color=color, alpha=0.85)

                # Dashed overall baseline
                if metric in overall:
                    ax.axhline(
                        overall[metric], color=color,
                        linestyle="--", linewidth=1.2, alpha=0.6,
                    )

            ax.set_xticks(x + width * 1.5)
            ax.set_xticklabels(groups, color="white", fontsize=10, rotation=20, ha="right")
            ax.set_title(f"Performance by {col.replace('_', ' ').title()}", color="white", fontsize=11, fontweight="bold")
            ax.set_ylabel("Score", color="white")
            ax.tick_params(colors="white")
            ax.spines[:].set_color("#475569")
            ax.set_ylim(0, 1.1)
            if col_idx == 0:
                ax.legend(facecolor="#1e293b", labelcolor="white", fontsize=8)

            # ── Bottom row: recall heatmap by group ────────────
            ax2 = axes[1, col_idx]
            ax2.set_facecolor("#1e293b")
            recall_vals = np.array([[groups_data[g].get("recall", 0)] for g in groups])
            sns.heatmap(
                recall_vals,
                annot=True, fmt=".3f",
                cmap="RdYlGn", vmin=0, vmax=1,
                ax=ax2,
                yticklabels=groups,
                xticklabels=["Recall"],
                cbar_kws={"shrink": 0.6},
            )
            ax2.set_title(f"Recall Heatmap — {col.replace('_', ' ').title()}", color="white", fontsize=10)
            ax2.tick_params(colors="white")

        plt.suptitle(
            "TrustLung AI — Fairness & Bias Analysis Dashboard",
            color="white", fontsize=14, fontweight="bold", y=1.01,
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=150)
            logger.info(f"Fairness dashboard saved → {save_path}")
        plt.show()
        plt.close(fig)

    def plot_subgroup_confusion_matrices(
        self,
        y_true:          np.ndarray,
        y_pred:          np.ndarray,
        demographics_df: pd.DataFrame,
        col:             str,
        save_path:       Optional[str] = None,
    ) -> None:
        """
        Plot confusion matrices for each subgroup in a demographic column.

        Args:
            y_true:          (N,) true labels.
            y_pred:          (N,) predicted labels.
            demographics_df: DataFrame with demographic columns.
            col:             Column to split on.
            save_path:       Output file path.
        """
        groups = sorted(demographics_df[col].unique())
        n = len(groups)
        fig, axes = plt.subplots(1, n, figsize=(n * 5, 5))
        fig.patch.set_facecolor("#0f172a")

        if n == 1:
            axes = [axes]

        for ax, group in zip(axes, groups):
            mask = (demographics_df[col] == group).values
            cm   = confusion_matrix(y_true[mask], y_pred[mask], labels=range(len(self.class_names)))
            ax.set_facecolor("#1e293b")
            sns.heatmap(
                cm, annot=True, fmt="d",
                xticklabels=self.class_names,
                yticklabels=self.class_names,
                cmap="Blues", ax=ax, cbar=False,
            )
            ax.set_title(f"{group} (n={mask.sum()})", color="white", fontsize=11)
            ax.set_xlabel("Predicted", color="white")
            ax.set_ylabel("True", color="white")
            ax.tick_params(colors="white")

        plt.suptitle(
            f"Confusion Matrices by {col.replace('_', ' ').title()}",
            color="white", fontsize=13, fontweight="bold",
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=150)
            logger.info(f"Subgroup confusion matrices saved → {save_path}")
        plt.show()
        plt.close(fig)
