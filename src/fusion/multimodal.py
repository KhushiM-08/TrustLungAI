"""
TrustLung AI — Module 6: Multi-Modal Fusion
============================================
Combines CT scan image predictions with clinical/tabular patient data
to improve diagnostic accuracy — mimicking real hospital workflows.

Fusion architecture:
  ┌─────────────────────────────────┐
  │  CT Scan Image                  │
  │  → CNN (MobileNetV2/EffNet)     │──► prob_img (3,)
  └─────────────────────────────────┘
                                         ┌──────────────────────────┐
  ┌─────────────────────────────────┐    │  Fusion Layer            │
  │  Clinical Data (age, smoking,   │    │  (Weighted Avg / Stack)  │──► Final Prediction
  │   gender, symptoms…)            │──► │                          │
  │  → XGBoost Classifier           │    └──────────────────────────┘
  └─────────────────────────────────┘
           prob_clinical (n_classes,)

Fusion Methods:
  1. Weighted Averaging  — lerp between image and clinical predictions
  2. Stacked Fusion      — train a meta-learner on concatenated probabilities
"""

import os
import logging
import joblib
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score

logger = logging.getLogger("TrustLungAI.fusion")


# ============================================================
# XGBOOST CLINICAL BRANCH
# ============================================================

class ClinicalBranch:
    """
    XGBoost-based classifier for tabular clinical data.

    Why XGBoost for clinical data?
      - Handles mixed feature types well
      - Built-in feature importance
      - Works excellently on small/medium tabular datasets
      - Interpretable via SHAP
    """

    def __init__(self, n_classes: int = 3, seed: int = 42):
        self.n_classes = n_classes
        self.seed      = seed
        self.model     = None
        self._build()

    def _build(self) -> None:
        """Initialize XGBoost model with sensible medical defaults."""
        try:
            from xgboost import XGBClassifier
            self.model = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="mlogloss",
                random_state=self.seed,
                n_jobs=-1,
                verbosity=0,
            )
            logger.info("XGBoost clinical branch initialized")
        except ImportError:
            # Fallback to RandomForest if XGBoost not available
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                random_state=self.seed,
                n_jobs=-1,
            )
            logger.warning("XGBoost not found, using RandomForest fallback")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        """
        Train the clinical branch.

        Args:
            X_train: (N, n_features) training features.
            y_train: (N,) integer class labels.
            X_val:   Optional validation features for early stopping.
            y_val:   Optional validation labels.
        """
        # XGBoost supports early stopping via eval_set
        try:
            if X_val is not None:
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )
            else:
                self.model.fit(X_train, y_train)
        except TypeError:
            self.model.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, self.model.predict(X_train))
        logger.info(f"Clinical branch trained. Train accuracy: {train_acc:.4f}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probabilities.

        Args:
            X: (N, n_features) feature array.

        Returns:
            (N, n_classes) probability array.
        """
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class indices."""
        return self.model.predict(X)

    def feature_importance(self, feature_names: Optional[List[str]] = None) -> pd.Series:
        """Return feature importance as a sorted Series."""
        try:
            imp = self.model.feature_importances_
        except AttributeError:
            return pd.Series()

        names = feature_names or [f"feature_{i}" for i in range(len(imp))]
        return pd.Series(imp, index=names).sort_values(ascending=False)

    def save(self, path: str) -> None:
        """Save clinical branch model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"Clinical branch saved → {path}")

    @classmethod
    def load(cls, path: str) -> "ClinicalBranch":
        """Load a saved clinical branch."""
        instance = cls.__new__(cls)
        instance.model = joblib.load(path)
        logger.info(f"Clinical branch loaded from: {path}")
        return instance


# ============================================================
# MULTI-MODAL FUSION
# ============================================================

class MultiModalFusion:
    """
    Fuses image-branch and clinical-branch predictions.

    Fusion modes:
      - "weighted": img_weight * prob_img + (1 - img_weight) * prob_clinical
      - "stacked":  train a meta LogisticRegression on [prob_img | prob_clinical]
    """

    def __init__(
        self,
        n_classes: int = 3,
        fusion_method: str = "weighted",
        img_weight: float = 0.6,
        seed: int = 42,
    ):
        """
        Args:
            n_classes:      Number of output classes.
            fusion_method:  "weighted" | "stacked"
            img_weight:     Weight for image branch in weighted fusion.
            seed:           Random seed.
        """
        self.n_classes     = n_classes
        self.fusion_method = fusion_method
        self.img_weight    = img_weight
        self.seed          = seed
        self.meta_clf      = None  # for stacked fusion

    def fuse(
        self,
        prob_img:      np.ndarray,
        prob_clinical: np.ndarray,
    ) -> np.ndarray:
        """
        Fuse image and clinical probabilities.

        Args:
            prob_img:      (N, n_classes) from CNN.
            prob_clinical: (N, n_classes) from XGBoost.

        Returns:
            (N, n_classes) fused probabilities.
        """
        if self.fusion_method == "weighted":
            fused = (self.img_weight * prob_img
                     + (1 - self.img_weight) * prob_clinical)
            return fused

        elif self.fusion_method == "stacked":
            if self.meta_clf is None:
                raise RuntimeError("Call fit_stacked() before fuse() in stacked mode.")
            combined = np.hstack([prob_img, prob_clinical])
            return self.meta_clf.predict_proba(combined)

        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

    def fit_stacked(
        self,
        prob_img_val:      np.ndarray,
        prob_clinical_val: np.ndarray,
        y_val:             np.ndarray,
    ) -> None:
        """
        Train the meta-classifier for stacked fusion on validation predictions.

        Args:
            prob_img_val:      (N_val, n_classes) image probabilities.
            prob_clinical_val: (N_val, n_classes) clinical probabilities.
            y_val:             (N_val,) true labels.
        """
        from sklearn.linear_model import LogisticRegression
        combined = np.hstack([prob_img_val, prob_clinical_val])
        self.meta_clf = LogisticRegression(C=1.0, max_iter=500, random_state=self.seed)
        self.meta_clf.fit(combined, y_val)
        cv_scores = cross_val_score(self.meta_clf, combined, y_val, cv=5, scoring="accuracy")
        logger.info(f"Stacked meta-clf CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    def find_optimal_weight(
        self,
        prob_img_val:      np.ndarray,
        prob_clinical_val: np.ndarray,
        y_val:             np.ndarray,
        step:              float = 0.05,
    ) -> float:
        """
        Grid-search for the optimal image_weight in weighted fusion.

        Args:
            prob_img_val:      (N, n_classes) image probabilities.
            prob_clinical_val: (N, n_classes) clinical probabilities.
            y_val:             (N,) true integer labels.
            step:              Search step size.

        Returns:
            Best image_weight value.
        """
        best_acc, best_w = 0.0, 0.6
        weights = np.arange(0.0, 1.0 + step, step)

        for w in weights:
            fused = w * prob_img_val + (1 - w) * prob_clinical_val
            preds = np.argmax(fused, axis=1)
            acc   = accuracy_score(y_val, preds)
            if acc > best_acc:
                best_acc, best_w = acc, w

        self.img_weight = best_w
        logger.info(f"Optimal fusion weight: img={best_w:.2f}, clinical={1-best_w:.2f} | Val acc={best_acc:.4f}")
        return best_w


# ============================================================
# FUSION COMPARISON VISUALIZATION
# ============================================================

def plot_fusion_comparison(
    results: Dict[str, Dict],
    class_names: List[str],
    save_path: Optional[str] = None,
) -> None:
    """
    Bar chart comparing image-only, clinical-only, and fused models.

    Args:
        results:     Dict of model_name → metrics dict (accuracy, f1, etc.).
        class_names: Class name strings.
        save_path:   Output file path.
    """
    model_names  = list(results.keys())
    metric_keys  = ["accuracy", "precision", "recall", "f1"]
    colors       = ["#60a5fa", "#f59e0b", "#4ade80", "#c084fc"]

    x     = np.arange(len(model_names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    for i, (metric, color) in enumerate(zip(metric_keys, colors)):
        vals = [results[m].get(metric, 0) for m in model_names]
        bars = ax.bar(x + i * width, vals, width, label=metric.capitalize(), color=color, alpha=0.85)
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", color="white", fontsize=8,
            )

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, color="white", fontsize=11)
    ax.set_ylabel("Score", color="white", fontsize=11)
    ax.set_title("Multi-Modal Fusion — Model Comparison", color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#475569")
    ax.legend(facecolor="#1e293b", labelcolor="white")
    ax.set_ylim(0, 1.15)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=150)
        logger.info(f"Fusion comparison plot saved → {save_path}")
    plt.show()
    plt.close(fig)


def plot_feature_importance(
    importance: pd.Series,
    top_n: int = 15,
    save_path: Optional[str] = None,
) -> None:
    """
    Horizontal bar chart for XGBoost feature importances.

    Args:
        importance: pd.Series with feature_name → importance score.
        top_n:      Show top N features.
        save_path:  Output file path.
    """
    top = importance.head(top_n)

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    bars = ax.barh(range(len(top)), top.values[::-1], color="#60a5fa", alpha=0.85)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index[::-1], color="white", fontsize=10)
    ax.set_xlabel("Feature Importance", color="white", fontsize=11)
    ax.set_title(f"Clinical Feature Importance (Top {top_n})", color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#475569")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=150)
        logger.info(f"Feature importance plot saved → {save_path}")
    plt.show()
    plt.close(fig)
