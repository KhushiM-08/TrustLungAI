"""
TrustLung AI — Module 4: Confidence-Aware AI (Monte Carlo Dropout)
===================================================================
Implements Bayesian uncertainty estimation via Monte Carlo Dropout.

Why uncertainty matters in medical AI:
  - A model can be WRONG with high confidence (dangerous)
  - Uncertainty flags cases needing human radiologist review
  - Builds trust: "I'm not sure — please check this one"
  - Reduces false positives by rejecting low-confidence predictions

Monte Carlo Dropout (Gal & Ghahramani, 2016):
  - Keep Dropout layers ACTIVE during inference
  - Run N stochastic forward passes
  - Mean prediction  → final class probabilities
  - Std deviation    → epistemic uncertainty
  - Entropy of mean  → total uncertainty
"""

import logging
from typing import Tuple, Dict, Optional, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logger = logging.getLogger("TrustLungAI.uncertainty")


# ============================================================
# MONTE CARLO DROPOUT PREDICTOR
# ============================================================

class MCDropoutPredictor:
    """
    Uncertainty-aware predictor using Monte Carlo Dropout.

    At inference time, dropout layers are kept ACTIVE (training=True),
    so each forward pass samples a different "thinned" network.
    Repeating this N times approximates Bayesian posterior inference.

    Attributes:
        model:       tf.keras.Model with Dropout layers.
        n_samples:   Number of MC forward passes.
        class_names: Human-readable class labels.
    """

    def __init__(
        self,
        model: "tf.keras.Model",
        n_samples: int = 50,
        class_names: Optional[List[str]] = None,
    ):
        self.model       = model
        self.n_samples   = n_samples
        self.class_names = class_names or ["Normal", "Benign", "Malignant"]

    def predict_with_uncertainty(
        self,
        image: np.ndarray,
    ) -> Dict:
        """
        Run Monte Carlo Dropout inference on a single image.

        Args:
            image: Preprocessed image (H, W, 3) or (1, H, W, 3).

        Returns:
            dict with keys:
              - mean_probs     : (n_classes,) mean probability vector
              - std_probs      : (n_classes,) std deviation per class
              - predicted_class: int — argmax of mean_probs
              - predicted_label: str — class name
              - confidence     : float — max mean probability
              - uncertainty    : float — predictive entropy [0, 1]
              - uncertainty_label: "Low" | "Medium" | "High"
              - all_samples    : (n_samples, n_classes) all MC predictions
        """
        import tensorflow as tf

        if image.ndim == 3:
            image = np.expand_dims(image, 0)
        img_tensor = tf.cast(image, tf.float32)

        # Run N stochastic forward passes
        # training=True keeps Dropout active
        all_preds = np.stack([
            self.model(img_tensor, training=True).numpy()[0]
            for _ in range(self.n_samples)
        ])  # Shape: (n_samples, n_classes)

        mean_probs = all_preds.mean(axis=0)
        std_probs  = all_preds.std(axis=0)

        predicted_class = int(np.argmax(mean_probs))
        confidence      = float(mean_probs[predicted_class])

        # Predictive entropy: H = -sum(p * log(p))
        # Higher entropy = more uncertain
        epsilon = 1e-8  # avoid log(0)
        entropy = float(-np.sum(mean_probs * np.log(mean_probs + epsilon)))
        # Normalize to [0, 1]: max entropy for n_classes = log(n_classes)
        max_entropy = np.log(len(self.class_names))
        uncertainty = entropy / max_entropy

        # Label uncertainty level
        if uncertainty < 0.25:
            uncertainty_label = "Low"
        elif uncertainty < 0.60:
            uncertainty_label = "Medium"
        else:
            uncertainty_label = "High"

        return {
            "mean_probs":       mean_probs,
            "std_probs":        std_probs,
            "predicted_class":  predicted_class,
            "predicted_label":  self.class_names[predicted_class],
            "confidence":       confidence,
            "uncertainty":      uncertainty,
            "uncertainty_label":uncertainty_label,
            "entropy":          entropy,
            "all_samples":      all_preds,
        }

    def predict_batch_with_uncertainty(
        self,
        images: np.ndarray,
        verbose: bool = True,
    ) -> List[Dict]:
        """
        Run MC Dropout on a batch of images.

        Args:
            images:  (N, H, W, 3) array.
            verbose: Show progress.

        Returns:
            List of prediction dicts (one per image).
        """
        results = []
        n = len(images)
        for i, img in enumerate(images):
            result = self.predict_with_uncertainty(img)
            results.append(result)
            if verbose and (i + 1) % 20 == 0:
                logger.info(f"MC Dropout: {i+1}/{n} done")
        return results

    def aggregate_batch_results(self, results: List[Dict]) -> Dict:
        """
        Aggregate a list of per-image prediction dicts into
        summary arrays suitable for evaluation.

        Returns:
            dict with:
              - y_pred     : (N,) predicted class indices
              - y_prob     : (N, n_classes) mean probabilities
              - confidence : (N,) confidence scores
              - uncertainty: (N,) uncertainty scores
              - unc_labels : (N,) uncertainty labels
        """
        return {
            "y_pred":      np.array([r["predicted_class"]  for r in results]),
            "y_prob":      np.stack([r["mean_probs"]        for r in results]),
            "confidence":  np.array([r["confidence"]        for r in results]),
            "uncertainty": np.array([r["uncertainty"]       for r in results]),
            "unc_labels":  np.array([r["uncertainty_label"] for r in results]),
        }


# ============================================================
# UNCERTAINTY VISUALIZATION
# ============================================================

def plot_uncertainty_distribution(
    uncertainties: np.ndarray,
    true_labels: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
) -> None:
    """
    Plot uncertainty distribution per class.
    Useful to check: are wrong predictions more uncertain?

    Args:
        uncertainties: (N,) uncertainty scores.
        true_labels:   (N,) integer class labels.
        class_names:   Class label strings.
        save_path:     File path to save figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0f172a")

    colors = ["#22c55e", "#f59e0b", "#ef4444"]  # Normal, Benign, Malignant

    # ── Left: box plot per class ───────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#1e293b")
    data_per_class = [uncertainties[true_labels == i] for i in range(len(class_names))]
    bp = ax.boxplot(data_per_class, patch_artist=True, labels=class_names, notch=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title("Uncertainty by Class", color="white", fontsize=12)
    ax.set_ylabel("Predictive Uncertainty", color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#475569")

    # ── Right: histogram of uncertainty ───────────────────────
    ax = axes[1]
    ax.set_facecolor("#1e293b")
    for i, (name, color) in enumerate(zip(class_names, colors)):
        mask = true_labels == i
        ax.hist(
            uncertainties[mask], bins=20, alpha=0.6,
            label=name, color=color, edgecolor="none",
        )
    ax.axvline(0.25, color="yellow", linestyle="--", linewidth=1.5, label="Low/Med threshold")
    ax.axvline(0.60, color="orange", linestyle="--", linewidth=1.5, label="Med/High threshold")
    ax.set_title("Uncertainty Distribution", color="white", fontsize=12)
    ax.set_xlabel("Predictive Uncertainty", color="white")
    ax.set_ylabel("Count", color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#475569")
    ax.legend(facecolor="#1e293b", labelcolor="white")

    plt.suptitle("Monte Carlo Dropout — Uncertainty Analysis", color="white", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=150)
        logger.info(f"Uncertainty distribution saved → {save_path}")
    plt.show()
    plt.close(fig)


def plot_confidence_vs_accuracy(
    confidences: np.ndarray,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10,
    save_path: Optional[str] = None,
) -> None:
    """
    Reliability (calibration) diagram:
    Shows whether high confidence ≈ high actual accuracy.
    Well-calibrated model → points near the diagonal.

    Args:
        confidences: (N,) confidence scores.
        y_pred:      (N,) predicted class indices.
        y_true:      (N,) true class indices.
        n_bins:      Number of confidence bins.
        save_path:   Output file path.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_acc, bin_conf, bin_count = [], [], []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() > 0:
            bin_acc.append((y_pred[mask] == y_true[mask]).mean())
            bin_conf.append(confidences[mask].mean())
            bin_count.append(mask.sum())
        else:
            bin_acc.append(0)
            bin_conf.append((lo + hi) / 2)
            bin_count.append(0)

    bin_acc  = np.array(bin_acc)
    bin_conf = np.array(bin_conf)

    # ECE (Expected Calibration Error)
    total = sum(bin_count)
    ece = sum(
        count / total * abs(acc - conf)
        for count, acc, conf in zip(bin_count, bin_acc, bin_conf)
        if count > 0
    )

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    # Ideal calibration line
    ax.plot([0, 1], [0, 1], "w--", linewidth=1.5, label="Perfect calibration")
    # Gap fill (over/under confidence)
    ax.bar(bin_conf, bin_acc, width=0.08, alpha=0.7, color="#60a5fa", label="Model accuracy")
    ax.bar(bin_conf, bin_conf, width=0.08, alpha=0.3, color="#f87171", label="Model confidence")

    ax.set_xlabel("Confidence", color="white", fontsize=12)
    ax.set_ylabel("Accuracy",   color="white", fontsize=12)
    ax.set_title(f"Reliability Diagram  (ECE = {ece:.3f})", color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#475569")
    ax.legend(facecolor="#1e293b", labelcolor="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=150)
        logger.info(f"Reliability diagram saved → {save_path}")
    plt.show()
    plt.close(fig)


def plot_mc_samples(
    all_samples: np.ndarray,
    class_names: List[str],
    title: str = "MC Dropout Sample Distribution",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot distribution of MC Dropout samples for one image.
    Shows how much variance there is in probability estimates.

    Args:
        all_samples: (n_samples, n_classes) array from MCDropoutPredictor.
        class_names: Class name strings.
        title:       Figure title.
        save_path:   Output file path.
    """
    n_classes = all_samples.shape[1]
    colors = ["#22c55e", "#f59e0b", "#ef4444"][:n_classes]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    for i, (name, color) in enumerate(zip(class_names, colors)):
        ax.hist(
            all_samples[:, i], bins=25, alpha=0.7,
            color=color, label=name, edgecolor="none",
        )
        ax.axvline(
            all_samples[:, i].mean(), color=color,
            linestyle="--", linewidth=2,
            label=f"{name} mean={all_samples[:,i].mean():.2f}",
        )

    ax.set_xlabel("Predicted Probability", color="white", fontsize=11)
    ax.set_ylabel("Frequency", color="white", fontsize=11)
    ax.set_title(title, color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#475569")
    ax.legend(facecolor="#1e293b", labelcolor="white", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=150)
    plt.show()
    plt.close(fig)
