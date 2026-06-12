"""
TrustLung AI — Module 3: Explainable AI (Grad-CAM)
====================================================
Implements Gradient-weighted Class Activation Mapping (Grad-CAM)
to visualize which regions of a CT scan the model focuses on.

Why Grad-CAM for medical AI?
  - Radiologists need to see *where* the model looks
  - Helps build trust in AI-assisted diagnosis
  - Exposes wrong focus areas (debugging)
  - Satisfies clinical and regulatory explainability requirements

Output: Side-by-side original + heatmap overlay images.
"""

import os
import logging
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

logger = logging.getLogger("TrustLungAI.explainability")


# ============================================================
# GRAD-CAM CORE
# ============================================================

class GradCAM:
    """
    Grad-CAM implementation for tf.keras models.

    Algorithm (Selvaraju et al., 2017):
      1. Forward pass: compute model output and activations at target conv layer.
      2. Compute gradient of the class score w.r.t. the conv layer activations.
      3. Global-average-pool the gradients → importance weights (α_k).
      4. Weighted combination of activation maps → raw CAM.
      5. Apply ReLU (keep only positive influence).
      6. Resize to input image size and normalize.
      7. Overlay as a heatmap on the original image.

    Usage:
        cam = GradCAM(model)
        heatmap = cam.compute(image_array, class_idx=2)
        overlay = cam.overlay_heatmap(original_image, heatmap)
    """

    def __init__(self, model: "tf.keras.Model", layer_name: Optional[str] = None):
        """
        Args:
            model:      Trained tf.keras.Model.
            layer_name: Name of the target convolutional layer.
                        If None, auto-detects the last Conv2D layer.
        """
        import tensorflow as tf
        self.model = model
        self.layer_name = layer_name or self._find_last_conv_layer()
        logger.info(f"Grad-CAM initialized on layer: {self.layer_name}")

    def _find_last_conv_layer(self) -> str:
        """
        Auto-detect the last convolutional layer in the model.
        Searches through nested models (e.g. MobileNetV2 base).
        """
        import tensorflow as tf

        # Search top-level model layers in reverse
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name
            # Search nested model (base model)
            if hasattr(layer, "layers"):
                for sublayer in reversed(layer.layers):
                    if isinstance(sublayer, tf.keras.layers.Conv2D):
                        return sublayer.name

        raise ValueError("No Conv2D layer found in the model.")

    def compute(
        self,
        image: np.ndarray,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute the Grad-CAM heatmap for a single image.

        Args:
            image:     Preprocessed image array of shape (H, W, 3),
                       already normalized. Will be expanded to (1, H, W, 3).
            class_idx: Class index to explain. If None, uses the predicted class.

        Returns:
            np.ndarray: Heatmap of shape (H, W), values in [0, 1].
        """
        import tensorflow as tf

        # Add batch dimension
        img_tensor = tf.cast(np.expand_dims(image, axis=0), tf.float32)

        # Build a sub-model that outputs (conv_activations, final_predictions)
        # We need to handle nested models (MobileNetV2 / EfficientNetB0)
        grad_model = self._build_grad_model()

        with tf.GradientTape() as tape:
            # Forward pass
            conv_outputs, predictions = grad_model(img_tensor)
            tape.watch(conv_outputs)

            if class_idx is None:
                class_idx = int(tf.argmax(predictions[0]))

            # Score for the target class
            class_score = predictions[:, class_idx]

        # Gradient of class score w.r.t. conv layer output
        grads = tape.gradient(class_score, conv_outputs)

        # Global average pooling of gradients → importance weights
        # Shape: (num_filters,)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight the activation maps by importance
        conv_outputs = conv_outputs[0]  # Remove batch dim: (H, W, C)
        cam = tf.reduce_sum(
            tf.multiply(pooled_grads, conv_outputs), axis=-1
        )

        # ReLU: only keep positive contributions
        cam = tf.nn.relu(cam)

        # Convert to numpy and normalize to [0, 1]
        cam = cam.numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to input image size
        cam = cv2.resize(cam, (image.shape[1], image.shape[0]))
        return cam.astype(np.float32)

    def _build_grad_model(self) -> "tf.keras.Model":
        """
        Build a model that returns both the target conv layer output
        and the final predictions in one forward pass.
        """
        import tensorflow as tf

        # Try to find the layer directly in the top-level model
        try:
            target_layer = self.model.get_layer(self.layer_name)
            return tf.keras.Model(
                inputs=self.model.inputs,
                outputs=[target_layer.output, self.model.output],
            )
        except ValueError:
            pass

        # Search nested models
        for layer in self.model.layers:
            if hasattr(layer, "layers"):
                try:
                    target_layer = layer.get_layer(self.layer_name)
                    # Build: input → (nested_conv_output, model_output)
                    grad_model = tf.keras.Model(
                        inputs=self.model.inputs,
                        outputs=[target_layer.output, self.model.output],
                    )
                    return grad_model
                except ValueError:
                    continue

        raise ValueError(f"Layer '{self.layer_name}' not found in model or sub-models.")

    @staticmethod
    def overlay_heatmap(
        original_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Overlay a Grad-CAM heatmap on the original CT scan image.

        Args:
            original_image: Original image array (H, W, 3), values in [0,1] or [0,255].
            heatmap:        Grad-CAM heatmap (H, W), values in [0,1].
            alpha:          Heatmap transparency (0=invisible, 1=opaque).
            colormap:       OpenCV colormap for the heatmap.

        Returns:
            np.ndarray: Blended overlay image, uint8.
        """
        # Ensure original image is uint8
        if original_image.dtype != np.uint8:
            img_display = (original_image * 255).clip(0, 255).astype(np.uint8)
        else:
            img_display = original_image.copy()

        # Convert heatmap to colored uint8
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
        colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

        # Ensure same size
        if colored_heatmap.shape[:2] != img_display.shape[:2]:
            colored_heatmap = cv2.resize(colored_heatmap, (img_display.shape[1], img_display.shape[0]))

        # Blend
        overlay = cv2.addWeighted(img_display, 1 - alpha, colored_heatmap, alpha, 0)
        return overlay


# ============================================================
# GRAD-CAM BATCH VISUALIZATION
# ============================================================

class GradCAMVisualizer:
    """
    High-level class for batch Grad-CAM generation and saving.

    Generates:
      - Side-by-side panels: Original | Heatmap | Overlay
      - Batch processing with auto-save
      - Summary grid of top predictions
    """

    def __init__(
        self,
        model: "tf.keras.Model",
        class_names: List[str],
        output_dir: str = "outputs/heatmaps",
        layer_name: Optional[str] = None,
    ):
        self.cam      = GradCAM(model, layer_name)
        self.model    = model
        self.class_names = class_names
        self.output_dir  = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def explain_single(
        self,
        image: np.ndarray,
        true_label: Optional[int] = None,
        save_path: Optional[str] = None,
        title: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, int, float]:
        """
        Generate and optionally save Grad-CAM for one image.

        Args:
            image:      Preprocessed image (H, W, 3).
            true_label: Ground truth class index (for display).
            save_path:  If provided, saves the figure to this path.
            title:      Figure title.

        Returns:
            (overlay, heatmap, predicted_class, confidence)
        """
        import tensorflow as tf

        # Get prediction
        pred = self.model.predict(np.expand_dims(image, 0), verbose=0)[0]
        pred_class = int(np.argmax(pred))
        confidence = float(pred[pred_class])

        # Compute Grad-CAM
        heatmap = self.cam.compute(image, class_idx=pred_class)
        overlay = GradCAM.overlay_heatmap(image, heatmap, alpha=0.5)

        # Build visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.patch.set_facecolor("#0f0f0f")

        titles_list = ["Original CT Scan", "Grad-CAM Heatmap", "Overlay (AI Focus)"]
        images_list = [
            (image * 255).clip(0, 255).astype(np.uint8),
            plt.cm.jet(heatmap)[:, :, :3],
            overlay,
        ]

        for ax, img, t in zip(axes, images_list, titles_list):
            ax.imshow(img, cmap="gray" if len(img.shape) == 2 else None)
            ax.set_title(t, color="white", fontsize=12, pad=8)
            ax.axis("off")

        # Add prediction info
        pred_name = self.class_names[pred_class]
        true_name = self.class_names[true_label] if true_label is not None else "Unknown"
        color = "lime" if (true_label is None or pred_class == true_label) else "red"

        sup_text = (
            f"Predicted: {pred_name}  (Confidence: {confidence:.1%})"
            + (f"  |  True Label: {true_name}" if true_label is not None else "")
        )
        fig.suptitle(title or sup_text, color=color, fontsize=13, fontweight="bold", y=1.01)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=150)
            logger.info(f"Heatmap saved → {save_path}")
        plt.close(fig)

        return overlay, heatmap, pred_class, confidence

    def explain_batch(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        paths: Optional[np.ndarray] = None,
        max_samples: int = 20,
    ) -> None:
        """
        Generate Grad-CAM for a batch of images and save all.

        Args:
            images:      Array of shape (N, H, W, 3).
            labels:      Integer labels of shape (N,).
            paths:       Optional image paths for naming files.
            max_samples: Maximum number of heatmaps to generate.
        """
        n = min(len(images), max_samples)
        logger.info(f"Generating {n} Grad-CAM explanations...")

        for i in range(n):
            name = f"gradcam_{i:04d}"
            if paths is not None and i < len(paths):
                stem = Path(str(paths[i])).stem
                name = f"gradcam_{stem}"
            save_path = str(self.output_dir / f"{name}.png")
            self.explain_single(images[i], true_label=int(labels[i]), save_path=save_path)

        logger.info(f"All Grad-CAM heatmaps saved to: {self.output_dir}")

    def explanation_grid(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        n_cols: int = 4,
        max_samples: int = 16,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Create a grid summary of Grad-CAM overlays.

        Args:
            images:      (N, H, W, 3) image array.
            labels:      (N,) integer labels.
            n_cols:      Grid columns.
            max_samples: Max images to include.
            save_path:   Output path for the grid image.
        """
        n = min(len(images), max_samples)
        n_rows = (n + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
        fig.patch.set_facecolor("#111111")
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()

        for i in range(n):
            ax = axes[i]
            pred = self.model.predict(np.expand_dims(images[i], 0), verbose=0)[0]
            pred_class  = int(np.argmax(pred))
            confidence  = float(pred[pred_class])
            heatmap     = self.cam.compute(images[i], class_idx=pred_class)
            overlay     = GradCAM.overlay_heatmap(images[i], heatmap)

            ax.imshow(overlay)
            color = "lime" if pred_class == int(labels[i]) else "red"
            ax.set_title(
                f"Pred: {self.class_names[pred_class]}\n{confidence:.1%}",
                color=color, fontsize=9,
            )
            ax.axis("off")

        # Hide unused subplots
        for j in range(n, len(axes)):
            axes[j].axis("off")

        plt.suptitle("Grad-CAM Explanation Grid", color="white", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", facecolor="#111111", dpi=150)
            logger.info(f"Grad-CAM grid saved → {save_path}")
        plt.show()
        plt.close(fig)
