"""
TrustLung AI — Training Pipeline
==================================
Handles the full training loop for both MobileNetV2 and EfficientNetB0:
  Phase 1 — Feature extraction (frozen base, train head only)
  Phase 2 — Fine-tuning (unfreeze top N layers, lower LR)

Includes:
  - Early stopping
  - Learning rate scheduling (ReduceLROnPlateau)
  - Model checkpointing
  - TensorBoard logging
  - Class-weighted training
  - Training history persistence
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger("TrustLungAI.trainer")


# ============================================================
# CALLBACKS FACTORY
# ============================================================

def get_callbacks(
    model_name: str,
    checkpoint_dir: str = "outputs/models",
    logs_dir: str = "outputs/logs",
    patience_early_stop: int = 10,
    patience_lr: int = 5,
    lr_factor: float = 0.5,
    min_lr: float = 1e-6,
) -> list:
    """
    Build a list of Keras callbacks for training.

    Callbacks:
      - ModelCheckpoint : saves best weights by val_loss
      - EarlyStopping   : stops training when val_loss stagnates
      - ReduceLROnPlateau: halves LR when val_loss plateaus
      - TensorBoard     : logs metrics for visualization
      - CSVLogger       : saves per-epoch metrics to CSV

    Args:
        model_name:          Name string used for file naming.
        checkpoint_dir:      Where to save .keras model checkpoints.
        logs_dir:            TensorBoard / CSV log directory.
        patience_early_stop: Epochs to wait before early stopping.
        patience_lr:         Epochs to wait before reducing LR.
        lr_factor:           Factor to multiply LR by on plateau.
        min_lr:              Minimum allowed learning rate.

    Returns:
        List of tf.keras.callbacks.
    """
    import tensorflow as tf

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_best.keras")
    csv_path        = os.path.join(logs_dir, f"{model_name}_history.csv")
    tb_log_dir      = os.path.join(logs_dir, "tensorboard", model_name)

    callbacks = [
        # ── Save best model ────────────────────────────────────
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            mode="min",
            verbose=1,
        ),
        # ── Stop when not improving ────────────────────────────
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience_early_stop,
            restore_best_weights=True,
            verbose=1,
            mode="min",
        ),
        # ── Reduce LR on plateau ───────────────────────────────
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=lr_factor,
            patience=patience_lr,
            min_lr=min_lr,
            verbose=1,
            mode="min",
        ),
        # ── TensorBoard ────────────────────────────────────────
        tf.keras.callbacks.TensorBoard(
            log_dir=tb_log_dir,
            histogram_freq=1,
            write_graph=True,
        ),
        # ── CSV Logger ─────────────────────────────────────────
        tf.keras.callbacks.CSVLogger(
            filename=csv_path,
            append=False,
        ),
    ]

    logger.info(f"Callbacks ready. Checkpoint → {checkpoint_path}")
    return callbacks


# ============================================================
# MAIN TRAINER CLASS
# ============================================================

class LungCancerTrainer:
    """
    Manages the two-phase training strategy:

    Phase 1 — Feature extraction:
        Base model is frozen.
        Only the classification head is trained.
        Higher learning rate (~1e-3).
        Typically 10–20 epochs.

    Phase 2 — Fine-tuning:
        Top N layers of the base model are unfrozen.
        Lower learning rate (~1e-4) to preserve pretrained features.
        Typically 20–30 more epochs.

    Usage:
        trainer = LungCancerTrainer(config)
        history = trainer.train(model, data_dict)
        trainer.save_history(history, "outputs/reports/history_mobilenet.json")
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Full project config dict (from configs/config.yaml).
        """
        self.config = config
        self.train_cfg = config["training"]
        self.model_cfg = config["model"]
        self.out_cfg   = config["outputs"]

    def train(
        self,
        model: "tf.keras.Model",
        data: dict,
        phase: str = "both",
    ) -> Dict:
        """
        Run the training loop.

        Args:
            model:  Compiled tf.keras.Model.
            data:   Dict with X_train, y_train, X_val, y_val, class_weights.
            phase:  "extract" | "finetune" | "both"

        Returns:
            Dict with 'phase1' and/or 'phase2' history objects.
        """
        import tensorflow as tf
        from src.models.architectures import unfreeze_top_layers

        model_name = model.name
        histories  = {}

        X_train, y_train = data["X_train"], data["y_train"]
        X_val,   y_val   = data["X_val"],   data["y_val"]
        class_weights    = data.get("class_weights", None)

        logger.info(f"Starting training for: {model_name}")
        logger.info(f"Train: {X_train.shape} | Val: {X_val.shape}")

        # ── Phase 1: Feature Extraction ────────────────────────
        if phase in ("extract", "both"):
            logger.info("=== Phase 1: Feature Extraction ===")
            callbacks_p1 = get_callbacks(
                model_name=f"{model_name}_phase1",
                checkpoint_dir=self.out_cfg["models_dir"],
                logs_dir=self.out_cfg["logs_dir"],
                patience_early_stop=self.train_cfg["early_stopping_patience"],
                patience_lr=self.train_cfg["reduce_lr_patience"],
                lr_factor=self.train_cfg["reduce_lr_factor"],
                min_lr=self.train_cfg["min_lr"],
            )

            history_p1 = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=min(25, self.train_cfg["epochs"]),
                batch_size=self.train_cfg["batch_size"],
                callbacks=callbacks_p1,
                class_weight=class_weights if self.train_cfg["class_weight"] else None,
                verbose=1,
            )
            histories["phase1"] = history_p1.history
            logger.info(f"Phase 1 complete. Best val_loss: {min(history_p1.history['val_loss']):.4f}")

        # ── Phase 2: Fine-Tuning ───────────────────────────────
        if phase in ("finetune", "both"):
            logger.info("=== Phase 2: Fine-Tuning ===")
            model = unfreeze_top_layers(
                model,
                n_layers=self.model_cfg["fine_tune_layers"],
                learning_rate=self.train_cfg["fine_tune_lr"],
            )

            callbacks_p2 = get_callbacks(
                model_name=f"{model_name}_phase2",
                checkpoint_dir=self.out_cfg["models_dir"],
                logs_dir=self.out_cfg["logs_dir"],
                patience_early_stop=self.train_cfg["early_stopping_patience"],
                patience_lr=self.train_cfg["reduce_lr_patience"],
                lr_factor=self.train_cfg["reduce_lr_factor"],
                min_lr=self.train_cfg["min_lr"],
            )

            history_p2 = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.train_cfg["epochs"],
                batch_size=self.train_cfg["batch_size"],
                callbacks=callbacks_p2,
                class_weight=class_weights if self.train_cfg["class_weight"] else None,
                verbose=1,
            )
            histories["phase2"] = history_p2.history
            logger.info(f"Phase 2 complete. Best val_loss: {min(history_p2.history['val_loss']):.4f}")

        # ── Save final model ───────────────────────────────────
        final_path = os.path.join(self.out_cfg["models_dir"], f"{model_name}_final.keras")
        model.save(final_path)
        logger.info(f"Final model saved → {final_path}")

        return histories

    @staticmethod
    def save_history(histories: dict, path: str) -> None:
        """Persist training histories to JSON for later plotting."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Convert numpy floats to Python floats for JSON serialization
        serializable = {}
        for phase, hist in histories.items():
            serializable[phase] = {
                k: [float(v) for v in vals]
                for k, vals in hist.items()
            }
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info(f"Training history saved → {path}")

    @staticmethod
    def load_history(path: str) -> dict:
        """Load previously saved training history."""
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def load_model(path: str) -> "tf.keras.Model":
        """
        Load a saved Keras model.

        Args:
            path: Path to .keras or SavedModel directory.

        Returns:
            Loaded tf.keras.Model.
        """
        import tensorflow as tf
        model = tf.keras.models.load_model(path)
        logger.info(f"Model loaded from: {path}")
        return model
