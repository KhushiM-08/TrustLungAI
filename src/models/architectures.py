"""
TrustLung AI — Module 2: Lightweight Model Architectures
=========================================================
Implements transfer learning models for lung cancer detection:
  - MobileNetV2   : ~3.4M params — ultra-lightweight for edge devices
  - EfficientNetB0: ~5.3M params — better accuracy / efficiency trade-off

Both use:
  - Pretrained ImageNet weights
  - Fine-tuning of top layers
  - Monte Carlo Dropout (Module 4)
  - GlobalAveragePooling instead of Flatten (smaller head)
"""

import os
import logging
from typing import Tuple, Optional

import numpy as np

logger = logging.getLogger("TrustLungAI.models")


# ============================================================
# TENSORFLOW / KERAS MODELS
# ============================================================

def build_mobilenetv2(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 3,
    dropout_rate: float = 0.4,
    pretrained: bool = True,
) -> "tf.keras.Model":
    """
    Build MobileNetV2-based lung cancer classifier.

    Architecture:
        MobileNetV2 (frozen) → GlobalAvgPool → Dense(256) →
        MC Dropout → BatchNorm → Dense(num_classes, softmax)

    Args:
        input_shape:  (H, W, C) — default (224, 224, 3).
        num_classes:  Number of output classes.
        dropout_rate: Dropout probability (also used for MC Dropout).
        pretrained:   Use ImageNet pretrained weights.

    Returns:
        tf.keras.Model (compiled).
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    from tensorflow.keras.applications import MobileNetV2

    weights = "imagenet" if pretrained else None

    # ── Base model (frozen initially) ─────────────────────────
    base = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=weights,
    )
    base.trainable = False  # Freeze for feature extraction phase

    # ── Classification head ────────────────────────────────────
    inputs = tf.keras.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    # Monte Carlo Dropout: training=True keeps it active at inference time
    x = layers.Dropout(dropout_rate)(x, training=True)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout_rate / 2)(x, training=True)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs, name="TrustLung_MobileNetV2")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")],
    )

    logger.info(f"MobileNetV2 built: {model.count_params():,} parameters")
    return model


def build_efficientnetb0(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 3,
    dropout_rate: float = 0.4,
    pretrained: bool = True,
) -> "tf.keras.Model":
    """
    Build EfficientNetB0-based lung cancer classifier.

    Architecture:
        EfficientNetB0 (frozen) → GlobalAvgPool → Dense(256) →
        MC Dropout → BatchNorm → Dense(num_classes, softmax)

    Args:
        input_shape:  (H, W, C).
        num_classes:  Number of output classes.
        dropout_rate: Dropout probability.
        pretrained:   Use ImageNet pretrained weights.

    Returns:
        tf.keras.Model (compiled).
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    from tensorflow.keras.applications import EfficientNetB0

    weights = "imagenet" if pretrained else None

    base = EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights=weights,
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x, training=True)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout_rate / 2)(x, training=True)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs, name="TrustLung_EfficientNetB0")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")],
    )

    logger.info(f"EfficientNetB0 built: {model.count_params():,} parameters")
    return model


def unfreeze_top_layers(model: "tf.keras.Model", n_layers: int = 20, learning_rate: float = 1e-4) -> "tf.keras.Model":
    """
    Unfreeze the top N layers of the base model for fine-tuning.
    Uses a lower learning rate to avoid destroying pretrained features.

    Args:
        model:         Keras model with frozen base.
        n_layers:      Number of layers from the end to unfreeze.
        learning_rate: Learning rate for fine-tuning phase.

    Returns:
        Recompiled model with partial unfreezing.
    """
    import tensorflow as tf

    # Find the base model (first layer that isn't Input/Functional)
    base_model = None
    for layer in model.layers:
        if hasattr(layer, "layers"):  # it's a nested model
            base_model = layer
            break

    if base_model is None:
        logger.warning("Could not find nested base model for unfreezing.")
        return model

    # Unfreeze top n_layers of base
    for layer in base_model.layers[-n_layers:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    logger.info(f"Fine-tuning: {trainable_count} trainable layers in base model")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")],
    )
    return model


def get_model(
    architecture: str = "EfficientNetB0",
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 3,
    dropout_rate: float = 0.4,
    pretrained: bool = True,
) -> "tf.keras.Model":
    """
    Model factory — returns the requested architecture.

    Args:
        architecture: "MobileNetV2" or "EfficientNetB0".
        input_shape:  (H, W, C).
        num_classes:  Output classes.
        dropout_rate: Dropout rate.
        pretrained:   ImageNet weights.

    Returns:
        Compiled tf.keras.Model.
    """
    arch = architecture.lower()
    if "mobilenet" in arch:
        return build_mobilenetv2(input_shape, num_classes, dropout_rate, pretrained)
    elif "efficientnet" in arch:
        return build_efficientnetb0(input_shape, num_classes, dropout_rate, pretrained)
    else:
        raise ValueError(f"Unknown architecture: {architecture}. Choose 'MobileNetV2' or 'EfficientNetB0'.")


# ============================================================
# MODEL SUMMARY HELPER
# ============================================================

def model_summary_dict(model: "tf.keras.Model") -> dict:
    """
    Return a dict summary of model metrics for reporting.

    Args:
        model: tf.keras.Model.

    Returns:
        dict with parameter counts and architecture info.
    """
    total_params     = model.count_params()
    trainable_params = sum(np.prod(w.shape) for w in model.trainable_weights)
    frozen_params    = total_params - trainable_params

    return {
        "model_name":        model.name,
        "total_params":      f"{total_params:,}",
        "trainable_params":  f"{trainable_params:,}",
        "frozen_params":     f"{frozen_params:,}",
        "input_shape":       str(model.input_shape),
        "output_shape":      str(model.output_shape),
    }
