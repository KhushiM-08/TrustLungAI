"""
TrustLung AI — train.py
========================
Main training script. Runs the full training pipeline:
  1. Load configuration
  2. Set seed for reproducibility
  3. Detect device (GPU/CPU)
  4. Load / generate dataset
  5. Build both models (MobileNetV2, EfficientNetB0)
  6. Train with Phase 1 (feature extraction) + Phase 2 (fine-tuning)
  7. Evaluate both models
  8. Generate training curves
  9. Save models and reports

Usage:
    python train.py
    python train.py --config configs/config.yaml --model MobileNetV2
    python train.py --synthetic    # Use synthetic demo data
"""

import os
import sys
import argparse
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.helpers import load_config, setup_logger, set_seed, get_device, ensure_dirs
from src.preprocessing.data_loader import load_dataset_from_directory, generate_synthetic_dataset
from src.models.architectures import get_model, model_summary_dict
from src.models.trainer import LungCancerTrainer
from src.models.uncertainty import MCDropoutPredictor
from src.evaluation.metrics import compute_metrics, plot_confusion_matrix, plot_roc_curves, build_comparison_table
from src.visualization.plots import plot_training_curves, plot_class_distribution, plot_prediction_distribution


def parse_args():
    parser = argparse.ArgumentParser(description="TrustLung AI — Training Pipeline")
    parser.add_argument("--config",    type=str, default="configs/config.yaml", help="Config file path")
    parser.add_argument("--model",     type=str, default=None, help="Model: MobileNetV2 | EfficientNetB0 | both")
    parser.add_argument("--synthetic", action="store_true",    help="Use synthetic demo data (no real dataset required)")
    parser.add_argument("--phase",     type=str, default="both", help="Training phase: extract | finetune | both")
    return parser.parse_args()


def train_model(model_arch: str, data: dict, config: dict, logger: logging.Logger) -> dict:
    """
    Full training + evaluation pipeline for one architecture.

    Args:
        model_arch: "MobileNetV2" or "EfficientNetB0"
        data:       Preprocessed dataset dict
        config:     Config dict
        logger:     Logger instance

    Returns:
        dict with model, metrics, history
    """
    model_cfg  = config["model"]
    train_cfg  = config["training"]
    data_cfg   = config["data"]
    out_cfg    = config["outputs"]

    image_size = tuple(data_cfg["image_size"])
    n_classes  = data_cfg["num_classes"]

    logger.info(f"\n{'='*60}")
    logger.info(f"  Training: {model_arch}")
    logger.info(f"{'='*60}")

    # ── Build model ────────────────────────────────────────────
    model = get_model(
        architecture=model_arch,
        input_shape=(*image_size, 3),
        num_classes=n_classes,
        dropout_rate=model_cfg["dropout_rate"],
        pretrained=model_cfg["pretrained"],
    )

    # Print model summary
    summary = model_summary_dict(model)
    for k, v in summary.items():
        logger.info(f"  {k}: {v}")

    # ── Train ──────────────────────────────────────────────────
    trainer  = LungCancerTrainer(config)
    histories = trainer.train(model, data, phase=args.phase)

    # Save training history
    history_path = os.path.join(out_cfg["reports_dir"], f"{model_arch}_history.json")
    trainer.save_history(histories, history_path)

    # ── Plot training curves ────────────────────────────────────
    curve_path = os.path.join(out_cfg["plots_dir"], f"{model_arch}_training_curves.png")
    plot_training_curves(histories, model_name=model_arch, save_path=curve_path)

    # ── Evaluate on test set ────────────────────────────────────
    logger.info(f"Evaluating {model_arch} on test set...")

    # MC Dropout prediction for uncertainty-aware evaluation
    mc_predictor = MCDropoutPredictor(
        model,
        n_samples=config["model"]["mc_dropout_samples"],
        class_names=data_cfg["class_names"],
    )
    mc_results = mc_predictor.predict_batch_with_uncertainty(
        data["X_test"], verbose=True
    )
    agg = mc_predictor.aggregate_batch_results(mc_results)

    y_true = data["y_test_raw"]
    y_pred = agg["y_pred"]
    y_prob = agg["y_prob"]

    metrics = compute_metrics(y_true, y_pred, y_prob, data_cfg["class_names"])

    # ── Confusion matrix ────────────────────────────────────────
    cm_path = os.path.join(out_cfg["plots_dir"], f"{model_arch}_confusion_matrix.png")
    plot_confusion_matrix(
        y_true, y_pred,
        class_names=data_cfg["class_names"],
        title=f"{model_arch} — Confusion Matrix",
        save_path=cm_path,
    )

    # ── ROC curves ─────────────────────────────────────────────
    roc_path = os.path.join(out_cfg["plots_dir"], f"{model_arch}_roc_curves.png")
    plot_roc_curves(
        y_true, y_prob,
        class_names=data_cfg["class_names"],
        title=f"{model_arch} — ROC Curves",
        save_path=roc_path,
    )

    logger.info(
        f"\n  {model_arch} Results:\n"
        f"    Accuracy:  {metrics['accuracy']:.4f}\n"
        f"    F1 Macro:  {metrics['f1_macro']:.4f}\n"
        f"    ROC-AUC:   {metrics['roc_auc_macro']:.4f}\n"
    )

    return {
        "model":    model,
        "metrics":  metrics,
        "history":  histories,
        "y_pred":   y_pred,
        "y_prob":   y_prob,
        "confidences":   agg["confidence"],
        "uncertainties": agg["uncertainty"],
    }


def main():
    global args
    args = parse_args()

    # ── Setup ──────────────────────────────────────────────────
    config = load_config(args.config)
    logger = setup_logger("TrustLungAI", log_dir=config["outputs"]["logs_dir"])

    # Create all output directories
    ensure_dirs([
        config["outputs"]["models_dir"],
        config["outputs"]["plots_dir"],
        config["outputs"]["reports_dir"],
        config["outputs"]["heatmaps_dir"],
        config["outputs"]["logs_dir"],
    ])

    # Reproducibility
    seed = config["project"]["seed"]
    set_seed(seed)
    logger.info(f"Random seed: {seed}")

    # Device
    device = get_device(config["project"]["device"])
    logger.info(f"Compute device: {device}")

    # ── Load data ──────────────────────────────────────────────
    data_cfg = config["data"]

    if args.synthetic:
        logger.info("Using SYNTHETIC data (demo mode)")
        data = generate_synthetic_dataset(
            n_samples=600,
            image_size=tuple(data_cfg["image_size"]),
            num_classes=data_cfg["num_classes"],
            seed=seed,
        )
    else:
        img_dir = data_cfg["raw_image_dir"]
        if not os.path.exists(img_dir):
            logger.error(f"Image directory not found: {img_dir}")
            logger.info("Run with --synthetic flag for a demo without real data.")
            sys.exit(1)

        logger.info(f"Loading real data from: {img_dir}")
        data = load_dataset_from_directory(
            root_dir=img_dir,
            image_size=tuple(data_cfg["image_size"]),
            augment_train=data_cfg["augment"],
            val_split=data_cfg["val_split"],
            test_split=data_cfg["test_split"],
            seed=seed,
            class_names=data_cfg["class_names"],
        )

    # Plot class distribution
    dist_path = os.path.join(config["outputs"]["plots_dir"], "class_distribution.png")
    plot_class_distribution(
        data["y_train_raw"],
        class_names=data_cfg["class_names"],
        title="Training Set Class Distribution",
        save_path=dist_path,
    )

    # ── Determine which models to train ────────────────────────
    model_choice = args.model or config["model"]["architecture"]
    if model_choice.lower() == "both":
        architectures = ["MobileNetV2", "EfficientNetB0"]
    else:
        architectures = [model_choice]

    logger.info(f"Training architectures: {architectures}")

    # ── Train each model ────────────────────────────────────────
    all_results = {}
    for arch in architectures:
        result = train_model(arch, data, config, logger)
        all_results[arch] = result

    # ── Model comparison table ─────────────────────────────────
    if len(all_results) > 1:
        comparison = build_comparison_table(
            {name: res["metrics"] for name, res in all_results.items()},
            save_path=os.path.join(config["outputs"]["reports_dir"], "model_comparison.csv"),
        )
        logger.info(f"\nModel Comparison:\n{comparison.to_string(index=False)}")

    logger.info("\n✅ Training complete. Check outputs/ directory for results.")


if __name__ == "__main__":
    main()
