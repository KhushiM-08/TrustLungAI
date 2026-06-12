"""
TrustLung AI — main.py
========================
Master orchestration script. Runs ALL 12 modules in sequence:

  Module 1  — Data Preprocessing
  Module 2  — Lightweight Model Training
  Module 3  — Grad-CAM Explainability
  Module 4  — Monte Carlo Dropout Uncertainty
  Module 5  — False Positive Reduction
  Module 6  — Multi-Modal Fusion
  Module 7  — Fairness & Bias Analysis
  Module 8  — Model Evaluation
  Module 9  — Visualization & Analytics
  Module 10 — Error Analysis
  Module 11 — Inference Benchmarking
  (Module 12 — Streamlit App: run separately with `streamlit run streamlit_app/app.py`)

Usage:
    python main.py                    # Full pipeline (real data)
    python main.py --synthetic        # Full pipeline (synthetic demo data)
    python main.py --skip-training    # Skip training, load saved models
    python main.py --modules 3,4,7   # Run only specific modules
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from src.utils.helpers import load_config, setup_logger, set_seed, get_device, ensure_dirs, print_metrics
from src.preprocessing.data_loader import (
    load_dataset_from_directory,
    generate_synthetic_dataset,
    ClinicalDataPreprocessor,
)
from src.models.architectures import get_model
from src.models.trainer import LungCancerTrainer
from src.models.uncertainty import MCDropoutPredictor, plot_uncertainty_distribution, plot_confidence_vs_accuracy
from src.models.fp_reduction import apply_confidence_threshold, ensemble_predict, ProbabilityCalibrator, false_positive_analysis
from src.explainability.gradcam import GradCAMVisualizer
from src.fusion.multimodal import ClinicalBranch, MultiModalFusion, plot_fusion_comparison, plot_feature_importance
from src.fairness.bias_analysis import FairnessAnalyzer
from src.evaluation.metrics import compute_metrics, plot_confusion_matrix, plot_roc_curves, plot_pr_curves, build_comparison_table
from src.evaluation.error_analysis import find_errors, plot_error_analysis
from src.visualization.plots import (
    plot_training_curves, plot_class_distribution,
    plot_prediction_distribution, plot_evaluation_dashboard,
)
from src.benchmarking.inference_benchmark import InferenceBenchmarker


def parse_args():
    parser = argparse.ArgumentParser(description="TrustLung AI — Full Pipeline")
    parser.add_argument("--config",         type=str,  default="configs/config.yaml")
    parser.add_argument("--synthetic",      action="store_true", help="Use synthetic demo data")
    parser.add_argument("--skip-training",  action="store_true", help="Skip training, load saved models")
    parser.add_argument("--modules",        type=str,  default=None, help="Comma-separated module numbers to run")
    return parser.parse_args()


def should_run(module_num: int, selected: list) -> bool:
    return not selected or module_num in selected


def main():
    args = parse_args()

    # ── Parse module selection ─────────────────────────────────
    selected_modules = []
    if args.modules:
        selected_modules = [int(m.strip()) for m in args.modules.split(",")]

    # ── Config & setup ─────────────────────────────────────────
    config = load_config(args.config)
    logger = setup_logger("TrustLungAI", log_dir=config["outputs"]["logs_dir"])
    logger.info("=" * 60)
    logger.info("  TrustLung AI — Full Pipeline")
    logger.info("=" * 60)

    seed = config["project"]["seed"]
    set_seed(seed)

    device = get_device(config["project"]["device"])
    logger.info(f"Device: {device}")

    data_cfg = config["data"]
    out_cfg  = config["outputs"]
    class_names = data_cfg["class_names"]

    ensure_dirs(list(out_cfg.values()))

    # ============================================================
    # MODULE 1 — DATA PREPROCESSING
    # ============================================================
    if should_run(1, selected_modules):
        logger.info("\n[ MODULE 1 ] Data Preprocessing")

        if args.synthetic:
            logger.info("  Generating synthetic dataset...")
            data = generate_synthetic_dataset(
                n_samples=600,
                image_size=tuple(data_cfg["image_size"]),
                num_classes=data_cfg["num_classes"],
                seed=seed,
            )
        else:
            img_dir = data_cfg["raw_image_dir"]
            if not os.path.exists(img_dir):
                logger.warning(f"  Image directory not found: {img_dir}. Falling back to synthetic.")
                data = generate_synthetic_dataset(600, tuple(data_cfg["image_size"]), data_cfg["num_classes"], seed)
            else:
                data = load_dataset_from_directory(
                    root_dir=img_dir,
                    image_size=tuple(data_cfg["image_size"]),
                    val_split=data_cfg["val_split"],
                    test_split=data_cfg["test_split"],
                    seed=seed,
                    class_names=class_names,
                )

        # Clinical data
        clinical_preprocessor = ClinicalDataPreprocessor()
        clinical_csv = data_cfg.get("raw_clinical_csv", "")
        if os.path.exists(clinical_csv):
            clinical_df = pd.read_csv(clinical_csv)
        else:
            logger.info("  Generating synthetic clinical data...")
            clinical_df = clinical_preprocessor.generate_demo_clinical_data(n_samples=500, seed=seed)

        X_clinical, y_clinical = clinical_preprocessor.fit_transform(clinical_df)

        plot_class_distribution(
            data["y_train_raw"], class_names,
            title="Training Set Class Distribution",
            save_path=os.path.join(out_cfg["plots_dir"], "class_distribution.png"),
        )
        logger.info(f"  Data ready: Train={len(data['X_train'])} | Val={len(data['X_val'])} | Test={len(data['X_test'])}")

    # ============================================================
    # MODULE 2 — MODEL TRAINING
    # ============================================================
    trained_models = {}
    all_metrics    = {}

    if should_run(2, selected_modules):
        logger.info("\n[ MODULE 2 ] Lightweight Model Training")
        trainer = LungCancerTrainer(config)

        for arch in ["MobileNetV2", "EfficientNetB0"]:
            if args.skip_training:
                model_path = os.path.join(out_cfg["models_dir"], f"TrustLung_{arch}_final.keras")
                if os.path.exists(model_path):
                    logger.info(f"  Loading saved model: {model_path}")
                    trained_models[arch] = trainer.load_model(model_path)
                    continue
                else:
                    logger.info(f"  No saved model found for {arch}, training from scratch...")

            model = get_model(
                architecture=arch,
                input_shape=(*data_cfg["image_size"], 3),
                num_classes=data_cfg["num_classes"],
                dropout_rate=config["model"]["dropout_rate"],
                pretrained=config["model"]["pretrained"],
            )
            histories = trainer.train(model, data, phase="both")
            trainer.save_history(histories, os.path.join(out_cfg["reports_dir"], f"{arch}_history.json"))

            plot_training_curves(
                histories, model_name=arch,
                save_path=os.path.join(out_cfg["plots_dir"], f"{arch}_training_curves.png"),
            )
            trained_models[arch] = model

    # ============================================================
    # MODULE 3 — EXPLAINABLE AI (Grad-CAM)
    # ============================================================
    if should_run(3, selected_modules) and trained_models:
        logger.info("\n[ MODULE 3 ] Explainable AI — Grad-CAM")
        for arch, model in trained_models.items():
            visualizer = GradCAMVisualizer(
                model=model,
                class_names=class_names,
                output_dir=os.path.join(out_cfg["heatmaps_dir"], arch),
            )
            n_heatmaps = min(config["explainability"]["save_top_k"], len(data["X_test"]))
            visualizer.explain_batch(
                data["X_test"][:n_heatmaps],
                data["y_test_raw"][:n_heatmaps],
                paths=data["paths_test"][:n_heatmaps],
                max_samples=n_heatmaps,
            )
            visualizer.explanation_grid(
                data["X_test"][:16], data["y_test_raw"][:16],
                save_path=os.path.join(out_cfg["plots_dir"], f"{arch}_gradcam_grid.png"),
            )

    # ============================================================
    # MODULE 4 — UNCERTAINTY ESTIMATION
    # ============================================================
    mc_results_all = {}
    if should_run(4, selected_modules) and trained_models:
        logger.info("\n[ MODULE 4 ] Confidence-Aware AI — MC Dropout")
        for arch, model in trained_models.items():
            mc_pred = MCDropoutPredictor(
                model,
                n_samples=config["model"]["mc_dropout_samples"],
                class_names=class_names,
            )
            mc_results = mc_pred.predict_batch_with_uncertainty(data["X_test"])
            agg = mc_pred.aggregate_batch_results(mc_results)
            mc_results_all[arch] = agg

            plot_uncertainty_distribution(
                agg["uncertainty"], data["y_test_raw"], class_names,
                save_path=os.path.join(out_cfg["plots_dir"], f"{arch}_uncertainty_dist.png"),
            )
            plot_confidence_vs_accuracy(
                agg["confidence"], agg["y_pred"], data["y_test_raw"],
                save_path=os.path.join(out_cfg["plots_dir"], f"{arch}_reliability_diagram.png"),
            )

    # ============================================================
    # MODULE 5 — FALSE POSITIVE REDUCTION
    # ============================================================
    if should_run(5, selected_modules) and mc_results_all:
        logger.info("\n[ MODULE 5 ] False Positive Reduction")
        for arch, agg in mc_results_all.items():
            y_pred_raw = agg["y_pred"]
            y_prob     = agg["y_prob"]

            # Confidence thresholding
            threshold = config["false_positive"]["confidence_threshold"]
            y_pred_reduced = apply_confidence_threshold(y_prob, threshold=threshold)

            false_positive_analysis(
                y_true=data["y_test_raw"],
                y_pred_before=y_pred_raw,
                y_pred_after=y_pred_reduced,
                class_names=class_names,
                save_path=os.path.join(out_cfg["plots_dir"], f"{arch}_fp_reduction.png"),
            )

            # Calibration
            calibrator = ProbabilityCalibrator(method=config["false_positive"]["calibration_method"])
            val_agg = MCDropoutPredictor(
                trained_models[arch],
                n_samples=20,
                class_names=class_names,
            ).aggregate_batch_results(
                MCDropoutPredictor(trained_models[arch], 20, class_names)
                .predict_batch_with_uncertainty(data["X_val"])
            )
            calibrator.fit(val_agg["y_prob"], data["y_val_raw"])
            cal_probs = calibrator.calibrate(y_prob)
            agg["y_prob_calibrated"] = cal_probs

    # ============================================================
    # MODULE 6 — MULTI-MODAL FUSION
    # ============================================================
    if should_run(6, selected_modules):
        logger.info("\n[ MODULE 6 ] Multi-Modal Fusion")

        # Use first available CNN model
        primary_model = list(trained_models.values())[0] if trained_models else None

        # Split clinical data to match image test set size
        n_test = len(data["X_test"])
        if len(X_clinical) >= n_test:
            X_clin_test  = X_clinical[-n_test:]
            y_clin_test  = y_clinical[-n_test:]
            X_clin_train = X_clinical[:-n_test]
            y_clin_train = y_clinical[:-n_test]
        else:
            # Repeat to match size
            reps = (n_test // len(X_clinical)) + 1
            X_clin_test  = np.tile(X_clinical, (reps, 1))[:n_test]
            y_clin_test  = np.tile(y_clinical, reps)[:n_test]
            X_clin_train = X_clinical
            y_clin_train = y_clinical

        # Clinical branch (XGBoost)
        clinical_branch = ClinicalBranch(n_classes=data_cfg["num_classes"], seed=seed)
        clinical_branch.fit(X_clin_train, y_clin_train)
        prob_clinical_test = clinical_branch.predict_proba(X_clin_test)

        # Image branch probabilities
        if primary_model is not None and mc_results_all:
            arch = list(mc_results_all.keys())[0]
            prob_img_test = mc_results_all[arch]["y_prob"][:n_test]
        else:
            prob_img_test = np.ones((n_test, data_cfg["num_classes"])) / data_cfg["num_classes"]

        # Ensure matching sizes
        min_n = min(len(prob_img_test), len(prob_clinical_test), len(data["y_test_raw"]))
        prob_img_test      = prob_img_test[:min_n]
        prob_clinical_test = prob_clinical_test[:min_n]
        y_test_fusion      = data["y_test_raw"][:min_n]

        # Fuse
        fusion = MultiModalFusion(n_classes=data_cfg["num_classes"], fusion_method="weighted", seed=seed)
        best_w = fusion.find_optimal_weight(
            prob_img_test[:min_n//2], prob_clinical_test[:min_n//2], y_test_fusion[:min_n//2]
        )
        fused_probs = fusion.fuse(prob_img_test, prob_clinical_test)
        y_pred_fused = np.argmax(fused_probs, axis=1)

        # Compute fusion metrics
        fusion_metrics = compute_metrics(
            y_test_fusion, y_pred_fused, fused_probs, class_names
        )
        clinical_preds  = clinical_branch.predict(X_clin_test[:min_n])
        clinical_metrics = compute_metrics(
            y_test_fusion, clinical_preds,
            prob_clinical_test, class_names
        )

        arch = list(trained_models.keys())[0] if trained_models else "CNN"
        img_metrics = compute_metrics(
            y_test_fusion, np.argmax(prob_img_test, axis=1),
            prob_img_test, class_names
        )

        fusion_comparison = {
            f"{arch} (Image Only)": img_metrics,
            "XGBoost (Clinical Only)": clinical_metrics,
            f"Fused (w={best_w:.2f})": fusion_metrics,
        }
        plot_fusion_comparison(
            fusion_comparison, class_names,
            save_path=os.path.join(out_cfg["plots_dir"], "fusion_comparison.png"),
        )
        plot_feature_importance(
            clinical_branch.feature_importance(clinical_preprocessor.feature_names),
            save_path=os.path.join(out_cfg["plots_dir"], "feature_importance.png"),
        )
        clinical_branch.save(os.path.join(out_cfg["models_dir"], "clinical_branch.joblib"))

    # ============================================================
    # MODULE 7 — FAIRNESS ANALYSIS
    # ============================================================
    if should_run(7, selected_modules) and mc_results_all:
        logger.info("\n[ MODULE 7 ] Fairness & Bias Analysis")

        # Use clinical data demographic columns for fairness analysis
        n_test = len(data["y_test_raw"])
        clin_df_demo = clinical_preprocessor.generate_demo_clinical_data(n_samples=n_test, seed=seed+1)

        # Build demographics DataFrame
        demo_df = pd.DataFrame({
            "age_group":     pd.cut(
                clin_df_demo["AGE"], bins=[0,40,55,70,120],
                labels=["<40","40-55","55-70","70+"]
            ).astype(str),
            "gender":        clin_df_demo["GENDER"].values,
            "smoking_status":clin_df_demo["SMOKING"].map({1:"Non-smoker",2:"Smoker"}).fillna("Unknown").values,
        }).reset_index(drop=True)

        analyzer = FairnessAnalyzer(
            class_names=class_names,
            demographic_columns=config["fairness"]["demographic_columns"],
        )

        arch = list(mc_results_all.keys())[0]
        agg  = mc_results_all[arch]

        fairness_results = analyzer.analyze(
            data["y_test_raw"][:n_test],
            agg["y_pred"][:n_test],
            demo_df,
        )
        fairness_df = analyzer.fairness_report_dataframe(fairness_results)
        fairness_path = os.path.join(out_cfg["reports_dir"], "fairness_report.csv")
        fairness_df.to_csv(fairness_path, index=False)
        logger.info(f"  Fairness report saved → {fairness_path}")

        analyzer.plot_fairness_dashboard(
            fairness_results,
            save_path=os.path.join(out_cfg["plots_dir"], "fairness_dashboard.png"),
        )
        analyzer.plot_subgroup_confusion_matrices(
            data["y_test_raw"][:n_test],
            agg["y_pred"][:n_test],
            demo_df, col="age_group",
            save_path=os.path.join(out_cfg["plots_dir"], "fairness_cm_age.png"),
        )

    # ============================================================
    # MODULE 8 — MODEL EVALUATION
    # ============================================================
    if should_run(8, selected_modules) and mc_results_all:
        logger.info("\n[ MODULE 8 ] Model Evaluation")
        for arch, agg in mc_results_all.items():
            metrics = compute_metrics(
                data["y_test_raw"], agg["y_pred"], agg["y_prob"], class_names
            )
            all_metrics[arch] = metrics
            print_metrics(metrics, title=f"{arch} — Test Metrics")

            plot_roc_curves(
                data["y_test_raw"], agg["y_prob"], class_names,
                title=f"{arch} — ROC Curves",
                save_path=os.path.join(out_cfg["plots_dir"], f"{arch}_roc.png"),
            )
            plot_pr_curves(
                data["y_test_raw"], agg["y_prob"], class_names,
                title=f"{arch} — Precision-Recall Curves",
                save_path=os.path.join(out_cfg["plots_dir"], f"{arch}_pr_curves.png"),
            )
            plot_confusion_matrix(
                data["y_test_raw"], agg["y_pred"], class_names,
                title=f"{arch} — Normalized Confusion Matrix",
                normalize=True,
                save_path=os.path.join(out_cfg["plots_dir"], f"{arch}_cm_normalized.png"),
            )
            plot_evaluation_dashboard(
                metrics, model_name=arch,
                save_path=os.path.join(out_cfg["plots_dir"], f"{arch}_eval_dashboard.png"),
            )

        # Save comparison
        if all_metrics:
            build_comparison_table(
                all_metrics,
                save_path=os.path.join(out_cfg["reports_dir"], "model_comparison.csv"),
            )

    # ============================================================
    # MODULE 9 — VISUALIZATION
    # ============================================================
    if should_run(9, selected_modules) and mc_results_all:
        logger.info("\n[ MODULE 9 ] Visualization & Analytics")
        for arch, agg in mc_results_all.items():
            plot_prediction_distribution(
                agg["y_prob"], data["y_test_raw"], class_names,
                title=f"{arch} — Prediction Distribution",
                save_path=os.path.join(out_cfg["plots_dir"], f"{arch}_pred_dist.png"),
            )

    # ============================================================
    # MODULE 10 — ERROR ANALYSIS
    # ============================================================
    if should_run(10, selected_modules) and mc_results_all:
        logger.info("\n[ MODULE 10 ] Error Analysis")
        for arch, agg in mc_results_all.items():
            error_df = find_errors(
                y_true=data["y_test_raw"],
                y_pred=agg["y_pred"],
                confidences=agg["confidence"],
                uncertainties=agg["uncertainty"],
                paths=data["paths_test"],
                class_names=class_names,
            )
            error_path = os.path.join(out_cfg["reports_dir"], f"{arch}_error_analysis.csv")
            error_df.to_csv(error_path, index=False)

            plot_error_analysis(
                data["y_test_raw"], agg["y_pred"],
                agg["confidence"], class_names,
                uncertainties=agg["uncertainty"],
                save_path=os.path.join(out_cfg["plots_dir"], f"{arch}_error_analysis.png"),
            )

    # ============================================================
    # MODULE 11 — INFERENCE BENCHMARKING
    # ============================================================
    if should_run(11, selected_modules) and trained_models:
        logger.info("\n[ MODULE 11 ] Inference Benchmarking")
        benchmarker = InferenceBenchmarker(
            n_warmup=config["benchmarking"]["num_warmup_runs"],
            n_runs=config["benchmarking"]["num_benchmark_runs"],
        )
        benchmark_results = []
        sample_img = data["X_test"][0]

        for arch, model in trained_models.items():
            model_path = os.path.join(out_cfg["models_dir"], f"{model.name}_final.keras")
            result = benchmarker.benchmark_single_image(model, sample_img, arch, model_path)
            benchmark_results.append(result)

        benchmarker.compare_models(
            benchmark_results,
            save_path=os.path.join(out_cfg["plots_dir"], "inference_benchmark.png"),
        )
        bench_df = benchmarker.generate_report(benchmark_results)
        bench_df.to_csv(
            os.path.join(out_cfg["reports_dir"], "benchmark_report.csv"), index=False
        )
        logger.info(f"\nBenchmark Report:\n{bench_df.to_string(index=False)}")

    logger.info("\n" + "=" * 60)
    logger.info("  ✅ TrustLung AI Pipeline Complete!")
    logger.info(f"  📂 Results saved to: outputs/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
