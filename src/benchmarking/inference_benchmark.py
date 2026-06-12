"""
TrustLung AI — Module 11: Inference Benchmarking
=================================================
Measures suitability of models for low-resource / rural healthcare deployment:
  - Inference latency (ms per image)
  - Memory usage (MB) during inference
  - Model file size (MB)
  - Throughput (images/second)
  - Comparison: MobileNetV2 vs EfficientNetB0

Target environment context:
  Rural health centers may have only basic laptops (Intel i5, 8GB RAM, no GPU).
  Models must be fast enough for near-real-time CT scan analysis.
  Typical acceptable latency: < 500ms/image on CPU.
"""

import os
import gc
import time
import logging
import platform
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psutil

logger = logging.getLogger("TrustLungAI.benchmarking")

BG_DARK  = "#0f172a"
BG_PANEL = "#1e293b"
BORDER   = "#475569"
TEXT     = "white"


# ============================================================
# MEMORY UTILITIES
# ============================================================

def get_memory_usage_mb() -> float:
    """Return current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)


def get_model_size_mb(model_path: str) -> float:
    """Return saved model file size in MB."""
    if not os.path.exists(model_path):
        return 0.0
    if os.path.isdir(model_path):
        # SavedModel directory: sum all files
        total = sum(
            os.path.getsize(os.path.join(dirpath, f))
            for dirpath, _, filenames in os.walk(model_path)
            for f in filenames
        )
        return total / (1024 ** 2)
    return os.path.getsize(model_path) / (1024 ** 2)


# ============================================================
# BENCHMARKER
# ============================================================

class InferenceBenchmarker:
    """
    Measures and reports model inference performance.

    Usage:
        benchmarker = InferenceBenchmarker()
        results = benchmarker.benchmark(model, image_array, model_name="EfficientNetB0")
        benchmarker.compare_models(results_dict, save_path="outputs/plots/benchmark.png")
    """

    def __init__(self, n_warmup: int = 5, n_runs: int = 100):
        """
        Args:
            n_warmup: Warm-up runs to prime the model/GPU cache.
            n_runs:   Number of timed runs for averaging.
        """
        self.n_warmup = n_warmup
        self.n_runs   = n_runs

    def benchmark_single_image(
        self,
        model:       "tf.keras.Model",
        image:       np.ndarray,
        model_name:  str = "Model",
        model_path:  Optional[str] = None,
    ) -> Dict:
        """
        Benchmark inference time and memory for a single image.

        Args:
            model:      Compiled tf.keras.Model.
            image:      Single image array (H, W, 3) or (1, H, W, 3).
            model_name: Label for logging/reporting.
            model_path: Path to saved model file for size measurement.

        Returns:
            dict with latency, memory, throughput, model_size metrics.
        """
        import tensorflow as tf

        if image.ndim == 3:
            image = np.expand_dims(image, 0)
        img_tensor = tf.cast(image, tf.float32)

        logger.info(f"Benchmarking: {model_name} (warmup={self.n_warmup}, runs={self.n_runs})")

        # ── Warmup ─────────────────────────────────────────────
        for _ in range(self.n_warmup):
            _ = model(img_tensor, training=False)

        # ── Memory before ──────────────────────────────────────
        gc.collect()
        mem_before = get_memory_usage_mb()

        # ── Timed runs ─────────────────────────────────────────
        latencies = []
        for _ in range(self.n_runs):
            t0 = time.perf_counter()
            _ = model(img_tensor, training=False)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)  # ms

        mem_after = get_memory_usage_mb()

        latency_mean = np.mean(latencies)
        latency_std  = np.std(latencies)
        latency_p95  = np.percentile(latencies, 95)
        throughput   = 1000.0 / latency_mean  # images/sec

        model_size = get_model_size_mb(model_path) if model_path else _estimate_model_size_mb(model)

        results = {
            "model_name":         model_name,
            "latency_mean_ms":    round(float(latency_mean),  2),
            "latency_std_ms":     round(float(latency_std),   2),
            "latency_p95_ms":     round(float(latency_p95),   2),
            "throughput_img_sec": round(float(throughput),    1),
            "memory_delta_mb":    round(float(mem_after - mem_before), 1),
            "model_size_mb":      round(float(model_size), 2),
            "n_params":           model.count_params(),
            "platform":           platform.processor(),
        }

        logger.info(
            f"  {model_name}: {latency_mean:.1f}ms ± {latency_std:.1f}ms "
            f"| {throughput:.0f} img/s | {model_size:.1f} MB"
        )
        return results

    def benchmark_batch(
        self,
        model:      "tf.keras.Model",
        images:     np.ndarray,
        batch_sizes: List[int] = [1, 8, 16, 32],
        model_name: str = "Model",
    ) -> pd.DataFrame:
        """
        Benchmark multiple batch sizes to find optimal throughput.

        Args:
            model:       Compiled tf.keras.Model.
            images:      (N, H, W, 3) array.
            batch_sizes: List of batch sizes to test.
            model_name:  Label for reporting.

        Returns:
            pd.DataFrame with per-batch-size results.
        """
        import tensorflow as tf
        rows = []

        for bs in batch_sizes:
            if bs > len(images):
                continue

            batch = images[:bs]
            tensor = tf.cast(batch, tf.float32)

            # Warmup
            for _ in range(3):
                _ = model(tensor, training=False)

            times = []
            for _ in range(max(10, self.n_runs // bs)):
                t0 = time.perf_counter()
                _ = model(tensor, training=False)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)

            lat_per_img = np.mean(times) / bs
            rows.append({
                "model":             model_name,
                "batch_size":        bs,
                "total_latency_ms":  round(float(np.mean(times)), 1),
                "per_image_ms":      round(float(lat_per_img), 2),
                "throughput_img_s":  round(1000.0 / lat_per_img, 1),
            })

        df = pd.DataFrame(rows)
        logger.info(f"Batch benchmark for {model_name}:\n{df.to_string(index=False)}")
        return df

    @staticmethod
    def compare_models(
        results_list: List[Dict],
        save_path:    Optional[str] = None,
    ) -> None:
        """
        Bar chart comparing benchmarking results across models.

        Args:
            results_list: List of dicts from benchmark_single_image().
            save_path:    Output file path.
        """
        if not results_list:
            return

        model_names = [r["model_name"]          for r in results_list]
        latencies   = [r["latency_mean_ms"]      for r in results_list]
        sizes       = [r["model_size_mb"]         for r in results_list]
        throughputs = [r["throughput_img_sec"]    for r in results_list]
        params      = [r["n_params"] / 1e6        for r in results_list]  # millions

        fig, axes = plt.subplots(1, 4, figsize=(16, 5))
        fig.patch.set_facecolor(BG_DARK)
        colors = ["#60a5fa", "#4ade80", "#f59e0b", "#c084fc"]

        datasets = [
            (latencies,   "Latency (ms/image)",     "Lower is better ↓"),
            (sizes,       "Model Size (MB)",         "Lower is better ↓"),
            (throughputs, "Throughput (img/sec)",    "Higher is better ↑"),
            (params,      "Parameters (Millions)",   "Lower is better ↓"),
        ]

        for ax, (data, ylabel, subtitle) in zip(axes, datasets):
            ax.set_facecolor(BG_PANEL)
            bars = ax.bar(model_names, data, color=colors[:len(model_names)], alpha=0.85)
            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(data) * 0.01,
                    f"{bar.get_height():.1f}",
                    ha="center", va="bottom", color=TEXT, fontsize=10, fontweight="bold",
                )
            ax.set_title(f"{ylabel}\n{subtitle}", color=TEXT, fontsize=10, fontweight="bold")
            ax.set_ylabel(ylabel, color=TEXT, fontsize=9)
            ax.tick_params(colors=TEXT)
            ax.spines[:].set_color(BORDER)
            for tick in ax.get_xticklabels():
                tick.set_color(TEXT)
                tick.set_fontsize(9)

        plt.suptitle("Inference Benchmarking — Deployment Readiness", color=TEXT, fontsize=13, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", facecolor=BG_DARK, dpi=150)
            logger.info(f"Benchmark comparison saved → {save_path}")
        plt.show()
        plt.close(fig)

    @staticmethod
    def generate_report(results_list: List[Dict]) -> pd.DataFrame:
        """Generate a tidy DataFrame report from benchmark results."""
        df = pd.DataFrame(results_list)
        cols = ["model_name", "latency_mean_ms", "latency_p95_ms",
                "throughput_img_sec", "model_size_mb", "n_params"]
        return df[[c for c in cols if c in df.columns]]


def _estimate_model_size_mb(model: "tf.keras.Model") -> float:
    """Estimate model size in MB from parameter count (4 bytes per float32)."""
    return model.count_params() * 4 / (1024 ** 2)
