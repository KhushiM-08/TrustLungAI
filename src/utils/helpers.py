"""
TrustLung AI — Utility Functions
=================================
Shared utilities used across all modules:
- config loading
- logging setup
- reproducibility (seed setting)
- device detection
- file/directory helpers
"""

import os
import sys
import random
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import yaml

# Suppress verbose TensorFlow / library warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# CONFIGURATION LOADER
# ============================================================

def load_config(config_path: str = "configs/config.yaml") -> dict:
    """
    Load the YAML configuration file.

    Args:
        config_path: Path to the config YAML file.

    Returns:
        dict: Parsed configuration dictionary.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


# ============================================================
# LOGGER SETUP
# ============================================================

def setup_logger(
    name: str = "TrustLungAI",
    log_dir: str = "outputs/logs",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Set up a logger that writes to both console and a log file.

    Args:
        name:    Logger name.
        log_dir: Directory to save log files.
        level:   Logging level (default: INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers when function is called multiple times
    if logger.handlers:
        return logger

    # Console handler (colored output)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_format)

    # File handler (full timestamps)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_format = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_format)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info(f"Logger initialized. Log file: {log_file}")
    return logger


# ============================================================
# REPRODUCIBILITY — SEED SETTER
# ============================================================

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for full reproducibility across:
    Python, NumPy, TensorFlow, and PyTorch (if available).

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # TensorFlow
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        # Enable deterministic operations (may slow training slightly)
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
    except ImportError:
        pass

    # PyTorch
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ============================================================
# DEVICE DETECTION
# ============================================================

def get_device(preference: str = "auto") -> str:
    """
    Detect the best available compute device.

    Args:
        preference: "auto" | "cpu" | "cuda" | "mps"

    Returns:
        str: Device string ("GPU", "CPU", or "MPS").
    """
    if preference == "cpu":
        return "CPU"

    # Check TensorFlow GPU
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus and preference in ("auto", "cuda"):
            # Enable memory growth to prevent OOM errors
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            return f"GPU ({len(gpus)} available)"
    except Exception:
        pass

    # Check PyTorch GPU / MPS
    try:
        import torch
        if torch.cuda.is_available() and preference in ("auto", "cuda"):
            return f"CUDA ({torch.cuda.get_device_name(0)})"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            if preference in ("auto", "mps"):
                return "MPS (Apple Silicon)"
    except Exception:
        pass

    return "CPU"


# ============================================================
# DIRECTORY HELPERS
# ============================================================

def ensure_dirs(paths: list) -> None:
    """Create a list of directories if they don't exist."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def get_output_path(config: dict, subdir: str, filename: str) -> str:
    """
    Build a full output file path from config.

    Args:
        config:   Config dictionary.
        subdir:   Sub-key under 'outputs' (e.g. 'plots_dir').
        filename: Output filename.

    Returns:
        str: Full file path.
    """
    base = config["outputs"][subdir]
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, filename)


# ============================================================
# METRICS PRETTY PRINTER
# ============================================================

def print_metrics(metrics: dict, title: str = "Evaluation Results") -> None:
    """
    Pretty-print a metrics dictionary to console.

    Args:
        metrics: Dict of metric_name -> value.
        title:   Section header.
    """
    width = 50
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")
    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"  {key:<25} {val:.4f}")
        else:
            print(f"  {key:<25} {val}")
    print(f"{'='*width}\n")


# ============================================================
# TIMESTAMP HELPER
# ============================================================

def timestamp() -> str:
    """Return current timestamp string for file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
