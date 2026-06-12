"""
TrustLung AI — Module 1: Data Preprocessing
============================================
Handles:
- CT scan image loading, resizing, normalization
- Albumentations-based augmentation pipeline
- Class balancing via oversampling / class weights
- Clinical tabular data preprocessing (UCI dataset)
- Train / Validation / Test splitting
- Missing value handling
- Feature encoding and scaling

Datasets:
  Images  : IQ-OTH/NCCD Lung Cancer Dataset (Normal / Benign / Malignant)
  Clinical: UCI Lung Cancer Dataset
"""

import os
import glob
import logging
from pathlib import Path
from typing import Tuple, Dict, Optional, List

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import albumentations as A
from albumentations.core.composition import Compose

logger = logging.getLogger("TrustLungAI.preprocessing")


# ============================================================
# AUGMENTATION PIPELINE (Albumentations)
# ============================================================

def get_train_augmentation(image_size: Tuple[int, int] = (224, 224)) -> Compose:
    """
    Training augmentation pipeline.
    Medical imaging augmentations are conservative — we avoid
    transformations that distort clinically relevant features.

    Args:
        image_size: Target (height, width).

    Returns:
        Albumentations Compose pipeline.
    """
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        # Horizontal flip — valid for CT axial slices
        A.HorizontalFlip(p=0.5),
        # Small rotation — lung orientation
        A.Rotate(limit=10, p=0.4),
        # Brightness/contrast — simulate scanner variability
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        # Gaussian noise — simulate low-dose CT noise
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
        # Slight blur — simulate motion artifact
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        # Elastic transforms — subtle anatomical variation
        A.ElasticTransform(alpha=30, sigma=5, alpha_affine=5, p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def get_val_augmentation(image_size: Tuple[int, int] = (224, 224)) -> Compose:
    """
    Validation/test augmentation — only resize and normalize.

    Args:
        image_size: Target (height, width).

    Returns:
        Albumentations Compose pipeline.
    """
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(
    path: str,
    augmentation: Optional[Compose] = None,
    image_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Load a single CT scan image, apply augmentation, return array.

    Args:
        path:        Path to image file.
        augmentation: Albumentations pipeline (None = basic resize+normalize).
        image_size:  Target (H, W).

    Returns:
        np.ndarray of shape (H, W, 3), float32.
    """
    # Read image
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load image: {path}")

    # Convert BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if augmentation is not None:
        augmented = augmentation(image=img)
        img = augmented["image"]
    else:
        # Fallback: resize and normalize manually
        img = cv2.resize(img, (image_size[1], image_size[0]))
        img = img.astype(np.float32) / 255.0

    return img.astype(np.float32)


def load_dataset_from_directory(
    root_dir: str,
    image_size: Tuple[int, int] = (224, 224),
    augment_train: bool = True,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Load CT scan images from a directory structured as:

        root_dir/
            Normal/       *.jpg / *.png
            Benign/       *.jpg / *.png
            Malignant/    *.jpg / *.png

    Returns a dictionary with train/val/test splits and metadata.

    Args:
        root_dir:      Root directory with class subdirectories.
        image_size:    Target image size.
        augment_train: Apply augmentations to training set.
        val_split:     Fraction for validation.
        test_split:    Fraction for test.
        seed:          Random seed.
        class_names:   Explicit class ordering (alphabetical if None).

    Returns:
        dict with keys: X_train, X_val, X_test, y_train, y_val, y_test,
                        class_names, class_weights, label_encoder
    """
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root_dir}")

    logger.info(f"Loading images from: {root_dir}")

    # Discover classes
    if class_names is None:
        class_names = sorted([
            d.name for d in root_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])
    logger.info(f"Classes found: {class_names}")

    # Build file paths and labels
    all_paths, all_labels = [], []
    for label_idx, class_name in enumerate(class_names):
        class_dir = root_dir / class_name
        if not class_dir.exists():
            logger.warning(f"Class directory missing: {class_dir}")
            continue
        # Support common image formats
        patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
        files = []
        for pattern in patterns:
            files.extend(glob.glob(str(class_dir / pattern)))
            files.extend(glob.glob(str(class_dir / pattern.upper())))
        logger.info(f"  {class_name}: {len(files)} images")
        all_paths.extend(files)
        all_labels.extend([label_idx] * len(files))

    all_paths = np.array(all_paths)
    all_labels = np.array(all_labels)

    # ── Split: train / val / test ─────────────────────────────
    # First split off test set
    paths_tv, paths_test, labels_tv, labels_test = train_test_split(
        all_paths, all_labels,
        test_size=test_split,
        stratify=all_labels,
        random_state=seed,
    )
    # Then split remaining into train/val
    val_ratio = val_split / (1.0 - test_split)
    paths_train, paths_val, labels_train, labels_val = train_test_split(
        paths_tv, labels_tv,
        test_size=val_ratio,
        stratify=labels_tv,
        random_state=seed,
    )

    logger.info(f"Split sizes — Train: {len(paths_train)} | Val: {len(paths_val)} | Test: {len(paths_test)}")

    # ── Load images ───────────────────────────────────────────
    train_aug = get_train_augmentation(image_size) if augment_train else get_val_augmentation(image_size)
    val_aug   = get_val_augmentation(image_size)

    def load_batch(paths, augmentation, split_name):
        images = []
        for i, p in enumerate(paths):
            try:
                img = load_image(p, augmentation, image_size)
                images.append(img)
            except Exception as e:
                logger.warning(f"Skipping {p}: {e}")
            if (i + 1) % 100 == 0:
                logger.info(f"  Loaded {i+1}/{len(paths)} [{split_name}]")
        return np.array(images, dtype=np.float32)

    logger.info("Loading training images...")
    X_train = load_batch(paths_train, train_aug, "train")
    logger.info("Loading validation images...")
    X_val   = load_batch(paths_val,   val_aug,   "val")
    logger.info("Loading test images...")
    X_test  = load_batch(paths_test,  val_aug,   "test")

    # ── Class weights for imbalance ───────────────────────────
    classes = np.unique(labels_train)
    weights = compute_class_weight("balanced", classes=classes, y=labels_train)
    class_weights = dict(zip(classes.tolist(), weights.tolist()))
    logger.info(f"Class weights: {class_weights}")

    # ── One-hot encode labels ─────────────────────────────────
    def to_onehot(labels, num_classes):
        onehot = np.zeros((len(labels), num_classes), dtype=np.float32)
        onehot[np.arange(len(labels)), labels] = 1.0
        return onehot

    num_classes = len(class_names)
    y_train_oh = to_onehot(labels_train, num_classes)
    y_val_oh   = to_onehot(labels_val,   num_classes)
    y_test_oh  = to_onehot(labels_test,  num_classes)

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train_oh, "y_val": y_val_oh, "y_test": y_test_oh,
        "y_train_raw": labels_train, "y_val_raw": labels_val, "y_test_raw": labels_test,
        "class_names": class_names,
        "class_weights": class_weights,
        "paths_train": paths_train, "paths_val": paths_val, "paths_test": paths_test,
    }


# ============================================================
# SYNTHETIC DATASET GENERATOR (for demo / testing)
# ============================================================

def generate_synthetic_dataset(
    n_samples: int = 600,
    image_size: Tuple[int, int] = (224, 224),
    num_classes: int = 3,
    seed: int = 42,
) -> Dict:
    """
    Generate a synthetic dataset for testing when real data
    is not available. Creates random images with class labels.

    Args:
        n_samples:   Total number of synthetic samples.
        image_size:  Image dimensions.
        num_classes: Number of classes.
        seed:        Random seed.

    Returns:
        Dict matching the structure of load_dataset_from_directory().
    """
    np.random.seed(seed)
    logger.info(f"Generating synthetic dataset: {n_samples} samples, {num_classes} classes")

    class_names = ["Normal", "Benign", "Malignant"][:num_classes]
    n_per_class = n_samples // num_classes

    X_all, y_all = [], []
    for cls_idx in range(num_classes):
        for _ in range(n_per_class):
            # Each class gets a slightly different mean brightness
            # to give the model something to learn
            mean = 0.3 + cls_idx * 0.2
            img = np.clip(np.random.normal(mean, 0.1, (*image_size, 3)), 0, 1)
            X_all.append(img.astype(np.float32))
            y_all.append(cls_idx)

    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all)

    # Shuffle
    idx = np.random.permutation(len(X_all))
    X_all, y_all = X_all[idx], y_all[idx]

    # Split
    split1 = int(0.70 * len(X_all))
    split2 = int(0.85 * len(X_all))
    X_train, y_train_raw = X_all[:split1], y_all[:split1]
    X_val,   y_val_raw   = X_all[split1:split2], y_all[split1:split2]
    X_test,  y_test_raw  = X_all[split2:], y_all[split2:]

    def to_onehot(labels, nc):
        oh = np.zeros((len(labels), nc), dtype=np.float32)
        oh[np.arange(len(labels)), labels] = 1.0
        return oh

    classes = np.unique(y_train_raw)
    weights = compute_class_weight("balanced", classes=classes, y=y_train_raw)
    class_weights = dict(zip(classes.tolist(), weights.tolist()))

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": to_onehot(y_train_raw, num_classes),
        "y_val":   to_onehot(y_val_raw,   num_classes),
        "y_test":  to_onehot(y_test_raw,  num_classes),
        "y_train_raw": y_train_raw, "y_val_raw": y_val_raw, "y_test_raw": y_test_raw,
        "class_names": class_names,
        "class_weights": class_weights,
        "paths_train": np.array([f"synthetic_{i}" for i in range(len(X_train))]),
        "paths_val":   np.array([f"synthetic_{i}" for i in range(len(X_val))]),
        "paths_test":  np.array([f"synthetic_{i}" for i in range(len(X_test))]),
    }


# ============================================================
# CLINICAL DATA PREPROCESSING (UCI Dataset)
# ============================================================

class ClinicalDataPreprocessor:
    """
    Preprocesses the UCI Lung Cancer clinical dataset.

    UCI features:
        GENDER, AGE, SMOKING, YELLOW_FINGERS, ANXIETY,
        PEER_PRESSURE, CHRONIC_DISEASE, FATIGUE, ALLERGY,
        WHEEZING, ALCOHOL_CONSUMING, COUGHING,
        SHORTNESS_OF_BREATH, SWALLOWING_DIFFICULTY, CHEST_PAIN,
        LUNG_CANCER (target: YES/NO)
    """

    def __init__(self):
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.fitted = False

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit and transform the raw clinical DataFrame.

        Args:
            df: Raw UCI DataFrame.

        Returns:
            (X, y) numpy arrays.
        """
        df = df.copy()
        df.columns = df.columns.str.upper().str.strip()

        # ── Handle missing values ──────────────────────────────
        # For binary columns (1/2 coding), fill with mode
        df.fillna(df.mode().iloc[0], inplace=True)
        logger.info(f"Clinical data shape after NA fill: {df.shape}")

        # ── Encode target ──────────────────────────────────────
        target_col = "LUNG_CANCER"
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found.")
        le_target = LabelEncoder()
        y = le_target.fit_transform(df[target_col].astype(str).str.upper())
        self.label_encoders["target"] = le_target
        df.drop(columns=[target_col], inplace=True)

        # ── Encode GENDER ──────────────────────────────────────
        if "GENDER" in df.columns:
            le_gender = LabelEncoder()
            df["GENDER"] = le_gender.fit_transform(df["GENDER"].astype(str))
            self.label_encoders["GENDER"] = le_gender

        # ── Add age groups for fairness analysis ──────────────
        if "AGE" in df.columns:
            df["AGE"] = pd.to_numeric(df["AGE"], errors="coerce").fillna(df["AGE"].median())
            df["age_group"] = pd.cut(
                df["AGE"], bins=[0, 40, 55, 70, 120],
                labels=["<40", "40-55", "55-70", "70+"]
            ).astype(str)
            le_age = LabelEncoder()
            df["age_group"] = le_age.fit_transform(df["age_group"])
            self.label_encoders["age_group"] = le_age

        # ── Add smoking categories ─────────────────────────────
        if "SMOKING" in df.columns:
            # UCI codes: 1=No, 2=Yes; remap for clarity
            df["smoking_status"] = df["SMOKING"].map({1: "Non-smoker", 2: "Smoker"}).fillna("Unknown")
            le_smoke = LabelEncoder()
            df["smoking_status"] = le_smoke.fit_transform(df["smoking_status"])
            self.label_encoders["smoking_status"] = le_smoke

        # ── Scale features ─────────────────────────────────────
        self.feature_names = df.columns.tolist()
        X = self.scaler.fit_transform(df.values.astype(float))
        self.fitted = True
        logger.info(f"Clinical features: {self.feature_names}")
        return X.astype(np.float32), y.astype(np.int32)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using already-fitted preprocessor."""
        if not self.fitted:
            raise RuntimeError("Call fit_transform() first.")
        df = df.copy()
        df.columns = df.columns.str.upper().str.strip()
        df.fillna(df.mode().iloc[0], inplace=True)
        if "LUNG_CANCER" in df.columns:
            df.drop(columns=["LUNG_CANCER"], inplace=True)
        if "GENDER" in df.columns:
            df["GENDER"] = self.label_encoders["GENDER"].transform(df["GENDER"].astype(str))
        return self.scaler.transform(df[self.feature_names].values.astype(float)).astype(np.float32)

    def generate_demo_clinical_data(self, n_samples: int = 500, seed: int = 42) -> pd.DataFrame:
        """
        Generate synthetic clinical data matching UCI format.
        Useful for demos without the real dataset.
        """
        np.random.seed(seed)
        n = n_samples
        df = pd.DataFrame({
            "GENDER":               np.random.choice(["M", "F"], n),
            "AGE":                  np.random.randint(30, 85, n),
            "SMOKING":              np.random.choice([1, 2], n),
            "YELLOW_FINGERS":       np.random.choice([1, 2], n),
            "ANXIETY":              np.random.choice([1, 2], n),
            "PEER_PRESSURE":        np.random.choice([1, 2], n),
            "CHRONIC_DISEASE":      np.random.choice([1, 2], n),
            "FATIGUE":              np.random.choice([1, 2], n),
            "ALLERGY":              np.random.choice([1, 2], n),
            "WHEEZING":             np.random.choice([1, 2], n),
            "ALCOHOL_CONSUMING":    np.random.choice([1, 2], n),
            "COUGHING":             np.random.choice([1, 2], n),
            "SHORTNESS_OF_BREATH":  np.random.choice([1, 2], n),
            "SWALLOWING_DIFFICULTY":np.random.choice([1, 2], n),
            "CHEST_PAIN":           np.random.choice([1, 2], n),
            "LUNG_CANCER":          np.random.choice(["YES", "NO"], n, p=[0.4, 0.6]),
        })
        return df
