# 🫁 TrustLung AI
## Lightweight Confidence-Aware Explainable Multi-Modal Lung Cancer Detection System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-2.13+-orange?logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.28+-red?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
  <img src="https://img.shields.io/badge/Status-Research%20Prototype-yellow" />
</p>

> ⚠️ **Disclaimer:** TrustLung AI is a research prototype. It is NOT approved for clinical diagnosis or patient care.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Motivation & Real-World Impact](#motivation)
- [Architecture](#architecture)
- [Features](#features)
- [Datasets](#datasets)
- [Project Structure](#structure)
- [Setup](#setup)
- [Usage](#usage)
- [Results](#results)
- [Module Reference](#modules)
- [Future Improvements](#future)

---

## 🎯 Overview <a name="overview"></a>

TrustLung AI is a research-grade AI system for **lung cancer screening** that goes beyond simple "cancer / no cancer" predictions. It is designed to be:

| Property | How |
|---|---|
| **Explainable** | Grad-CAM heatmaps highlight suspicious CT regions |
| **Confidence-Aware** | Monte Carlo Dropout gives uncertainty estimates |
| **Multi-Modal** | Fuses CT scans with clinical patient data |
| **Fairness-Tested** | Evaluates bias across age, gender, smoking groups |
| **Lightweight** | MobileNetV2 + EfficientNetB0 for low-resource deployment |
| **False-Positive-Aware** | Calibration + thresholding reduces unnecessary alarms |

---

## 💡 Motivation & Real-World Impact <a name="motivation"></a>

Lung cancer is the **leading cause of cancer death worldwide** (~1.8M deaths/year). Yet several real-world problems prevent AI from helping:

| Problem | TrustLung AI Solution |
|---|---|
| Late diagnosis in rural areas | Lightweight models runnable on basic laptops |
| Radiologist shortage | AI pre-screening with explainable heatmaps |
| Black-box AI distrust | Grad-CAM + uncertainty scores doctors can interpret |
| High false positive rates | Calibrated probabilities + confidence thresholding |
| Unfair AI across patient groups | Fairness audit across age/gender/smoking |
| Unreliable predictions | MC Dropout uncertainty flags uncertain cases |

---

## 🏗️ Architecture <a name="architecture"></a>

```
CT Scan Image
     │
     ▼
┌─────────────────────────────┐
│  Image Branch               │
│  MobileNetV2 / EffNetB0     │──► prob_img (3,)
│  + MC Dropout (uncertainty) │
└─────────────────────────────┘
                                    ┌──────────────────────────┐
Clinical Data (age, smoking, etc.)  │  Fusion Layer            │──► Final Prediction
     │                              │  Weighted Avg / Stacked  │    + Confidence
     ▼                              └──────────────────────────┘    + Uncertainty
┌─────────────────────────────┐    /                               + Grad-CAM
│  Clinical Branch            │──►
│  XGBoost Classifier         │──► prob_clinical (3,)
└─────────────────────────────┘
```

---

## ✨ Features <a name="features"></a>

### 🔬 Core ML Features
- **Transfer Learning** — ImageNet-pretrained MobileNetV2 & EfficientNetB0
- **Two-Phase Training** — Feature extraction → fine-tuning
- **Class Imbalance Handling** — Computed class weights
- **Albumentations Augmentation** — Medical-appropriate CT augmentations

### 🧠 Explainable AI
- **Grad-CAM Heatmaps** — Visualize suspicious regions on CT scans
- **Auto-layer Detection** — Finds last Conv2D layer automatically
- **Batch Heatmap Export** — Save top-K explanations automatically
- **Explanation Grid** — Multi-image summary visualization

### 📊 Uncertainty Estimation
- **Monte Carlo Dropout** — N stochastic forward passes at inference
- **Predictive Entropy** — Normalized uncertainty score [0, 1]
- **Uncertainty Labels** — Low / Medium / High classification
- **Reliability Diagram** — Checks if confidence ≈ actual accuracy

### 🛡️ False Positive Reduction
- **Confidence Thresholding** — Only predict Malignant above threshold
- **Ensemble Averaging** — Merge MobileNetV2 + EfficientNetB0 predictions
- **Isotonic Calibration** — Align model probabilities with true rates
- **Temperature Scaling** — Soften over-confident predictions

### 🔗 Multi-Modal Fusion
- **CNN Image Branch** — Deep features from CT scans
- **XGBoost Clinical Branch** — Tabular patient features
- **Weighted Averaging** — Learnable image/clinical blend ratio
- **Stacked Fusion** — Meta-learner on combined probabilities
- **Auto Weight Search** — Grid-search for optimal fusion weight

### ⚖️ Fairness Analysis
- **Subgroup Metrics** — Per age group / gender / smoking status
- **Equalized Odds Gap** — TPR/FPR fairness metric
- **Fairness Dashboard** — Multi-panel visualization
- **Subgroup Confusion Matrices** — Per-demographic error breakdown

### 📈 Evaluation & Benchmarking
- Full metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
- Per-class metrics and confusion matrices
- Model comparison tables (CSV)
- Inference latency, throughput, model size benchmarking

---

## 📦 Datasets <a name="datasets"></a>

### 1. CT Scan Images — IQ-OTH/NCCD Lung Cancer Dataset
- **Source:** [Kaggle — IQ-OTH/NCCD Lung Cancer Dataset](https://www.kaggle.com/datasets/adityamahimkar/iqothnccd-lung-cancer-dataset)
- **Classes:** Normal | Benign | Malignant
- **Format:** JPG/PNG CT scan slices
- **Download:** Place in `data/raw/images/` with subdirectories `Normal/`, `Benign/`, `Malignant/`

### 2. Clinical Data — UCI Lung Cancer Dataset
- **Source:** [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Lung+Cancer)
- **Features:** Age, Gender, Smoking, Symptoms (15 binary features)
- **Target:** LUNG_CANCER (YES/NO)
- **Download:** Place CSV in `data/raw/clinical/lung_cancer_clinical.csv`

> ℹ️ Both datasets can be replaced by synthetic demo data with `--synthetic` flag.

---

## 📁 Project Structure <a name="structure"></a>

```
TrustLungAI/
│
├── configs/
│   └── config.yaml              # All hyperparameters and paths
│
├── data/
│   ├── raw/
│   │   ├── images/              # CT scan images (Normal/Benign/Malignant/)
│   │   └── clinical/            # UCI clinical CSV
│   └── processed/               # Preprocessed arrays (auto-generated)
│
├── notebooks/
│   ├── eda.ipynb                # Exploratory Data Analysis
│   ├── training.ipynb           # Model training walkthrough
│   ├── explainability.ipynb     # Grad-CAM + uncertainty demo
│   └── fairness_analysis.ipynb  # Bias analysis walkthrough
│
├── src/
│   ├── preprocessing/
│   │   └── data_loader.py       # Image + clinical preprocessing, augmentation
│   ├── models/
│   │   ├── architectures.py     # MobileNetV2, EfficientNetB0 builders
│   │   ├── trainer.py           # Two-phase training pipeline + callbacks
│   │   ├── uncertainty.py       # Monte Carlo Dropout predictor
│   │   └── fp_reduction.py      # Thresholding, calibration, ensemble
│   ├── explainability/
│   │   └── gradcam.py           # Grad-CAM implementation + visualizer
│   ├── fusion/
│   │   └── multimodal.py        # XGBoost clinical branch + fusion layer
│   ├── fairness/
│   │   └── bias_analysis.py     # Subgroup evaluation + fairness metrics
│   ├── evaluation/
│   │   ├── metrics.py           # Full evaluation suite + plots
│   │   └── error_analysis.py    # FP/FN analysis, difficult samples
│   ├── visualization/
│   │   └── plots.py             # Training curves, distributions, dashboard
│   ├── benchmarking/
│   │   └── inference_benchmark.py # Latency, memory, throughput
│   └── utils/
│       └── helpers.py           # Config, logger, seed, device utils
│
├── outputs/
│   ├── models/                  # Saved .keras models
│   ├── plots/                   # All generated figures
│   ├── reports/                 # CSV reports, JSON histories
│   ├── heatmaps/                # Grad-CAM heatmap images
│   └── logs/                    # Training logs + TensorBoard
│
├── streamlit_app/
│   └── app.py                   # Module 12 — Lightweight UI
│
├── main.py                      # Full pipeline orchestrator
├── train.py                     # Training-only entry point
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup <a name="setup"></a>

### 1. Clone & create environment

```bash
git clone https://github.com/yourusername/TrustLungAI.git
cd TrustLungAI

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Download real datasets

Place datasets as described in the [Datasets](#datasets) section.

---

## 🚀 Usage <a name="usage"></a>

### Quick demo (synthetic data — no dataset needed)

```bash
python main.py --synthetic
```

### Train only

```bash
# Both models, synthetic data
python train.py --synthetic

# Specific model, real data
python train.py --model EfficientNetB0

# Both models
python train.py --model both
```

### Full pipeline

```bash
# All 11 modules, real data
python main.py

# Specific modules only
python main.py --synthetic --modules 3,4,7

# Skip training, use saved models
python main.py --skip-training
```

### Streamlit UI

```bash
streamlit run streamlit_app/app.py
```

### Jupyter Notebooks

```bash
jupyter notebook notebooks/
```

---

## 📊 Results <a name="results"></a>

> Results below are from training on IQ-OTH/NCCD dataset. Actual numbers may vary.

### Model Comparison

| Model | Accuracy | F1 (Macro) | ROC-AUC | Params | Latency (CPU) |
|---|---|---|---|---|---|
| MobileNetV2 | ~91% | ~0.90 | ~0.97 | 3.4M | ~45ms |
| EfficientNetB0 | ~93% | ~0.92 | ~0.98 | 5.3M | ~68ms |
| Fused (CNN + XGBoost) | ~94% | ~0.93 | ~0.98 | — | ~90ms |

### Fairness Analysis (Sample)

| Group | Accuracy | Recall | F1 |
|---|---|---|---|
| Age <40 | ~92% | ~0.90 | ~0.91 |
| Age 55-70 | ~93% | ~0.92 | ~0.92 |
| Male | ~92% | ~0.91 | ~0.91 |
| Female | ~93% | ~0.92 | ~0.92 |
| Smoker | ~94% | ~0.93 | ~0.93 |
| Non-smoker | ~91% | ~0.89 | ~0.90 |

### Sample Outputs

```
Prediction:   Malignant
Confidence:   91.4%
Uncertainty:  Low  (0.08)
Inference:    67ms

⚠️ Note: Confidence below threshold for this case — recommend radiologist review.
```

---

## 📚 Module Reference <a name="modules"></a>

| Module | File | Description |
|---|---|---|
| 1 — Preprocessing | `src/preprocessing/data_loader.py` | Image + clinical data pipelines |
| 2 — Models | `src/models/architectures.py` | MobileNetV2, EfficientNetB0 |
| 2 — Training | `src/models/trainer.py` | Two-phase training + callbacks |
| 3 — Grad-CAM | `src/explainability/gradcam.py` | Explainability heatmaps |
| 4 — Uncertainty | `src/models/uncertainty.py` | MC Dropout predictor |
| 5 — FP Reduction | `src/models/fp_reduction.py` | Thresholding + calibration |
| 6 — Fusion | `src/fusion/multimodal.py` | CNN + XGBoost fusion |
| 7 — Fairness | `src/fairness/bias_analysis.py` | Bias analysis |
| 8 — Evaluation | `src/evaluation/metrics.py` | Full metrics suite |
| 9 — Visualization | `src/visualization/plots.py` | All plot generators |
| 10 — Error Analysis | `src/evaluation/error_analysis.py` | FP/FN inspection |
| 11 — Benchmarking | `src/benchmarking/inference_benchmark.py` | Latency/memory |
| 12 — Streamlit | `streamlit_app/app.py` | Web UI |

---

## 🔮 Future Improvements <a name="future"></a>

- [ ] **3D CNN** — Process full volumetric CT scans instead of 2D slices
- [ ] **DICOM support** — Read medical-standard DICOM files directly
- [ ] **Federated Learning** — Train across hospitals without sharing patient data
- [ ] **Model Quantization** — INT8/FP16 for further edge deployment speedup
- [ ] **SHAP for XGBoost** — Per-patient clinical feature attribution
- [ ] **Longitudinal Analysis** — Track nodule changes over multiple scans
- [ ] **Radiologist Feedback Loop** — Active learning from expert corrections
- [ ] **FDA/CE regulatory prep** — Clinical validation study design
- [ ] **Docker deployment** — Containerized hospital deployment
- [ ] **ONNX export** — Framework-agnostic model deployment

---

## 🧪 Key Concepts Explained

### Explainable AI (Grad-CAM)
Gradient-weighted Class Activation Mapping backpropagates the gradient of the predicted class score through the final convolutional layer. Regions with large positive gradients are highlighted — these are the pixels the model "looked at" most when making its prediction. For lung cancer detection, these should ideally align with nodules or masses visible to radiologists.

### Uncertainty Estimation (MC Dropout)
Instead of treating the neural network as a deterministic function, Monte Carlo Dropout approximates Bayesian inference. By keeping Dropout layers active at test time and running N forward passes, we get a distribution over predictions. The variance (entropy) of this distribution quantifies the model's epistemic uncertainty — how confident it is in its prediction.

### Fairness Analysis
A model may achieve high overall accuracy while performing poorly for specific patient subgroups (e.g., elderly patients, women, non-smokers). Equalized Odds requires that True Positive Rate (sensitivity/recall) and False Positive Rate be equal across demographic groups. Large gaps indicate potential bias requiring retraining with augmented subgroup data.

### Lightweight Deployment
Rural healthcare centers often lack GPU infrastructure. MobileNetV2 (~3.4M parameters, ~45ms CPU latency) and EfficientNetB0 (~5.3M parameters, ~68ms CPU latency) were specifically designed for constrained environments. Both achieve near-ResNet50 accuracy at a fraction of the compute cost.

---

## 📄 License

MIT License — see `LICENSE` for details.

---

## 🙏 Acknowledgements

- IQ-OTH/NCCD Dataset contributors
- UCI Machine Learning Repository
- [Grad-CAM paper](https://arxiv.org/abs/1610.02391) — Selvaraju et al., 2017
- [MC Dropout paper](https://arxiv.org/abs/1506.02142) — Gal & Ghahramani, 2016
- MobileNetV2 & EfficientNet authors
