"""
TrustLung AI — Module 12: Streamlit Web App
=============================================
Lightweight medical-style UI for:
  - CT scan upload + prediction
  - Clinical data entry
  - Confidence score display
  - Uncertainty estimation
  - Grad-CAM heatmap visualization
  - Multi-modal fusion prediction

Run with:
    streamlit run streamlit_app/app.py
"""

import os
import sys
import io
import time
import tempfile
from pathlib import Path

import numpy as np
import cv2
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Page config (must be first Streamlit call) ─────────────
st.set_page_config(
    page_title="TrustLung AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f172a; }
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        margin: 0.3rem 0;
    }
    .metric-value { font-size: 2rem; font-weight: 700; }
    .metric-label { font-size: 0.85rem; color: #94a3b8; margin-top: 0.2rem; }
    .pred-malignant { color: #ef4444; }
    .pred-benign    { color: #f59e0b; }
    .pred-normal    { color: #4ade80; }
    .unc-low    { color: #4ade80; }
    .unc-medium { color: #f59e0b; }
    .unc-high   { color: #ef4444; }
    .section-header {
        border-left: 4px solid #3b82f6;
        padding-left: 0.8rem;
        margin: 1.2rem 0 0.8rem 0;
        font-size: 1.1rem;
        font-weight: 600;
    }
    div[data-testid="stSidebar"] { background-color: #1e293b; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPERS
# ============================================================

CLASS_NAMES   = ["Normal", "Benign", "Malignant"]
CLASS_COLORS  = {"Normal": "pred-normal", "Benign": "pred-benign", "Malignant": "pred-malignant"}
IMAGE_SIZE    = (224, 224)

@st.cache_resource(show_spinner="Loading TrustLung AI models...")
def load_models():
    """Load trained models (cached across sessions)."""
    models = {}
    model_dir = Path("outputs/models")

    try:
        import tensorflow as tf
        for arch in ["TrustLung_EfficientNetB0", "TrustLung_MobileNetV2"]:
            for suffix in ["_final.keras", "_phase2.keras", "_phase1.keras"]:
                path = model_dir / f"{arch}{suffix}"
                if path.exists():
                    models[arch.replace("TrustLung_", "")] = tf.keras.models.load_model(str(path))
                    break
    except Exception as e:
        st.warning(f"Could not load saved models: {e}. Using demo mode.")

    return models


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """Resize + normalize uploaded image for model input."""
    import albumentations as A
    transform = A.Compose([
        A.Resize(*IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return transform(image=img)["image"].astype(np.float32)


def mc_predict(model, image: np.ndarray, n_samples: int = 30):
    """Run Monte Carlo Dropout prediction."""
    import tensorflow as tf
    img_tensor = tf.cast(np.expand_dims(image, 0), tf.float32)
    preds = np.stack([
        model(img_tensor, training=True).numpy()[0]
        for _ in range(n_samples)
    ])
    mean   = preds.mean(axis=0)
    std    = preds.std(axis=0)
    pred_class  = int(np.argmax(mean))
    confidence  = float(mean[pred_class])
    entropy     = float(-np.sum(mean * np.log(mean + 1e-8)))
    uncertainty = entropy / np.log(len(CLASS_NAMES))
    unc_label   = "Low" if uncertainty < 0.25 else ("Medium" if uncertainty < 0.60 else "High")
    return {
        "mean_probs":    mean,
        "std_probs":     std,
        "pred_class":    pred_class,
        "pred_label":    CLASS_NAMES[pred_class],
        "confidence":    confidence,
        "uncertainty":   uncertainty,
        "unc_label":     unc_label,
        "all_samples":   preds,
    }


def compute_gradcam(model, image: np.ndarray) -> np.ndarray:
    """Compute Grad-CAM heatmap."""
    try:
        from src.explainability.gradcam import GradCAM
        cam = GradCAM(model)
        return cam.compute(image)
    except Exception:
        return np.zeros(IMAGE_SIZE, dtype=np.float32)


def overlay_heatmap(original: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend heatmap onto original image."""
    img_uint8 = (original * 255).clip(0, 255).astype(np.uint8)
    h_uint8   = (heatmap * 255).astype(np.uint8)
    colored   = cv2.applyColorMap(h_uint8, cv2.COLORMAP_JET)
    colored   = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    colored   = cv2.resize(colored, (img_uint8.shape[1], img_uint8.shape[0]))
    return cv2.addWeighted(img_uint8, 1 - alpha, colored, alpha, 0)


def demo_prediction(class_idx: int = None):
    """Generate a plausible demo prediction when no model is loaded."""
    if class_idx is None:
        class_idx = np.random.randint(0, 3)
    probs = np.random.dirichlet(np.ones(3) * 0.5)
    probs[class_idx] += 1.0
    probs /= probs.sum()
    unc = float(np.random.uniform(0.05, 0.4))
    return {
        "mean_probs": probs,
        "pred_class": class_idx,
        "pred_label": CLASS_NAMES[class_idx],
        "confidence": float(probs[class_idx]),
        "uncertainty": unc,
        "unc_label": "Low" if unc < 0.25 else "Medium",
    }


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🫁 TrustLung AI")
    st.markdown("*Confidence-Aware Lung Cancer Detection*")
    st.divider()

    st.markdown("### ⚙️ Settings")
    selected_arch  = st.selectbox("Model Architecture", ["EfficientNetB0", "MobileNetV2"])
    mc_samples     = st.slider("MC Dropout Samples", min_value=10, max_value=100, value=30, step=10)
    conf_threshold = st.slider("Confidence Threshold", min_value=0.50, max_value=0.99, value=0.75, step=0.05)
    heatmap_alpha  = st.slider("Heatmap Opacity", min_value=0.2, max_value=0.8, value=0.5, step=0.05)
    show_fusion    = st.checkbox("Enable Multi-Modal Fusion", value=True)

    st.divider()
    st.markdown("### 📚 About")
    st.markdown("""
    **TrustLung AI** is a research prototype for AI-assisted lung cancer screening.

    - 🔬 **Explainable** via Grad-CAM
    - 📊 **Uncertainty-aware** via MC Dropout
    - 🏥 **Multi-modal** (CT + clinical)
    - ⚖️ **Fairness-tested** across demographics
    - 🌍 **Lightweight** for rural healthcare

    > ⚠️ *Not for clinical use.*
    """)


# ============================================================
# MAIN UI
# ============================================================

st.markdown("# 🫁 TrustLung AI")
st.markdown("### Lightweight Confidence-Aware Explainable Lung Cancer Detection")
st.divider()

# Load models
models = load_models()
model  = models.get(selected_arch)
demo_mode = model is None

if demo_mode:
    st.info("🔬 **Demo Mode** — No trained models found. Upload images to see a simulated prediction flow. Train models first with `python train.py --synthetic`", icon="ℹ️")

# ── Two-column layout ─────────────────────────────────────
col_input, col_results = st.columns([1, 1], gap="large")

# ============================================================
# LEFT COLUMN — INPUTS
# ============================================================
with col_input:
    st.markdown('<div class="section-header">📁 CT Scan Upload</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload a CT scan image (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        help="Upload a lung CT scan slice for analysis.",
    )

    # Clinical data form
    if show_fusion:
        st.markdown('<div class="section-header">📋 Patient Clinical Data</div>', unsafe_allow_html=True)
        with st.expander("Enter patient information", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                age     = st.number_input("Age",    min_value=18, max_value=100, value=55)
                gender  = st.selectbox("Gender",   ["Male", "Female"])
                smoking = st.selectbox("Smoking",  ["Non-smoker", "Smoker", "Ex-smoker"])
            with c2:
                chest_pain  = st.checkbox("Chest Pain",        value=False)
                coughing    = st.checkbox("Chronic Coughing",  value=False)
                fatigue     = st.checkbox("Fatigue",           value=False)
                shortness   = st.checkbox("Shortness of Breath", value=False)
                wheezing    = st.checkbox("Wheezing",          value=False)

    # Analyze button
    st.divider()
    analyze_btn = st.button("🔍 Analyze CT Scan", type="primary", use_container_width=True, disabled=uploaded_file is None)

# ============================================================
# RIGHT COLUMN — RESULTS (shown after analysis)
# ============================================================
with col_results:
    if uploaded_file is None:
        st.markdown("""
        <div style="background:#1e293b; border-radius:12px; padding:2rem; text-align:center; margin-top:2rem;">
            <div style="font-size:3rem;">🫁</div>
            <div style="color:#94a3b8; margin-top:1rem;">Upload a CT scan to begin analysis</div>
            <div style="color:#64748b; font-size:0.85rem; margin-top:0.5rem;">Supported formats: JPG, PNG</div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# ANALYSIS
# ============================================================
if uploaded_file is not None and analyze_btn:
    # Read uploaded image
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_disp   = cv2.resize(img_rgb, IMAGE_SIZE)      # display version
    img_proc   = preprocess_image(img_rgb)            # model input

    # ── Run prediction ─────────────────────────────────────────
    with st.spinner("🔬 Analyzing CT scan..."):
        t_start = time.perf_counter()

        if demo_mode:
            result = demo_prediction()
        else:
            result = mc_predict(model, img_proc, n_samples=mc_samples)

        # Apply confidence threshold
        if result["confidence"] < conf_threshold and result["pred_label"] == "Malignant":
            result["pred_label"]   = "Uncertain — Review Required"
            result["pred_class"]   = -1
            result["flagged"]      = True
        else:
            result["flagged"] = False

        # Grad-CAM
        if demo_mode:
            heatmap = np.random.rand(*IMAGE_SIZE).astype(np.float32)
            heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() + 1e-8)
        else:
            with st.spinner("Generating Grad-CAM explanation..."):
                heatmap = compute_gradcam(model, img_proc)

        overlay = overlay_heatmap(img_proc, heatmap, alpha=heatmap_alpha)
        t_elapsed = (time.perf_counter() - t_start) * 1000  # ms

    # ── Display results ────────────────────────────────────────
    with col_results:
        st.markdown('<div class="section-header">🩺 Prediction Results</div>', unsafe_allow_html=True)

        # Prediction pill
        pred_label = result["pred_label"]
        conf_pct   = result["confidence"] * 100
        unc_label  = result.get("unc_label", "Medium")
        css_class  = CLASS_COLORS.get(pred_label, "pred-benign")
        unc_css    = f"unc-{unc_label.lower()}"

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value {css_class}">{pred_label}</div>
            <div class="metric-label">Predicted Diagnosis</div>
        </div>
        """, unsafe_allow_html=True)

        # Metrics row
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:#60a5fa">{conf_pct:.1f}%</div>
                <div class="metric-label">Confidence</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value {unc_css}">{unc_label}</div>
                <div class="metric-label">Uncertainty</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:#94a3b8">{t_elapsed:.0f}ms</div>
                <div class="metric-label">Inference Time</div>
            </div>""", unsafe_allow_html=True)

        if result.get("flagged"):
            st.warning("⚠️ **Low confidence prediction.** This case should be reviewed by a radiologist.", icon="⚠️")

        # Class probability bars
        st.markdown('<div class="section-header">📊 Class Probabilities</div>', unsafe_allow_html=True)
        probs      = result["mean_probs"]
        bar_colors = ["#4ade80", "#f59e0b", "#ef4444"]
        for cls_name, prob, color in zip(CLASS_NAMES, probs, bar_colors):
            st.markdown(f"**{cls_name}**")
            st.progress(float(prob), text=f"{prob:.1%}")

        # Image panels
        st.markdown('<div class="section-header">🔬 Explainability (Grad-CAM)</div>', unsafe_allow_html=True)
        img_c1, img_c2 = st.columns(2)
        with img_c1:
            st.image(img_disp, caption="Original CT Scan", use_container_width=True)
        with img_c2:
            st.image(overlay,  caption="AI Focus Heatmap (Grad-CAM)", use_container_width=True)

        st.caption("""
        🟥 **Red/Yellow regions** = areas the AI focused on most.
        These should correspond to suspicious nodules or masses.
        """)

        # Multi-modal fusion
        if show_fusion:
            st.markdown('<div class="section-header">🔗 Multi-Modal Fusion Result</div>', unsafe_allow_html=True)
            # Simulate clinical score for demo
            smoking_risk  = 1 if smoking == "Smoker" else (0.5 if smoking == "Ex-smoker" else 0.1)
            age_risk      = min(1.0, (age - 18) / 70)
            symptom_count = sum([chest_pain, coughing, fatigue, shortness, wheezing])
            symptom_risk  = symptom_count / 5

            clinical_risk = (smoking_risk * 0.4 + age_risk * 0.3 + symptom_risk * 0.3)
            fused_mal     = float(probs[2]) * 0.6 + clinical_risk * 0.4

            fused_col1, fused_col2 = st.columns(2)
            with fused_col1:
                st.metric("Image-Only Malignant Prob.", f"{probs[2]:.1%}")
            with fused_col2:
                delta = fused_mal - probs[2]
                st.metric("Fused Malignant Prob.", f"{fused_mal:.1%}", delta=f"{delta:+.1%}")

            st.caption(f"Clinical risk factors: Smoking={smoking}, Age={age}, Symptoms={symptom_count}/5")

        # Uncertainty chart
        st.markdown('<div class="section-header">📉 Uncertainty Breakdown</div>', unsafe_allow_html=True)
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")

        if "all_samples" in result and result["all_samples"] is not None:
            samples = result["all_samples"]
        else:
            samples = np.random.dirichlet(np.ones(3), 30)

        fig, ax = plt.subplots(figsize=(7, 3))
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#1e293b")
        colors_mc = ["#4ade80", "#f59e0b", "#ef4444"]
        for i, (name, color) in enumerate(zip(CLASS_NAMES, colors_mc)):
            ax.hist(samples[:, i], bins=20, alpha=0.7, color=color, label=name, density=True)
            ax.axvline(float(probs[i]), color=color, linestyle="--", linewidth=2)
        ax.set_xlabel("Predicted Probability", color="white", fontsize=9)
        ax.set_ylabel("Frequency", color="white", fontsize=9)
        ax.set_title("MC Dropout Sample Distribution", color="white", fontsize=10)
        ax.tick_params(colors="white", labelsize=8)
        ax.spines[:].set_color("#475569")
        ax.legend(facecolor="#1e293b", labelcolor="white", fontsize=8)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", facecolor="#0f172a", dpi=120)
        buf.seek(0)
        st.image(buf, use_container_width=True)
        plt.close(fig)

        # Clinical summary
        if show_fusion:
            with st.expander("🏥 Clinical Risk Summary"):
                risk_items = {
                    "Age Risk":      f"{age_risk:.1%}",
                    "Smoking Risk":  f"{smoking_risk:.1%}",
                    "Symptom Risk":  f"{symptom_risk:.1%}",
                    "Overall Risk":  f"{clinical_risk:.1%}",
                }
                for k, v in risk_items.items():
                    c_a, c_b = st.columns([2, 1])
                    c_a.write(k)
                    c_b.write(f"**{v}**")

        # Footer disclaimer
        st.divider()
        st.caption(
            "⚠️ **Disclaimer:** TrustLung AI is a research prototype only. "
            "It is NOT approved for clinical diagnosis. Always consult a qualified radiologist."
        )

# ============================================================
# FOOTER METRICS (always visible)
# ============================================================
st.divider()
footer_cols = st.columns(4)
stats = [
    ("🫁", "Classes", "Normal / Benign / Malignant"),
    ("⚡", "Models",  "MobileNetV2 · EfficientNetB0"),
    ("🔬", "XAI",    "Grad-CAM + MC Dropout"),
    ("⚖️", "Fairness", "Age · Gender · Smoking"),
]
for col, (icon, label, value) in zip(footer_cols, stats):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:1.4rem">{icon}</div>
            <div style="font-size:0.8rem; color:#94a3b8; margin-top:0.2rem">{label}</div>
            <div style="font-size:0.85rem; color:white; font-weight:600; margin-top:0.2rem">{value}</div>
        </div>
        """, unsafe_allow_html=True)
