import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="HemoJaundice AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, .stApp {
        background-color: #060b13;
        color: #e0e6f0;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .main-header {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 700;
        font-size: 2.8rem;
        color: #e0f2fe;
        text-align: center;
        margin-bottom: 0.2rem;
        background: linear-gradient(135deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 400;
        color: #94a3b8;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        padding: 1.8rem 1.5rem;
        box-shadow: 0 20px 40px -12px rgba(0,0,0,0.6);
        transition: transform 0.2s ease;
        height: 100%;
    }
    .glass-card:hover {
        transform: translateY(-4px);
        border-color: rgba(56, 189, 248, 0.3);
    }

    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.6rem;
        font-weight: 600;
        color: #f8fafc;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #94a3b8;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    .metric-unit {
        font-size: 1rem;
        color: #94a3b8;
        margin-left: 4px;
    }
    .ci-text {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        color: #94a3b8;
        background: rgba(0,0,0,0.3);
        padding: 0.2rem 0.8rem;
        border-radius: 40px;
        display: inline-block;
    }

    .badge-warning {
        background: #f59e0b;
        color: #0f172a;
        font-weight: 700;
        padding: 0.25rem 1rem;
        border-radius: 40px;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-danger {
        background: #ef4444;
        color: #f1f5f9;
        font-weight: 700;
        padding: 0.25rem 1rem;
        border-radius: 40px;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-success {
        background: #22c55e;
        color: #0f172a;
        font-weight: 700;
        padding: 0.25rem 1rem;
        border-radius: 40px;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .stSelectbox, .stFileUploader {
        background: rgba(255,255,255,0.02);
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .stSelectbox label, .stFileUploader label {
        color: #94a3b8 !important;
        font-weight: 600 !important;
    }

    .plot-container {
        background: rgba(0,0,0,0.2);
        border-radius: 20px;
        padding: 0.5rem;
        border: 1px solid rgba(255,255,255,0.04);
    }
</style>
""", unsafe_allow_html=True)

# ---------- HELPER FUNCTIONS ----------

def get_hsv_ranges(mode):
    """Return lower and upper HSV bounds for each screening mode."""
    if mode == "👀 Sclera / Eye White (Jaundice Screening)":
        lower = np.array([0, 0, 150], dtype=np.uint8)
        upper = np.array([180, 50, 255], dtype=np.uint8)
    elif mode == "👁️ Palpebral Conjunctiva (Inner Eye - Anemia Screening)":
        lower = np.array([0, 50, 100], dtype=np.uint8)
        upper = np.array([10, 255, 255], dtype=np.uint8)
    elif mode == "🖐️ Fingernail Bed / Inner Lip (General Tissue Pallor)":
        lower = np.array([0, 20, 100], dtype=np.uint8)
        upper = np.array([20, 150, 255], dtype=np.uint8)
    else:
        lower = np.array([0, 0, 0], dtype=np.uint8)
        upper = np.array([180, 255, 255], dtype=np.uint8)
    return lower, upper

def extract_segmented_roi_features(image, mode):
    """
    Extract 8 features from the ROI defined by adaptive HSV thresholding.
    Returns (features, mask) where features is a list of 8 floats:
    [R_mean, G_mean, B_mean, L_chroma, a_chroma, b_chroma, rg_ratio, pallor_val]
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower, upper = get_hsv_ranges(mode)
    mask = cv2.inRange(hsv, lower, upper)

    # Morphological cleaning
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # If mask is empty, fallback to whole image
    if np.sum(mask) < 10:
        mask = np.ones(image.shape[:2], dtype=np.uint8) * 255

    masked = cv2.bitwise_and(image, image, mask=mask)
    pixels = masked[mask > 0]

    if len(pixels) == 0:
        pixels = image.reshape(-1, 3)

    R_mean = np.mean(pixels[:, 0])
    G_mean = np.mean(pixels[:, 1])
    B_mean = np.mean(pixels[:, 2])

    # LAB conversion with proper scaling
    lab_image = cv2.cvtColor(masked, cv2.COLOR_RGB2LAB)
    lab_pixels = lab_image[mask > 0]
    if len(lab_pixels) == 0:
        lab_pixels = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).reshape(-1, 3)

    L_vals = lab_pixels[:, 0] * (100.0 / 255.0)
    a_vals = lab_pixels[:, 1] - 128.0
    b_vals = lab_pixels[:, 2] - 128.0

    L_chroma = np.mean(L_vals)
    a_chroma = np.mean(a_vals)
    b_chroma = np.mean(b_vals)

    rg_ratio = R_mean / (G_mean + 1e-6)
    pallor_val = 1.0 - (R_mean / 255.0)

    features = [R_mean, G_mean, B_mean, L_chroma, a_chroma, b_chroma, rg_ratio, pallor_val]
    return features, mask

def generate_synthetic_data(n_samples=300):
    """Generate synthetic calibration data mapping features to bilirubin and hemoglobin."""
    np.random.seed(42)
    data = []
    for _ in range(n_samples):
        R = np.random.uniform(40, 220)
        G = np.random.uniform(30, 200)
        B = np.random.uniform(20, 180)
        L = (0.299*R + 0.587*G + 0.114*B) / 255.0 * 100
        a = np.random.uniform(-50, 50)
        b = np.random.uniform(-50, 50)
        rg_ratio = R / (G + 1e-6)
        pallor_val = 1.0 - (R / 255.0)

        bilirubin = 0.5 + 0.02 * (b + 50) + 0.5 * pallor_val + np.random.normal(0, 0.3)
        bilirubin = np.clip(bilirubin, 0.2, 20.0)

        hemoglobin = 14.0 + 0.02 * (a + 20) - 0.5 * pallor_val + np.random.normal(0, 0.5)
        hemoglobin = np.clip(hemoglobin, 5.0, 18.0)

        data.append([R, G, B, L, a, b, rg_ratio, pallor_val, bilirubin, hemoglobin])

    columns = ['R','G','B','L','a','b','rg_ratio','pallor_val','bilirubin','hemoglobin']
    return pd.DataFrame(data, columns=columns)

def train_and_save_models():
    """Train RandomForest models on synthetic data and save to disk."""
    df = generate_synthetic_data(300)
    X = df[['R','G','B','L','a','b','rg_ratio','pallor_val']].values
    y_bili = df['bilirubin'].values
    y_hb = df['hemoglobin'].values

    model_bili = RandomForestRegressor(n_estimators=50, random_state=42)
    model_hb = RandomForestRegressor(n_estimators=50, random_state=42)
    model_bili.fit(X, y_bili)
    model_hb.fit(X, y_hb)

    joblib.dump(model_bili, 'bili_model.pkl')
    joblib.dump(model_hb, 'hb_model.pkl')
    return model_bili, model_hb

def load_models():
    """Load models from disk or train if missing."""
    bili_path = 'bili_model.pkl'
    hb_path = 'hb_model.pkl'
    if os.path.exists(bili_path) and os.path.exists(hb_path):
        model_bili = joblib.load(bili_path)
        model_hb = joblib.load(hb_path)
    else:
        model_bili, model_hb = train_and_save_models()
    return model_bili, model_hb

def predict_with_uncertainty(model, features):
    """
    Predict target and 95% CI using ensemble tree predictions.
    Returns (mean, lower, upper)
    """
    features = np.array(features).reshape(1, -1)
    tree_preds = np.array([tree.predict(features) for tree in model.estimators_]).flatten()
    mean = np.mean(tree_preds)
    std = np.std(tree_preds)
    ci = 1.96 * std
    lower = max(mean - ci, 0)
    upper = mean + ci
    return mean, lower, upper

# ---------- MAIN APP ----------
st.markdown("<div class='main-header'>HemoJaundice AI</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Computer Vision · Machine Learning · Precision Screening</div>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🧪 Screening Mode")
    mode = st.selectbox(
        "Select anatomical site",
        ["👀 Sclera / Eye White (Jaundice Screening)",
         "👁️ Palpebral Conjunctiva (Inner Eye - Anemia Screening)",
         "🖐️ Fingernail Bed / Inner Lip (General Tissue Pallor)"],
        index=0
    )

    st.markdown("---")
    uploaded_file = st.file_uploader("Upload a photo (JPEG/PNG)", type=['jpg', 'jpeg', 'png'])

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.8rem; color:#64748b;'>
    <b>Model status:</b> Synthetic calibration active.<br>
    <b>Confidence:</b> 95% ensemble interval.
    </div>
    """, unsafe_allow_html=True)

# Main content
if uploaded_file is not None:
    try:
        # Load image and convert to RGB NumPy array
        pil_img = Image.open(uploaded_file)
        img_rgb = np.array(pil_img.convert('RGB'))
    except Exception as e:
        st.error(f"Error loading image: {e}")
        st.stop()

    # Extract features and mask
    with st.spinner("Analyzing tissue..."):
        features, mask = extract_segmented_roi_features(img_rgb, mode)
        # Prepare mask overlay (3-channel for visualization)
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        overlay = cv2.addWeighted(img_rgb, 0.7, mask_3ch, 0.3, 0)

    # Load models
    model_bili, model_hb = load_models()

    # Predictions
    bili_mean, bili_low, bili_high = predict_with_uncertainty(model_bili, features)
    hb_mean, hb_low, hb_high = predict_with_uncertainty(model_hb, features)

    # Determine clinical alerts
    jaundice_alert = "⚠️ Elevated" if bili_mean > 1.2 else "✅ Normal"
    anemia_alert = "⚠️ Low" if hb_mean < 12.0 else "✅ Normal"

    # ---------- LAYOUT ----------
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("### 📷 Original & Segmentation")
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.image(img_rgb, caption="Original", use_container_width=True)
        with col_img2:
            st.image(overlay, caption="ROI Mask (overlay)", use_container_width=True)

        # Show feature vector as a small table
        feat_names = ['R', 'G', 'B', 'L*', 'a*', 'b*', 'R/G', 'Pallor']
        feat_vals = [f"{v:.1f}" for v in features]
        st.markdown("**Extracted features**")
        cols = st.columns(4)
        for i, (name, val) in enumerate(zip(feat_names, feat_vals)):
            cols[i % 4].metric(label=name, value=val, delta=None)

    with col_right:
        st.markdown("### 🧬 Clinical Estimates")

        # Bilirubin card
        with st.container():
            st.markdown(f"""
            <div class='glass-card'>
                <div class='metric-label'>💛 Bilirubin (Total)</div>
                <div>
                    <span class='metric-value'>{bili_mean:.2f}</span>
                    <span class='metric-unit'>mg/dL</span>
                </div>
                <div style='margin-top: 0.5rem;'>
                    <span class='ci-text'>95% CI: {bili_low:.2f} – {bili_high:.2f}</span>
                </div>
                <div style='margin-top: 0.8rem;'>
                    <span class='badge-{"danger" if bili_mean > 1.2 else "success"}'>
                        {jaundice_alert}
                    </span>
                    <span style='margin-left: 0.8rem; font-size:0.8rem; color:#94a3b8;'>
                        {"> 1.2 mg/dL indicates jaundice" if bili_mean > 1.2 else "Within normal range"}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Hemoglobin card
        with st.container():
            st.markdown(f"""
            <div class='glass-card'>
                <div class='metric-label'>🩸 Hemoglobin (Estimated)</div>
                <div>
                    <span class='metric-value'>{hb_mean:.1f}</span>
                    <span class='metric-unit'>g/dL</span>
                </div>
                <div style='margin-top: 0.5rem;'>
                    <span class='ci-text'>95% CI: {hb_low:.1f} – {hb_high:.1f}</span>
                </div>
                <div style='margin-top: 0.8rem;'>
                    <span class='badge-{"danger" if hb_mean < 12.0 else "success"}'>
                        {anemia_alert}
                    </span>
                    <span style='margin-left: 0.8rem; font-size:0.8rem; color:#94a3b8;'>
                        {"< 12.0 g/dL suggests anemia" if hb_mean < 12.0 else "Within normal range"}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Uncertainty distribution chart (Plotly)
        st.markdown("**📊 Ensemble Prediction Distribution**")
        features_arr = np.array(features).reshape(1, -1)
        tree_preds_bili = np.array([tree.predict(features_arr) for tree in model_bili.estimators_]).flatten()
        tree_preds_hb = np.array([tree.predict(features_arr) for tree in model_hb.estimators_]).flatten()

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=tree_preds_bili,
            name='Bilirubin',
            opacity=0.7,
            marker_color='#fbbf24',
            nbinsx=20,
            histnorm='probability density'
        ))
        fig.add_trace(go.Histogram(
            x=tree_preds_hb,
            name='Hemoglobin',
            opacity=0.7,
            marker_color='#f87171',
            nbinsx=20,
            histnorm='probability density',
            yaxis='y2'
        ))

        fig.add_vline(x=bili_mean, line_width=2, line_dash="dash", line_color="#fbbf24",
                      annotation_text=f"μ={bili_mean:.2f}")
        fig.add_vline(x=hb_mean, line_width=2, line_dash="dash", line_color="#f87171",
                      annotation_text=f"μ={hb_mean:.1f}", annotation_position="top")

        fig.update_layout(
            template="plotly_dark",
            height=300,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(title="Value", gridcolor='#1e293b'),
            yaxis=dict(title="Density", gridcolor='#1e293b'),
            yaxis2=dict(title="Density", overlaying='y', side='right', showgrid=False),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

else:
    st.info("👆 Upload a photo using the sidebar to begin screening.", icon="ℹ️")
    st.markdown("""
    <div style='display: flex; justify-content: center; margin-top: 2rem;'>
        <div style='background: rgba(255,255,255,0.02); border-radius: 32px; padding: 2rem 3rem; border: 1px dashed #334155; text-align: center;'>
            <span style='font-size: 3rem;'>🧬</span>
            <p style='color: #64748b; margin-top: 0.5rem;'>Select a screening site and upload an image<br>to receive real‑time clinical predictions.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #475569; font-size: 0.8rem;'>"
    "HemoJaundice AI · Powered by Random Forest Ensemble · Synthetic calibration model."
    "</div>",
    unsafe_allow_html=True
)
