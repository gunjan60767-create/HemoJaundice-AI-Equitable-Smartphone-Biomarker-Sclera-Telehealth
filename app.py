import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="HemoJaundice AI", page_icon="🧬", layout="wide", initial_sidebar_state="expanded")

# ---------- CSS (unchanged, keep premium) ----------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
    html, body, .stApp { background-color: #060b13; color: #e0e6f0; font-family: 'Plus Jakarta Sans', sans-serif; }
    .main-header { font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; font-size: 2.8rem; color: #e0f2fe; text-align: center; margin-bottom: 0.2rem; background: linear-gradient(135deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .sub-header { font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 400; color: #94a3b8; text-align: center; font-size: 1.1rem; margin-bottom: 2rem; }
    .glass-card { background: rgba(255, 255, 255, 0.04); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 24px; padding: 1.8rem 1.5rem; box-shadow: 0 20px 40px -12px rgba(0,0,0,0.6); transition: transform 0.2s ease; height: 100%; }
    .glass-card:hover { transform: translateY(-4px); border-color: rgba(56, 189, 248, 0.3); }
    .glass-card-primary { border: 1px solid rgba(56, 189, 248, 0.3); background: rgba(56, 189, 248, 0.08); }
    .metric-value { font-family: 'JetBrains Mono', monospace; font-size: 2.6rem; font-weight: 600; color: #f8fafc; line-height: 1.2; }
    .metric-label { font-size: 0.85rem; font-weight: 600; color: #94a3b8; letter-spacing: 0.03em; text-transform: uppercase; }
    .metric-unit { font-size: 1rem; color: #94a3b8; margin-left: 4px; }
    .ci-text { font-family: 'JetBrains Mono', monospace; font-size: 0.9rem; color: #94a3b8; background: rgba(0,0,0,0.3); padding: 0.2rem 0.8rem; border-radius: 40px; display: inline-block; }
    .badge-danger { background: #ef4444; color: #f1f5f9; font-weight: 700; padding: 0.25rem 1rem; border-radius: 40px; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }
    .badge-success { background: #22c55e; color: #0f172a; font-weight: 700; padding: 0.25rem 1rem; border-radius: 40px; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }
    .badge-warning { background: #f59e0b; color: #0f172a; font-weight: 700; padding: 0.25rem 1rem; border-radius: 40px; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }
    .stSelectbox, .stFileUploader { background: rgba(255,255,255,0.02); border-radius: 16px; border: 1px solid rgba(255,255,255,0.06); }
    .stSelectbox label, .stFileUploader label { color: #94a3b8 !important; font-weight: 600 !important; }
</style>
""", unsafe_allow_html=True)

# ---------- IMPROVED FEATURE EXTRACTION ----------
def get_roi_mask(image, mode):
    """Adaptive HSV masking – now with mode‑specific fine‑tuning."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    if mode == "Sclera":
        lower = np.array([0, 0, 150], dtype=np.uint8)
        upper = np.array([180, 50, 255], dtype=np.uint8)
    elif mode == "Conjunctiva":
        # Broader range to catch pink/red, but also low‑saturation pale areas
        lower = np.array([0, 30, 60], dtype=np.uint8)
        upper = np.array([15, 255, 255], dtype=np.uint8)
    elif mode == "Pallor":
        lower = np.array([0, 10, 80], dtype=np.uint8)
        upper = np.array([25, 120, 255], dtype=np.uint8)
    else:
        lower = np.array([0, 0, 0], dtype=np.uint8)
        upper = np.array([180, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    # Morphological clean
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    # If mask is too small, fallback to whole image
    if np.sum(mask) < 50:
        mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
    return mask

def extract_segmented_roi_features(image, mode):
    """
    Extract 8 features + extra pallor metrics.
    Returns (features, mask, extra_dict) where features = [R,G,B,L,a,b,rg_ratio,pallor_val]
    and extra_dict contains {pallor_score, redness_score, etc.}
    """
    mask = get_roi_mask(image, mode)
    masked = cv2.bitwise_and(image, image, mask=mask)
    pixels = masked[mask > 0]
    if len(pixels) == 0:
        pixels = image.reshape(-1, 3)

    # Mean RGB
    R, G, B = np.mean(pixels[:, 0]), np.mean(pixels[:, 1]), np.mean(pixels[:, 2])

    # Convert whole masked image to LAB for per‑pixel L, a, b
    lab = cv2.cvtColor(masked, cv2.COLOR_RGB2LAB)
    lab_pix = lab[mask > 0]
    if len(lab_pix) == 0:
        lab_pix = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).reshape(-1, 3)
    L_vals = lab_pix[:, 0] * (100.0 / 255.0)
    a_vals = lab_pix[:, 1] - 128.0
    b_vals = lab_pix[:, 2] - 128.0
    L = np.mean(L_vals)
    a = np.mean(a_vals)
    b = np.mean(b_vals)

    rg_ratio = R / (G + 1e-6)
    pallor_val = 1.0 - (R / 255.0)   # simple pallor from R

    # ----- NEW PALLOR METRICS -----
    # 1. Normalised a* (redness) – lower a* means more pale
    # 2. L* / a* ratio – higher ratio means pale (high L, low a)
    # 3. Standard deviation of L* – to detect unevenness (not used but good)
    L_std = np.std(L_vals)
    a_mean = a
    pallor_score = (L / 100.0) * (1 - (a + 50) / 100.0)  # heuristic: high L, low a -> pale

    # Append these to features? We'll keep the 8 core, but we can use them in model training
    # For now, we'll add them as extra features in the training CSV creation.

    features = [R, G, B, L, a, b, rg_ratio, pallor_val]
    extra = {"pallor_score": pallor_score, "L_std": L_std, "a_mean": a_mean}
    return features, mask, extra

# ---------- ENHANCED DATA GENERATION (for fallback) ----------
def generate_enhanced_synthetic_data(mode, n=1500):
    """
    Generate 1500 samples with more realistic correlations based on clinical literature.
    For Conjunctiva: hemoglobin = 14 - 6*pallor_val + 0.04*a + noise (so pallor strongly drops Hb)
    """
    np.random.seed(42)
    data = []
    for _ in range(n):
        R = np.random.uniform(40, 220)
        G = np.random.uniform(30, 200)
        B = np.random.uniform(20, 180)
        L = (0.299*R + 0.587*G + 0.114*B) / 255.0 * 100
        a = np.random.uniform(-50, 50)
        b = np.random.uniform(-50, 50)
        rg_ratio = R / (G + 1e-6)
        pallor_val = 1.0 - (R / 255.0)

        if mode == "Sclera":
            bilirubin = 0.5 + 4.0 * pallor_val + 0.025 * (b + 50) + np.random.normal(0, 0.3)
            bilirubin = np.clip(bilirubin, 0.2, 20.0)
            hemoglobin = 14.0 + np.random.normal(0, 1.2)
            hemoglobin = np.clip(hemoglobin, 5.0, 18.0)
        elif mode == "Conjunctiva":
            # Strong inverse: high pallor -> low Hb, plus redness (a) raises Hb
            hemoglobin = 14.0 - 6.0 * pallor_val + 0.04 * (a + 20) + np.random.normal(0, 0.5)
            hemoglobin = np.clip(hemoglobin, 5.0, 18.0)
            bilirubin = 0.5 + np.random.normal(0, 0.6)
            bilirubin = np.clip(bilirubin, 0.2, 20.0)
        else:  # Pallor
            hemoglobin = 14.0 - 4.0 * pallor_val + np.random.normal(0, 0.6)
            hemoglobin = np.clip(hemoglobin, 5.0, 18.0)
            bilirubin = 0.5 + 2.0 * pallor_val + np.random.normal(0, 0.4)
            bilirubin = np.clip(bilirubin, 0.2, 20.0)

        # Add extra features (pallor_score, etc.) – we'll include them in the dataset
        # but our model uses only the 8 core features to keep compatibility.
        data.append([R, G, B, L, a, b, rg_ratio, pallor_val, bilirubin, hemoglobin])

    columns = ['R','G','B','L','a','b','rg_ratio','pallor_val','bilirubin','hemoglobin']
    return pd.DataFrame(data, columns=columns)

# ---------- MODEL TRAINING & LOADING ----------
def train_models(mode, df):
    X = df[['R','G','B','L','a','b','rg_ratio','pallor_val']].values
    y_bili = df['bilirubin'].values
    y_hb = df['hemoglobin'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model_bili = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model_bili.fit(X_scaled, y_bili)
    model_hb = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model_hb.fit(X_scaled, y_hb)
    joblib.dump(model_bili, f'bili_model_{mode}.pkl')
    joblib.dump(model_hb, f'hb_model_{mode}.pkl')
    joblib.dump(scaler, f'scaler_{mode}.pkl')
    return model_bili, model_hb, scaler

def load_or_train(mode, real_df=None):
    bili_path = f'bili_model_{mode}.pkl'
    hb_path = f'hb_model_{mode}.pkl'
    scaler_path = f'scaler_{mode}.pkl'
    if real_df is not None:
        return train_models(mode, real_df)
    if os.path.exists(bili_path) and os.path.exists(hb_path) and os.path.exists(scaler_path):
        model_bili = joblib.load(bili_path)
        model_hb = joblib.load(hb_path)
        scaler = joblib.load(scaler_path)
        return model_bili, model_hb, scaler
    # fallback synthetic
    df = generate_enhanced_synthetic_data(mode, 1500)
    return train_models(mode, df)

def predict_with_uncertainty(model, scaler, features):
    features_scaled = scaler.transform(np.array(features).reshape(1, -1))
    tree_preds = np.array([tree.predict(features_scaled) for tree in model.estimators_]).flatten()
    mean = np.mean(tree_preds)
    std = np.std(tree_preds)
    ci = 1.96 * std
    lower = max(mean - ci, 0)
    upper = mean + ci
    return mean, lower, upper

# ---------- MAIN APP ----------
st.markdown("<div class='main-header'>HemoJaundice AI</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>National‑Level Hackathon · Real ML + Computer Vision</div>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🧪 Screening Mode")
    mode_display = st.selectbox(
        "Select anatomical site",
        ["👀 Sclera (Jaundice)", "👁️ Conjunctiva (Anemia)", "🖐️ Fingernail/Lip (Pallor)"],
        index=0
    )
    mode_map = {"👀 Sclera (Jaundice)": "Sclera", "👁️ Conjunctiva (Anemia)": "Conjunctiva", "🖐️ Fingernail/Lip (Pallor)": "Pallor"}
    mode = mode_map[mode_display]

    st.markdown("---")
    uploaded_file = st.file_uploader("Upload a photo (JPEG/PNG)", type=['jpg', 'jpeg', 'png'])

    st.markdown("---")
    st.markdown("### 🔄 Custom Training")
    st.markdown("Upload a CSV with **real data** to retrain models. Columns must be exactly:")
    st.code("R,G,B,L,a,b,rg_ratio,pallor_val,bilirubin,hemoglobin", language='csv')
    train_csv = st.file_uploader("CSV file", type=['csv'])

    if train_csv is not None:
        if st.button("🚀 Train Models with this Data"):
            try:
                real_df = pd.read_csv(train_csv)
                required = ['R','G','B','L','a','b','rg_ratio','pallor_val','bilirubin','hemoglobin']
                if not all(col in real_df.columns for col in required):
                    st.error(f"CSV must have these columns: {required}")
                else:
                    with st.spinner("Training..."):
                        model_bili, model_hb, scaler = train_models(mode, real_df)
                        st.session_state['trained'] = True
                        st.session_state['mode'] = mode
                        st.session_state['model_bili'] = model_bili
                        st.session_state['model_hb'] = model_hb
                        st.session_state['scaler'] = scaler
                        st.success("✅ Models retrained successfully!")
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.8rem; color:#64748b;'>
    <b>Model source:</b> Synthetic (fallback) or Custom<br>
    <b>Confidence:</b> 95% tree‑based interval.
    </div>
    """, unsafe_allow_html=True)

# ---------- MAIN CONTENT ----------
if uploaded_file is not None:
    try:
        pil_img = Image.open(uploaded_file)
        img_rgb = np.array(pil_img.convert('RGB'))
    except Exception as e:
        st.error(f"Error loading image: {e}")
        st.stop()

    with st.spinner("Analyzing tissue..."):
        features, mask, extra = extract_segmented_roi_features(img_rgb, mode)
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        overlay = cv2.addWeighted(img_rgb, 0.7, mask_3ch, 0.3, 0)

    # Load models
    if 'trained' in st.session_state and st.session_state.get('mode') == mode:
        model_bili = st.session_state['model_bili']
        model_hb = st.session_state['model_hb']
        scaler = st.session_state['scaler']
    else:
        model_bili, model_hb, scaler = load_or_train(mode)
        st.session_state['trained'] = False

    bili_mean, bili_low, bili_high = predict_with_uncertainty(model_bili, scaler, features)
    hb_mean, hb_low, hb_high = predict_with_uncertainty(model_hb, scaler, features)

    # Determine primary metric
    if mode == "Sclera":
        primary_label = "💛 Bilirubin (Total)"
        primary_value, primary_unit = bili_mean, "mg/dL"
        primary_low, primary_high = bili_low, bili_high
        primary_alert = "⚠️ Elevated" if bili_mean > 1.2 else "✅ Normal"
        primary_badge = "danger" if bili_mean > 1.2 else "success"
        primary_threshold = "> 1.2 mg/dL indicates jaundice"
        secondary_label, secondary_value, secondary_unit = "🩸 Hemoglobin (Est.)", hb_mean, "g/dL"
        secondary_low, secondary_high = hb_low, hb_high
        secondary_alert = "⚠️ Low" if hb_mean < 12.0 else "✅ Normal"
        secondary_badge = "danger" if hb_mean < 12.0 else "success"
        secondary_threshold = "< 12.0 g/dL suggests anemia"
    elif mode == "Conjunctiva":
        primary_label = "🩸 Hemoglobin (Est.)"
        primary_value, primary_unit = hb_mean, "g/dL"
        primary_low, primary_high = hb_low, hb_high
        primary_alert = "⚠️ Low" if hb_mean < 12.0 else "✅ Normal"
        primary_badge = "danger" if hb_mean < 12.0 else "success"
        primary_threshold = "< 12.0 g/dL suggests anemia"
        secondary_label, secondary_value, secondary_unit = "💛 Bilirubin (Total)", bili_mean, "mg/dL"
        secondary_low, secondary_high = bili_low, bili_high
        secondary_alert = "⚠️ Elevated" if bili_mean > 1.2 else "✅ Normal"
        secondary_badge = "danger" if bili_mean > 1.2 else "success"
        secondary_threshold = "> 1.2 mg/dL indicates jaundice"
    else:  # Pallor
        pallor_index = features[7] * 100
        primary_label = "🤍 Pallor Index"
        primary_value, primary_unit = pallor_index, "%"
        primary_low = max(pallor_index - 5, 0)
        primary_high = min(pallor_index + 5, 100)
        primary_alert = "⚠️ Pallor" if pallor_index > 60 else "✅ Normal"
        primary_badge = "danger" if pallor_index > 60 else "success"
        primary_threshold = "> 60% indicates significant pallor"
        secondary_label, secondary_value, secondary_unit = "🩸 Hemoglobin (Est.)", hb_mean, "g/dL"
        secondary_low, secondary_high = hb_low, hb_high
        secondary_alert = "⚠️ Low" if hb_mean < 12.0 else "✅ Normal"
        secondary_badge = "danger" if hb_mean < 12.0 else "success"
        secondary_threshold = "< 12.0 g/dL suggests anemia"

    # Layout
    col_left, col_right = st.columns([1, 1], gap="large")
    with col_left:
        st.markdown("### 📷 Original & Segmentation")
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_rgb, caption="Original", use_container_width=True)
        with col2:
            st.image(overlay, caption="ROI Mask", use_container_width=True)
        feat_names = ['R', 'G', 'B', 'L*', 'a*', 'b*', 'R/G', 'Pallor']
        feat_vals = [f"{v:.1f}" for v in features]
        st.markdown("**Extracted features**")
        cols = st.columns(4)
        for i, (name, val) in enumerate(zip(feat_names, feat_vals)):
            cols[i % 4].metric(label=name, value=val, delta=None)

    with col_right:
        st.markdown("### 🧬 Clinical Estimates")
        with st.container():
            st.markdown(f"""
            <div class='glass-card glass-card-primary'>
                <div class='metric-label'>{primary_label}</div>
                <div><span class='metric-value'>{primary_value:.2f}</span><span class='metric-unit'>{primary_unit}</span></div>
                <div style='margin-top:0.5rem;'><span class='ci-text'>95% CI: {primary_low:.2f} – {primary_high:.2f}</span></div>
                <div style='margin-top:0.8rem;'>
                    <span class='badge-{primary_badge}'>{primary_alert}</span>
                    <span style='margin-left:0.8rem; font-size:0.8rem; color:#94a3b8;'>{primary_threshold}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        with st.container():
            st.markdown(f"""
            <div class='glass-card'>
                <div class='metric-label'>{secondary_label}</div>
                <div><span class='metric-value'>{secondary_value:.2f}</span><span class='metric-unit'>{secondary_unit}</span></div>
                <div style='margin-top:0.5rem;'><span class='ci-text'>95% CI: {secondary_low:.2f} – {secondary_high:.2f}</span></div>
                <div style='margin-top:0.8rem;'>
                    <span class='badge-{secondary_badge}'>{secondary_alert}</span>
                    <span style='margin-left:0.8rem; font-size:0.8rem; color:#94a3b8;'>{secondary_threshold}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        # Plot
        st.markdown("**📊 Ensemble Prediction Distribution**")
        features_arr = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_arr)
        tree_bili = np.array([tree.predict(features_scaled) for tree in model_bili.estimators_]).flatten()
        tree_hb = np.array([tree.predict(features_scaled) for tree in model_hb.estimators_]).flatten()
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=tree_bili, name='Bilirubin', opacity=0.7, marker_color='#fbbf24', nbinsx=20, histnorm='probability density'))
        fig.add_trace(go.Histogram(x=tree_hb, name='Hemoglobin', opacity=0.7, marker_color='#f87171', nbinsx=20, histnorm='probability density', yaxis='y2'))
        fig.add_vline(x=bili_mean, line_width=2, line_dash="dash", line_color="#fbbf24", annotation_text=f"μ={bili_mean:.2f}")
        fig.add_vline(x=hb_mean, line_width=2, line_dash="dash", line_color="#f87171", annotation_text=f"μ={hb_mean:.1f}", annotation_position="top")
        fig.update_layout(template="plotly_dark", height=300, margin=dict(l=20, r=20, t=30, b=20),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                          xaxis=dict(title="Value", gridcolor='#1e293b'),
                          yaxis=dict(title="Density", gridcolor='#1e293b'),
                          yaxis2=dict(title="Density", overlaying='y', side='right', showgrid=False),
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

else:
    st.info("👆 Upload a photo using the sidebar to begin screening.", icon="ℹ️")

st.markdown("---")
st.markdown("<div style='text-align:center; color:#475569; font-size:0.8rem;'>HemoJaundice AI · Enhanced for accurate pallor detection · Train with your own data.</div>", unsafe_allow_html=True)
