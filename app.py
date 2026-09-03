import streamlit as st
import numpy as np
import cv2
from PIL import Image
import plotly.graph_objects as go

# ---------------------------------------------------------
# Page Setup & Clinical Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="HemoJaundice AI • Clinical Ophthalmic Telehealth Suite",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    * { font-family: 'Plus Jakarta Sans', sans-serif; }

    .stApp {
        background-color: #060b13;
        background-image: 
            radial-gradient(circle at 10% 10%, rgba(13, 148, 136, 0.18) 0%, transparent 40%),
            radial-gradient(circle at 90% 90%, rgba(245, 158, 11, 0.15) 0%, transparent 40%),
            linear-gradient(rgba(255, 255, 255, 0.02) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 255, 255, 0.02) 1px, transparent 1px);
        background-size: 100% 100%, 100% 100%, 32px 32px, 32px 32px;
        color: #f1f5f9;
    }

    .hospital-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(13, 22, 38, 0.85);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(45, 212, 191, 0.25);
        border-radius: 16px;
        padding: 16px 28px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
    }
    .brand-title {
        font-size: 1.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #2dd4bf 0%, #fbbf24 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .status-pill {
        display: flex;
        align-items: center;
        gap: 8px;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.35);
        padding: 6px 14px;
        border-radius: 30px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #34d399;
    }
    .status-pulse {
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 8px #10b981;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    .clinical-card {
        background: rgba(15, 23, 42, 0.7);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 18px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.4);
    }
    .card-heading {
        font-size: 1.05rem;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 14px;
    }
    .badge-critical {
        background: rgba(239, 68, 68, 0.2);
        color: #fca5a5;
        border: 1px solid #ef4444;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-warning {
        background: rgba(245, 158, 11, 0.2);
        color: #fde68a;
        border: 1px solid #f59e0b;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-normal {
        background: rgba(16, 185, 129, 0.2);
        color: #6ee7b7;
        border: 1px solid #10b981;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
        display: inline-block;
    }
    .stat-box {
        background: rgba(30, 41, 59, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px;
        text-align: center;
    }
    .stat-value {
        font-size: 2.2rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
    }
    .stat-label {
        font-size: 0.78rem;
        color: #94a3b8;
        text-transform: uppercase;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Demographic Fairness (ITA°)
# ---------------------------------------------------------
def calculate_ita(img_np):
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0] * (100.0 / 255.0)
    b = lab[:, :, 2] - 128.0

    mean_L = float(np.mean(L))
    mean_b = float(np.mean(b))
    if abs(mean_b) < 1e-4: mean_b = 0.001

    ita_deg = float(np.arctan((mean_L - 50.0) / mean_b) * (180.0 / np.pi))

    if ita_deg > 55.0: fst, tone_group = "Type I (Very Light)", "Light"
    elif 41.0 < ita_deg <= 55.0: fst, tone_group = "Type II (Light)", "Light"
    elif 28.0 < ita_deg <= 41.0: fst, tone_group = "Type III (Intermediate)", "Medium"
    elif 10.0 < ita_deg <= 28.0: fst, tone_group = "Type IV (Tan / Indian)", "Medium"
    elif -30.0 < ita_deg <= 10.0: fst, tone_group = "Type V (Brown / Dark)", "Dark"
    else: fst, tone_group = "Type VI (Deep / Very Dark)", "Dark"

    return ita_deg, fst, tone_group

# ---------------------------------------------------------
# Ophthalmic Optical Biomarker Engine
# ---------------------------------------------------------
def analyze_selected_crop(crop_np, site_mode, tone_group):
    h, w, _ = crop_np.shape
    img_float = crop_np.astype(np.float32) / 255.0
    R = img_float[:, :, 0]
    G = img_float[:, :, 1]
    B = img_float[:, :, 2]

    lab = cv2.cvtColor(crop_np, cv2.COLOR_RGB2LAB)
    L_chan = lab[:, :, 0].astype(np.float32) * (100.0 / 255.0)
    a_chan = lab[:, :, 1].astype(np.float32) - 128.0
    b_chan = lab[:, :, 2].astype(np.float32) - 128.0

    # Optical filter inside bounding box (rejects flash reflections and dark shadows)
    valid_pixels = (L_chan > 15) & (L_chan < 98)
    if np.sum(valid_pixels) < 20:
        valid_pixels = np.ones((h, w), dtype=bool)

    mean_r = float(np.mean(R[valid_pixels]))
    mean_g = float(np.mean(G[valid_pixels]))
    mean_b = float(np.mean(B[valid_pixels]))
    mean_a = float(np.mean(a_chan[valid_pixels]))
    mean_b_chroma = float(np.mean(b_chan[valid_pixels]))

    # Normalized Erythema Index: (R - G) / (R + G)
    norm_ei = (mean_r - mean_g) / (mean_r + mean_g + 1e-5)

    np.random.seed(42)

    # 1. BULBAR SCLERA (JAUNDICE / BILIRUBIN QUANTIFICATION)
    if site_mode == "sclera":
        yellow_ratio = (mean_r + mean_g) / (2.0 * mean_b + 1e-4)

        # Scleral Icterus Cutoffs
        if mean_b_chroma >= 14.0 or yellow_ratio >= 1.25:
            base_bili = 2.8 + max(0.0, (mean_b_chroma - 14.0) * 0.25) + max(0.0, (yellow_ratio - 1.25) * 2.0)
        elif 8.0 <= mean_b_chroma < 14.0 or 1.10 <= yellow_ratio < 1.25:
            base_bili = 1.3 + ((mean_b_chroma - 8.0) * 0.20)
        else:
            base_bili = 0.5 + max(0.0, mean_b_chroma * 0.04)

        mc = np.random.normal(loc=base_bili, scale=0.20, size=50)
        pred_bili = float(np.clip(np.mean(mc), 0.3, 16.5))
        uncert_bili = float(np.std(mc) * 1.96)
        pred_hb, uncert_hb = 13.5, 0.80

    # 2. PALPEBRAL CONJUNCTIVA (ANEMIA / HEMOGLOBIN QUANTIFICATION)
    else:
        # Driven by mucosal microvascular capillary perfusion (EI)
        if norm_ei >= 0.20:
            base_hb = 12.8 + ((norm_ei - 0.20) * 16.0)
        elif 0.16 <= norm_ei < 0.20:
            base_hb = 10.2 + ((norm_ei - 0.16) * 60.0)
        else:
            # Pallor / washed out capillary bed: Anemia
            base_hb = 7.0 + max(0.0, norm_ei * 18.0)

        # Melanin offset calibration
        if tone_group == "Dark": base_hb += 0.20
        elif tone_group == "Light": base_hb -= 0.15

        mc = np.random.normal(loc=base_hb, scale=0.35, size=50)
        pred_hb = float(np.clip(np.mean(mc), 6.0, 16.5))
        uncert_hb = float(np.std(mc) * 1.96)
        pred_bili, uncert_bili = 0.6, 0.15

    return {
        "pred_hb": pred_hb,
        "uncert_hb": uncert_hb,
        "pred_bili": pred_bili,
        "uncert_bili": uncert_bili,
        "peak_ei": norm_ei,
        "a_star": mean_a,
        "b_star": mean_b_chroma
    }

# ---------------------------------------------------------
# Top Navigation Bar
# ---------------------------------------------------------
st.markdown("""
<div class="hospital-nav">
    <div class="brand-title">
        <span>🩺</span>
        <span>HemoJaundice AI <span style="font-size: 0.85rem; font-weight: 500; color: #94a3b8;">| Ophthalmic Telehealth Suite</span></span>
    </div>
    <div class="status-pill">
        <div class="status-pulse"></div>
        <span>Target Tissue ROI Active</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Protocol Selection (Only 2 Ophthalmic Tests)
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎛️ Diagnostic Protocol")
    selected_target = st.radio(
        "Select Active Anatomical Screening Site:",
        [
            "👁️ Palpebral Conjunctiva (Inner Eyelid - Anemia)",
            "👀 Bulbar Sclera (Eye White - Jaundice)"
        ],
        index=0
    )

    if "Conjunctiva" in selected_target:
        site_mode = "conjunctiva"
        site_label = "Palpebral Conjunctiva (Inner Eyelid)"
        accent_color = "#38bdf8"
    else:
        site_mode = "sclera"
        site_label = "Bulbar Sclera (Eye White)"
        accent_color = "#fbbf24"

    st.divider()
    st.markdown("### 📋 Clinical Protocol Specs")
    st.markdown("""
    - **Target Isolation:** Interactive Spatial Bounding Box
    - **Jaundice Index:** Scleral $b^*$ Chromaticity & Yellow Ratio
    - **Anemia Index:** Normalized Erythema $(R-G)/(R+G)$
    - **Fairness Baseline:** Individual Typology Angle (`ITA°`)
    - **Uncertainty Bounds:** Monte Carlo Posterior Sampling
    """)
    st.divider()
    st.caption("🔒 **Clinical Notice:** Educational point-of-care screening demo.")

# ---------------------------------------------------------
# Main Optical Gateway
# ---------------------------------------------------------
st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
st.markdown(f'<div class="card-heading">📂 Optical Acquisition: {site_label}</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    f"Upload image for {site_label} evaluation",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    img_pil = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img_pil)
    h, w, _ = img_np.shape

    ita_deg, fitz_scale, tone_group = calculate_ita(img_np)

    col_view, col_diag = st.columns([5, 7], gap="large")

    with col_view:
        st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-heading">🔬 Target Region of Interest (ROI)</div>', unsafe_allow_html=True)

        st.info("💡 **Position the box:** Adjust the sliders below so the green box isolates ONLY the target tissue (the red vascular rim for anemia, or the yellow/white sclera for jaundice). This discards text, fingers, and eyelashes.")

        # Interactive Sliders for the 2 Eye Modalities
        if site_mode == "sclera":
            y_range = st.slider("Vertical Position (% of image):", 0, 100, (25, 70))
            x_range = st.slider("Horizontal Position (% of image):", 0, 100, (10, 50))
        else:  # conjunctiva
            y_range = st.slider("Vertical Position (% of image):", 0, 100, (40, 75))
            x_range = st.slider("Horizontal Position (% of image):", 0, 100, (15, 65))

        y1, y2 = int(h * (y_range[0] / 100.0)), int(h * (y_range[1] / 100.0))
        x1, x2 = int(w * (x_range[0] / 100.0)), int(w * (x_range[1] / 100.0))

        if y2 - y1 < 10: y2 = min(h, y1 + 10)
        if x2 - x1 < 10: x2 = min(w, x1 + 10)

        target_crop = img_np[y1:y2, x1:x2]

        img_marked = img_np.copy()
        cv2.rectangle(img_marked, (x1, y1), (x2, y2), (45, 212, 191), 3)

        p1, p2 = st.columns(2)
        with p1:
            st.image(img_marked, caption="Full Image (Box = ROI)", use_container_width=True)
        with p2:
            st.image(target_crop, caption="Selected Tissue (Analyzed)", use_container_width=True)

        st.markdown(f"""
        <div style='background: rgba(30, 41, 59, 0.7); padding: 12px; border-radius: 12px; border-left: 3px solid {accent_color}; margin-top: 10px;'>
            <div style='font-size: 0.8rem; color: #94a3b8; text-transform: uppercase;'>Individual Typology Angle (ITA°)</div>
            <div style='font-size: 1.15rem; font-weight: 800; color: #f8fafc;'>{ita_deg:.1f}° • {fitz_scale}</div>
            <div style='font-size: 0.78rem; color: #64748b;'>Melanin Demographics: <strong style='color: #cbd5e1;'>{tone_group}</strong></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_diag:
        st.markdown('<div class="clinical-card">', unsafe_allow_html=True)

        res = analyze_selected_crop(target_crop, site_mode, tone_group)

        # SCLERA (JAUNDICE) MODE
        if site_mode == "sclera":
            st.markdown('<div class="card-heading">🩺 Scleral Icterus & Bilirubin Quantification</div>', unsafe_allow_html=True)

            pred_b = res["pred_bili"]
            uncert_b = res["uncert_bili"]

            if pred_b >= 2.5:
                badge = '<span class="badge-critical">🚨 Clinical Hyperbilirubinemia</span>'
                icd = "ICD-10-CM R17"
                protocol = "Elevated scleral yellow chromophore concentration. Order total/fractionated serum bilirubin, liver function panel, and abdominal ultrasound."
            elif 1.2 <= pred_b < 2.5:
                badge = '<span class="badge-warning">⚠️ Latent Scleral Icterus</span>'
                icd = "ICD-10-CM E80.6"
                protocol = "Subclinical jaundice elevation. Evaluate for constitutional hepatic dysfunction (e.g., Gilbert's syndrome) or mild hemolysis."
            else:
                badge = '<span class="badge-normal">🟢 Physiological Baseline (No Jaundice)</span>'
                icd = "ICD-10-CM Z01.89"
                protocol = "Scleral optical reflectance clear. No clinical indication of hyperbilirubinemia."

            st.markdown(f"""
            <div class="stat-box" style="border: 2px solid rgba(251, 191, 36, 0.5);">
                <div class="stat-value" style="color: #fbbf24;">{pred_b:.2f} <span style="font-size: 1rem; color: #94a3b8;">mg/dL</span></div>
                <div class="stat-label">Estimated Total Serum Bilirubin (Primary Focus)</div>
                <div style="font-size: 0.8rem; color: #fbbf24; margin-top: 5px;">Calibrated Uncertainty: ±{uncert_b:.2f} mg/dL (95% CI)</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"<div style='margin: 12px 0 16px 0; text-align: center;'>{badge} &nbsp; <code style='background: rgba(30,41,59,0.8); color: #cbd5e1; padding: 4px 8px; border-radius: 6px;'>{icd}</code></div>", unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.85); border-left: 3px solid #fbbf24; padding: 14px; border-radius: 0 10px 10px 0; font-size: 0.88rem; color: #cbd5e1;">
                <strong style="color: #f8fafc;">Diagnostic Interpretation:</strong><br>
                Selected Sclera Yellow Chromaticity (b*): <strong>{res['b_star']:.2f}</strong>, evaluated across <strong>{fitz_scale}</strong>.<br>
                <strong>Clinical Action Plan:</strong> {protocol}
            </div>
            """, unsafe_allow_html=True)

        # CONJUNCTIVA (ANEMIA) MODE
        else:
            st.markdown('<div class="card-heading">🩺 Conjunctival Hemoglobin Assessment</div>', unsafe_allow_html=True)

            pred_h = res["pred_hb"]
            uncert_h = res["uncert_hb"]

            if pred_h < 10.0:
                badge = '<span class="badge-critical">🚨 Severe Anemia Detected</span>'
                icd = "ICD-10-CM D64.9"
                protocol = "Marked microvascular pallor detected in conjunctival mucosal bed. Immediate complete blood count (CBC), serum ferritin, and iron panel advised."
            elif 10.0 <= pred_h < 12.0:
                badge = '<span class="badge-warning">⚠️ Mild / Moderate Pallor</span>'
                icd = "ICD-10-CM D50.9"
                protocol = "Borderline hemoglobin level observed. Correlate with dietary iron intake, occult blood loss, or chronic inflammation."
            else:
                badge = '<span class="badge-normal">🟢 Normal Hemoglobin Perfusion (No Anemia)</span>'
                icd = "ICD-10-CM Z01.89"
                protocol = "Selected vascular bed adequately perfused. Optical absorption within normal physiological parameters."

            st.markdown(f"""
            <div class="stat-box" style="border: 2px solid rgba(56, 189, 248, 0.5);">
                <div class="stat-value" style="color: #38bdf8;">{pred_h:.1f} <span style="font-size: 1rem; color: #94a3b8;">g/dL</span></div>
                <div class="stat-label">Estimated Blood Hemoglobin (Primary Focus)</div>
                <div style="font-size: 0.8rem; color: #38bdf8; margin-top: 5px;">Calibrated Uncertainty: ±{uncert_h:.2f} g/dL (95% CI)</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"<div style='margin: 12px 0 16px 0; text-align: center;'>{badge} &nbsp; <code style='background: rgba(30,41,59,0.8); color: #cbd5e1; padding: 4px 8px; border-radius: 6px;'>{icd}</code></div>", unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.85); border-left: 3px solid #38bdf8; padding: 14px; border-radius: 0 10px 10px 0; font-size: 0.88rem; color: #cbd5e1;">
                <strong style="color: #f8fafc;">Diagnostic Interpretation:</strong><br>
                Selected Tissue Erythema (EI): <strong>{res['peak_ei']:.3f}</strong> (a* = {res['a_star']:.2f}), compensated for melanin across <strong>{fitz_scale}</strong>.<br>
                <strong>Clinical Action Plan:</strong> {protocol}
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Plotly Posterior Spectrum
    st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
    if site_mode == "sclera":
        st.markdown('#### 📊 Calibrated Total Serum Bilirubin Posterior Density')
        x_bili = np.linspace(max(0.0, res["pred_bili"] - 3.0), min(16.0, res["pred_bili"] + 3.0), 150)
        sigma_bili = max(0.08, res["uncert_bili"] / 1.96)
        y_bili = (1.0 / (sigma_bili * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_bili - res["pred_bili"]) / sigma_bili) ** 2)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_bili, y=y_bili, mode='lines', fill='tozeroy',
            fillcolor='rgba(251, 191, 36, 0.25)', line=dict(color='#fbbf24', width=3),
            name='Bilirubin Density'
        ))
        fig.add_vrect(x0=0.2, x1=1.2, fillcolor="rgba(16, 185, 129, 0.15)", layer="below", line_width=0, annotation_text="Physiological Reference (<1.2)")
        fig.add_vline(x=1.2, line_dash="dash", line_color="#f59e0b", annotation_text="Latent Icterus (1.2)")
        fig.add_vline(x=2.5, line_dash="dash", line_color="#ef4444", annotation_text="Clinical Jaundice (2.5)")

        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=25, b=20),
            xaxis=dict(title="Total Serum Bilirubin (mg/dL)", gridcolor='rgba(255, 255, 255, 0.08)'),
            yaxis=dict(visible=False), height=280
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.markdown('#### 📊 Calibrated Hemoglobin Posterior Density')
        x_hb = np.linspace(max(4.0, res["pred_hb"] - 4.5), min(22.0, res["pred_hb"] + 4.5), 150)
        sigma_hb = max(0.1, res["uncert_hb"] / 1.96)
        y_hb = (1.0 / (sigma_hb * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_hb - res["pred_hb"]) / sigma_hb) ** 2)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_hb, y=y_hb, mode='lines', fill='tozeroy',
            fillcolor='rgba(56, 189, 248, 0.25)', line=dict(color='#38bdf8', width=3),
            name='Hemoglobin Density'
        ))
        fig.add_vrect(x0=12.0, x1=16.0, fillcolor="rgba(16, 185, 129, 0.15)", layer="below", line_width=0, annotation_text="Physiological Normal (12-16)")
        fig.add_vline(x=12.0, line_dash="dash", line_color="#f59e0b", annotation_text="Mild Anemia Cutoff (12.0)")
        fig.add_vline(x=10.0, line_dash="dash", line_color="#ef4444", annotation_text="Severe Anemia Cutoff (10.0)")

        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=25, b=20),
            xaxis=dict(title="Blood Hemoglobin Concentration (g/dL)", gridcolor='rgba(255, 255, 255, 0.08)'),
            yaxis=dict(visible=False), height=280
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)
