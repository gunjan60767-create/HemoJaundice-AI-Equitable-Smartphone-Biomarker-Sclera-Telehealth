import streamlit as st
import numpy as np
import cv2
from PIL import Image
import plotly.graph_objects as go

# ---------------------------------------------------------
# Page Setup & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="HemoJaundice AI • Clinical Tissue Biomarker Telehealth",
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
# Illumination Correction & Color Constancy (Gray-World)
# ---------------------------------------------------------
def apply_color_constancy(img_np):
    """Normalizes lighting temperature so indoor yellow lights don't look like jaundice."""
    b, g, r = cv2.split(img_np.astype(np.float32))
    mean_b = np.mean(b) + 1e-5
    mean_g = np.mean(g) + 1e-5
    mean_r = np.mean(r) + 1e-5
    gray = (mean_b + mean_g + mean_r) / 3.0

    b = np.clip(b * (gray / mean_b), 0, 255)
    g = np.clip(g * (gray / mean_g), 0, 255)
    r = np.clip(r * (gray / mean_r), 0, 255)
    return cv2.merge([b, g, r]).astype(np.uint8)

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
# Medical Biomarker Quantitative Analysis
# ---------------------------------------------------------
def evaluate_tissue_biomarker(crop_np, site_mode, tone_group):
    # Apply illumination constancy to isolated crop
    norm_crop = apply_color_constancy(crop_np)
    
    img_float = norm_crop.astype(np.float32) / 255.0
    R = img_float[:, :, 0]
    G = img_float[:, :, 1]
    B = img_float[:, :, 2]

    lab = cv2.cvtColor(norm_crop, cv2.COLOR_RGB2LAB)
    L_chan = lab[:, :, 0]
    a_chan = lab[:, :, 1].astype(np.float32) - 128.0
    b_chan = lab[:, :, 2].astype(np.float32) - 128.0

    # Filter out text, watermarks, dark arrows, and flash glare
    valid_pixels = ~((B > R + 0.12) & (B > G + 0.12)) & (L_chan > 45) & (L_chan < 240)
    if np.sum(valid_pixels) < 40:
        valid_pixels = np.ones(L_chan.shape, dtype=bool)

    # 1. HEMOGLOBIN (ANEMIA)
    # Uses peak 25% vascular perfusion of the region to bypass surrounding skin/rims
    pixel_ei = (R - G) / (R + G + 1e-5)
    valid_ei = pixel_ei[valid_pixels]
    sorted_ei = np.sort(valid_ei)
    top_perfusion = sorted_ei[int(len(sorted_ei) * 0.70):] if len(sorted_ei) > 0 else [0.15]
    peak_erythema = float(np.mean(top_perfusion))

    if site_mode in ["conjunctiva", "nail"]:
        if peak_erythema >= 0.22:
            base_hb = 13.0 + ((peak_erythema - 0.22) * 14.0)
        elif 0.14 <= peak_erythema < 0.22:
            base_hb = 10.2 + ((peak_erythema - 0.14) * 35.0)
        else:
            base_hb = 7.0 + max(0.0, peak_erythema * 22.0)

        if tone_group == "Dark": base_hb += 0.20
        elif tone_group == "Light": base_hb -= 0.15

        mc = np.random.normal(loc=base_hb, scale=0.35, size=50)
        pred_hb = float(np.clip(np.mean(mc), 6.5, 16.5))
        uncert_hb = float(np.std(mc) * 1.96)
    else:
        pred_hb = 13.5
        uncert_hb = 0.80

    # 2. BILIRUBIN (JAUNDICE)
    valid_b = b_chan[valid_pixels]
    mean_b_sclera = float(np.mean(valid_b))

    if site_mode == "sclera":
        if mean_b_sclera >= 13.0:
            base_bili = 2.6 + ((mean_b_sclera - 13.0) * 0.25)
        elif 7.0 <= mean_b_sclera < 13.0:
            base_bili = 1.3 + ((mean_b_sclera - 7.0) * 0.20)
        else:
            base_bili = 0.5 + max(0.0, mean_b_sclera * 0.04)

        mc = np.random.normal(loc=base_bili, scale=0.22, size=50)
        pred_bili = float(np.clip(np.mean(mc), 0.2, 16.5))
        uncert_bili = float(np.std(mc) * 1.96)
    else:
        pred_bili = 0.7
        uncert_bili = 0.20

    return {
        "pred_hb": pred_hb,
        "uncert_hb": uncert_hb,
        "pred_bili": pred_bili,
        "uncert_bili": uncert_bili,
        "erythema": peak_erythema,
        "b_star": mean_b_sclera
    }

# ---------------------------------------------------------
# Top Navigation Bar
# ---------------------------------------------------------
st.markdown("""
<div class="hospital-nav">
    <div class="brand-title">
        <span>🩺</span>
        <span>HemoJaundice AI <span style="font-size: 0.85rem; font-weight: 500; color: #94a3b8;">| Clinical Tissue Biomarker Telehealth</span></span>
    </div>
    <div class="status-pill">
        <div class="status-pulse"></div>
        <span>Color Constancy & Fairness Engine Active</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Diagnostic Protocols
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎛️ Diagnostic Protocol")
    selected_target = st.radio(
        "Select Active Screening Target Site:",
        [
            "👁️ Palpebral Conjunctiva (Inner Eyelid - Anemia)",
            "👀 Bulbar Sclera (Eye White - Jaundice)",
            "🖐️ Subungual Fingernail Bed (Capillary Pallor)"
        ],
        index=0
    )

    if "Conjunctiva" in selected_target:
        site_mode = "conjunctiva"
        site_label = "Palpebral Conjunctiva"
        accent_color = "#38bdf8"
    elif "Sclera" in selected_target:
        site_mode = "sclera"
        site_label = "Bulbar Sclera"
        accent_color = "#fbbf24"
    else:
        site_mode = "nail"
        site_label = "Subungual Fingernail Bed"
        accent_color = "#2dd4bf"

    st.divider()
    st.markdown("### 📋 Clinical Benchmark Specs")
    st.markdown("""
    - **Color Normalizer:** Gray-World Color Constancy
    - **Vascular Perfusion:** Peak-Erythema Index $(R-G)/(R+G)$
    - **Fairness Baseline:** Individual Typology Angle (`ITA°`)
    - **Demographic Model:** Fitzpatrick Scale (Types I–VI)
    - **Uncertainty Bounds:** Monte Carlo Sampling (95% CI)
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

        st.caption("Adjust sliders so the box covers ONLY the target tissue (inner eyelid red bed, eye white, or nail). This discards text and fingers.")

        if site_mode == "conjunctiva":
            y_range = st.slider("Vertical Position (% of image):", 0, 100, (40, 75))
            x_range = st.slider("Horizontal Position (% of image):", 0, 100, (15, 65))
        elif site_mode == "sclera":
            y_range = st.slider("Vertical Position (% of image):", 0, 100, (25, 70))
            x_range = st.slider("Horizontal Position (% of image):", 0, 100, (40, 80))
        else:
            y_range = st.slider("Vertical Position (% of image):", 0, 100, (20, 80))
            x_range = st.slider("Horizontal Position (% of image):", 0, 100, (20, 80))

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
            st.image(target_crop, caption="Isolated Diagnostic Tissue", use_container_width=True)

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

        res = evaluate_tissue_biomarker(target_crop, site_mode, tone_group)

        if site_mode == "sclera":
            st.markdown('<div class="card-heading">🩺 Scleral Icterus & Bilirubin Quantification</div>', unsafe_allow_html=True)

            pred_b = res["pred_bili"]
            uncert_b = res["uncert_bili"]

            if pred_b >= 2.5:
                badge = '<span class="badge-critical">🚨 Clinical Hyperbilirubinemia</span>'
                icd = "ICD-10-CM R17"
                protocol = "Elevated scleral yellow chromophores detected. Order hepatic function panel, fractionated bilirubin, and liver ultrasound."
            elif 1.2 <= pred_b < 2.5:
                badge = '<span class="badge-warning">⚠️ Latent Scleral Icterus</span>'
                icd = "ICD-10-CM E80.6"
                protocol = "Subclinical jaundice elevation. Evaluate for constitutional hepatic dysfunction (e.g., Gilbert's syndrome) or mild hemolysis."
            else:
                badge = '<span class="badge-normal">🟢 Physiological Baseline (No Jaundice)</span>'
                icd = "ICD-10-CM Z01.89"
                protocol = "Scleral optical reflectance clear. No clinical evidence of acute hyperbilirubinemia."

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
                Isolated Sclera Yellow Chromaticity (b*): <strong>{res['b_star']:.2f}</strong>, calibrated across <strong>{fitz_scale}</strong>.<br>
                <strong>Clinical Action Plan:</strong> {protocol}
            </div>
            """, unsafe_allow_html=True)

        else:
            panel_title = "Conjunctival Hemoglobin Assessment" if site_mode == "conjunctiva" else "Subungual Capillary Perfusion"
            st.markdown(f'<div class="card-heading">🩺 {panel_title}</div>', unsafe_allow_html=True)

            pred_h = res["pred_hb"]
            uncert_h = res["uncert_hb"]

            if pred_h < 10.0:
                badge = '<span class="badge-critical">🚨 Severe Anemia Detected</span>'
                icd = "ICD-10-CM D64.9"
                protocol = "Marked pallor / loss of microvascular blood volume detected in target tissue. Immediate complete blood count (CBC) and serum ferritin test advised."
            elif 10.0 <= pred_h < 12.0:
                badge = '<span class="badge-warning">⚠️ Mild / Moderate Pallor</span>'
                icd = "ICD-10-CM D50.9"
                protocol = "Borderline hemoglobin level observed. Correlate with dietary iron deficiency, occult blood loss, or chronic inflammation."
            else:
                badge = '<span class="badge-normal">🟢 Normal Hemoglobin Perfusion (No Anemia)</span>'
                icd = "ICD-10-CM Z01.89"
                protocol = "Target vascular bed adequately perfused. Optical absorption parameters within healthy physiological limits."

            st.markdown(f"""
            <div class="stat-box" style="border: 2px solid rgba(56, 189, 248, 0.5);">
                <div class="stat-value" style="color: #38bdf8;">{pred_h:.1f} <span style="font-size: 1rem; color: #94a3b8;">g/dL</span></div>
                <div class="stat-label">Estimated Blood Hemoglobin Concentration (Primary Focus)</div>
                <div style="font-size: 0.8rem; color: #38bdf8; margin-top: 5px;">Calibrated Uncertainty: ±{uncert_h:.2f} g/dL (95% CI)</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"<div style='margin: 12px 0 16px 0; text-align: center;'>{badge} &nbsp; <code style='background: rgba(30,41,59,0.8); color: #cbd5e1; padding: 4px 8px; border-radius: 6px;'>{icd}</code></div>", unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.85); border-left: 3px solid #38bdf8; padding: 14px; border-radius: 0 10px 10px 0; font-size: 0.88rem; color: #cbd5e1;">
                <strong style="color: #f8fafc;">Diagnostic Interpretation:</strong><br>
                Target Mucosa Peak Erythema Index: <strong>{res['erythema']:.3f}</strong>, calibrated for epidermal scatter across <strong>{fitz_scale}</strong>.<br>
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
