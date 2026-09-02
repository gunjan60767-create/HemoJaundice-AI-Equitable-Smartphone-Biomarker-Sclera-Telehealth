# 🩸 HemoJaundice AI: Equitable Smartphone Biomarker & Sclera Telehealth

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://hemojaundice-ai-equitable-smartphone-biomarker-sclera-telehea.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An Equitable Healthcare Computer Vision & Telehealth platform engineered to perform non-invasive **Anemia (Hemoglobin g/dL)** and **Jaundice (Bilirubin mg/dL)** screening from smartphone camera captures of the palpebral conjunctiva, sclera, and skin.

--- 

## 🌐 Live Application
👉 **[Test Live Web Demo](https://hemojaundice-ai-equitable-smartphone-biomarker-sclera-telehea.streamlit.app)**

---

## ✨ Key Technical Highlights
- **Skin-Tone Stratified Fairness:** Computes Individual Typology Angle (`ITA°`) in CIE $L^*a^*b^*$ space to dynamically categorize skin pigmentation across **Fitzpatrick Types I through VI** and eliminate demographic bias.
- **Calibrated Uncertainty Estimation:** Uses Monte Carlo sampling to compute empirical 95% Confidence Intervals ($\pm 	ext{CI}$) for both Hemoglobin and Bilirubin estimates.
- **Dual Biomarker Extraction:** Evaluates the Erythema/Pallor optical index and $b^*$ chromaticity spectrum to flag latent anemia and scleral icterus.

---

## 🛠️ Tech Stack
- **Computer Vision:** OpenCV (`cv2`), NumPy, Pillow
- **Calibration & Uncertainty:** Monte Carlo Sampling, CIE $L^*a^*b^*$ Colorimetry
- **Frontend & Analytics:** Streamlit, Plotly Interactive Visualizations
