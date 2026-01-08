import streamlit as st
import sys
import os
import shap
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference.score import predict_with_features
from explainability.explain import explain_prediction
from explainability.feature_names import FEATURE_NAMES
from explainability.modality_shap import modality_contributions
from explainability.clinical_text import clinical_explanation
from utils.risk_bands import risk_band

st.set_page_config(
    page_title="NeuroRisk AI",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ====================== STYLES ======================
st.markdown("""
<style>
header[data-testid="stHeader"] { display: none; }

html, body, section[data-testid="stAppViewContainer"], div[data-testid="stApp"] {
    background: radial-gradient(circle at top, #0b1020 0%, #020617 70%) !important;
    color: #e5e7eb;
}

div[data-testid="stVerticalBlock"],
div[data-testid="stHorizontalBlock"],
div[data-testid="stContainer"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

.custom-header {
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 64px;
    background: radial-gradient(circle at top, #0b1020 0%, #020617 70%);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 9999;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}

.custom-header-title {
    font-size: 2.6rem;
    font-weight: 800;
    color: #93c5fd;
    letter-spacing: 0.05em;
}

.block-container { padding-top: 90px !important; }

.subtitle-box-wrapper {
    display: flex;
    justify-content: center;
    margin-bottom: 2.4rem;
}

.subtitle-box {
    padding: 1.4rem 3.2rem;
    border-radius: 22px;
    background: rgba(255,255,255,0.06);
    box-shadow: 0 18px 40px rgba(0,0,0,0.55);
    backdrop-filter: blur(12px);
    text-align: center;
}

.subtitle-text {
    font-size: 1.1rem;
    color: #93c5fd;
    font-weight: 600;
}

.subtitle-small {
    font-size: 0.9rem;
    color: #9ca3af;
}

.upload-title {
    font-size: 1.05rem;
    font-weight: 600;
    margin-top: 2rem;
    margin-bottom: 0.5rem;
    color: #93c5fd;
}

section[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.05);
    border-radius: 14px;
    padding: 1rem;
    box-shadow: 0 8px 22px rgba(0,0,0,0.35);
}

.predict-btn button {
    width: 100%;
    height: 3.2rem;
    font-size: 1.1rem;
    font-weight: 700;
    background: linear-gradient(135deg, #60a5fa, #38bdf8);
    color: #020617;
    border-radius: 14px;
}

.result-box {
    margin-top: 2.5rem;
    padding: 1.8rem;
    border-radius: 18px;
    background: rgba(147,197,253,0.12);
    text-align: center;
    box-shadow: 0 14px 30px rgba(0,0,0,0.4);
}

.footer-note {
    font-size: 0.9rem;
    color: #9ca3af;
    text-align: center;
    margin-top: 3.2rem;
}
</style>

<div class="custom-header">
    <div class="custom-header-title">NeuroRisk AI</div>
</div>
""", unsafe_allow_html=True)

# ====================== SUBTITLE ======================
st.markdown("""
<div class="subtitle-box-wrapper">
  <div class="subtitle-box">
    <div class="subtitle-text">
      Early Parkinson’s Risk Screening using Speech & Handwriting AI
    </div>
    <div class="subtitle-small">
      Research-grade early risk awareness — not a medical diagnosis
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ====================== INPUTS ======================
st.markdown('<div class="upload-title">🎙️ Upload Speech Sample (.wav / .flac)</div>', unsafe_allow_html=True)
audio = st.file_uploader("", type=["wav", "flac"], label_visibility="collapsed")

st.markdown('<div class="upload-title">✍️ Upload Handwriting Image (.png)</div>', unsafe_allow_html=True)
image = st.file_uploader("", type=["png"], label_visibility="collapsed")

# ====================== PREDICT BUTTON ======================
st.markdown('<div class="predict-btn">', unsafe_allow_html=True)
predict_clicked = st.button("🔍 Predict Risk")
st.markdown('</div>', unsafe_allow_html=True)

# ====================== INFERENCE ======================
if predict_clicked:

    if not audio or not image:
        st.warning("Please upload both speech and handwriting samples.")
    else:
        with open("temp_audio.wav", "wb") as f:
            f.write(audio.read())
        with open("temp_image.png", "wb") as f:
            f.write(image.read())

        with st.spinner("Analyzing multimodal biomarkers..."):
            risk, X = predict_with_features("temp_audio.wav", "temp_image.png")

        band, color = risk_band(risk)

        st.markdown(f"""
        <div class="result-box">
            <h3>Estimated Neurological Risk</h3>
            <h1>{risk*100:.2f}%</h1>
            <p style="color:{color}; font-weight:700;">{band}</p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(float(risk))

        # ====================== SHAP ======================
        st.subheader("Why did the model predict this?")
        shap_values = explain_prediction(X)
        values = shap_values[0][:, 1]

        fig, ax = plt.subplots(figsize=(10, 5))
        shap.waterfall_plot(
            shap.Explanation(
                values=values,
                base_values=0,
                data=X[0],
                feature_names=FEATURE_NAMES
            ),
            show=False
        )
        st.pyplot(fig)

        # ====================== MODALITY ======================
        contrib = modality_contributions(shap_values)

        st.subheader("Modality Contribution")
        st.write(f"🗣️ Speech Biomarkers: {contrib['speech']*100:.1f}%")
        st.write(f"✍️ Handwriting Biomarkers: {contrib['handwriting']*100:.1f}%")

        # ====================== CLINICAL ======================
        st.subheader("Clinical Interpretation")
        st.info(clinical_explanation(risk, contrib))

# ====================== FOOTER ======================
st.markdown("""
<div class="footer-note">
⚠️ This AI system provides early screening insights only.<br>
Always consult a qualified neurologist for medical diagnosis.
</div>
""", unsafe_allow_html=True)
