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

st.set_page_config(page_title="NeuroRisk AI", layout="centered")

st.title("🧠 NeuroRisk AI")
st.write("Early Parkinson’s Risk Screening (Not a Diagnosis)")

audio = st.file_uploader("Upload Speech (.wav / .flac)", type=["wav", "flac"])
image = st.file_uploader("Upload Handwriting Image (.png)", type=["png"])

if audio and image:
    with open("temp_audio", "wb") as f:
        f.write(audio.read())

    with open("temp_image.png", "wb") as f:
        f.write(image.read())

    risk, X = predict_with_features("temp_audio", "temp_image.png")

    band, color = risk_band(risk)

    st.subheader("Risk Assessment")
    st.progress(float(risk))
    st.markdown(
        f"**Estimated Risk:** {risk*100:.2f}%  \n"
        f"**Category:** :{color}[{band}]"
    )

    shap_values = explain_prediction(X)
    values = shap_values[0][:, 1]

    st.subheader("Why did the model predict this?")

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

    contrib = modality_contributions(shap_values)

    st.subheader("Modality Contribution")
    st.write(f"🗣️ Speech Biomarkers: {contrib['speech']*100:.1f}%")
    st.write(f"✍️ Handwriting Biomarkers: {contrib['handwriting']*100:.1f}%")

    st.subheader("Clinical Interpretation")
    st.info(clinical_explanation(risk, contrib))
