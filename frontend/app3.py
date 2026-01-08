import streamlit as st
import sys
import os
import shap
import matplotlib.pyplot as plt
from PIL import Image

from audiorecorder import audiorecorder
from streamlit_drawable_canvas import st_canvas

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference.score import predict_with_features
from explainability.explain import explain_prediction
from explainability.feature_names import FEATURE_NAMES
from explainability.modality_shap import modality_contributions
from explainability.clinical_text import clinical_explanation
from utils.risk_bands import risk_band


# ====================== SESSION STATE INIT ======================
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None

if "image_path" not in st.session_state:
    st.session_state.image_path = None


# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="NeuroRisk AI",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ====================== HEADER ======================
st.markdown("""
<h1 style="text-align:center; color:#93c5fd;">NeuroRisk AI</h1>
<p style="text-align:center; color:#9ca3af;">
Early Parkinson’s Risk Screening using Speech & Handwriting AI<br>
<em>Research-grade screening — not a medical diagnosis</em>
</p>
""", unsafe_allow_html=True)


# ====================== AUDIO INPUT ======================
st.markdown("## 🎙️ Speech Input")

audio_mode = st.radio(
    "Speech Mode",
    ["Upload Audio", "Record Audio"],
    horizontal=True,
    label_visibility="collapsed"
)

if audio_mode == "Upload Audio":
    audio_file = st.file_uploader(
        "Upload speech (.wav / .flac)",
        type=["wav", "flac"],
        label_visibility="collapsed"
    )
    if audio_file:
        path = "temp_audio.wav"
        with open(path, "wb") as f:
            f.write(audio_file.read())
        st.session_state.audio_path = path
        st.success("Audio uploaded successfully")

else:
    audio_segment = audiorecorder("🎤 Start Recording", "⏹️ Stop Recording")
    if audio_segment:
        path = "temp_audio.wav"
        audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)
        audio_segment.export(path, format="wav")
        st.session_state.audio_path = path
        st.success("Audio recorded and saved")


# ====================== HANDWRITING INPUT ======================
st.markdown("## ✍️ Handwriting Input")

write_mode = st.radio(
    "Handwriting Mode",
    ["Upload Image", "Write on Canvas"],
    horizontal=True,
    label_visibility="collapsed"
)

if write_mode == "Upload Image":
    image_file = st.file_uploader(
        "Upload handwriting (.png)",
        type=["png"],
        label_visibility="collapsed"
    )
    if image_file:
        path = "temp_image.png"
        with open(path, "wb") as f:
            f.write(image_file.read())
        st.session_state.image_path = path
        st.success("Image uploaded successfully")

else:
    canvas = st_canvas(
        fill_color="rgba(255,255,255,0)",
        stroke_width=4,
        stroke_color="#ffffff",
        background_color="#000000",
        height=280,
        width=600,
        drawing_mode="freedraw",
        key="handwriting_canvas"
    )

    if st.button("💾 Save Handwriting"):
        if canvas.image_data is None:
            st.warning("Canvas is empty")
        else:
            img = Image.fromarray(canvas.image_data.astype("uint8"), "RGBA")
            path = "temp_image.png"
            img.convert("RGB").save(path)
            st.session_state.image_path = path
            st.success("Handwriting saved")


# ====================== PREDICTION ======================
st.markdown("## 🔍 Risk Prediction")

if st.button("Predict Risk"):
    if not st.session_state.audio_path or not st.session_state.image_path:
        st.warning("Please provide both speech and handwriting samples.")
    else:
        with st.spinner("Analyzing multimodal biomarkers..."):
            risk, X = predict_with_features(
                st.session_state.audio_path,
                st.session_state.image_path
            )

        band, color = risk_band(risk)

        st.markdown(f"""
        <div style="padding:1.5rem; border-radius:14px; background:rgba(147,197,253,0.12); text-align:center;">
            <h3>Estimated Neurological Risk</h3>
            <h1>{risk*100:.2f}%</h1>
            <p style="color:{color}; font-weight:700;">{band}</p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(float(risk))

        # SHAP
        st.subheader("Why did the model predict this?")
        shap_values = explain_prediction(X)
        values = shap_values[0][:, 1]

        fig, _ = plt.subplots(figsize=(10, 5))
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
        st.write(f"🗣️ Speech: {contrib['speech']*100:.1f}%")
        st.write(f"✍️ Handwriting: {contrib['handwriting']*100:.1f}%")

        st.subheader("Clinical Interpretation")
        st.info(clinical_explanation(risk, contrib))


# ====================== FOOTER ======================
st.markdown("""
<p style="text-align:center; color:#9ca3af; margin-top:2rem;">
⚠️ This AI system provides early screening insights only.<br>
Always consult a qualified neurologist for diagnosis.
</p>
""", unsafe_allow_html=True)
