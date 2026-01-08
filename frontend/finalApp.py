import streamlit as st
import requests
from PIL import Image
from audiorecorder import audiorecorder
from streamlit_drawable_canvas import st_canvas

API_BASE = "https://parkinson-api-c8cxg6b9dwcdhzgx.centralindia-01.azurewebsites.net"

# ====================== SESSION STATE ======================
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None
if "image_path" not in st.session_state:
    st.session_state.image_path = None

if "_audio_just_saved" not in st.session_state:
    st.session_state._audio_just_saved = False
if "_image_just_saved" not in st.session_state:
    st.session_state._image_just_saved = False

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="NeuroRisk AI",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ====================== UI STYLING ======================
st.markdown("""
<style>

header[data-testid="stHeader"] {
    background-color: #e4efef;
    border-bottom: 1px solid #cbdede;
    height: 64px;
    position: relative;
}

header[data-testid="stHeader"]::before {
    content: "NeuroRisk AI";
    position: absolute;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
    font-size: 2.4rem;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: 0.05em;
    pointer-events: none;
}

html, body, section[data-testid="stAppViewContainer"], div[data-testid="stApp"] {
    background-color: #e4efef !important;
    color: #334155;
}

.block-container {
    padding-top: 2rem !important;
}

.subtitle-box-wrapper {
    display: flex;
    justify-content: center;
    margin-bottom: 2.2rem;
}

.subtitle-box {
    padding: 1.4rem 3.2rem;
    border-radius: 22px;
    background: #d6ecec;
    text-align: center;
    margin-top: 2rem;
}

.subtitle-text {
    font-size: 1.1rem;
    color: #0f172a;
    font-weight: 600;
}

.subtitle-small {
    font-size: 0.9rem;
    color: #475569;
}

.upload-title {
    font-size: 1.05rem;
    font-weight: 600;
    margin-top: 2rem;
    margin-bottom: 0.5rem;
    color: #0f172a;
}

section[data-testid="stFileUploader"],
div[data-testid="stRadio"] {
    background: #d6ecec !important;
    border-radius: 14px;
    padding: 1rem;
    border: 2px dashed #9ccccc;
}

.predict-btn button {
    width: 100%;
    height: 3.2rem;
    font-size: 1.1rem;
    font-weight: 700;
    background: #5fd3c7;
    color: #043c3c;
    border-radius: 14px;
    border: none;
}

.predict-btn button:hover {
    background: #4fc3b7;
}

.result-box {
    margin-top: 2.5rem;
    padding: 1.8rem;
    border-radius: 18px;
    background: #d6ecec;
    text-align: center;
}

.footer-note {
    font-size: 0.9rem;
    color: #475569;
    text-align: center;
    margin-top: 3.2rem;
}

</style>
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

# ====================== SPEECH INPUT ======================
st.markdown('<div class="upload-title">🎙️ Speech Input</div>', unsafe_allow_html=True)

audio_mode = st.radio(
    "Speech Mode",
    ["Upload Audio", "Record Audio"],
    horizontal=True,
    label_visibility="collapsed"
)

if audio_mode == "Upload Audio":
    audio = st.file_uploader("", type=["wav", "flac"], label_visibility="collapsed")
    if audio and st.session_state.audio_path is None:
        with open("temp_audio.wav", "wb") as f:
            f.write(audio.read())
        st.session_state.audio_path = "temp_audio.wav"
        st.session_state._audio_just_saved = True
else:
    audio_segment = audiorecorder("🎤 Start Recording", "⏹️ Stop Recording")
    if audio_segment and st.session_state.audio_path is None:
        audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)
        audio_segment.export("temp_audio.wav", format="wav")
        st.session_state.audio_path = "temp_audio.wav"
        st.session_state._audio_just_saved = True

if st.session_state._audio_just_saved:
    st.success("Audio recorded and saved")
    st.session_state._audio_just_saved = False

# ====================== HANDWRITING INPUT ======================
st.markdown('<div class="upload-title">✍️ Handwriting Input</div>', unsafe_allow_html=True)

write_mode = st.radio(
    "Handwriting Mode",
    ["Upload Image", "Write on Canvas"],
    horizontal=True,
    label_visibility="collapsed"
)

if write_mode == "Upload Image":
    image = st.file_uploader("", type=["png"], label_visibility="collapsed")
    if image and st.session_state.image_path is None:
        with open("temp_image.png", "wb") as f:
            f.write(image.read())
        st.session_state.image_path = "temp_image.png"
        st.session_state._image_just_saved = True
else:
    eraser_mode = st.checkbox("Erase", key="eraser_toggle")

    canvas = st_canvas(
        fill_color="rgba(255,255,255,0)",
        stroke_width=20 if eraser_mode else 4,
        stroke_color="#ffffff" if eraser_mode else "#0f172a",
        background_color="#ffffff",
        height=300,
        width=700,
        drawing_mode="freedraw",
        key="handwriting_canvas",
        display_toolbar=False
    )

    if st.button("💾 Save Handwriting") and st.session_state.image_path is None:
        if canvas.image_data is not None:
            img = Image.fromarray(canvas.image_data.astype("uint8"), "RGBA")
            img.convert("RGB").save("temp_image.png")
            st.session_state.image_path = "temp_image.png"
            st.session_state._image_just_saved = True

if st.session_state._image_just_saved:
    st.success("Handwriting saved")
    st.session_state._image_just_saved = False

# ====================== PREDICT ======================
st.markdown('<div class="predict-btn">', unsafe_allow_html=True)
predict_clicked = st.button("🔍 Predict Risk")
st.markdown('</div>', unsafe_allow_html=True)

if predict_clicked:
    if not st.session_state.audio_path or not st.session_state.image_path:
        st.warning("Please provide both speech and handwriting samples.")
    else:
        with st.spinner("Analyzing multimodal biomarkers..."):
            with open(st.session_state.audio_path, "rb") as a, open(st.session_state.image_path, "rb") as i:
                files = {
                    "audio": ("audio.wav", a, "audio/wav"),
                    "image": ("image.png", i, "image/png")
                }
                response = requests.post(f"{API_BASE}/predict", files=files)

        if response.status_code == 200:
            result = response.json()
            risk = result["risk_score"]

            st.markdown(f"""
            <div class="result-box">
                <h3>Estimated Neurological Risk</h3>
                <h1>{risk*100:.2f}%</h1>
                <p style="font-weight:700;">{result["risk_band"]}</p>
            </div>
            """, unsafe_allow_html=True)

            st.progress(float(risk))

            explain_res = requests.post(
                f"{API_BASE}/agent/explain",
                json=result
            )

            if explain_res.status_code == 200:
                st.subheader("🧠 Clinical Explanation")
                st.info(explain_res.json()["explanation"])

# ====================== FOOTER ======================
st.markdown("""
<div class="footer-note">
⚠️ This AI system provides early screening insights only.<br>
Always consult a qualified neurologist for diagnosis.
</div>
""", unsafe_allow_html=True)
