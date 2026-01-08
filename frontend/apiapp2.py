import streamlit as st
import requests
from PIL import Image
from audiorecorder import audiorecorder
from streamlit_drawable_canvas import st_canvas

API_BASE = "https://neurorisk-api-dnetg9g4f2hkenhn.centralindia-01.azurewebsites.net"

if "audio_path" not in st.session_state:
    st.session_state.audio_path = None

if "image_path" not in st.session_state:
    st.session_state.image_path = None

st.set_page_config(
    page_title="NeuroRisk AI",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ====================== SOFT TEAL STONE THEME (LOCKED) ======================
st.markdown("""
<style>
    .stApp {
        background-color: #e4efef;
    }

    h1, h2, h3 {
        color: #0f172a;
    }

    p, label, span {
        color: #334155;
    }

    section[data-testid="stFileUploader"],
    div[data-testid="stRadio"],
    div[data-testid="stAlert"] {
        background-color: #d3e4e4;
        padding: 1.2rem;
        border-radius: 18px;
    }

    .stButton > button {
        background-color: #0f766e;
        color: #ffffff;
        border-radius: 16px;
        padding: 0.75rem 1.8rem;
        font-weight: 600;
        border: none;
    }

    .stButton > button:hover {
        background-color: #115e59;
        color: #ffffff;
    }

    .stProgress > div > div {
        background-color: #0f766e;
    }
</style>
""", unsafe_allow_html=True)

# ====================== HEADER ======================
st.markdown("""
<h1 style="text-align:center;">NeuroRisk AI</h1>
<p style="text-align:center;">
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
        stroke_color="#0f172a",
        background_color="#d3e4e4",
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
            with open(st.session_state.audio_path, "rb") as a, open(st.session_state.image_path, "rb") as i:
                files = {
                    "audio": ("audio.wav", a, "audio/wav"),
                    "image": ("image.png", i, "image/png")
                }
                response = requests.post(f"{API_BASE}/predict", files=files)

        if response.status_code == 200:
            result = response.json()
            risk = result["risk_score"]
            band = result["risk_band"]

            st.markdown(f"""
            <div style="
                padding:1.8rem;
                border-radius:20px;
                background:#d3e4e4;
                text-align:center;
            ">
                <h3>Estimated Neurological Risk</h3>
                <h1>{risk*100:.2f}%</h1>
                <p style="font-weight:700;">{band}</p>
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
<p style="text-align:center; color:#475569; margin-top:2.5rem;">
⚠️ This AI system provides early screening insights only.<br>
Always consult a qualified neurologist for diagnosis.
</p>
""", unsafe_allow_html=True)
