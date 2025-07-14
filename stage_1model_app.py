import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from huggingface_hub import hf_hub_download
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av, time, io, wave, struct
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model as keras_load

# ───── App Setup ─────────────────────────────────────────────────
st.set_page_config(page_title="Drosophila Stage Detection", layout="centered")
st.title("🪰 Drosophila Stage Detection (Live + Upload)")
st.write("Match a stage to get visual and audio alerts.")

# ───── Beep Sound ────────────────────────────────────────────────
def make_beep():
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(44100)
        dur, freq, vol = 0.2, 800, 0.8
        samples = [int(vol*32767*np.sin(2*np.pi*freq*t/44100)) for t in range(int(44100*dur))]
        wf.writeframes(b''.join(struct.pack('<h', s) for s in samples))
    return buf.getvalue()

BEEP_WAV = make_beep()

# ───── Model & Labels ────────────────────────────────────────────
HF_REPO_ID = "RishiPTrial/my-model-name"
MODEL_FILE = "drosophila_inceptionv3_classifier.h5"
LABELS = ["egg","1st instar","2nd instar","3rd instar","white pupa","brown pupa","eye pupa"]
sel = st.selectbox("🎯 Stage to alert on:", LABELS)

@st.cache_resource
def load_model():
    path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILE)
    return keras_load(path, compile=False), 299

model, input_size = load_model()

def predict(pil):
    img = pil.resize((input_size, input_size)).convert("RGB")
    arr = preprocess_input(np.array(img, np.float32))[None]
    preds = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    return LABELS[idx], preds[idx]

# ───── Upload Image ─────────────────────────────────────────────
st.subheader("📷 Upload Image")
up = st.file_uploader("Upload image:", type=["jpg", "png", "jpeg"])
if up:
    pil = Image.open(up).convert("RGB")
    st.image(pil, use_column_width=True)
    label, conf = predict(pil)
    st.success(f"Prediction: {label} ({conf:.1%})")
    if label == sel:
        st.audio(BEEP_WAV, format="audio/wav")
        st.markdown(f"<div style='background:red;padding:10px;color:white;font-size:20px;'>⚠️ Matched: {label}</div>", unsafe_allow_html=True)

# ───── Live Prediction State ─────────────────────────────────────
if "live_label" not in st.session_state:
    st.session_state["live_label"] = ""
    st.session_state["live_conf"] = 0.0
    st.session_state["last_beep_time"] = 0.0

# ───── Live Camera ──────────────────────────────────────────────
st.subheader("📹 Live Camera")

class Processor(VideoProcessorBase):
    def __init__(self):
        self.last_time = 0

    def recv(self, frame: av.VideoFrame):
        img = frame.to_ndarray(format="rgb24")
        pil = Image.fromarray(img)
        label, conf = predict(pil)

        # Save to session state (not UI!)
        st.session_state["live_label"] = label
        st.session_state["live_conf"] = conf
        st.session_state["live_updated"] = time.time()

        # Draw overlay
        draw = ImageDraw.Draw(pil)
        draw.text((10, 10), f"{label} ({conf:.0%})", fill="red")

        return av.VideoFrame.from_ndarray(np.array(pil), format="rgb24")

webrtc_streamer(
    key="live",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=Processor,
    async_processing=True,
)

# ───── Poll & Beep if matched ────────────────────────────────────
if "live_updated" in st.session_state:
    label = st.session_state["live_label"]
    conf = st.session_state["live_conf"]
    now = time.time()
    if label == sel and (now - st.session_state["last_beep_time"] > 3):
        st.audio(BEEP_WAV, format="audio/wav")
        st.markdown(f"<div style='background:red;padding:10px;color:white;font-size:20px;'>⚠️ LIVE Match: {label} ({conf:.1%})</div>", unsafe_allow_html=True)
        st.session_state["last_beep_time"] = now

    # Periodic rerun (simulate real-time polling)
    st.experimental_rerun()
