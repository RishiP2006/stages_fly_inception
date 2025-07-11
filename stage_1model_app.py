import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from huggingface_hub import hf_hub_download
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import io
import wave
import struct
import time

st.set_page_config(page_title="Drosophila Stage Detection (Live + Upload)", layout="centered")
st.title("Drosophila Stage Detection")
st.write("Upload an image or use your webcam. Select a stage, and get a beep when it matches.")

# --- Model & Labels ---
HF_REPO_ID = "RishiPTrial/my-model-name"
MODEL_FILE = "drosophila_inceptionv3_classifier.h5"
STAGE_LABELS = [
    "egg","1st instar","2nd instar","3rd instar",
    "white pupa","brown pupa","eye pupa"
]

# --- Dropdown target stage ---
selected_stage = st.selectbox("🎯 Select stage to alert on:", STAGE_LABELS)

# --- Prepare beep WAV data once ---
def make_beep_wav(duration=0.6, freq=660, volume=0.8, sr=44100):
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        samples = [
            int(volume * 32767 * np.sin(2 * np.pi * freq * t / sr))
            for t in range(int(sr * duration))
        ]
        wf.writeframes(b''.join(struct.pack('<h', s) for s in samples))
    return buf.getvalue()

beep_wav = make_beep_wav()

if 'beep_live' not in st.session_state:
    st.session_state['beep_live'] = False

# --- Load model ---
@st.cache_resource(show_spinner=False)
def load_model():
    path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILE)
    from tensorflow.keras.models import load_model as lm
    return {"model": lm(path, compile=False), "input_size": 299}

model_info = load_model()
model = model_info["model"]

# --- Preprocess & classify helpers ---
def preprocess_image(img, size):
    from tensorflow.keras.applications.inception_v3 import preprocess_input
    arr = img.resize((size, size)).convert("RGB")
    x = np.asarray(arr, np.float32)
    return preprocess_input(x)

def classify(arr):
    preds = model.predict(np.expand_dims(arr, 0), verbose=0)
    idx = int(np.argmax(preds, axis=1)[0])
    return STAGE_LABELS[idx], float(preds[0][idx])

# --- Image upload section ---
st.subheader("📷 Upload Image")
file = st.file_uploader("", type=["jpg","jpeg","png"])
if file:
    img = Image.open(file).convert("RGB")
    st.image(img, use_column_width=True)
    arr = preprocess_image(img, model_info["input_size"])
    label, conf = classify(arr)
    st.success(f"Prediction: {label} ({conf:.1%})")
    if label == selected_stage:
        st.audio(beep_wav, format="audio/wav", start_time=0)

# --- Live webcam section ---
st.subheader("📹 Live Camera Detection")

class LiveProcessor(VideoProcessorBase):
    def __init__(self):
        self.size = model_info["input_size"]
        self.last_time = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="rgb24")
        pil = Image.fromarray(img)
        arr = preprocess_image(pil, self.size)
        label, conf = classify(arr)

        draw = ImageDraw.Draw(pil)
        draw.text((10, 10), f"{label} ({conf:.0%})", fill="red")

        now = time.time()
        # match + 2s cooldown
        if label == selected_stage and (now - self.last_time) > 2:
            self.last_time = now
            st.session_state['beep_live'] = True

        return av.VideoFrame.from_ndarray(np.array(pil), format="rgb24")

ctx = webrtc_streamer(
    key="dros-stage-live",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=LiveProcessor,
    async_processing=True
)

# --- After streaming, if flag is set, play beep and rerun ---
if st.session_state['beep_live']:
    st.session_state['beep_live'] = False
    st.audio(beep_wav, format="audio/wav", start_time=0)
    # Force a rerun so that the audio component is attached to a fresh render
    st.experimental_rerun()

st.markdown("---")
st.write(f"Model: `{HF_REPO_ID}/{MODEL_FILE}`")
