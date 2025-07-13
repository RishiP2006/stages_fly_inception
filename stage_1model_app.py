import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from huggingface_hub import hf_hub_download
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import base64
import time

# ─── Page Setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Drosophila Stage Detection", layout="centered")
st.title("Drosophila Stage Detection (Live + Upload)")
st.write("Upload an image or use your webcam. Select a stage, and get a beep when it matches.")

# ─── Model & Labels ───────────────────────────────────────────────────────────
HF_REPO_ID = "RishiPTrial/my-model-name"
MODEL_FILE = "drosophila_inceptionv3_classifier.h5"
STAGE_LABELS = [
    "egg", "1st instar", "2nd instar", "3rd instar",
    "white pupa", "brown pupa", "eye pupa"
]

# ─── Dropdown to select target stage ─────────────────────────────────────────
selected_stage = st.selectbox("🎯 Select stage to alert on:", STAGE_LABELS)

# ─── Shared flag for live beeps ──────────────────────────────────────────────
if "live_match" not in st.session_state:
    st.session_state["live_match"] = False

# ─── Beep generation & playback ──────────────────────────────────────────────
def beep_base64():
    import io, wave, struct
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(44100)
        duration, freq, volume = 0.5, 700, 0.8
        samples = [
            int(volume * 32767 * np.sin(2 * np.pi * freq * t / 44100))
            for t in range(int(44100 * duration))
        ]
        f.writeframes(b''.join(struct.pack('<h', s) for s in samples))
    return base64.b64encode(buffer.getvalue()).decode()

def play_beep():
    """Inject an audio tag to play the beep once."""
    b64 = beep_base64()
    html = f"""
    <audio autoplay>
      <source src="data:audio/wav;base64,{b64}" type="audio/wav">
    </audio>
    """
    st.components.v1.html(html, height=0)

# ─── Load Keras model ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILE)
    from tensorflow.keras.models import load_model as lm
    return lm(path, compile=False), 299

model, input_size = load_model()

# ─── Preprocess & classify helpers ────────────────────────────────────────────
from tensorflow.keras.applications.inception_v3 import preprocess_input

def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.resize((input_size, input_size)).convert("RGB")
    arr = np.asarray(img, np.float32)
    return preprocess_input(arr)

def classify(arr: np.ndarray):
    preds = model.predict(arr[np.newaxis], verbose=0)[0]
    idx = int(np.argmax(preds))
    return STAGE_LABELS[idx], float(preds[idx])

# ─── Upload Image Section ─────────────────────────────────────────────────────
st.subheader("📷 Upload Image for Detection")
uploaded = st.file_uploader(
    label="Choose an image...",
    type=["jpg","jpeg","png"],
    label_visibility="visible"
)
if uploaded:
    pil = Image.open(uploaded).convert("RGB")
    st.image(pil, use_column_width=True)
    arr = preprocess_image(pil)
    label, conf = classify(arr)
    st.success(f"✅ Prediction: **{label}** ({conf:.1%})")
    if label == selected_stage:
        play_beep()

# ─── Live Camera Section ──────────────────────────────────────────────────────
st.subheader("📹 Live Camera Detection")
video_placeholder = st.empty()

class LiveProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_time = 0.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="rgb24")
        pil = Image.fromarray(img)
        arr = preprocess_image(pil)
        label, conf = classify(arr)

        # Draw overlay
        draw = ImageDraw.Draw(pil)
        draw.text((10,10), f"{label} ({conf:.0%})", fill="red")

        # If match and >2s since last, set flag
        now = time.time()
        if label == selected_stage and (now - self.last_time) > 2.0:
            self.last_time = now
            st.session_state["live_match"] = True

        return av.VideoFrame.from_ndarray(np.array(pil), format="rgb24")

webrtc_streamer(
    key="live-dros-stage",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=LiveProcessor,
    async_processing=True,
    video_frame_callback=lambda frame: video_placeholder.image(
        frame.to_ndarray(format="rgb24"),
        channels="RGB"
    )
)

# ─── After live stream: play beep if flagged ────────────────────────────────
if st.session_state["live_match"]:
    st.session_state["live_match"] = False
    play_beep()

st.markdown("---")
st.write(f"Model: `{HF_REPO_ID}/{MODEL_FILE}`")
