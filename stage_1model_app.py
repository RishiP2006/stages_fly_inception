import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from huggingface_hub import hf_hub_download
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import base64
import time

st.set_page_config(page_title="Drosophila Stage Detection", layout="centered")
st.title("Drosophila Stage Detection (Live + Upload)")
st.write("Using deployed InceptionV3 model for inference.")

# === Hugging Face model info ===
HF_REPO_ID = "RishiPTrial/my-model-name"
MODEL_FILE = "drosophila_inceptionv3_classifier.h5"

# === Life stage labels ===
STAGE_LABELS = [
    "egg",
    "1st instar",
    "2nd instar",
    "3rd instar",
    "white pupa",
    "brown pupa",
    "eye pupa"
]

# === Dropdown ===
selected_stage = st.selectbox("🎯 Select stage to detect", STAGE_LABELS)

# === Audio beep logic ===
def play_beep():
    beep = """
    <audio autoplay>
        <source src="data:audio/wav;base64,{0}" type="audio/wav">
    </audio>
    """.format(beep_base64())
    st.components.v1.html(beep, height=0)

def beep_base64():
    import io
    import wave
    import struct
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(44100)
        duration = 0.2
        freq = 880
        volume = 0.8
        samples = [
            int(volume * 32767 * np.sin(2 * np.pi * freq * t / 44100))
            for t in range(int(44100 * duration))
        ]
        f.writeframes(b''.join([struct.pack('<h', s) for s in samples]))
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

# === Load model from Hugging Face ===
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILE)
    except Exception as e:
        st.error(f"Error downloading model: {e}")
        return None
    try:
        import keras
        from keras.models import load_model
        model = load_model(path, compile=False)
        return {"model": model, "input_size": 299}
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model_info = load_model()
if model_info is None:
    st.stop()
model = model_info["model"]

# === Image preprocessing ===
def preprocess_image(img: Image.Image, size: int):
    from keras.applications.inception_v3 import preprocess_input
    arr = img.resize((size, size)).convert("RGB")
    x = np.asarray(arr).astype(np.float32)
    x = preprocess_input(x)
    return x

# === Prediction logic ===
def classify(model, arr: np.ndarray):
    x = np.expand_dims(arr, axis=0)
    return model.predict(x)

def interpret_class(preds):
    arr = np.asarray(preds)
    if arr.ndim == 2 and arr.shape[1] == len(STAGE_LABELS):
        idx = int(np.argmax(arr, axis=1)[0])
        return STAGE_LABELS[idx], float(arr[0][idx])
    return None, None

# === Upload image section ===
st.subheader("📷 Upload Image for Detection")
img_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
if img_file:
    pil = Image.open(img_file).convert("RGB")
    st.image(pil, use_column_width=True)
    arr = preprocess_image(pil, model_info["input_size"])
    preds = classify(model, arr)
    label, conf = interpret_class(preds)
    st.write("🔍 Raw prediction:", preds)
    if label:
        st.success(f"✅ Prediction: {label} ({conf:.1%})")
        if label == selected_stage:
            play_beep()

# === Live camera processing ===
class LiveProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.size = model_info["input_size"]
        self.last_beep_time = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="rgb24")
        pil = Image.fromarray(img)
        arr = preprocess_image(pil, self.size)
        preds = classify(self.model, arr)
        label, conf = interpret_class(preds)

        draw = ImageDraw.Draw(pil)
        if label:
            draw.text((10, 10), f"{label} ({conf:.0%})", fill="red")
            if label == selected_stage:
                now = time.time()
                if now - self.last_beep_time > 2:  # Beep at most every 2s
                    self.last_beep_time = now
                    play_beep()
        return av.VideoFrame.from_ndarray(np.array(pil), format="rgb24")

st.subheader("📹 Live Camera Detection")
webrtc_streamer(
    key="live-dros-stage",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=LiveProcessor,
    async_processing=True,
)

st.markdown("---")
st.write(f"Model: `{HF_REPO_ID}/{MODEL_FILE}`")
