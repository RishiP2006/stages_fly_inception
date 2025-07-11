import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from huggingface_hub import hf_hub_download
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

st.set_page_config(page_title="Drosophila Stage Detection", layout="centered")
st.title("Drosophila Stage Detection")
st.write("Using the single deployed model for inference.")

# Hugging Face model info
HF_REPO_ID = "RishiPTrial/my-model-name"
MODEL_FILE = "drosophila_inceptionv3_classifier.h5"

# Life stage labels
STAGE_LABELS = [
    "egg",
    "1st instar",
    "2nd instar",
    "3rd instar",
    "white pupa",
    "brown pupa",
    "eye pupa"
]

# 1️⃣ Dropdown for user‑selected target stage
selected_stage = st.selectbox("Select stage to alert on:", STAGE_LABELS)

@st.cache_data(show_spinner=False)
def check_ultralytics():
    try:
        import ultralytics
        version = ultralytics.__version__ if hasattr(ultralytics, "__version__") else "unknown"
        st.info(f"Ultralytics installed, version: {version}")
        return True
    except Exception as e:
        st.warning(f"Ultralytics import failed: {e}")
        return False

_ULTRA_AVAILABLE = check_ultralytics()

@st.cache_resource(show_spinner=False)
def load_model():
    try:
        path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILE)
    except Exception as e:
        st.error(f"Error downloading model: {e}")
        return None

    if MODEL_FILE.lower().endswith(".h5"):
        from tensorflow.keras.models import load_model as lm
        model = lm(path, compile=False)
        return {"model": model, "type": "classification", "framework": "keras", "input_size": 299}
    return None

model_info = load_model()
if not model_info:
    st.stop()
model = model_info["model"]

def preprocess_image(img: Image.Image, size: int):
    from tensorflow.keras.applications.inception_v3 import preprocess_input
    arr = img.resize((size, size)).convert("RGB")
    x = np.asarray(arr).astype(np.float32)
    x = preprocess_input(x)
    return x

def classify(model, arr: np.ndarray):
    x = np.expand_dims(arr, axis=0)
    return model.predict(x)

def interpret_class(preds):
    if preds is None: return None, None
    arr = np.asarray(preds)
    if arr.ndim == 2 and arr.shape[1] == len(STAGE_LABELS):
        idx = int(np.argmax(arr, axis=1)[0])
        return STAGE_LABELS[idx], float(arr[0][idx])
    return None, None

# Helper to generate a short beep tone
@st.cache_data
def make_beep(duration_s=0.2, freq=440, sr=22050):
    t = np.linspace(0, duration_s, int(sr*duration_s), False)
    tone = 0.5 * np.sin(2 * np.pi * freq * t)
    return tone, sr

beep_wave, beep_sr = make_beep()

st.subheader("Upload Image")
img_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
if img_file:
    pil = Image.open(img_file).convert("RGB")
    st.image(pil, use_column_width=True)
    arr = preprocess_image(pil, model_info["input_size"])
    preds = classify(model, arr)
    st.write("Raw model prediction:", preds)
    label, conf = interpret_class(preds)
    if label:
        st.success(f"Prediction: {label} ({conf:.1%})")
        # 2️⃣ If prediction matches selected_stage → beep
        if label == selected_stage:
            st.audio(beep_wave, sample_rate=beep_sr)

st.markdown("---")
st.write(f"- Model from: {HF_REPO_ID} / {MODEL_FILE}")
