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

# Life stage labels (✅ FIXED comma issue!)
STAGE_LABELS = [
    "egg",
    "1st instar",
    "2nd instar",
    "3rd instar",
    "white pupa",
    "brown pupa",
    "eye pupa"
]

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
        try:
            import keras
            from keras.models import load_model
            model = load_model(path, compile=False)
            return {"model": model, "type": "classification", "framework": "keras", "input_size": 299}
        except Exception as e:
            st.error(f"Failed loading Keras model: {e}")
            return None
    return None

model_info = load_model()
if not model_info:
    st.stop()
model = model_info["model"]

# Helper: Preprocessing
def preprocess_image(img: Image.Image, size: int):
    from keras.applications.inception_v3 import preprocess_input
    arr = img.resize((size, size)).convert("RGB")
    x = np.asarray(arr).astype(np.float32)
    x = preprocess_input(x)
    return x

# Helper: Prediction
def classify(model, arr: np.ndarray):
    x = np.expand_dims(arr, axis=0)
    try:
        import keras
        if isinstance(model, keras.Model):
            return model.predict(x)
    except Exception:
        pass
    return None

# ✅ Fixed interpretation: no softmax
def interpret_class(preds):
    if preds is None:
        return None, None
    arr = np.asarray(preds)
    if arr.ndim == 2 and arr.shape[1] == len(STAGE_LABELS):
        idx = int(np.argmax(arr, axis=1)[0])
        return STAGE_LABELS[idx], float(arr[0][idx])
    return None, None

# 🔴 Optional: Live detection code (not used here)
class GenderProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.info = model_info
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="rgb24")
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        if self.info["type"] == "classification":
            arr = preprocess_image(pil, self.info["input_size"])
            preds = classify(self.model, arr)
            label, conf = interpret_class(preds)
            if label:
                draw.text((10,10), f"{label} ({conf:.0%})", fill="red")
        return av.VideoFrame.from_ndarray(np.array(pil), format="rgb24")

# UI: Upload image
st.subheader("Upload Image")
img_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
if img_file:
    pil = Image.open(img_file).convert("RGB")
    st.image(pil, use_column_width=True)
    if model_info["type"] == "classification":
        arr = preprocess_image(pil, model_info["input_size"])
        preds = classify(model, arr)
        st.write("Raw model prediction:", preds)
        label, conf = interpret_class(preds)
        if label:
            st.success(f"Prediction: {label} ({conf:.1%})")

# (Optional) Live camera - only if needed
st.subheader("📸 Live Camera Detection")
webrtc_streamer(
     key="live-gender-detect",
     mode=WebRtcMode.SENDRECV,
     media_stream_constraints={"video": True, "audio": False},
     video_processor_factory=GenderProcessor,
     async_processing=True,
)

st.markdown("---")
st.write(f"- Model from: {HF_REPO_ID} / {MODEL_FILE}")
