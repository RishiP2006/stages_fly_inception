import streamlit as st
st.set_page_config(layout="centered")
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model as lm
from tensorflow.keras.applications.inception_v3 import preprocess_input

# ─── Model Load ───────────────────────────
HF_REPO_ID = "RishiPTrial/my-model-name"
MODEL_FILE = "drosophila_inceptionv3_classifier.h5"
STAGE_LABELS = [
    "egg", "1st instar", "2nd instar", "3rd instar",
    "white pupa", "brown pupa", "eye pupa"
]

@st.cache_resource
def load_model():
    path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILE)
    return lm(path, compile=False), 299

model, input_size = load_model()

def preprocess_image(pil: Image.Image):
    pil = pil.resize((input_size, input_size)).convert("RGB")
    arr = np.asarray(pil, np.float32)
    return preprocess_input(arr)

def classify(pil: Image.Image):
    arr = preprocess_image(pil)
    preds = model.predict(arr[np.newaxis], verbose=0)[0]
    idx = int(np.argmax(preds))
    return STAGE_LABELS[idx], float(preds[idx])

# ─── UI Setup ─────────────────────────────

st.title("Live Drosophila Detection")
st.subheader("📹 Live Camera Detection with Stable Prediction")

# ─── Session State to Display Result ─────
if "stable_prediction" not in st.session_state:
    st.session_state["stable_prediction"] = "Waiting..."

# ─── Video Processor ─────────────────────
class StableProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_label = None
        self.count = 0
        self.stable_label = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="rgb24")
        pil = Image.fromarray(img)

        label, conf = classify(pil)

        # Check stability
        if label == self.last_label:
            self.count += 1
        else:
            self.last_label = label
            self.count = 1

        # Update stable label
        if self.count >= 3:
            self.stable_label = label

        # Draw prediction with black background box
        draw = ImageDraw.Draw(pil)
        text = f"{label} ({conf:.0%})"

        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=36)
        except:
            font = ImageFont.load_default()

        text_size = draw.textsize(text, font=font)
        x, y = 10, 10
        draw.rectangle([x, y, x + text_size[0] + 10, y + text_size[1] + 10], fill="black")
        draw.text((x + 5, y + 5), text, fill="red", font=font)

        # Save stable prediction
        if self.stable_label:
            st.session_state["stable_prediction"] = self.stable_label

        return av.VideoFrame.from_ndarray(np.array(pil), format="rgb24")

# ─── Stream Setup ────────────────────────
webrtc_ctx = webrtc_streamer(
    key="live",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=StableProcessor,
    async_processing=True
)

# ─── Show Stable Label ───────────────────
st.markdown("### 🧠 Stable Prediction (after 3 consistent frames):")
st.success(st.session_state.get("stable_prediction", "Waiting..."))
