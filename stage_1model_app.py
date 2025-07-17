import streamlit as st
st.set_page_config(layout="centered")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model as lm
from tensorflow.keras.applications.inception_v3 import preprocess_input

# ─── Load Model ────────────────────────────
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

# ─── UI ─────────────────────────────────────
st.title("Live Drosophila Detection")
st.subheader("📹 Live Camera Detection")

placeholder = st.empty()

# ─── Video Processor ────────────────────────
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.pred_label = "Loading..."
        self.confidence = 0.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        self.frame_count += 1
        img = frame.to_ndarray(format="rgb24")

        # Run prediction every 10 frames to reduce lag
        if self.frame_count % 10 == 0:
            pil_img = Image.fromarray(img)
            label, conf = classify(pil_img)
            self.pred_label = label
            self.confidence = conf

        # Draw prediction
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        font_size = 30
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()

        text = f"{self.pred_label} ({self.confidence:.0%})"
        text_size = draw.textbbox((0, 0), text, font=font)
        text_width = text_size[2] - text_size[0]
        text_height = text_size[3] - text_size[1]

        # Draw black rectangle background
        draw.rectangle([10, 10, 10 + text_width + 10, 10 + text_height + 10], fill="black")
        draw.text((15, 15), text, font=font, fill="white")

        # Update text in Streamlit (optional)
        placeholder.success(f"🧠 Prediction: **{self.pred_label}** ({self.confidence:.0%})")

        return av.VideoFrame.from_ndarray(np.array(pil_img), format="rgb24")

# ─── Start Stream ───────────────────────────
webrtc_ctx = webrtc_streamer(
    key="live",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 640},
            "height": {"ideal": 480},
            "frameRate": {"ideal": 30}
        },
        "audio": False
    },
    video_processor_factory=VideoProcessor,
    async_processing=True
)
