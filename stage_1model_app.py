import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from huggingface_hub import hf_hub_download
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import time

st.set_page_config(page_title="Drosophila Stage Detection", layout="centered")
st.title("🪰 Drosophila Stage Detection")
st.write("Upload an image or use your webcam. Select a stage, and get a BIG visual ALERT when it matches!")

# --- Model & Labels ---
HF_REPO_ID = "RishiPTrial/my-model-name"
MODEL_FILE = "drosophila_inceptionv3_classifier.h5"
STAGE_LABELS = [
    "egg","1st instar","2nd instar","3rd instar",
    "white pupa","brown pupa","eye pupa"
]

# --- Dropdown target stage ---
selected_stage = st.selectbox("🎯 Select stage to alert on:", STAGE_LABELS)

# --- Session flag for live match alert ---
if "live_alert" not in st.session_state:
    st.session_state["live_alert"] = False

# --- Load Keras Model ---
@st.cache_resource
def load_model():
    path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILE)
    from tensorflow.keras.models import load_model as lm
    return lm(path, compile=False), 299

model, input_size = load_model()

# --- Helpers ---
from tensorflow.keras.applications.inception_v3 import preprocess_input
def preprocess_image(img):
    img = img.resize((input_size, input_size)).convert("RGB")
    arr = np.asarray(img, np.float32)
    return preprocess_input(arr)

def predict(arr):
    pred = model.predict(arr[np.newaxis], verbose=0)[0]
    idx = int(np.argmax(pred))
    return STAGE_LABELS[idx], float(pred[idx])

# --- Upload section ---
st.subheader("📷 Upload Image")
uploaded = st.file_uploader("", type=["jpg","jpeg","png"])
if uploaded:
    img = Image.open(uploaded)
    st.image(img, use_column_width=True)
    arr = preprocess_image(img)
    label, conf = predict(arr)
    st.success(f"Prediction: {label} ({conf:.1%})")

# --- Live camera section ---
st.subheader("📹 Live Camera Detection")
video_box = st.empty()             # placeholder for video
alert_box = st.empty()             # placeholder for alert banner

class CamProcessor(VideoProcessorBase):
    def __init__(self):
        self.last = 0
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="rgb24")
        pil = Image.fromarray(img)
        arr = preprocess_image(pil)
        label, conf = predict(arr)

        # Draw prediction text
        draw = ImageDraw.Draw(pil)
        draw.text((10,10), f"{label} {conf:.0%}", fill="red")

        # Trigger visual alert if match and cooldown >1s
        now = time.time()
        if label == selected_stage and now - self.last > 1.0:
            self.last = now
            st.session_state["live_alert"] = True

        return av.VideoFrame.from_ndarray(np.array(pil), format="rgb24")

webrtc_streamer(
    key="dros_live",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video":True, "audio":False},
    video_processor_factory=CamProcessor,
    async_processing=True,
    video_frame_callback=lambda frame: video_box.image(frame.to_ndarray(format="rgb24"))
)

# --- Show visual alert if flagged ---
if st.session_state["live_alert"]:
    alert_box.markdown(
        "<div style='background-color:#ff4b4b;color:white;padding:20px;"
        "font-size:32px;text-align:center;border-radius:8px;'>"
        "🚨 MATCHED: " + selected_stage.upper() + " 🚨</div>",
        unsafe_allow_html=True
    )
    # Clear after 1 second
    time.sleep(1.0)
    alert_box.empty()
    st.session_state["live_alert"] = False

st.markdown("---")
st.write(f"Model: `{HF_REPO_ID}/{MODEL_FILE}`")
