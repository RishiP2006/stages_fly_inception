import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from huggingface_hub import hf_hub_download
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av, time, io, wave, struct

st.set_page_config(page_title="Drosophila Stage Detection", layout="centered")
st.title("🪰 Drosophila Stage Detection")
st.write("Upload an image or use your webcam. Select a stage, and see a BIG ALERT when it matches!")

# --- Model & Labels ---
HF_REPO_ID = "RishiPTrial/my-model-name"
MODEL_FILE  = "drosophila_inceptionv3_classifier.h5"
STAGE_LABELS = [
    "egg","1st instar","2nd instar","3rd instar",
    "white pupa","brown pupa","eye pupa"
]

# --- Dropdown target stage ---
selected_stage = st.selectbox("🎯 Select stage to alert on:", STAGE_LABELS)

# --- State for triggering the alert banner ---
if "show_alert" not in st.session_state:
    st.session_state["show_alert"] = False

# --- Load model once ---
@st.cache_resource
def load_model():
    path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILE)
    from tensorflow.keras.models import load_model as lm
    return lm(path, compile=False), 299

model, input_size = load_model()

# --- Preprocess & predict helpers ---
from tensorflow.keras.applications.inception_v3 import preprocess_input
def preprocess_image(img):
    img = img.resize((input_size,input_size)).convert("RGB")
    arr = np.asarray(img, np.float32)
    return preprocess_input(arr)

def predict(arr):
    preds = model.predict(arr[np.newaxis], verbose=0)[0]
    idx   = int(np.argmax(preds))
    return STAGE_LABELS[idx], float(preds[idx])

# --- Function to set alert flag ---
def trigger_alert():
    st.session_state["show_alert"] = True

# --- Display visual alert if flagged ---
def show_alert_banner():
    if st.session_state["show_alert"]:
        st.markdown(
            "<div style='background-color:#ff4b4b;color:white;padding:20px;"
            "font-size:32px;text-align:center;border-radius:8px;margin:10px 0;'>"
            f"🚨 MATCHED: {selected_stage.upper()} 🚨"
            "</div>",
            unsafe_allow_html=True
        )
        # clear after short pause
        time.sleep(1)
        st.session_state["show_alert"] = False

# --- Upload section ---
st.subheader("📷 Upload Image")
uploaded = st.file_uploader(
    label="Choose an image...",
    type=["jpg","jpeg","png"],
    label_visibility="visible"
)
if uploaded:
    img = Image.open(uploaded)
    st.image(img, use_column_width=True)
    arr = preprocess_image(img)
    label, conf = predict(arr)
    st.success(f"Prediction: **{label}** ({conf:.1%})")
    if label == selected_stage:
        trigger_alert()

# Show alert for upload (if any)
show_alert_banner()

# --- Live webcam section ---
st.subheader("📹 Live Camera Detection")
video_placeholder = st.empty()

class CamProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_time = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="rgb24")
        pil = Image.fromarray(img)
        arr = preprocess_image(pil)
        label, conf = predict(arr)

        # draw prediction
        draw = ImageDraw.Draw(pil)
        draw.text((10,10), f"{label} ({conf:.0%})", fill="red")

        # if match and cooldown >1s, set alert
        now = time.time()
        if label == selected_stage and (now - self.last_time) > 1.0:
            self.last_time = now
            trigger_alert()

        return av.VideoFrame.from_ndarray(np.array(pil), format="rgb24")

webrtc_streamer(
    key="dros_live",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video":True, "audio":False},
    video_processor_factory=CamProcessor,
    async_processing=True,
    video_frame_callback=lambda frame: video_placeholder.image(
        frame.to_ndarray(format="rgb24"),
        channels="RGB"
    )
)

# Show alert for live (if any)
show_alert_banner()

st.markdown("---")
st.write(f"Model: `{HF_REPO_ID}/{MODEL_FILE}`")
