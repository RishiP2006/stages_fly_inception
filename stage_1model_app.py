import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from huggingface_hub import hf_hub_download
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av, io, wave, struct, time

st.set_page_config(page_title="Drosophila Stage Detection", layout="centered")
st.title("🪰 Drosophila Stage Detection")
st.write("Upload an image or use your webcam. Select a stage, and hear a beep when it matches!")

# --- Config ---
HF_REPO_ID = "RishiPTrial/my-model-name"
MODEL_FILE = "drosophila_inceptionv3_classifier.h5"
STAGE_LABELS = [
    "egg","1st instar","2nd instar","3rd instar",
    "white pupa","brown pupa","eye pupa"
]

# --- Dropdown ---
selected_stage = st.selectbox("🎯 Select stage to alert on:", STAGE_LABELS)

# --- Generate beep WAV once ---
def make_beep_wav(duration=0.2, freq=800, sr=44100, volume=0.8):
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        samples = (volume * np.sin(2*np.pi*freq*np.arange(sr*duration)/sr)).astype(np.float32)
        # convert to int16
        ints = (samples * 32767).astype(np.int16)
        wf.writeframes(ints.tobytes())
    return buf.getvalue()

beep_wav = make_beep_wav(duration=0.2, freq=800)

# --- Session flag for live beep ---
if 'beep_live' not in st.session_state:
    st.session_state['beep_live'] = False

# --- Load model from HF ---
@st.cache_resource
def load_model():
    path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILE)
    from tensorflow.keras.models import load_model as lm
    return lm(path, compile=False), 299

model, input_size = load_model()

# --- Helpers ---
from tensorflow.keras.applications.inception_v3 import preprocess_input
def preprocess_image(img):
    img = img.resize((input_size,input_size)).convert("RGB")
    arr = np.asarray(img, np.float32)
    return preprocess_input(arr)

def predict(arr):
    pred = model.predict(arr[np.newaxis,:,:,:], verbose=0)[0]
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
    if label == selected_stage:
        st.audio(beep_wav, format='audio/wav')

# --- Live camera section ---
st.subheader("📹 Live Camera Detection")

class CamProcessor(VideoProcessorBase):
    def __init__(self):
        self.last = 0
    def recv(self, frame):
        img = frame.to_ndarray(format="rgb24")
        pil = Image.fromarray(img)
        arr = preprocess_image(pil)
        label, conf = predict(arr)
        draw = ImageDraw.Draw(pil)
        draw.text((10,10), f"{label} {conf:.0%}", fill="red")
        now = time.time()
        if label == selected_stage and now - self.last > 1.0:
            self.last = now
            st.session_state['beep_live'] = True
        return av.VideoFrame.from_ndarray(np.array(pil), format="rgb24")

webrtc_streamer(
    key="dros_live",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video":True, "audio":False},
    video_processor_factory=CamProcessor,
    async_processing=True
)

# After camera, if beep flag, play and rerun
if st.session_state['beep_live']:
    st.session_state['beep_live'] = False
    st.audio(beep_wav, format='audio/wav')
    st.experimental_rerun()

st.markdown("---")
st.write(f"Model: `{HF_REPO_ID}/{MODEL_FILE}`")
