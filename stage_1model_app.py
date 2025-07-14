import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from huggingface_hub import hf_hub_download
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av, time, io, wave, struct

# ─── Page Setup ────────────────
st.set_page_config(page_title="Drosophila Stage Detection", layout="centered")
st.title("Drosophila Stage Detection (Live + Upload)")
st.write("Upload an image or use your webcam. Select a stage, and get a beep when it matches.")

# ─── Model & Labels ─────────────
HF_REPO_ID = "RishiPTrial/my-model-name"
MODEL_FILE = "drosophila_inceptionv3_classifier.h5"
STAGE_LABELS = [
    "egg","1st instar","2nd instar","3rd instar",
    "white pupa","brown pupa","eye pupa"
]
selected_stage = st.selectbox("🎯 Select stage to alert on:", STAGE_LABELS)

if "live_match" not in st.session_state:
    st.session_state["live_match"] = False

# ─── Beep ───────────────────────
def make_beep_wav():
    buf = io.BytesIO()
    with wave.open(buf,'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(44100)
        duration, freq, volume = 0.2, 800, 0.8
        samples = [int(volume*32767*np.sin(2*np.pi*freq*t/44100))
                   for t in range(int(44100*duration))]
        wf.writeframes(b''.join(struct.pack('<h', s) for s in samples))
    return buf.getvalue()
beep_wav = make_beep_wav()
def play_beep():
    st.audio(beep_wav, format="audio/wav", start_time=0)

# ─── Load Model ─────────────────
@st.cache_resource
def load_model():
    path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILE)
    from tensorflow.keras.models import load_model as lm
    return lm(path, compile=False), 299

model, input_size = load_model()

from tensorflow.keras.applications.inception_v3 import preprocess_input
def preprocess_image(img):
    img = img.resize((input_size,input_size)).convert("RGB")
    arr = np.asarray(img, np.float32)
    return preprocess_input(arr)

def classify(arr):
    preds = model.predict(arr[np.newaxis], verbose=0)[0]
    idx = int(np.argmax(preds))
    return STAGE_LABELS[idx], float(preds[idx])

# ─── Image Upload ───────────────
st.subheader("📷 Upload Image for Detection")
uploaded = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])
if uploaded:
    pil = Image.open(uploaded).convert("RGB")
    st.image(pil, use_column_width=True)
    arr = preprocess_image(pil)
    label, conf = classify(arr)
    st.success(f"✅ Prediction: **{label}** ({conf:.1%})")
    if label == selected_stage:
        play_beep()

# ─── Live Camera ────────────────
st.subheader("📹 Live Camera Detection")
class LiveProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.last_time   = 0.0
        self.last_label  = None
        self.last_conf   = 0.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="rgb24")
        pil = Image.fromarray(img)
        self.frame_count += 1

        # Only classify every 3rd frame
        if self.frame_count % 3 == 0:
            arr = preprocess_image(pil)
            label, conf = classify(arr)
            self.last_label, self.last_conf = label, conf

            # beep flag
            now = time.time()
            if label == selected_stage and now - self.last_time > 2.0:
                self.last_time = now
                st.session_state["live_match"] = True

        # draw overlay using last_label, last_conf
        if self.last_label is not None:
            draw = ImageDraw.Draw(pil)
            text = f"{self.last_label} ({self.last_conf:.0%})"
            draw.text((10,10), text, fill="red")

        return av.VideoFrame.from_ndarray(np.array(pil), format="rgb24")

webrtc_streamer(
    key="live-dros-stage",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video":True, "audio":False},
    video_processor_factory=LiveProcessor,
    async_processing=True,  # allow dropping frames
)

# ─── Play beep if flagged ───────
if st.session_state["live_match"]:
    st.session_state["live_match"] = False
    play_beep()

st.markdown("---")
st.write(f"Model: `{HF_REPO_ID}/{MODEL_FILE}`")
