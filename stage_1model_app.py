import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av, time

# ─── Page Setup ─────────────────────────────────────────
st.set_page_config(page_title="Stable Live Prediction", layout="centered")
st.title("Live Drosophila Stage (Stable 3‑Frame)")

# ─── Model & Helpers ────────────────────────────────────
# (reuse your load_model(), preprocess_image(), classify() from before)

@st.cache_resource
def load_model():
    # ... your huggingface load here ...
    model = ...  # loaded Keras model
    return model, 299

model, input_size = load_model()

def preprocess_image(pil: Image.Image):
    img = pil.resize((input_size, input_size)).convert("RGB")
    arr = np.asarray(img, np.float32)
    from tensorflow.keras.applications.inception_v3 import preprocess_input
    return preprocess_input(arr)

def classify(arr: np.ndarray):
    preds = model.predict(arr[np.newaxis], verbose=0)[0]
    idx = int(np.argmax(preds))
    label = STAGE_LABELS[idx]
    return label

# ─── Session State for stability ────────────────────────
if "live_last_label" not in st.session_state:
    st.session_state["live_last_label"] = None
    st.session_state["live_count"]      = 0
    st.session_state["stable_label"]    = None

# ─── Live Processor ─────────────────────────────────────
STAGE_LABELS = [
    "egg","1st instar","2nd instar","3rd instar",
    "white pupa","brown pupa","eye pupa"
]

class StableProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="rgb24")
        pil = Image.fromarray(img)
        label = classify(preprocess_image(pil))

        # stability logic
        if label == st.session_state["live_last_label"]:
            st.session_state["live_count"] += 1
        else:
            st.session_state["live_last_label"] = label
            st.session_state["live_count"] = 1

        if st.session_state["live_count"] >= 3:
            st.session_state["stable_label"] = label

        # draw overlay every frame
        draw = ImageDraw.Draw(pil)
        draw.text((10,10), label, fill="red")

        return av.VideoFrame.from_ndarray(np.array(pil), format="rgb24")

# ─── Stream & Display ───────────────────────────────────
webrtc_ctx = webrtc_streamer(
    key="stable-live",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=StableProcessor,
    async_processing=True
)

# ─── Show the 3‑frame‑stable result ─────────────────────
st.markdown("### Stable prediction (3 consecutive frames):")
stable = st.session_state.get("stable_label")
if stable:
    st.success(f"🔒 {stable}")
else:
    st.info("Waiting for a stable 3‑frame prediction…")
