import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from huggingface_hub import hf_hub_download
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, WebRtcMode
import av, time, io, wave, struct, base64
from tensorflow.keras.applications.inception_v3 import preprocess_input

# ─── Page Setup ───────────────────────────────────────────────────────────
st.set_page_config(page_title="Drosophila Stage Detection", layout="centered")
st.title("Drosophila Stage Detection (Live + Upload)")
st.write("First, unlock audio; then upload or go live to hear beeps!")

# ─── Audio Unlock Section ───────────────────────────────────────────────────
if "audio_unlocked" not in st.session_state:
    st.session_state["audio_unlocked"] = False

def make_silence_wav():
    buf = io.BytesIO()
    with wave.open(buf,'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(44100)
        wf.writeframes(b'')  # no data = silence
    return base64.b64encode(buf.getvalue()).decode()

SILENCE_B64 = make_silence_wav()

if not st.session_state["audio_unlocked"]:
    st.warning("🔊 Click below to enable sound alerts")
    if st.button("Enable Sound Alerts"):
        # play a silent clip to unlock the audio context
        st.components.v1.html(f"""
            <audio id="unlock" autoplay>
              <source src="data:audio/wav;base64,{SILENCE_B64}" type="audio/wav">
            </audio>""", height=0)
        st.session_state["audio_unlocked"] = True
        st.experimental_rerun()
    st.stop()

# ─── Embed persistent <audio> for beeps ──────────────────────────────────────
def make_beep_b64():
    buf = io.BytesIO()
    with wave.open(buf,'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(44100)
        d,f,v = 0.2,800,0.8
        s = [int(v*32767*np.sin(2*np.pi*f*t/44100)) for t in range(int(44100*d))]
        wf.writeframes(b''.join(struct.pack('<h', x) for x in s))
    return base64.b64encode(buf.getvalue()).decode()

BEEP_B64 = make_beep_b64()

# inject once
st.components.v1.html(f"""
  <audio id="beeper" src="data:audio/wav;base64,{BEEP_B64}"></audio>
""", height=0)

def beep():
    st.components.v1.html("""
      <script>
        document.getElementById('beeper').play();
      </script>
    """, height=0)

# ─── Model & Labels ─────────────────────────────────────────────────────────
HF_REPO_ID = "RishiPTrial/my-model-name"
MODEL_FILE = "drosophila_inceptionv3_classifier.h5"
STAGES = ["egg","1st instar","2nd instar","3rd instar","white pupa","brown pupa","eye pupa"]
sel = st.selectbox("Select stage to alert on:", STAGES)

# ─── Load model ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    p = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILE)
    from tensorflow.keras.models import load_model as lm
    return lm(p, compile=False), 299
model, sz = load_model()

def predict(img_pil):
    arr = preprocess_input(np.asarray(img_pil.resize((sz,sz)).convert("RGB"), np.float32)[None])
    preds = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    return STAGES[idx], preds[idx]

# ─── Upload Section ─────────────────────────────────────────────────────────
st.subheader("📷 Upload Image")
up = st.file_uploader("", type=["jpg","png","jpeg"])
if up:
    im = Image.open(up).convert("RGB")
    st.image(im, use_column_width=True)
    label, conf = predict(im)
    st.success(f"{label} ({conf:.1%})")
    if label==sel:
        beep()

# ─── Live Section ───────────────────────────────────────────────────────────
st.subheader("📹 Live Camera")
class Proc(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame):
        img = frame.to_ndarray(format="rgb24")
        pil = Image.fromarray(img)
        lbl, cf = predict(pil)
        draw = ImageDraw.Draw(pil)
        draw.text((10,10), f"{lbl} {cf:.0%}", fill="red")
        if lbl==sel:
            beep()
        return av.VideoFrame.from_ndarray(np.array(pil), format="rgb24")

webrtc_streamer(
    key="live",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video":True,"audio":False},
    video_processor_factory=Proc,
    async_processing=True
)
