import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from huggingface_hub import hf_hub_download
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av, time, itertools
from tensorflow.keras.applications.inception_v3 import preprocess_input

# ─── Page Setup ─────────────────────────────────────────────────
st.set_page_config(page_title="Drosophila Stage Detection", layout="centered")
st.title("🪰 Drosophila Stage Detection (Live + Upload)")
st.write("Allow notifications once, then any match will pop up a desktop notification + on‑screen alert.")

# ─── Notification Permission Request ────────────────────────────
if "notif_enabled" not in st.session_state:
    st.session_state["notif_enabled"] = False

if not st.session_state["notif_enabled"]:
    st.warning("🔔 Click to enable desktop notifications")
    if st.button("Enable Notifications"):
        st.experimental_set_query_params(_=itertools.count())  # force a re‑run
        st.write("""
        <script>
          Notification.requestPermission().then(status => {
            window.parent.postMessage({ notif: status }, "*");
          });
        </script>
        """, unsafe_allow_html=True)
        st.session_state["notif_enabled"] = True
        st.experimental_rerun()
    st.stop()

# ─── JS Alert Function ───────────────────────────────────────────
def notify_js(title, message):
    js = f"""
    <script>
      if (Notification.permission === 'granted') {{
        new Notification("{title}", {{ body: "{message}" }});
      }}
    </script>
    """
    st.components.v1.html(js, height=0)

# ─── On‑screen banner ─────────────────────────────────────────────
def show_banner(msg):
    st.markdown(f"""
    <div style="background:#ff4b4b;color:white;padding:10px;
                border-radius:8px;margin:10px 0;font-size:18px;">
      🚨 {msg}
    </div>
    """, unsafe_allow_html=True)

# ─── Model & Helpers ─────────────────────────────────────────────
HF_REPO_ID = "RishiPTrial/my-model-name"
MODEL_FILE = "drosophila_inceptionv3_classifier.h5"
STAGES = ["egg","1st instar","2nd instar","3rd instar","white pupa","brown pupa","eye pupa"]
sel = st.selectbox("Select stage to alert on:", STAGES)

@st.cache_resource
def load_model():
    p = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILE)
    from tensorflow.keras.models import load_model as lm
    return lm(p, compile=False), 299

model, sz = load_model()

def predict(pil):
    img = np.asarray(pil.resize((sz,sz)).convert("RGB"), np.float32)
    arr = preprocess_input(img[None])
    preds = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    return STAGES[idx], preds[idx]

# ─── Upload Section ──────────────────────────────────────────────
st.subheader("📷 Upload Image")
up = st.file_uploader("", type=["jpg","jpeg","png"])
if up:
    pil = Image.open(up).convert("RGB")
    st.image(pil, use_column_width=True)
    lbl, cf = predict(pil)
    st.success(f"{lbl} ({cf:.1%})")
    if lbl == sel:
        notify_js("Drosophila Match!", f"{lbl} ({cf:.1%})")
        show_banner(f"Upload matched: {lbl} ({cf:.1%})")

# ─── Live Section ────────────────────────────────────────────────
st.subheader("📹 Live Camera Detection")
class P(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame):
        img = frame.to_ndarray(format="rgb24")
        pil = Image.fromarray(img)
        lbl, cf = predict(pil)
        draw = ImageDraw.Draw(pil)
        draw.text((10,10), f"{lbl} ({cf:.0%})", fill="red")
        if lbl == sel:
            notify_js("Drosophila Live Match!", f"{lbl} ({cf:.1%})")
            show_banner(f"Live matched: {lbl} ({cf:.1%})")
        return av.VideoFrame.from_ndarray(np.array(pil), format="rgb24")

webrtc_streamer(
    key="live",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video":True,"audio":False},
    video_processor_factory=P,
    async_processing=True
)
