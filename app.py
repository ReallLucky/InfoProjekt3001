import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import joblib
import tempfile
import os
import plotly.graph_objects as go

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =========================
# CONFIG
# =========================
WINDOW_SIZE = 15
STEP_SIZE   = 5
MODEL_PATH  = "pose_landmarker_full.task"

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    knn           = joblib.load("eskrima_knn_model.pkl")
    scaler        = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return knn, scaler, label_encoder

# =========================
# MEDIAPIPE
# =========================
@st.cache_resource
def load_landmarker():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options      = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
    )
    return vision.PoseLandmarker.create_from_options(options)

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    layout="wide",
    page_title="Eskrima AI Coach",
    page_icon=":material/swords:"
)

st.markdown("""
<style>

/* ===== Hide Streamlit Top Bar / Toolbar ===== */
header {
    visibility: hidden;
    height: 0px;
}

/* Hide hamburger menu */
#MainMenu {
    visibility: hidden;
}

/* Hide footer */
footer {
    visibility: hidden;
}

/* Remove top padding */
.block-container {
    padding-top: 1rem;
}

/* Optional: remove deploy button spacing */
[data-testid="stToolbar"] {
    display: none;
}

@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&family=Share+Tech+Mono&display=swap');

[data-testid="stApp"] {
    background: #08090f;
    color: #dde2f0;
    font-family: 'Rajdhani', sans-serif;
}

[data-testid="stSidebar"],
[data-testid="collapsedControl"] {
    display: none !important;
}

h1, h2, h3, h4 {
    font-family: 'Rajdhani', sans-serif !important;
    letter-spacing: 0.04em;
}

/* Tips button */
div[data-testid="stButton"] button {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px;
    color: rgba(200,215,255,0.7);
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.9rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    transition: background 0.15s, border-color 0.15s;
}
div[data-testid="stButton"] button:hover {
    background: rgba(60,120,255,0.12);
    border-color: rgba(60,120,255,0.4);
    color: #aaccff;
}

[data-testid="stFileUploader"] section {
    border: 1.5px dashed rgba(60,120,255,0.35) !important;
    border-radius: 14px !important;
    background: rgba(40,80,255,0.04) !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"] section:hover {
    border-color: rgba(60,120,255,0.65) !important;
}

[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #1a5fff, #55aaff) !important;
    border-radius: 999px !important;
}

[data-testid="stProgress"] {
    text-align: center;
}

.score-card {
    background: linear-gradient(135deg, rgba(30,80,200,0.18) 0%, rgba(0,0,0,0) 60%);
    border: 1px solid rgba(60,120,255,0.2);
    border-radius: 16px;
    padding: 26px 30px 22px;
    margin-bottom: 20px;
}
.score-label {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: rgba(140,170,255,0.55);
    margin-bottom: 4px;
}
.score-number {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 4.2rem;
    font-weight: 700;
    line-height: 1;
    background: linear-gradient(90deg, #5599ff, #aad4ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.score-verdict-ok  { color: #44dd88; font-size: 1rem; font-weight: 600; margin-top: 8px; }
.score-verdict-bad { color: #ff5544; font-size: 1rem; font-weight: 600; margin-top: 8px; }
.score-meta {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.73rem;
    color: rgba(180,200,255,0.3);
    margin-top: 10px;
}

.stat-row { display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }
.stat-pill {
    flex: 1;
    min-width: 80px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 10px 14px;
    font-family: 'Rajdhani', sans-serif;
}
.stat-pill .sp-val { font-size: 1.5rem; font-weight: 700; line-height: 1.1; font-family: 'Rajdhani', sans-serif !important; }
.stat-pill .sp-lbl { font-size: 0.72rem; color: rgba(180,200,255,0.4); margin-top: 2px; }
.stat-ok   .sp-val { color: #44dd88; }
.stat-err  .sp-val { color: #ff6655; }
.stat-idle .sp-val { color: #7799dd; }

.section-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: rgba(160,185,255,0.5);
    margin: 20px 0 10px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    padding-bottom: 6px;
}

b {
    font-family: 'Rajdhani', sans-serif;
}

/* =========================
   VIDEO FIT SCREEN
========================= */

[data-testid="stVideo"] {
    height: calc(100vh - 120px);
    display: flex;
    align-items: center;
    justify-content: center;
    position: sticky;
    top: 1rem;
}

/* wrapper */
[data-testid="stVideo"] > div {
    width: 100%;
    height: 100%;
}

/* actual video */
[data-testid="stVideo"] video {
    width: 100%;
    height: 100%;
    max-height: calc(100vh - 120px);
    object-fit: contain;

    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.07);
    background: #000;

    display: block;
}
</style>
""", unsafe_allow_html=True)

# =========================
# TIPS DIALOG
# =========================
@st.dialog("Tips")
def tips_dialog():
    st.markdown("""
### :material/checklist: Recording tips
- Film from a **side or 45° angle**
- Keep your **full body in frame**
- Use **good, even lighting**
- Supported formats: mp4, mov, avi
- 720p or higher recommended

---
### <span style="color:#aaccff;">:material/label:</span> Labels

<div style="margin-top:8px; line-height:1.8">

<span style="color:#44dd88;">:material/check_circle:</span>
<b style="color:#44dd88;"> Correct</b><br>

<span style="color:#ff6655;">:material/cancel:</span>
<b style="color:#ff6655;"> Too Early</b><br>

<span style="color:#ffaa22;">:material/warning:</span>
<b style="color:#ffaa22;"> Shoulder Back</b><br>

<span style="color:#7799dd;">:material/pause_circle:</span>
<b style="color:#7799dd;"> Idle</b>

</div>
""", unsafe_allow_html=True)

# =========================
# HELPERS
# =========================
def get_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps and fps > 1 else 30.0


def extract_frames(video_path, progress_placeholder, start_progress):
    cap        = cv2.VideoCapture(video_path)
    landmarker = load_landmarker()
    frames     = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_range = 0.6  # from 0.2 to 0.8
    i = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = landmarker.detect(mp_image)

        if result.pose_landmarks:
            lm  = result.pose_landmarks[0]
            row = []
            for p in lm:
                row.extend([p.x, p.y, p.z])
            frames.append(row)

        i += 1
        if total_frames > 0:
            progress = start_progress + (i / total_frames) * progress_range
            progress_placeholder.progress(progress, text=" Extracting poses… ")

    cap.release()
    return frames


def create_sequences(frames):
    sequences = []
    for i in range(0, len(frames) - WINDOW_SIZE, STEP_SIZE):
        sequences.append(np.array(frames[i:i + WINDOW_SIZE]).flatten())
    return np.array(sequences)


def classify_label(l):
    if "correct"       in l: return "correct"
    if "too_early"     in l: return "too_early"
    if "shoulder_back" in l: return "shoulder_back"
    return "idle"


# =========================
# PROCESS (cached by video bytes)
# =========================
def process_video(video_bytes, progress_placeholder):
    progress_placeholder.progress(0, text=" Analysing video… ")
    knn, scaler, label_encoder = load_model()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        path = tmp.name

    progress_placeholder.progress(0.1, text=" Preparing video… ")
    fps = get_fps(path)
    progress_placeholder.progress(0.2, text=" Extracting poses… ")
    frames = extract_frames(path, progress_placeholder, 0.2)

    if len(frames) < WINDOW_SIZE:
        return None, None, fps, path

    progress_placeholder.progress(0.8, text=" Classifying… ")
    sequences = create_sequences(frames)
    X = scaler.transform(sequences)
    preds = knn.predict(X)
    labels = label_encoder.inverse_transform(preds)

    frame_time = STEP_SIZE / fps
    timestamps = [i * frame_time for i in range(len(labels))]

    progress_placeholder.progress(1.0, text=" Done! ")
    return labels, timestamps, fps, path


# =========================
# TIMELINE CHART
# =========================
def build_chart(labels, timestamps):
    STYLE = {
        "correct":       (1.0,  "#44dd88", "Correct"),
        "too_early":     (-1.0, "#ff5544", "Too Early"),
        "shoulder_back": (-0.5, "#ffaa22", "Shoulder Back"),
        "idle":          (0.0,  "#334466", "Idle"),
    }

    values, colors, texts = [], [], []
    for l in labels:
        k = classify_label(l)
        val, col, txt = STYLE[k]
        values.append(val)
        colors.append(col)
        texts.append(txt)

    fig = go.Figure()

    fig.add_hrect(y0=0.6,  y1=1.4,  fillcolor="rgba(50,220,120,0.05)", line_width=0)
    fig.add_hrect(y0=-1.4, y1=-0.2, fillcolor="rgba(255,80,60,0.05)",  line_width=0)

    fig.add_trace(go.Scatter(
        x=timestamps, y=values,
        mode="lines",
        line=dict(color="rgba(255,255,255,0.08)", width=1.5),
        showlegend=False,
    ))

    fig.add_trace(go.Scatter(
        x=timestamps, y=values,
        mode="markers",
        marker=dict(color=colors, size=8, line=dict(width=0.8, color="rgba(255,255,255,0.25)")),
        text=texts,
        hovertemplate="<b>%{text}</b><br>t = %{x:.2f} s<extra></extra>",
        showlegend=False,
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(180,200,255,0.6)", size=11, family='Rajdhani, sans-serif'),
        xaxis=dict(title="Time (s)", gridcolor="rgba(255,255,255,0.04)", zeroline=False),
        yaxis=dict(
            tickvals=[-1, -0.5, 0, 1],
            ticktext=["Too Early", "Shoulder Back", "Idle", "Correct"],
            gridcolor="rgba(255,255,255,0.04)",
            zeroline=False,
            range=[-1.5, 1.5],
        ),
        margin=dict(l=10, r=10, t=6, b=36),
        height=280,
    )
    return fig


# =========================
# HEADER
# =========================
title_col, tips_col = st.columns([8, 1])

with title_col:
    st.markdown("# :material/swords: Eskrima AI Coach")
    st.markdown(
        "<p style='color:rgba(180,200,255,0.4);font-size:.9rem;margin-top:-10px;margin-bottom:20px'>"
        "Upload a strike clip — the model scores your form frame by frame."
        "</p>",
        unsafe_allow_html=True,
    )

with tips_col:
    st.markdown("<div style='margin-top:14px'>", unsafe_allow_html=True)
    if st.button(
            "Tips",
            icon=":material/lightbulb:",
            use_container_width=True
    ):
        tips_dialog()
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# UPLOAD
# =========================
uploaded_file = st.file_uploader(
    "Upload your strike video",
    type=["mp4", "mov", "avi"],
    label_visibility="collapsed",
)

# =========================
# PROCESS & CACHE IN SESSION STATE
# =========================
if uploaded_file:
    video_bytes = uploaded_file.read()
    file_id     = uploaded_file.file_id

    if st.session_state.get("file_id") != file_id:
        progress_placeholder = st.empty()
        labels, timestamps, fps, video_path = process_video(video_bytes, progress_placeholder)
        progress_placeholder.empty()

        st.session_state.file_id     = file_id
        st.session_state.labels      = labels
        st.session_state.timestamps  = timestamps
        st.session_state.fps         = fps
        st.session_state.video_path  = video_path
        st.session_state.video_bytes = video_bytes  # ← store bytes for recovery

    # Restore temp file if the OS deleted it between reruns
    if not os.path.exists(st.session_state.video_path):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(st.session_state.video_bytes)
            st.session_state.video_path = tmp.name

# Render from session state so Tips button reruns don't wipe the UI
if st.session_state.get("labels") is not None and uploaded_file:

    labels     = st.session_state.labels
    timestamps = st.session_state.timestamps
    fps        = st.session_state.fps
    video_path = st.session_state.video_path

    if labels is None:
        st.error(
            "⚠️ Not enough pose data detected. Make sure your full body is visible "
            "and the lighting is adequate.",
            icon=":material/block:",
        )
        st.stop()

    # ── Scoring ──────────────────────────────────
    total         = len(labels)
    correct_count = sum("correct"        in l for l in labels)
    error_count   = sum("too_early"      in l or "shoulder_back" in l for l in labels)
    idle_count    = sum(classify_label(l) == "idle" for l in labels)
    correct_ratio = correct_count / total if total else 0
    score         = round(correct_ratio * 100, 1)

    verdict_class = "score-verdict-ok"  if correct_ratio >= 0.5 else "score-verdict-bad"
    verdict_text = """
    <span style="display:flex;align-items:center;gap:6px;">
        <span style="color:#44dd88;">●</span>
        Good execution
    </span>
    """ if correct_ratio >= 0.5 else """
<span style="display:flex;align-items:center;gap:6px;">
    <span style="color:#ff5544;">●</span>
    Needs improvement
</span>
"""

    # ── Layout ───────────────────────────────────
    left, right = st.columns([1, 1], gap="large")

    with left:
        # Score card
        st.markdown(f"""
<div class="score-card">
  <div class="score-label">Execution score</div>
  <div class="score-number">{score}%</div>
  <div class="{verdict_class}">{verdict_text}</div>
  <div class="score-meta">{total} windows &nbsp;·&nbsp; source {fps:.0f} fps</div>
</div>""", unsafe_allow_html=True)

        # Stat pills
        correct_pct = round(correct_count / total * 100) if total else 0
        error_pct   = round(error_count   / total * 100) if total else 0

        st.markdown(f"""
<div class="stat-row">
  <div class="stat-pill stat-ok">
    <div class="sp-val">{correct_pct}%</div>
    <div class="sp-lbl">Correct</div>
  </div>
  <div class="stat-pill stat-err">
    <div class="sp-val">{error_pct}%</div>
    <div class="sp-lbl">Errors</div>
  </div>
</div>""", unsafe_allow_html=True)

        # Timeline
        st.markdown('<div class="section-title">Movement timeline</div>', unsafe_allow_html=True)
        st.plotly_chart(build_chart(labels, timestamps), use_container_width=True)

    with right:
        st.markdown('<div class="section-title">Your video</div>', unsafe_allow_html=True)
        st.video(video_path)