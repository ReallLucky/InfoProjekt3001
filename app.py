import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import joblib
import tempfile
import time

import plotly.graph_objects as go
import tensorflow as tf

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =========================
# CONFIG — must match train.py
# =========================
WINDOW_SIZE       = 20
STEP_SIZE         = 5
ANALYSIS_FPS      = 15
RESIZE_W, RESIZE_H = 640, 360
CONFIDENCE_THRESH = 0.55

N_LANDMARKS    = 33
N_COORDS       = 3
RAW_FEATURES   = N_LANDMARKS * N_COORDS      # 99
TOTAL_FEATURES = RAW_FEATURES * 2            # 198

# =========================
# LOAD ARTIFACTS
# =========================
@st.cache_resource
def load_artifacts():
    model         = tf.keras.models.load_model("eskrima_lstm_model.keras")
    scaler        = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, scaler, label_encoder

# =========================
# MEDIAPIPE — VIDEO MODE
# One instance per video analysis run.
# =========================
def make_landmarker():
    base_options = python.BaseOptions(model_asset_path="pose_landmarker_full.task")
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


# =========================
# NORMALIZATION (identical to train.py)
# =========================
def normalize_landmarks(row: np.ndarray) -> np.ndarray:
    pts           = row.reshape(N_LANDMARKS, N_COORDS)
    left_hip      = pts[23]
    right_hip     = pts[24]
    left_shoulder = pts[11]
    origin        = (left_hip + right_hip) / 2.0
    scale         = np.linalg.norm(left_shoulder - origin) + 1e-6
    return ((pts - origin) / scale).flatten()


def add_velocity(frames: list[np.ndarray]) -> list[np.ndarray]:
    result = []
    for i, f in enumerate(frames):
        vel = frames[i] - frames[i - 1] if i > 0 else np.zeros(RAW_FEATURES)
        result.append(np.concatenate([f, vel]))
    return result


# =========================
# FRAME EXTRACTION (VIDEO mode)
# =========================
def extract_frames(video_path: str, progress_cb=None) -> list[np.ndarray]:
    cap        = cv2.VideoCapture(video_path)
    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval   = max(1, int(round(source_fps / ANALYSIS_FPS)))

    landmarker = make_landmarker()
    frames     = []
    idx        = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if idx % interval == 0:
            small = cv2.resize(frame, (RESIZE_W, RESIZE_H))
            rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            mp_im = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int((idx / source_fps) * 1000)

            result = landmarker.detect_for_video(mp_im, ts_ms)
            if result.pose_landmarks:
                lm  = result.pose_landmarks[0]
                raw = np.array([[p.x, p.y, p.z] for p in lm]).flatten()
                frames.append(normalize_landmarks(raw))

            if progress_cb:
                progress_cb(min(idx / total, 0.9))

        idx += 1

    cap.release()
    landmarker.close()
    return frames


# =========================
# INFERENCE
# =========================
def run_inference(frames, model, scaler, label_encoder, source_fps):
    frames_with_vel = add_velocity(frames)
    sequences       = []
    frame_indices   = []

    for i in range(0, len(frames_with_vel) - WINDOW_SIZE, STEP_SIZE):
        window = np.array(frames_with_vel[i : i + WINDOW_SIZE])
        sequences.append(window)
        frame_indices.append(i)

    if not sequences:
        return [], [], []

    X = np.array(sequences)                   # (N, WINDOW, FEATURES)
    N, T, F = X.shape
    X_flat  = scaler.transform(X.reshape(N, T * F))
    X       = X_flat.reshape(N, T, F)

    proba      = model.predict(X, batch_size=32, verbose=0)
    confidence = proba.max(axis=1)
    pred_idx   = np.argmax(proba, axis=1)
    labels     = label_encoder.inverse_transform(pred_idx)

    # Convert frame index → timestamp in seconds
    # frames were sampled at ANALYSIS_FPS, so each frame = 1/ANALYSIS_FPS s
    timestamps = [fi / ANALYSIS_FPS for fi in frame_indices]

    return labels, timestamps, confidence


# =========================
# SCORING
# =========================
def compute_score(labels, confidence):
    correct_windows = sum(
        1 for l, c in zip(labels, confidence)
        if "correct" in l and c >= CONFIDENCE_THRESH
    )
    counted_windows = sum(
        1 for c in confidence if c >= CONFIDENCE_THRESH
    )
    ratio = correct_windows / counted_windows if counted_windows else 0
    return round(ratio * 100, 1), ratio


def describe_errors(labels, timestamps, confidence):
    """Return list of (timestamp, error_type) for high-confidence errors only."""
    errors = []
    for l, t, c in zip(labels, timestamps, confidence):
        if c < CONFIDENCE_THRESH or "correct" in l or "idle" in l:
            continue
        if "too_early" in l:
            errors.append((t, "Moved too early"))
        elif "shoulder_back" in l:
            errors.append((t, "Shoulder pulled back"))
    # Collapse nearby errors (within 1s) to avoid flooding
    collapsed = []
    last_t    = -999
    for t, msg in errors:
        if t - last_t > 1.0:
            collapsed.append((t, msg))
            last_t = t
    return collapsed


# =========================
# PLOTTING
# =========================
def build_timeline(labels, timestamps, confidence):
    COLOR_MAP = {
        "idle":           ("blue",   0.0),
        "correct":        ("green",  1.0),
        "too_early":      ("red",   -1.0),
        "shoulder_back":  ("orange",-0.5),
    }

    colors, values, texts, ts_plot = [], [], [], []

    for l, t, c in zip(labels, timestamps, confidence):
        if c < CONFIDENCE_THRESH:
            # Show low-confidence windows as faded gray
            colors.append("rgba(150,150,150,0.3)")
            values.append(0)
            texts.append(f"Low confidence ({c:.0%})")
        else:
            matched = next((k for k in COLOR_MAP if k in l), None)
            col, val = COLOR_MAP.get(matched, ("gray", 0))
            colors.append(col)
            values.append(val)
            texts.append(l.replace("_", " ").title() + f" ({c:.0%})")
        ts_plot.append(t)

    fig = go.Figure()

    # Shaded regions
    fig.add_hrect(y0=0.7,  y1=1.3,  fillcolor="rgba(0,200,80,0.07)",  line_width=0)
    fig.add_hrect(y0=-1.3, y1=-0.3, fillcolor="rgba(220,50,50,0.07)", line_width=0)

    # Line
    fig.add_trace(go.Scatter(
        x=ts_plot, y=values,
        mode="lines",
        line=dict(color="rgba(255,255,255,0.2)", width=1),
        showlegend=False,
    ))

    # Markers
    fig.add_trace(go.Scatter(
        x=ts_plot, y=values,
        mode="markers",
        marker=dict(color=colors, size=9, line=dict(width=1, color="white")),
        text=texts,
        hovertemplate="<b>%{text}</b><br>t = %{x:.2f}s<extra></extra>",
        showlegend=False,
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="monospace"),
        xaxis=dict(title="Time (s)", gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(
            tickvals=[-1, -0.5, 0, 1],
            ticktext=["Too Early", "Shoulder Back", "Idle", "Correct"],
            gridcolor="rgba(255,255,255,0.06)",
        ),
        margin=dict(l=10, r=10, t=10, b=40),
        height=320,
    )

    return fig


# =========================
# PAGE CONFIG + STYLE
# =========================
st.set_page_config(layout="wide", page_title="Eskrima AI Coach")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Share+Tech+Mono&display=swap');

html, body, [data-testid="stApp"] {
    background: #07080f;
    color: #e8eaf0;
}

[data-testid="stApp"] {
    background:
        radial-gradient(ellipse 80% 40% at 50% 100%, rgba(0,120,255,0.12), transparent),
        radial-gradient(ellipse 40% 60% at 90% 10%, rgba(255,60,0,0.06), transparent),
        #07080f;
}

h1, h2, h3 {
    font-family: 'Rajdhani', sans-serif !important;
    letter-spacing: 0.05em;
}

code, .mono { font-family: 'Share Tech Mono', monospace; }

/* Score box */
.score-box {
    background: linear-gradient(135deg, rgba(0,120,255,0.15), rgba(0,0,0,0));
    border: 1px solid rgba(0,120,255,0.3);
    border-radius: 12px;
    padding: 24px 28px;
    margin-bottom: 20px;
    font-family: 'Rajdhani', sans-serif;
}
.score-number {
    font-size: 4rem;
    font-weight: 700;
    line-height: 1;
    background: linear-gradient(90deg, #4af, #07f);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.verdict-ok  { color: #3f9; font-size: 1.2rem; font-weight: 600; }
.verdict-bad { color: #f54; font-size: 1.2rem; font-weight: 600; }

/* Error item */
.error-item {
    background: rgba(220,50,50,0.08);
    border-left: 3px solid #f54;
    border-radius: 4px;
    padding: 8px 14px;
    margin: 6px 0;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.85rem;
}

/* Upload zone */
[data-testid="stFileUploader"] {
    border: 1px dashed rgba(0,120,255,0.35) !important;
    border-radius: 12px;
    background: rgba(0,120,255,0.04) !important;
}

/* Progress bar */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #07f, #4af) !important;
}

/* Metric */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.03);
    border-radius: 8px;
    padding: 12px;
}

video {
    width: 100%;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.08);
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("# 🥋 Eskrima AI Coach")
st.markdown(
    "<span style='opacity:.5;font-size:.9rem'>Upload a strike video — the model will analyze "
    "your form frame-by-frame and score your execution.</span>",
    unsafe_allow_html=True
)
st.markdown("---")

# =========================
# SIDEBAR — tips
# =========================
with st.sidebar:
    st.markdown("### 📋 Recording tips")
    st.markdown("""
- Film from a **side or 45° angle**
- Keep your **full body visible**
- Ensure **good lighting** (avoid backlight)
- Supported: `mp4`, `mov`, `avi`
- Best results: 720p or higher
    """)
    st.markdown("### 🏷️ What the model detects")
    st.markdown("""
| Label | Meaning |
|---|---|
| ✅ Correct | Proper form |
| ❌ Too Early | Premature movement |
| ⚠️ Shoulder Back | Shoulder pulled behind hip |
| ⬜ Idle | No strike detected |
    """)

# =========================
# UPLOAD
# =========================
uploaded = st.file_uploader(
    "Upload your strike video",
    type=["mp4", "mov", "avi"],
    help="Keep the clip focused on one or two strikes for best results."
)

if uploaded:
    model, scaler, label_encoder = load_artifacts()

    video_bytes = uploaded.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        video_path = tmp.name

    # =========================
    # PROGRESS PROCESSING
    # =========================
    status      = st.empty()
    progress_bar = st.progress(0)

    status.markdown("⏳ **Extracting pose landmarks…**")

    def progress_cb(v):
        progress_bar.progress(v)

    t0     = time.time()
    frames = extract_frames(video_path, progress_cb=progress_cb)

    status.markdown("🧠 **Running LSTM inference…**")
    progress_bar.progress(0.92)

    cap     = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    if len(frames) < WINDOW_SIZE:
        progress_bar.empty()
        status.error(
            f"⚠️ Only {len(frames)} frames detected — not enough for analysis. "
            "Check lighting and ensure your full body is visible."
        )
        st.stop()

    labels, timestamps, confidence = run_inference(frames, model, scaler, label_encoder, src_fps)
    elapsed = time.time() - t0

    progress_bar.progress(1.0)
    status.empty()
    progress_bar.empty()

    # =========================
    # COMPUTE RESULTS
    # =========================
    score, ratio  = compute_score(labels, confidence)
    errors        = describe_errors(labels, timestamps, confidence)
    verdict_html  = (
        f"<span class='verdict-ok'>✅ Good execution</span>"
        if ratio >= 0.5 else
        f"<span class='verdict-bad'>❌ Needs improvement</span>"
    )

    # =========================
    # LAYOUT (CLEAN VERSION)
    # =========================
    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        # SCORE CARD
        st.markdown(f"""
    <div class="score-box">
      <div style="opacity:.6;font-size:.85rem;margin-bottom:4px">
        EXECUTION SCORE
      </div>
      <div class="score-number">{score}%</div>
      <div style="margin-top:8px">{verdict_html}</div>
      <div style="margin-top:6px;opacity:.45;font-size:.8rem">
        {len(labels)} analyzed windows ·
        {sum(c >= CONFIDENCE_THRESH for c in confidence)} high-confidence
      </div>
    </div>
    """, unsafe_allow_html=True)

        # QUICK STATS ONLY
        st.markdown("### 📊 Breakdown")

        m1, m2, m3 = st.columns(3)

        correct_pct = sum("correct" in l for l in labels) / max(len(labels), 1) * 100
        error_pct = sum("too_early" in l or "shoulder_back" in l for l in labels) / max(len(labels), 1) * 100
        avg_conf = np.mean(confidence) * 100 if len(confidence) else 0

        m1.metric("Correct", f"{correct_pct:.0f}%")
        m2.metric("Errors", f"{error_pct:.0f}%")
        m3.metric("Confidence", f"{avg_conf:.0f}%")

        # TIMELINE ONLY (MAIN VISUAL)
        st.markdown("### ⏱️ Movement Timeline")
        fig = build_timeline(labels, timestamps, confidence)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("### 🎥 Video Analysis")

        # Responsive video container (fills column cleanly)
        st.markdown(
            """
            <style>
            video {
                width: 100% !important;
                height: auto !important;
                max-height: 75vh;
                border-radius: 12px;
                border: 1px solid rgba(255,255,255,0.08);
                object-fit: contain;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.video(video_path)

        st.markdown("""
        <div style="
            margin-top:12px;
            padding:12px;
            background:rgba(255,255,255,0.03);
            border-radius:10px;
            font-size:0.85rem;
            opacity:0.8;
        ">
        💡 Tip: Best results come from side-angle recordings with full body visibility.
        </div>
        """, unsafe_allow_html=True)