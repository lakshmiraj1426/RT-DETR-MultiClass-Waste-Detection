"""
TrashScan AI — RT-DETR powered waste detection
================================================
A production-styled Streamlit front end for an Ultralytics RT-DETR trash
detection model, featuring dark glassmorphism UI, animated loading,
screenshot downloads, and fully functional Dashboard / History / Comparison
/ About pages.
"""

import io
import time
import datetime
import csv
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# ──────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="TrashScan AI · Waste Detection",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAX_DIM        = 1280
DETECT_FLOOR   = 0.05
DEFAULT_CONF   = 0.50

PALETTE = ["#00FF88", "#00CFFF", "#FF6B6B", "#FFD93D", "#C77DFF", "#FF9A3C"]

AVAILABLE_MODELS = {
    "Multiple Waste Dataset": "best.pt",
}

# ──────────────────────────────────────────────────────────────────────────
# DARK GLASSMORPHISM CSS
# ──────────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

/* ── Tokens ─────────────────────────────────────────────────────────── */
:root {
  --bg:          #0A0F0D;
  --bg2:         #0D1411;
  --surface:     rgba(255,255,255,0.04);
  --glass:       rgba(0,255,136,0.06);
  --border:      rgba(0,255,136,0.15);
  --border-soft: rgba(255,255,255,0.07);
  --neon:        #00FF88;
  --neon2:       #00CFFF;
  --neon3:       #C77DFF;
  --text:        #E8F5EE;
  --text-muted:  #6B8F78;
  --danger:      #FF6B6B;
  --warn:        #FFD93D;
  --card-glow:   0 0 30px rgba(0,255,136,0.08), 0 4px 24px rgba(0,0,0,0.5);
}

/* ── Base ───────────────────────────────────────────────────────────── */
html, body, .stApp {
  background: var(--bg) !important;
  color: var(--text);
  font-family: 'Inter', sans-serif;
}
#MainMenu, footer, header { visibility: hidden; }

/* ── Animated scanline background ──────────────────────────────────── */
.stApp::before {
  content: '';
  position: fixed;
  inset: 0;
  background:
    repeating-linear-gradient(
      0deg,
      transparent,
      transparent 2px,
      rgba(0,255,136,0.015) 2px,
      rgba(0,255,136,0.015) 4px
    );
  pointer-events: none;
  z-index: 0;
}

/* ── Typography ─────────────────────────────────────────────────────── */
h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; }

/* ── Sidebar ────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0D1A14 0%, #081009 100%) !important;
  border-right: 1px solid var(--border);
  box-shadow: 4px 0 30px rgba(0,255,136,0.05);
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }
section[data-testid="stSidebar"] h3 {
  font-family: 'Space Grotesk', sans-serif;
  color: var(--neon) !important;
}
section[data-testid="stSidebar"] hr { border-color: var(--border-soft); }
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] small { color: var(--text-muted) !important; }

/* Radio buttons */
section[data-testid="stSidebar"] [data-testid="stRadio"] label {
  transition: all 0.2s ease;
  border-radius: 6px;
  padding: 2px 6px;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
  color: var(--neon) !important;
}

/* Sliders */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
  background: var(--neon) !important;
  box-shadow: 0 0 10px var(--neon) !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] div[data-testid="stTickBar"] {
  background: var(--neon) !important;
}

/* ── Buttons ─────────────────────────────────────────────────────────── */
.stButton > button, .stDownloadButton > button {
  background: linear-gradient(135deg, rgba(0,255,136,0.12), rgba(0,207,255,0.08)) !important;
  border: 1px solid var(--border) !important;
  color: var(--neon) !important;
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.82rem !important;
  font-weight: 500 !important;
  letter-spacing: 0.06em !important;
  border-radius: 8px !important;
  padding: 0.5rem 1.2rem !important;
  transition: all 0.25s ease !important;
  box-shadow: 0 0 0 0 rgba(0,255,136,0.3) !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  background: linear-gradient(135deg, rgba(0,255,136,0.22), rgba(0,207,255,0.14)) !important;
  box-shadow: 0 0 16px rgba(0,255,136,0.25), 0 0 32px rgba(0,207,255,0.1) !important;
  transform: translateY(-1px) !important;
  border-color: var(--neon) !important;
}

/* ── File uploader ───────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
  border: 2px dashed var(--border) !important;
  border-radius: 12px !important;
  background: var(--surface) !important;
  backdrop-filter: blur(8px) !important;
  transition: all 0.3s ease !important;
}
[data-testid="stFileUploader"]:hover {
  border-color: var(--neon) !important;
  box-shadow: 0 0 20px rgba(0,255,136,0.1) !important;
}

/* ── Images ─────────────────────────────────────────────────────────── */
div[data-testid="stImage"] img {
  border-radius: 10px;
  border: 1px solid var(--border);
  box-shadow: var(--card-glow);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}
div[data-testid="stImage"] img:hover {
  transform: scale(1.01);
  box-shadow: 0 0 40px rgba(0,255,136,0.18), 0 8px 32px rgba(0,0,0,0.6);
}

/* ── Selectbox / Radio ───────────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {
  background: var(--surface) !important;
  border: 1px solid var(--border-soft) !important;
  color: var(--text) !important;
  border-radius: 8px !important;
}

/* ── Expander ────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
  border: 1px solid var(--border-soft) !important;
  border-radius: 10px !important;
  background: var(--surface) !important;
}

/* ── Info / Warning boxes ─────────────────────────────────────────────── */
[data-testid="stAlert"] {
  background: rgba(0,255,136,0.06) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  color: var(--text) !important;
}

/* ── Tables ──────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
  border: 1px solid var(--border-soft) !important;
  border-radius: 10px !important;
  overflow: hidden;
}

/* ── Progress bar ────────────────────────────────────────────────────── */
[data-testid="stProgressBar"] > div > div {
  background: linear-gradient(90deg, var(--neon), var(--neon2)) !important;
  box-shadow: 0 0 12px rgba(0,255,136,0.5) !important;
}

/* ── Tabs ────────────────────────────────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
  gap: 8px;
  border-bottom: 1px solid var(--border-soft) !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
  background: transparent !important;
  border: 1px solid transparent !important;
  border-radius: 8px 8px 0 0 !important;
  color: var(--text-muted) !important;
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.8rem !important;
  transition: all 0.2s ease !important;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover {
  color: var(--neon) !important;
  border-color: var(--border) !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
  color: var(--neon) !important;
  border-color: var(--border) !important;
  background: var(--glass) !important;
}

/* ── Spinner override ────────────────────────────────────────────────── */
[data-testid="stSpinner"] {
  color: var(--neon) !important;
}

/* ── Custom component classes ─────────────────────────────────────────── */

/* Hero */
.hero {
  padding: 0.5rem 0 1.5rem 0;
  border-bottom: 1px solid var(--border-soft);
  margin-bottom: 1.5rem;
  animation: fadeInUp 0.6s ease both;
}
.hero-eyebrow {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.72rem;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--neon);
  margin-bottom: 0.5rem;
  display: flex;
  align-items: center;
  gap: 0.6rem;
}
.status-dot {
  width: 8px; height: 8px; border-radius: 50%; display: inline-block;
  animation: pulseGlow 2s ease-in-out infinite;
}
.status-dot.online  { background: var(--neon);  box-shadow: 0 0 8px var(--neon); }
.status-dot.offline { background: var(--danger); box-shadow: 0 0 8px var(--danger); animation: none; }
.hero-title {
  font-size: 2.4rem; font-weight: 700; margin: 0;
  background: linear-gradient(135deg, var(--neon) 0%, var(--neon2) 60%, var(--neon3) 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
}
.hero-sub {
  font-family: 'Inter', sans-serif;
  color: var(--text-muted);
  font-size: 1rem;
  margin-top: 0.4rem;
  max-width: 680px;
  line-height: 1.65;
}

/* Glass card */
.glass-card {
  background: var(--surface);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid var(--border-soft);
  border-radius: 14px;
  padding: 1.4rem 1.6rem;
  box-shadow: var(--card-glow);
  animation: fadeInUp 0.5s ease both;
}
.glass-card:hover {
  border-color: rgba(0,255,136,0.25);
  box-shadow: 0 0 40px rgba(0,255,136,0.1), 0 8px 32px rgba(0,0,0,0.5);
  transition: all 0.3s ease;
}

/* Panel labels */
.panel-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.72rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--neon);
  margin-bottom: 0.5rem;
  opacity: 0.85;
}

/* Empty state */
.empty-state {
  border: 2px dashed var(--border);
  border-radius: 16px;
  background: var(--glass);
  backdrop-filter: blur(8px);
  padding: 5rem 2rem;
  text-align: center;
  color: var(--text);
  transition: all 0.35s ease;
  animation: fadeInUp 0.7s ease both;
}
.empty-state:hover {
  border-color: var(--neon);
  background: rgba(0,255,136,0.08);
  box-shadow: 0 0 30px rgba(0,255,136,0.12);
  transform: translateY(-4px);
}
.empty-state .icon {
  font-size: 3.2rem;
  margin-bottom: 1rem;
  display: block;
  animation: float 3s ease-in-out infinite;
}
.empty-state strong { color: var(--neon); font-family: 'Space Grotesk', sans-serif; }
.empty-state p { color: var(--text-muted); margin-top: 0.4rem; font-size: 0.95rem; }

/* Readout bar */
.readout {
  background: linear-gradient(135deg, rgba(0,20,12,0.9), rgba(0,10,18,0.9));
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.2rem 1.6rem;
  display: flex;
  flex-wrap: wrap;
  gap: 0;
  margin-top: 1.2rem;
  box-shadow: var(--card-glow);
  animation: fadeInUp 0.5s ease both;
}
.readout-cell {
  flex: 1;
  min-width: 130px;
  padding: 0 1.4rem;
  border-left: 1px solid rgba(0,255,136,0.12);
}
.readout-cell:first-child { border-left: none; padding-left: 0; }
.readout-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.65rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--text-muted);
  margin-bottom: 0.3rem;
}
.readout-value {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 1.45rem;
  font-weight: 600;
  color: var(--neon);
  text-shadow: 0 0 12px rgba(0,255,136,0.4);
}

/* Class badges */
.badge {
  display: inline-block;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.72rem;
  padding: 0.22rem 0.6rem;
  border-radius: 6px;
  background: rgba(0,255,136,0.1);
  color: var(--neon);
  border: 1px solid rgba(0,255,136,0.2);
  margin: 0.2rem 0.3rem 0.2rem 0;
  transition: all 0.2s ease;
}
.badge:hover {
  background: rgba(0,255,136,0.2);
  box-shadow: 0 0 10px rgba(0,255,136,0.2);
}

/* Metric cards */
.metric-card {
  background: var(--surface);
  backdrop-filter: blur(12px);
  border: 1px solid var(--border-soft);
  border-radius: 12px;
  padding: 1.2rem 1.4rem;
  text-align: center;
  transition: all 0.3s ease;
  animation: fadeInUp 0.5s ease both;
}
.metric-card:hover {
  border-color: var(--border);
  box-shadow: 0 0 24px rgba(0,255,136,0.1);
  transform: translateY(-2px);
}
.metric-icon { font-size: 1.8rem; margin-bottom: 0.4rem; }
.metric-value {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 2rem;
  font-weight: 700;
  color: var(--neon);
  text-shadow: 0 0 14px rgba(0,255,136,0.35);
}
.metric-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.68rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--text-muted);
  margin-top: 0.2rem;
}

/* History row */
.hist-row {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 0.75rem 1rem;
  border: 1px solid var(--border-soft);
  border-radius: 10px;
  margin-bottom: 0.6rem;
  background: var(--surface);
  transition: all 0.2s ease;
}
.hist-row:hover {
  border-color: var(--border);
  box-shadow: 0 0 16px rgba(0,255,136,0.07);
}
.hist-filename {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.82rem;
  color: var(--text);
  flex: 1;
}
.hist-badge {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.72rem;
  padding: 0.18rem 0.5rem;
  border-radius: 5px;
  background: rgba(0,255,136,0.1);
  color: var(--neon);
  border: 1px solid rgba(0,255,136,0.2);
}
.hist-time {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.68rem;
  color: var(--text-muted);
}

/* Section headings */
.section-heading {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.6rem;
  font-weight: 700;
  margin-bottom: 0.2rem;
  background: linear-gradient(90deg, var(--text) 40%, var(--neon) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.section-sub {
  color: var(--text-muted);
  font-size: 0.95rem;
  margin-bottom: 1.5rem;
}

/* Divider */
.neon-divider {
  border: none;
  border-top: 1px solid var(--border-soft);
  margin: 1.5rem 0;
}

/* About tech badge */
.tech-badge {
  display: inline-block;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.8rem;
  padding: 0.3rem 0.8rem;
  border-radius: 6px;
  background: rgba(0,207,255,0.08);
  color: var(--neon2);
  border: 1px solid rgba(0,207,255,0.2);
  margin: 0.2rem 0.3rem 0.2rem 0;
}

/* Loading bar */
.loading-bar-wrap {
  width: 100%;
  height: 3px;
  background: rgba(0,255,136,0.1);
  border-radius: 2px;
  overflow: hidden;
  margin: 0.5rem 0 1rem 0;
}
.loading-bar {
  height: 100%;
  width: 40%;
  background: linear-gradient(90deg, transparent, var(--neon), var(--neon2), transparent);
  border-radius: 2px;
  animation: shimmerBar 1.4s ease-in-out infinite;
}

/* ── Keyframes ────────────────────────────────────────────────────────── */
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(18px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulseGlow {
  0%, 100% { box-shadow: 0 0 6px var(--neon); opacity: 1; }
  50%       { box-shadow: 0 0 16px var(--neon), 0 0 28px rgba(0,255,136,0.4); opacity: 0.7; }
}
@keyframes float {
  0%, 100% { transform: translateY(0); }
  50%       { transform: translateY(-8px); }
}
@keyframes shimmerBar {
  0%   { transform: translateX(-200%); }
  100% { transform: translateX(500%); }
}
@keyframes spin {
  to { transform: rotate(360deg); }
}

/* Sidebar logo area */
.sidebar-logo {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.15rem;
  font-weight: 700;
  color: var(--neon) !important;
  letter-spacing: -0.01em;
  padding: 0.5rem 0 0.25rem 0;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.sidebar-logo .dot {
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--neon);
  box-shadow: 0 0 8px var(--neon);
  display: inline-block;
  animation: pulseGlow 2s ease-in-out infinite;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ──────────────────────────────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history = []   # list of dicts: {filename, count, avg_conf, elapsed_ms, classes, timestamp, annotated_bytes}

if "total_scans" not in st.session_state:
    st.session_state.total_scans = 0

if "total_objects" not in st.session_state:
    st.session_state.total_objects = 0

if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Detection"

if "dataset_choice" not in st.session_state:
    st.session_state.dataset_choice = list(AVAILABLE_MODELS.keys())[0]

if "confidence_threshold" not in st.session_state:
    st.session_state.confidence_threshold = DEFAULT_CONF

if "comparison_source" not in st.session_state:
    st.session_state.comparison_source = "Upload fresh images"

# ──────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ──────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    from ultralytics import RTDETR
    return RTDETR(model_path)

# ──────────────────────────────────────────────────────────────────────────
# IMAGE UTILITIES
# ──────────────────────────────────────────────────────────────────────────

def resize_image(image: Image.Image, max_dim: int = MAX_DIM) -> Image.Image:
    w, h = image.size
    longest = max(w, h)
    if longest <= max_dim:
        return image
    scale = max_dim / longest
    return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

def _load_font(size: int):
    for candidate in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            continue
    return ImageFont.load_default()

def image_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# ──────────────────────────────────────────────────────────────────────────
# INFERENCE
# ──────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_inference(image_bytes: bytes, _model):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = resize_image(image)
    start = time.perf_counter()
    results = _model.predict(image, conf=DETECT_FLOOR, verbose=False)[0]
    elapsed_ms = (time.perf_counter() - start) * 1000

    if len(results.boxes):
        boxes   = results.boxes.xyxy.cpu().numpy()
        confs   = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)
    else:
        boxes   = np.empty((0, 4))
        confs   = np.empty((0,))
        classes = np.empty((0,), dtype=int)

    return image, boxes, confs, classes, results.names, elapsed_ms

def draw_detections(image, boxes, confs, classes, names, threshold, palette):
    annotated = image.copy()
    draw      = ImageDraw.Draw(annotated)
    font_size = max(14, int(min(image.size) * 0.022))
    font      = _load_font(font_size)

    kept           = confs >= threshold
    detected_cls   = set()
    conf_total     = 0.0

    for box, conf, cls in zip(boxes[kept], confs[kept], classes[kept]):
        color      = palette[int(cls) % len(palette)]
        label_name = names.get(int(cls), str(cls)) if isinstance(names, dict) else names[int(cls)]
        detected_cls.add(label_name)
        conf_total += float(conf)

        x1, y1, x2, y2 = box
        # Rounded border (draw filled + outer rect for rounded feel)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        label   = f"{label_name}  {conf:.0%}"
        bbox    = draw.textbbox((0, 0), label, font=font)
        tw, th  = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad     = 5
        lbl_y0  = max(0, y1 - th - 2 * pad)
        draw.rectangle([x1, lbl_y0, x1 + tw + 2 * pad, lbl_y0 + th + 2 * pad], fill=color)
        draw.text((x1 + pad, lbl_y0 + pad // 2), label, fill="black", font=font)

    count    = int(kept.sum())
    avg_conf = (conf_total / count) if count else 0.0
    return annotated, count, detected_cls, avg_conf

# ──────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        "<div class='sidebar-logo'><span class='dot'></span>TrashScan AI</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<p style='color:#4a7a5a;font-size:0.72rem;font-family:IBM Plex Mono,monospace;letter-spacing:0.1em;margin-bottom:0.5rem;'>RT-DETR · Waste Detection</p>", unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["Detection", "Dashboard", "History", "Comparison", "About"],
        key="nav_page",
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:rgba(0,255,136,0.1);margin:0.8rem 0;'>", unsafe_allow_html=True)

    st.markdown("### ⚙️ Detection controls")
    st.caption("Tune how results are filtered.")

    confidence = st.slider(
        "Confidence threshold",
        min_value=0.1,
        max_value=1.0,
        value=DEFAULT_CONF,
        step=0.05,
        help="Only show detections at or above this confidence score.",
        key="confidence_threshold",
    )

    st.markdown("<hr style='border-color:rgba(0,255,136,0.1);margin:0.8rem 0;'>", unsafe_allow_html=True)

    with st.expander("ℹ️ System info"):
        st.markdown(f"**Resize cap** — long edge ≤ {MAX_DIM}px")
        try:
            import torch
            device = "GPU 🚀" if torch.cuda.is_available() else "CPU 🧠"
        except Exception:
            device = "unknown"
        st.markdown(f"**Device** — {device}")
        st.markdown(f"**Session scans** — {st.session_state.total_scans}")
        st.markdown(f"**Objects found** — {st.session_state.total_objects}")

# ──────────────────────────────────────────────────────────────────────────
# ── DASHBOARD PAGE
# ──────────────────────────────────────────────────────────────────────────

if page == "Dashboard":
    st.markdown(
        "<div class='section-heading'>📊 Dashboard</div>"
        "<div class='section-sub'>Session-level detection analytics and performance summary.</div>",
        unsafe_allow_html=True,
    )

    hist = st.session_state.history

    # ── Top metric cards
    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ("🖼️", str(st.session_state.total_scans), "Images Scanned"),
        ("🔍", str(st.session_state.total_objects), "Objects Detected"),
        (
            "📈",
            f"{(sum(h['avg_conf'] for h in hist) / len(hist)):.0%}" if hist else "—",
            "Avg Confidence",
        ),
        (
            "⚡",
            f"{(sum(h['elapsed_ms'] for h in hist) / len(hist)):.0f} ms" if hist else "—",
            "Avg Inference",
        ),
    ]
    for col, (icon, val, lbl) in zip([c1, c2, c3, c4], metrics):
        with col:
            st.markdown(
                f"""<div class='metric-card'>
                  <div class='metric-icon'>{icon}</div>
                  <div class='metric-value'>{val}</div>
                  <div class='metric-label'>{lbl}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)

    if not hist:
        st.markdown(
            """<div class='empty-state'>
              <span class='icon'>📡</span>
              <strong>No scans yet in this session</strong>
              <p>Run some detections on the Detection page and come back here to see analytics.</p>
            </div>""",
            unsafe_allow_html=True,
        )
        st.stop()

    # ── Class frequency chart
    from collections import Counter
    all_classes = []
    for h in hist:
        all_classes.extend(h.get("classes", []))

    if all_classes:
        import pandas as pd
        class_counts = Counter(all_classes)
        df_classes   = pd.DataFrame(
            {"Class": list(class_counts.keys()), "Count": list(class_counts.values())}
        ).sort_values("Count", ascending=False).head(12)

        col_left, col_right = st.columns([3, 2])
        with col_left:
            st.markdown("<div class='panel-label'>Most detected classes</div>", unsafe_allow_html=True)
            try:
                import altair as alt
                chart = (
                    alt.Chart(df_classes)
                    .mark_bar(
                        cornerRadiusTopLeft=5,
                        cornerRadiusTopRight=5,
                        color=alt.gradient(
                            "linear",
                            stops=[
                                alt.GradientStop(color="#00FF88", offset=0),
                                alt.GradientStop(color="#00CFFF", offset=1),
                            ],
                            x1=1, x2=1, y1=1, y2=0,
                        ),
                    )
                    .encode(
                        x=alt.X("Count:Q", axis=alt.Axis(labelColor="#6B8F78", gridColor="rgba(255,255,255,0.05)")),
                        y=alt.Y("Class:N", sort="-x", axis=alt.Axis(labelColor="#E8F5EE")),
                        tooltip=["Class", "Count"],
                    )
                    .properties(background="transparent", height=300)
                    .configure_axis(domainColor="rgba(255,255,255,0.1)", tickColor="rgba(255,255,255,0.1)")
                    .configure_view(strokeWidth=0)
                )
                st.altair_chart(chart, use_container_width=True)
            except Exception:
                st.bar_chart(df_classes.set_index("Class"))

        with col_right:
            st.markdown("<div class='panel-label'>Top class breakdown</div>", unsafe_allow_html=True)
            total = sum(class_counts.values())
            for cls, cnt in list(class_counts.most_common(8)):
                pct = cnt / total
                st.markdown(f"<span class='badge'>{cls}</span>", unsafe_allow_html=True)
                st.progress(pct)

    st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)

    # ── Recent activity feed
    st.markdown("<div class='panel-label'>Recent scans</div>", unsafe_allow_html=True)
    for h in reversed(hist[-6:]):
        ts = h["timestamp"].strftime("%H:%M:%S") if isinstance(h["timestamp"], datetime.datetime) else h["timestamp"]
        st.markdown(
            f"""<div class='hist-row'>
              <div class='hist-filename'>{h['filename']}</div>
              <span class='hist-badge'>{h['count']} obj</span>
              <span class='hist-badge'>{h['avg_conf']:.0%} conf</span>
              <span class='hist-time'>{ts}</span>
            </div>""",
            unsafe_allow_html=True,
        )
    st.stop()

# ──────────────────────────────────────────────────────────────────────────
# ── HISTORY PAGE
# ──────────────────────────────────────────────────────────────────────────

if page == "History":
    st.markdown(
        "<div class='section-heading'>🗂 Scan History</div>"
        "<div class='section-sub'>All detections performed in this session. Expand any row to view the annotated result.</div>",
        unsafe_allow_html=True,
    )

    hist = st.session_state.history
    if not hist:
        st.markdown(
            """<div class='empty-state'>
              <span class='icon'>📂</span>
              <strong>History is empty</strong>
              <p>Your scan results will appear here after you upload and detect images.</p>
            </div>""",
            unsafe_allow_html=True,
        )
        st.stop()

    # ── Export CSV
    import pandas as pd
    df_hist = pd.DataFrame([
        {
            "Filename": h["filename"],
            "Objects Detected": h["count"],
            "Avg Confidence": f"{h['avg_conf']:.2%}",
            "Inference (ms)": f"{h['elapsed_ms']:.1f}",
            "Detected Classes": ", ".join(sorted(h.get("classes", []))),
            "Timestamp": h["timestamp"].strftime("%Y-%m-%d %H:%M:%S") if isinstance(h["timestamp"], datetime.datetime) else h["timestamp"],
        }
        for h in hist
    ])

    csv_buf = io.StringIO()
    df_hist.to_csv(csv_buf, index=False)
    st.download_button(
        label="⬇ Export history as CSV",
        data=csv_buf.getvalue().encode("utf-8"),
        file_name=f"trashscan_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Expandable rows
    for i, h in enumerate(reversed(hist)):
        ts = h["timestamp"].strftime("%Y-%m-%d %H:%M:%S") if isinstance(h["timestamp"], datetime.datetime) else h["timestamp"]
        with st.expander(f"🖼️  {h['filename']}  ·  {h['count']} objects  ·  {ts}"):
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.markdown(
                    f"""<div class='readout' style='flex-direction:column;gap:0.7rem;'>
                      <div><div class='readout-label'>Objects</div><div class='readout-value'>{h['count']}</div></div>
                      <div><div class='readout-label'>Avg Conf</div><div class='readout-value'>{h['avg_conf']:.0%}</div></div>
                      <div><div class='readout-label'>Inference</div><div class='readout-value'>{h['elapsed_ms']:.0f} ms</div></div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                if h.get("classes"):
                    st.markdown("<br>" + "".join(f"<span class='badge'>{c}</span>" for c in sorted(h["classes"])), unsafe_allow_html=True)
            with col_b:
                if h.get("annotated_bytes"):
                    st.markdown("<div class='panel-label'>Annotated Result</div>", unsafe_allow_html=True)
                    st.image(h["annotated_bytes"], use_container_width=True)
                    st.download_button(
                        label="⬇ Download annotated image",
                        data=h["annotated_bytes"],
                        file_name=f"trashscan_{h['filename']}",
                        mime="image/png",
                        key=f"hist_dl_{i}",
                    )
    st.stop()

# ──────────────────────────────────────────────────────────────────────────
# ── COMPARISON PAGE
# ──────────────────────────────────────────────────────────────────────────

if page == "Comparison":
    st.markdown(
        "<div class='section-heading'>⚖️ Comparison</div>"
        "<div class='section-sub'>Compare two images side-by-side with full detection analysis.</div>",
        unsafe_allow_html=True,
    )

    hist = st.session_state.history

    mode = st.radio(
        "Source",
        ["Upload fresh images", "Pick from session history"],
        horizontal=True,
        label_visibility="visible",
        key="comparison_source",
    )

    ann_a = ann_b = None
    info_a = info_b = {}

    if mode == "Upload fresh images":
        ca, cb = st.columns(2)
        with ca:
            st.markdown("<div class='panel-label'>Image A</div>", unsafe_allow_html=True)
            file_a = st.file_uploader("Image A", type=["jpg","jpeg","png","webp"], key="cmp_a", label_visibility="collapsed")
        with cb:
            st.markdown("<div class='panel-label'>Image B</div>", unsafe_allow_html=True)
            file_b = st.file_uploader("Image B", type=["jpg","jpeg","png","webp"], key="cmp_b", label_visibility="collapsed")

        if file_a and file_b:
            # Load model for comparison
            dataset_choice_cmp = st.selectbox("Model", list(AVAILABLE_MODELS.keys()), key="cmp_model")
            if st.button("🔍 Run Comparison", key="run_comparison"):
                mdl = load_model(AVAILABLE_MODELS[dataset_choice_cmp])
                class_map_cmp = _get_class_map(dataset_choice_cmp)

                with st.spinner("Running inference on both images…"):
                    img_a, bx_a, cf_a, cl_a, nm_a, ms_a = run_inference(file_a.getvalue(), mdl)
                    img_b, bx_b, cf_b, cl_b, nm_b, ms_b = run_inference(file_b.getvalue(), mdl)

                ann_a, cnt_a, cls_a, avg_a = draw_detections(img_a, bx_a, cf_a, cl_a, nm_a, confidence, PALETTE)
                ann_b, cnt_b, cls_b, avg_b = draw_detections(img_b, bx_b, cf_b, cl_b, nm_b, confidence, PALETTE)
                info_a = {"count": cnt_a, "classes": cls_a, "avg_conf": avg_a, "elapsed_ms": ms_a, "name": file_a.name}
                info_b = {"count": cnt_b, "classes": cls_b, "avg_conf": avg_b, "elapsed_ms": ms_b, "name": file_b.name}

    else:
        if len(hist) < 2:
            st.info("You need at least 2 scans in session history. Run some detections first!")
            st.stop()
        names_list = [h["filename"] for h in hist]
        sel_a = st.selectbox("Select Image A", names_list, key="sel_a")
        sel_b = st.selectbox("Select Image B", [n for n in names_list if n != sel_a] or names_list, key="sel_b")
        if st.button("⚖️ Compare", key="run_history_compare"):
            h_a = next((h for h in hist if h["filename"] == sel_a), None)
            h_b = next((h for h in hist if h["filename"] == sel_b), None)
            if h_a and h_b and h_a.get("annotated_bytes") and h_b.get("annotated_bytes"):
                ann_a   = Image.open(io.BytesIO(h_a["annotated_bytes"]))
                ann_b   = Image.open(io.BytesIO(h_b["annotated_bytes"]))
                info_a  = {**h_a, "name": h_a["filename"]}
                info_b  = {**h_b, "name": h_b["filename"]}

    if ann_a is not None and ann_b is not None:
        st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)
        ca, cb = st.columns(2, gap="large")
        with ca:
            st.markdown(f"<div class='panel-label'>🅰 {info_a.get('name','Image A')}</div>", unsafe_allow_html=True)
            st.image(ann_a, use_container_width=True)
            st.markdown(
                f"""<div class='readout'>
                  <div class='readout-cell'><div class='readout-label'>Objects</div><div class='readout-value'>{info_a['count']}</div></div>
                  <div class='readout-cell'><div class='readout-label'>Avg Conf</div><div class='readout-value'>{info_a['avg_conf']:.0%}</div></div>
                  <div class='readout-cell'><div class='readout-label'>Inference</div><div class='readout-value'>{info_a['elapsed_ms']:.0f}ms</div></div>
                </div>""",
                unsafe_allow_html=True,
            )
            if info_a.get("classes"):
                st.markdown("".join(f"<span class='badge'>{c}</span>" for c in sorted(info_a["classes"])), unsafe_allow_html=True)
        with cb:
            st.markdown(f"<div class='panel-label'>🅱 {info_b.get('name','Image B')}</div>", unsafe_allow_html=True)
            st.image(ann_b, use_container_width=True)
            st.markdown(
                f"""<div class='readout'>
                  <div class='readout-cell'><div class='readout-label'>Objects</div><div class='readout-value'>{info_b['count']}</div></div>
                  <div class='readout-cell'><div class='readout-label'>Avg Conf</div><div class='readout-value'>{info_b['avg_conf']:.0%}</div></div>
                  <div class='readout-cell'><div class='readout-label'>Inference</div><div class='readout-value'>{info_b['elapsed_ms']:.0f}ms</div></div>
                </div>""",
                unsafe_allow_html=True,
            )
            if info_b.get("classes"):
                st.markdown("".join(f"<span class='badge'>{c}</span>" for c in sorted(info_b["classes"])), unsafe_allow_html=True)

        # ── Unique class diff
        cls_a_set = set(info_a.get("classes", []))
        cls_b_set = set(info_b.get("classes", []))
        only_in_a = cls_a_set - cls_b_set
        only_in_b = cls_b_set - cls_a_set
        in_both   = cls_a_set & cls_b_set

        st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)
        st.markdown("<div class='panel-label'>Class Diff</div>", unsafe_allow_html=True)
        dc1, dc2, dc3 = st.columns(3)
        with dc1:
            st.markdown("**Only in A**")
            if only_in_a:
                st.markdown("".join(f"<span class='badge'>{c}</span>" for c in sorted(only_in_a)), unsafe_allow_html=True)
            else:
                st.caption("—")
        with dc2:
            st.markdown("**In Both**")
            if in_both:
                st.markdown("".join(f"<span class='badge'>{c}</span>" for c in sorted(in_both)), unsafe_allow_html=True)
            else:
                st.caption("—")
        with dc3:
            st.markdown("**Only in B**")
            if only_in_b:
                st.markdown("".join(f"<span class='badge'>{c}</span>" for c in sorted(only_in_b)), unsafe_allow_html=True)
            else:
                st.caption("—")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────
# ── ABOUT PAGE
# ──────────────────────────────────────────────────────────────────────────

if page == "About":
    st.markdown(
        "<div class='section-heading'>ℹ️ About TrashScan AI</div>"
        "<div class='section-sub'>Open-source waste detection powered by state-of-the-art real-time detection transformers.</div>",
        unsafe_allow_html=True,
    )

    ca, cb = st.columns([3, 2])
    with ca:
        st.markdown("""
<div class='glass-card'>
<h3 style='color:#00FF88;font-family:Space Grotesk,sans-serif;margin-top:0;'>🔬 What is TrashScan AI?</h3>
<p style='color:#b0c8bb;line-height:1.75;'>
TrashScan AI is a real-time waste classification and localization system built on top of RT-DETR 
(Real-Time Detection Transformer), a state-of-the-art transformer-based object detector that 
achieves accuracy comparable to DINO-DETR while running at real-time speeds.
</p>
<p style='color:#b0c8bb;line-height:1.75;'>
Upload photos of waste scenes and the model draws bounding boxes around individual items, 
classifying them into recyclable, compostable, hazardous, or general waste streams.
</p>
</div>
""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
<div class='glass-card'>
<h3 style='color:#00CFFF;font-family:Space Grotesk,sans-serif;margin-top:0;'>🏗️ Architecture</h3>
<p style='color:#b0c8bb;line-height:1.75;'>
RT-DETR uses a hybrid CNN + Transformer backbone for efficient multi-scale feature extraction,
followed by an efficient hybrid encoder and an IoU-aware query selection mechanism that 
significantly improves detection performance without anchor design.
</p>
<ul style='color:#b0c8bb;line-height:2;'>
  <li>Backbone: ResNet / HGNetv2</li>
  <li>Encoder: Hybrid AIFI + CCFM</li>
  <li>Decoder: Transformer with learnable queries</li>
  <li>Post-processing: NMS-free end-to-end detection</li>
</ul>
</div>
""", unsafe_allow_html=True)

    with cb:
        st.markdown("""
<div class='glass-card'>
<h3 style='color:#C77DFF;font-family:Space Grotesk,sans-serif;margin-top:0;'>📦 Available Models</h3>
<hr style='border-color:rgba(255,255,255,0.07);'>
<p style='color:#00FF88;font-family:IBM Plex Mono,monospace;font-size:0.85rem;'>Multiple Waste Dataset</p>
<p style='color:#b0c8bb;font-size:0.85rem;line-height:1.6;'>24 classes including cardboard, plastic bottles, glass, medical waste, and more. Trained on a diverse multi-source waste dataset.</p>
<hr style='border-color:rgba(255,255,255,0.07);'>
<p style='color:#00FF88;font-family:IBM Plex Mono,monospace;font-size:0.85rem;'>TACO Dataset</p>
<p style='color:#b0c8bb;font-size:0.85rem;line-height:1.6;'>10 classes focused on outdoor litter: bottles, cans, cigarettes, cups, and wrappers. Trained on the TACO (Trash Annotations in Context) benchmark.</p>
</div>
""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
<div class='glass-card'>
<h3 style='color:#FFD93D;font-family:Space Grotesk,sans-serif;margin-top:0;'>🛠️ Tech Stack</h3>
""", unsafe_allow_html=True)
        techs = ["Ultralytics", "RT-DETR", "PyTorch", "Streamlit", "Pillow", "NumPy", "Altair", "Python 3.10+"]
        st.markdown("".join(f"<span class='tech-badge'>{t}</span>" for t in techs) + "</div>", unsafe_allow_html=True)

    st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)
    
    st.markdown("""
<div class='glass-card'>
<h3 style='color:#FF9A3C;font-family:Space Grotesk,sans-serif;margin-top:0;'>📊 Inference Benchmarks</h3>
<p style='color:#b0c8bb;font-size:0.9rem;line-height:1.6;'>
Impact of Intersection over Union (IoU) and Confidence thresholds on Average Precision (AP) and Non-Maximum Suppression (NMS) execution time.
</p>
</div>
<br>
""", unsafe_allow_html=True)

    ca, cb = st.columns(2)
    with ca:
        st.markdown("""
        <div class='glass-card' style='padding: 1rem;'>
        <table style="width:100%; text-align:center; border-collapse: collapse; color: #E8F5EE; font-family:'IBM Plex Mono', monospace; font-size: 0.9rem;">
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.2);">
                <th style="padding: 10px;">IoU thr.<br><span style="font-size:0.75rem; color:#6B8F78;">(Conf=0.001)</span></th>
                <th style="padding: 10px; border-left: 1px solid rgba(255,255,255,0.2);">AP<br><span style="font-size:0.75rem; color:#6B8F78;">(%)</span></th>
                <th style="padding: 10px;">NMS<br><span style="font-size:0.75rem; color:#6B8F78;">(ms)</span></th>
            </tr>
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                <td style="padding: 10px;">0.5</td>
                <td style="border-left: 1px solid rgba(255,255,255,0.2);">52.1</td>
                <td>2.24</td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                <td style="padding: 10px;">0.6</td>
                <td style="border-left: 1px solid rgba(255,255,255,0.2);">52.6</td>
                <td>2.29</td>
            </tr>
            <tr>
                <td style="padding: 10px;">0.8</td>
                <td style="border-left: 1px solid rgba(255,255,255,0.2);">52.8</td>
                <td>2.46</td>
            </tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

    with cb:
        st.markdown("""
        <div class='glass-card' style='padding: 1rem;'>
        <table style="width:100%; text-align:center; border-collapse: collapse; color: #E8F5EE; font-family:'IBM Plex Mono', monospace; font-size: 0.9rem;">
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.2);">
                <th style="padding: 10px;">Conf thr.<br><span style="font-size:0.75rem; color:#6B8F78;">(IoU=0.7)</span></th>
                <th style="padding: 10px; border-left: 1px solid rgba(255,255,255,0.2);">AP<br><span style="font-size:0.75rem; color:#6B8F78;">(%)</span></th>
                <th style="padding: 10px;">NMS<br><span style="font-size:0.75rem; color:#6B8F78;">(ms)</span></th>
            </tr>
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                <td style="padding: 10px;">0.001</td>
                <td style="border-left: 1px solid rgba(255,255,255,0.2);">52.9</td>
                <td>2.36</td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                <td style="padding: 10px;">0.01</td>
                <td style="border-left: 1px solid rgba(255,255,255,0.2);">52.4</td>
                <td>1.73</td>
            </tr>
            <tr>
                <td style="padding: 10px;">0.05</td>
                <td style="border-left: 1px solid rgba(255,255,255,0.2);">51.2</td>
                <td>1.06</td>
            </tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='glass-card'>
    <h3 style='color:#00CFFF;font-family:Space Grotesk,sans-serif;margin-top:0;font-size:1.2rem;'>🚀 Model Variants Performance</h3>
    <div style='overflow-x:auto; margin-top: 1rem;'>
    <table style="width:100%; text-align:center; border-collapse: collapse; color: #E8F5EE; font-family:'IBM Plex Mono', monospace; font-size: 0.85rem; white-space: nowrap;">
        <tr style="border-bottom: 1px solid rgba(255,255,255,0.2);">
            <th style="padding: 10px; text-align:left;">Model</th>
            <th style="padding: 10px;">Backbone</th>
            <th style="padding: 10px; border-left: 1px solid rgba(255,255,255,0.2);">#Epochs</th>
            <th style="padding: 10px;">#Params (M)</th>
            <th style="padding: 10px;">GFLOPs</th>
            <th style="padding: 10px;">FPS<sub style="color:#6B8F78;">bs=1</sub></th>
            <th style="padding: 10px; border-left: 1px solid rgba(255,255,255,0.2);">AP<sup style="color:#6B8F78;">val</sup></th>
            <th style="padding: 10px;">AP<sup style="color:#6B8F78;">val</sup><sub style="color:#6B8F78;">50</sub></th>
            <th style="padding: 10px;">AP<sup style="color:#6B8F78;">val</sup><sub style="color:#6B8F78;">75</sub></th>
            <th style="padding: 10px;">AP<sup style="color:#6B8F78;">val</sup><sub style="color:#6B8F78;">S</sub></th>
            <th style="padding: 10px;">AP<sup style="color:#6B8F78;">val</sup><sub style="color:#6B8F78;">M</sub></th>
            <th style="padding: 10px;">AP<sup style="color:#6B8F78;">val</sup><sub style="color:#6B8F78;">L</sub></th>
        </tr>
        <tr style="border-bottom: 1px solid rgba(255,255,255,0.05); background: rgba(0,255,136,0.03);">
            <td colspan="12" style="padding: 8px 10px; text-align:left; font-style: italic; color:#00CFFF;">Real-time End-to-end Object Detector (ours)</td>
        </tr>
        <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
            <td style="padding: 10px; text-align:left;">RT-DETR</td>
            <td style="padding: 10px;">R50</td>
            <td style="padding: 10px; border-left: 1px solid rgba(255,255,255,0.2);">72</td>
            <td style="padding: 10px;">42</td>
            <td style="padding: 10px;">136</td>
            <td style="padding: 10px; font-weight:bold; color:var(--neon);">108</td>
            <td style="padding: 10px; border-left: 1px solid rgba(255,255,255,0.2); font-weight:bold; color:var(--neon);">53.1</td>
            <td style="padding: 10px; font-weight:bold; color:var(--neon);">71.3</td>
            <td style="padding: 10px; font-weight:bold; color:var(--neon);">57.7</td>
            <td style="padding: 10px;">34.8</td>
            <td style="padding: 10px;">58.0</td>
            <td style="padding: 10px;">70.0</td>
        </tr>
        <tr>
            <td style="padding: 10px; text-align:left;">RT-DETR</td>
            <td style="padding: 10px;">R101</td>
            <td style="padding: 10px; border-left: 1px solid rgba(255,255,255,0.2);">72</td>
            <td style="padding: 10px;">76</td>
            <td style="padding: 10px;">259</td>
            <td style="padding: 10px; font-weight:bold; color:var(--neon);">74</td>
            <td style="padding: 10px; border-left: 1px solid rgba(255,255,255,0.2); font-weight:bold; color:var(--neon);">54.3</td>
            <td style="padding: 10px; font-weight:bold; color:var(--neon);">72.7</td>
            <td style="padding: 10px;">58.6</td>
            <td style="padding: 10px;">36.0</td>
            <td style="padding: 10px;">58.8</td>
            <td style="padding: 10px; font-weight:bold; color:var(--neon);">72.1</td>
        </tr>
    </table>
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br><div class='neon-divider'></div>", unsafe_allow_html=True)

    st.markdown("""
<div class='glass-card' style='text-align:center;'>
  <p style='color:#6B8F78;font-family:IBM Plex Mono,monospace;font-size:0.78rem;letter-spacing:0.1em;'>
    TrashScan AI · Built with ♻️ and ❤️ · RT-DETR × Ultralytics × Streamlit
  </p>
</div>
""", unsafe_allow_html=True)
    st.stop()

# ──────────────────────────────────────────────────────────────────────────
# ── DETECTION PAGE (main)
# ──────────────────────────────────────────────────────────────────────────

def _get_class_map(dataset_choice: str) -> dict:
    if dataset_choice == "TACO Dataset":
        return {
            0: "Bottle", 1: "Bottle Cap", 2: "Can", 3: "Cigarette", 4: "Cup",
            5: "Lid", 6: "Plastic Bag & Wrapper", 7: "Pop Tab", 8: "Straw", 9: "Other",
        }
    return {
        0: "cardboard", 1: "carton packaging", 2: "cigarette", 3: "clean paper",
        4: "clear plastic", 5: "contaminated paper", 6: "food packaging", 7: "food scraps",
        8: "glass", 9: "medical waste", 10: "metal", 11: "paper bag", 12: "paper cup",
        13: "plastic bottle", 14: "plastic container", 15: "plastic cup", 16: "plastic lid",
        17: "plastic packaging", 18: "plastic utensil", 19: "printed cardboard",
        20: "sanitary waste", 21: "straw", 22: "styrofoam", 23: "wood",
    }

# ── Model selection
dataset_choice = st.radio(
    "Select Dataset",
    list(AVAILABLE_MODELS.keys()),
    horizontal=True,
    index=0,
    key="dataset_choice",
)
selected_model_path = AVAILABLE_MODELS[dataset_choice]
class_map = _get_class_map(dataset_choice)

# ── Model loading with smoother spinner
model = None
model_error = None

with st.spinner("Loading model…"):
    try:
        model = load_model(selected_model_path)
    except Exception as exc:
        model_error = str(exc)

status_class = "online"  if model is not None else "offline"
status_text  = "Model online" if model is not None else "Model unavailable"

# ── Hero
st.markdown(
    f"""
    <div class="hero">
      <div class="hero-eyebrow">
        <span class="status-dot {status_class}"></span>{status_text} · RT-DETR · {dataset_choice}
      </div>
      <p class="hero-title">♻️ TrashScan AI</p>
      <p class="hero-sub">Upload a batch of photos and TrashScan locates and classifies litter in seconds —
      adjust the confidence threshold on the left to fine-tune what counts as a detection.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if model_error:
    st.error(
        f"Couldn't load the model from `{selected_model_path}`. Check the path and weights file.\n\nDetails: {model_error}"
    )
    st.stop()

# ── Supported classes chip row
classes_list_str = " &nbsp; ".join([f"<span class='badge'>{n.title()}</span>" for n in class_map.values()])
st.markdown(
    f"<div class='glass-card' style='padding:0.9rem 1.2rem;'>"
    f"<div class='panel-label' style='margin-bottom:0.5rem;'>Supported classes · {len(class_map)}</div>"
    f"{classes_list_str}"
    f"</div>",
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)

# ── Uploader
uploaded_files = st.file_uploader(
    "Upload Images",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
    help="Select one or multiple images",
    label_visibility="visible",
    key="upload_images",
)
st.markdown("<br>", unsafe_allow_html=True)

if not uploaded_files:
    st.markdown(
        """<div class="empty-state">
          <span class="icon">📷</span>
          <strong>No photos scanned yet</strong>
          <p>Drag & drop or click above to upload waste images — batch upload supported.</p>
        </div>""",
        unsafe_allow_html=True,
    )
    st.stop()

# ── Batch processing loop
for idx, file in enumerate(uploaded_files):

    st.markdown(
        f"<div style='font-family:IBM Plex Mono,monospace;font-size:0.8rem;color:#4a7a5a;letter-spacing:0.1em;margin-bottom:0.3rem;'>"
        f"FILE {idx+1}/{len(uploaded_files)}</div>"
        f"<h3 style='margin-top:0;color:#E8F5EE;font-family:Space Grotesk,sans-serif;'>📄 {file.name}</h3>",
        unsafe_allow_html=True,
    )

    image_bytes = file.getvalue()

    # Animated loading bar while inference runs
    loading_ph = st.empty()
    loading_ph.markdown(
        "<div class='loading-bar-wrap'><div class='loading-bar'></div></div>",
        unsafe_allow_html=True,
    )

    try:
        with st.spinner(f"🔍 Scanning {file.name}…"):
            base_image, boxes, confs, classes_arr, raw_names, elapsed_ms = run_inference(
                image_bytes, model
            )
        loading_ph.empty()

        annotated_image, count, detected_classes, avg_conf = draw_detections(
            base_image, boxes, confs, classes_arr, raw_names, confidence, PALETTE
        )
        st.toast(f"✅ {file.name} — {count} object(s) found", icon="♻️")

    except Exception as exc:
        loading_ph.empty()
        st.error(f"Detection failed: {file.name}\n\nDetails: {exc}")
        continue

    # ── Store in history / update session counters
    ann_bytes = image_to_bytes(annotated_image)
    st.session_state.history.append({
        "filename":       file.name,
        "count":          count,
        "avg_conf":       avg_conf,
        "elapsed_ms":     elapsed_ms,
        "classes":        list(detected_classes),
        "timestamp":      datetime.datetime.now(),
        "annotated_bytes": ann_bytes,
    })
    st.session_state.total_scans   += 1
    st.session_state.total_objects += count

    # ── Before / After panels
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("<div class='panel-label'>Original</div>", unsafe_allow_html=True)
        st.image(base_image, use_container_width=True)
    with col2:
        st.markdown("<div class='panel-label'>Detected</div>", unsafe_allow_html=True)
        st.image(annotated_image, use_container_width=True)

    # ── SCREENSHOT / DOWNLOAD BUTTON
    st.download_button(
        label="📸 Download Annotated Screenshot",
        data=ann_bytes,
        file_name=f"trashscan_{Path(file.name).stem}_annotated.png",
        mime="image/png",
        key=f"dl_{idx}_{file.name}",
    )

    # ── Scan readout
    st.markdown(
        f"""<div class="readout">
          <div class="readout-cell">
            <div class="readout-label">Objects detected</div>
            <div class="readout-value">{count}</div>
          </div>
          <div class="readout-cell">
            <div class="readout-label">Avg. confidence</div>
            <div class="readout-value">{avg_conf:.0%}</div>
          </div>
          <div class="readout-cell">
            <div class="readout-label">Inference time</div>
            <div class="readout-value">{elapsed_ms:.0f} ms</div>
          </div>
          <div class="readout-cell">
            <div class="readout-label">Threshold</div>
            <div class="readout-value">{confidence:.0%}</div>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Class badges
    if detected_classes:
        st.markdown(
            "<br>" + "".join(f"<span class='badge'>{c}</span>" for c in sorted(detected_classes)),
            unsafe_allow_html=True,
        )
    elif count == 0:
        st.info("No objects met the current confidence threshold.")

    # ── Confidence breakdown bar
    if count > 0:
        st.markdown("<br><div class='panel-label'>Avg confidence score</div>", unsafe_allow_html=True)
        st.progress(avg_conf)

    st.markdown("<br><hr style='border-color:rgba(0,255,136,0.07);'><br>", unsafe_allow_html=True)