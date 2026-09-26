# """
# VisionAssist - Minimal Streamlit Dashboard (Phase 1)
# Provides a live video feed, status strip, and basic controls.
# Uses the same session state objects as the original dashboard for consistency.
# """

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# flake8: noqa: E402
import logging
logger = logging.getLogger("VisionAssist.Dashboard")

import time
import cv2
import numpy as np

try:
    import streamlit as st
except ImportError:
    st = None

from config import ProductMode, PHONE_STREAM_URL
from input.webcam import OpenCVWebcam
from input.phone_stream import PhoneStreamCamera
from perception.object_detection import ObjectDetector
from perception.ocr import OCRReader
from intelligence.context_engine import ContextEngine
from intelligence.priority_engine import PriorityEngine
from intelligence.response_generator import ResponseGenerator
from intelligence.ask_engine import AskEngine
from audio.tts import TextToSpeechEngine
from utils.drawing import draw_visual_annotations

def inject_custom_css():
    """Inject dark theme and accessibility styling."""
    st.markdown(
        """
        <style>
        .stApp {background-color: #101828; color: #ECEAE4; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;}
        .header-title {font-size: 2.1rem; font-weight: 800; color: #FFFFFF; margin-bottom: 2px;}
        .header-subtitle {font-size: 0.95rem; color: #94A3B8; margin-bottom: 14px;}
        .status-strip {background-color: #1E293B; border: 1px solid #334155; padding: 10px 18px; font-size: 0.88rem; color: #94A3B8; border-radius: 8px; display: flex; flex-wrap: wrap; gap: 16px;}
        .status-strip .dot {height: 9px; width: 9px; background-color: #10B981; border-radius: 50%; display: inline-block; margin-right: 5px;}
        .audio-banner {background: linear-gradient(90deg, #09373B 0%, #0E5359 100%); border: 1px solid #16808C; border-radius: 10px; padding: 16px 22px; margin-bottom: 16px; color: #E6FFFA; font-weight: 600; font-size: 1.25rem; box-shadow: 0 4px 16px rgba(0,0,0,0.4);} 
        </style>
        """,
        unsafe_allow_html=True,
    )

def init_session_state():
    """Initialize Streamlit session state objects if not present."""
    if "detector" not in st.session_state:
        st.session_state.detector = ObjectDetector()
    if "ocr" not in st.session_state:
        st.session_state.ocr = OCRReader()
    if "context_engine" not in st.session_state:
        st.session_state.context_engine = ContextEngine()
    if "priority_engine" not in st.session_state:
        st.session_state.priority_engine = PriorityEngine()
    if "response_generator" not in st.session_state:
        st.session_state.response_generator = ResponseGenerator()
    if "ask_engine" not in st.session_state:
        st.session_state.ask_engine = AskEngine()
    if "tts" not in st.session_state:
        st.session_state.tts = TextToSpeechEngine()
    if "last_speech" not in st.session_state:
        st.session_state.last_speech = "VisionAssist online. Audio active."
    if "speech_history" not in st.session_state:
        st.session_state.speech_history = []
    if "language" not in st.session_state:
        st.session_state.language = "en"
    if "active_stage_override" not in st.session_state:
        st.session_state.active_stage_override = None

def render_team_page():
    """Render a simple Team page with member cards and contact info."""
    st.markdown("<div class='header-title'>🚀 Team</div>", unsafe_allow_html=True)
    st.markdown("""
    <style>
    .team-grid {display: flex; flex-wrap: wrap; gap: 20px; justify-content: center;}
    .member-card {background:#1E293B; border-radius:10px; padding:16px; width:200px; text-align:center;}
    .member-card img {border-radius:50%; width:100px; height:100px; object-fit:cover;}
    .member-card h4 {margin:8px 0 4px; color:#FFF;}
    .member-card p {margin:0; color:#94A3B8; font-size:0.9rem;}
    </style>
    <div class='team-grid'>
      <div class='member-card'>
        <img src='https://via.placeholder.com/100' alt='Member 1'>
        <h4>Adrit Das</h4>
        <p>Lead Engineer</p>
      </div>
      <div class='member-card'>
        <img src='https://via.placeholder.com/100' alt='Member 2'>
        <h4>Jane Smith</h4>
        <p>AI Researcher</p>
      </div>
      <div class='member-card'>
        <img src='https://via.placeholder.com/100' alt='Member 3'>
        <h4>John Doe</h4>
        <p>UX Designer</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

def render_dashboard():
    if st is None:
        raise ImportError("Streamlit is not installed. Run 'pip install streamlit'.")
    st.set_page_config(page_title="VisionAssist • Dashboard", page_icon="👁️", layout="wide")
    inject_custom_css()
    init_session_state()

    # Navigation sidebar
    with st.sidebar:
        st.header("🎛️ Controls")
        page = st.radio("Navigate", ["Dashboard", "Team"], index=0)
        if page == "Team":
            # Render team page and exit early
            render_team_page()
            return
        
        feed_source = st.radio(
            "Camera Feed Source",
            options=["Phone Camera Stream", "Laptop Webcam", "Curated Test Scene"],
            index=2,
        )
        stream_url = PHONE_STREAM_URL
        if feed_source == "Phone Camera Stream":
            stream_url = st.text_input("Phone Stream URL", value=PHONE_STREAM_URL)
        test_scene = st.selectbox("Demo Scene", ["Classroom", "Stairs", "Room Sign", "Collision"], index=0)
        st.divider()
        mode_option = st.selectbox(
            "Operating Mode",
            options=[ProductMode.OBSTACLE_AWARENESS.value, ProductMode.QUICK_LOOK.value, ProductMode.READ.value, ProductMode.ASK.value, ProductMode.SAFETY_ALERT.value],
        )
        selected_mode = ProductMode(mode_option)

    # Frame acquisition
    frame = None
    if feed_source == "Laptop Webcam":
        if "webcam" not in st.session_state:
            st.session_state.webcam = OpenCVWebcam(fallback_to_mock=True)
            st.session_state.webcam.open()
        ret, frame = st.session_state.webcam.read_frame()
        if not ret:
            st.warning("Webcam not available – using simulated feed.")
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "SIMULATED WEBCAM", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
    elif feed_source == "Phone Camera Stream":
        last_url = st.session_state.get("last_url")
        if "phone_cam" not in st.session_state or last_url != stream_url:
            try:
                st.session_state.phone_cam = PhoneStreamCamera(stream_url)
                st.session_state.phone_cam.open()
                st.session_state.last_url = stream_url
            except Exception as e:
                logger.warning(f"Phone stream unavailable ({e}); using placeholder.")
                st.session_state.phone_cam = None
        if st.session_state.phone_cam:
            ret, frame = st.session_state.phone_cam.read_frame()
            if not ret:
                st.warning("Phone stream failed – using placeholder.")
                frame = None
        if frame is None:
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Phone stream unavailable", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            frame = placeholder
    else:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, f"Demo: {test_scene}", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

    # Run detection pipeline if a frame is available
    context_items = []
    ocr_items = []
    prioritized = None
    if frame is not None:
        detections = st.session_state.detector.detect(frame)
        context_items = st.session_state.context_engine.process_detections(
            detections=detections, frame_shape=frame.shape, mode=selected_mode
        )
        # OCR handling for read/ask modes
        if selected_mode in (ProductMode.READ, ProductMode.ASK):
            ocr_items = st.session_state.ocr.read_text(frame)
        # Priority selection
        prioritized = st.session_state.priority_engine.select_top_item(
            context_items=context_items, mode=selected_mode, ocr_items=ocr_items
        )
        annotated = draw_visual_annotations(
            frame=frame,
            context_items=context_items,
            current_audio=st.session_state.last_speech,
            mode_label=selected_mode.name,
        )
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    else:
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)

    # Status strip (matches original spec)
    st.markdown(
        f"""
        <div class="status-strip">
            <span><span class="dot"></span>Camera: <b>Connected</b></span>
            <span><span class="dot"></span>Detection: <b>{'Running' if frame is not None else 'Idle'}</b></span>
            <span><span class="dot"></span>OCR: <b>{'Ready' if selected_mode in (ProductMode.READ, ProductMode.ASK) else 'Disabled'}</b></span>
            <span><span class="dot"></span>TTS: <b>Active</b></span>
            <span>Feed: <b>{feed_source}</b></span>
            <span>Mode: <b>{selected_mode.name}</b></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Three‑column layout
    col_live, col_objects, col_priority = st.columns([3, 2, 2])

    with col_live:
        st.subheader("📹 Live View")
        st.image(rgb, caption=f"Camera Input: {feed_source}")

    with col_objects:
        st.subheader("📦 Detected Objects")
        if context_items:
            # Build a simple table of objects
            table_data = []
            for item in context_items:
                obj = item.get("candidate", {})
                label = obj.get("object", "-")
                distance = obj.get("distance", "-")
                position = obj.get("position_desc", "-")
                table_data.append({"Object": label, "Distance": distance, "Position": position})
            st.table(table_data)
        else:
            st.write("No objects detected.")

    with col_priority:
        st.subheader("⚡ Priority Matrix")
        if prioritized:
            pri = prioritized
            cand = pri.get("candidate", {})
            st.write(f"**Object:** {cand.get('object', '-')}")
            st.write(f"**Priority:** {pri.get('priority', '-')}")
            st.write(f"**Critical:** {pri.get('is_critical', False)}")
        else:
            st.write("No prioritized item.")

    # Optional audio banner (bottom)
    st.markdown(
        f"""
        <div class="audio-banner">
            <span>🔊 CURRENT AUDIO:</span> "{st.session_state.last_speech}" 
        </div>
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    render_dashboard()
