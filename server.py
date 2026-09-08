"""
VisionAssist - Web Server & Real-Time Processing Gateway
=========================================================
Provides real-time WebSocket and HTTP endpoints for the VisionAssist
desktop/web frontend.

Connects the plain HTML/CSS/JS frontend with the existing Python
perception (YOLOv8 + OCR) and intelligence (Context, Priority, Ask) pipeline.
"""

import base64
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import ProductMode, DEFAULT_MODE, PHONE_STREAM_URL
from perception.object_detection import ObjectDetector
from perception.ocr import OCRReader
from intelligence.context_engine import ContextEngine
from intelligence.priority_engine import PriorityEngine
from intelligence.response_generator import ResponseGenerator
from intelligence.ask_engine import AskEngine
from utils.logger import setup_logger

logger = setup_logger("VisionAssist.Server")

app = FastAPI(title="VisionAssist Web API", version="1.0.0")

# Enable CORS for local network and browser devices
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "ui" / "web"
ASSETS_DIR = WEB_DIR / "assets"

# Initialize Core AI Pipeline Singletons
logger.info("Initializing VisionAssist Core Pipeline...")
detector = ObjectDetector()
ocr = OCRReader()
context_engine = ContextEngine()
priority_engine = PriorityEngine(cooldown_seconds=2.5)
response_generator = ResponseGenerator()
ask_engine = AskEngine()
logger.info("VisionAssist Core Pipeline loaded and ready.")


class FramePayload(BaseModel):
    image: str  # Base64 data URL
    mode: str = "obstacle"
    query: Optional[str] = None
    force_refresh: bool = False


def decode_base64_image(data_url: str) -> Optional[np.ndarray]:
    """Decodes a base64 DataURL or raw base64 string to an OpenCV BGR frame."""
    try:
        if "," in data_url:
            data_url = data_url.split(",", 1)[1]
        img_bytes = base64.b64decode(data_url)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        logger.error(f"Failed to decode base64 frame: {e}")
        return None


def process_frame_pipeline(
    frame: np.ndarray,
    mode_str: str = "obstacle",
    query: Optional[str] = None,
    force_refresh: bool = False
) -> Dict[str, Any]:
    """
    Executes the full pipeline:
    Frame -> Perception (YOLO / OCR) -> Context Engine -> Priority Engine -> Response Generator
    """
    t_start = time.perf_counter()
    h, w = frame.shape[:2]

    # Map string to ProductMode
    mode_map = {
        "quick_look": ProductMode.QUICK_LOOK,
        "obstacle": ProductMode.OBSTACLE_AWARENESS,
        "read": ProductMode.READ,
        "ask": ProductMode.ASK,
        "safety": ProductMode.SAFETY_ALERT,
        "safety_alert": ProductMode.SAFETY_ALERT,
    }
    mode = mode_map.get(mode_str.lower(), ProductMode.OBSTACLE_AWARENESS)

    # 1. Perception Layer
    detections = detector.detect(frame)
    ocr_items = []
    if mode in (ProductMode.READ, ProductMode.ASK):
        ocr_items = ocr.read_text(frame)

    # 2. Context Engine
    context_items = context_engine.process_detections(
        detections=detections,
        frame_shape=frame.shape,
        mode=mode
    )

    # 3. Priority & Reasoning
    speech_text = ""
    risk_level = "low"
    primary_label = ""
    primary_distance = 0.0
    is_critical = False

    if mode == ProductMode.ASK:
        active_q = query or "What is in front of me?"
        ans = ask_engine.ask(
            query=active_q,
            frame=frame,
            context_items=context_items,
            ocr_items=ocr_items
        )
        speech_text = ans or "The path ahead appears clear."
        risk_level = "medium"
        primary_label = "Query"
    else:
        top_item = priority_engine.select_top_item(
            context_items=context_items,
            mode=mode,
            ocr_items=ocr_items,
            force_refresh=force_refresh,
            user_query=query
        )

        if top_item:
            res = response_generator.generate(top_item, language="en")
            speech_text = res.get("text", "")
            is_critical = top_item.get("is_critical", False) or (mode == ProductMode.SAFETY_ALERT)

            candidate = top_item.get("candidate")
            if candidate:
                primary_label = candidate.get("object", "")
                primary_distance = candidate.get("distance", 0.0)
                risk_score = candidate.get("risk_score", 0.0)
                if is_critical or risk_score >= 65.0:
                    risk_level = "high"
                elif risk_score >= 35.0:
                    risk_level = "medium"
                else:
                    risk_level = "low"
            elif top_item.get("type") == "ocr":
                primary_label = "Text"
                risk_level = "low"
            elif top_item.get("type") == "quick_look":
                primary = top_item.get("primary", {})
                primary_label = primary.get("object", "Scene")
                primary_distance = primary.get("distance", 0.0)
                risk_level = "medium"

    # Default speech text fallback if nothing critical
    if not speech_text:
        if mode == ProductMode.OBSTACLE_AWARENESS:
            speech_text = ""  # Quiet guardian concept
        elif mode == ProductMode.READ:
            speech_text = "Looking for readable signs or text."
        elif mode == ProductMode.SAFETY_ALERT:
            speech_text = "Area clear. No immediate collision hazard."
        else:
            speech_text = "Looking..."

    latency_ms = (time.perf_counter() - t_start) * 1000.0

    return {
        "type": "response",
        "mode": mode.value,
        "message": speech_text,
        "risk": risk_level,
        "is_critical": is_critical,
        "label": primary_label,
        "distance": round(primary_distance, 1),
        "detections": context_items,
        "ocr_items": ocr_items,
        "latency_ms": round(latency_ms, 1),
        "timestamp": time.time(),
        "system_status": {
            "camera": "Active",
            "processing": "Running",
            "connection": "Connected"
        }
    }


# WebSocket endpoint for real-time video streaming & response
@app.websocket("/ws/process")
async def websocket_process(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected to /ws/process")
    try:
        while True:
            data_text = await websocket.receive_text()
            try:
                msg = json.loads(data_text)
                msg_type = msg.get("type", "frame")

                if msg_type == "reset_cooldown":
                    priority_engine.reset_cooldown()
                    await websocket.send_json({"type": "ack", "status": "cooldown_reset"})
                    continue

                if msg_type == "frame":
                    img_data = msg.get("image", "")
                    mode = msg.get("mode", "obstacle")
                    query = msg.get("query", None)
                    force = msg.get("force", False)

                    if not img_data:
                        continue

                    frame = decode_base64_image(img_data)
                    if frame is None:
                        continue

                    # Process frame
                    result = process_frame_pipeline(
                        frame=frame,
                        mode_str=mode,
                        query=query,
                        force_refresh=force
                    )
                    await websocket.send_json(result)

            except json.JSONDecodeError:
                pass
            except Exception as e:
                logger.error(f"Error handling websocket frame: {e}")
                await websocket.send_json({
                    "type": "error",
                    "error": str(e),
                    "system_status": {
                        "camera": "Active",
                        "processing": "Error",
                        "connection": "Connected"
                    }
                })

    except WebSocketDisconnect:
        logger.info("Client disconnected from /ws/process")
    except Exception as e:
        logger.error(f"WebSocket unhandled error: {e}")


@app.post("/api/process_frame")
async def api_process_frame(payload: FramePayload):
    """HTTP endpoint fallback for frame processing."""
    frame = decode_base64_image(payload.image)
    if frame is None:
        return JSONResponse(status_code=400, content={"error": "Invalid base64 image data"})

    result = process_frame_pipeline(
        frame=frame,
        mode_str=payload.mode,
        query=payload.query,
        force_refresh=payload.force_refresh
    )
    return JSONResponse(content=result)


@app.get("/api/status")
async def api_status():
    """System health check and perception metadata."""
    return {
        "status": "online",
        "detector_live": detector.is_live,
        "model": detector.model_name,
        "default_mode": DEFAULT_MODE.value,
        "cooldown_seconds": priority_engine.cooldown_seconds,
        "phone_stream_url": PHONE_STREAM_URL,
    }


# Static Files and Page Routes
if WEB_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/")
async def get_index():
    index_file = WEB_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>VisionAssist UI loading...</h1>")


@app.get("/app")
async def get_app_view():
    index_file = WEB_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>VisionAssist UI loading...</h1>")


def run_web_server(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    logger.info(f"Starting VisionAssist Web Server on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_web_server()
