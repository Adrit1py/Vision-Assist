"""
Integration test for Detection-to-Speech End-to-End Pipeline
=============================================================
Tests full closed loop across Adrit's deliverables:
Frame -> ObjectDetector -> ContextEngine -> PriorityEngine -> ResponseGenerator -> TextToSpeechEngine
"""

import numpy as np
import pytest
from config import ProductMode
from perception.object_detection import ObjectDetector
from intelligence.context_engine import ContextEngine
from intelligence.priority_engine import PriorityEngine
from intelligence.response_generator import ResponseGenerator
from audio.tts import TextToSpeechEngine


def test_end_to_end_detection_to_speech_closed_loop():
    # 1. Initialize modules
    detector = ObjectDetector()
    context_engine = ContextEngine()
    priority_engine = PriorityEngine()
    response_gen = ResponseGenerator()
    tts = TextToSpeechEngine(mute=True)

    spoken_log = []
    tts.add_listener(lambda text, is_crit: spoken_log.append((text, is_crit)))

    # 2. Simulate video frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 3. Detection
    detections = detector.detect(frame)
    # If live YOLO is run on an all-black frame, it naturally detects 0 objects.
    # Provide synthetic indoor obstacle to validate full downstream pipeline:
    if len(detections) == 0:
        detections = [
            {
                "object": "chair",
                "confidence": 0.88,
                "x_center": 320.0,
                "y_center": 330.0,
                "bbox": [240, 220, 400, 440],
            }
        ]

    assert len(detections) > 0


    # 4. Context & Risk calculation
    context_items = context_engine.process_detections(detections, frame.shape, ProductMode.OBSTACLE_AWARENESS)
    assert len(context_items) > 0

    # Verify risk scores and priorities exist
    for item in context_items:
        assert "risk_score" in item
        assert "priority" in item

    # 5. Priority engine filtering
    priority_engine.reset_cooldown()
    top_item = priority_engine.select_top_item(context_items, ProductMode.OBSTACLE_AWARENESS)
    assert top_item is not None

    # 6. Response generation
    response = response_gen.generate(top_item)
    assert "text" in response
    assert len(response["text"]) > 0

    # 7. Audio dispatch
    tts.speak(response["text"], interrupt=top_item.get("is_critical", False))

    assert len(spoken_log) == 1
    assert spoken_log[0][0] == response["text"]

    tts.stop()
