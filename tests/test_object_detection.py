"""
Unit tests for Object Detection Module (YOLOv8 Wrapper)
=======================================================
Verifies team interface contract:
{ "object": str, "confidence": float, "x_center": float, "y_center": float, "bbox": [x1, y1, x2, y2] }
"""

import numpy as np
import pytest
from perception.object_detection import ObjectDetector


def test_object_detector_initialization():
    detector = ObjectDetector(conf_thresh=0.45)
    assert detector is not None
    assert detector.conf_thresh == 0.45


def test_object_detector_empty_frame():
    detector = ObjectDetector()
    results = detector.detect(None)
    assert results == []


def test_object_detector_output_contract():
    detector = ObjectDetector()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = detector.detect(dummy_frame)

    assert isinstance(results, list)
    if len(results) > 0:
        item = results[0]
        assert "object" in item
        assert isinstance(item["object"], str)
        assert "confidence" in item
        assert isinstance(item["confidence"], float)
        assert "x_center" in item
        assert isinstance(item["x_center"], (float, int))
        assert "y_center" in item
        assert isinstance(item["y_center"], (float, int))
        assert "bbox" in item
        assert isinstance(item["bbox"], list)
        assert len(item["bbox"]) == 4

        # Bounds check
        x1, y1, x2, y2 = item["bbox"]
        assert 0 <= x1 <= 640
        assert 0 <= y1 <= 480
        assert 0 <= x2 <= 640
        assert 0 <= y2 <= 480


def test_object_detector_warmup_and_benchmark():
    detector = ObjectDetector()
    warmup_time = detector.warmup(frame_size=(240, 320, 3))
    assert warmup_time > 0.0

    bench = detector.benchmark(num_frames=3, frame_size=(240, 320, 3))
    assert "avg_latency_ms" in bench
    assert "fps" in bench
    assert bench["fps"] > 0
