"""
Unit tests for Optical Character Recognition (OCR) Reader
=========================================================
Verifies text extraction, CLAHE contrast enhancement, and bounding box contract:
{ "text": str, "confidence": float, "bbox": [x1, y1, x2, y2], "x_center": float, "y_center": float }
"""

import cv2
import numpy as np
import pytest
from perception.ocr import OCRReader


def test_ocr_reader_initialization():
    ocr = OCRReader(languages=["en"])
    assert ocr is not None
    assert "en" in ocr.languages


def test_ocr_reader_empty_frame():
    ocr = OCRReader()
    results = ocr.read_text(None)
    assert results == []


def test_ocr_preprocessing_clahe():
    ocr = OCRReader()
    dummy_bgr = np.ones((100, 100, 3), dtype=np.uint8) * 128
    enhanced = ocr.preprocess_image(dummy_bgr)
    assert enhanced is not None
    assert len(enhanced.shape) == 2  # Converted to 2D grayscale


def test_ocr_reader_output_contract():
    ocr = OCRReader()
    dummy_frame = np.ones((200, 400, 3), dtype=np.uint8) * 255
    cv2.putText(dummy_frame, "ROOM 204", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    results = ocr.read_text(dummy_frame)
    assert isinstance(results, list)
    assert len(results) > 0

    item = results[0]
    assert "text" in item
    assert isinstance(item["text"], str)
    assert "confidence" in item
    assert "bbox" in item
    assert len(item["bbox"]) == 4
    assert "x_center" in item
    assert "y_center" in item


def test_ocr_warmup_and_benchmark():
    ocr = OCRReader()
    warmup_ms = ocr.warmup(frame_size=(200, 300, 3))
    assert warmup_ms > 0

    bench = ocr.benchmark(num_iterations=2)
    assert "avg_latency_ms" in bench
    assert bench["avg_latency_ms"] > 0
