"""
Unit tests for Spatial Position & Monocular Depth Estimation
============================================================
Verifies 3-zone and 7-zone spatial positioning, pinhole projection distance math,
and speech formatting.
"""

import pytest
from perception.position import (
    classify_position,
    calculate_bounding_box_center,
    get_position_offset_description,
    PositionClassifier,
)
from perception.depth import (
    estimate_distance_and_proximity,
    format_distance_speech,
    DepthEstimator,
)


def test_position_classification():
    frame_width = 640

    # Left: x < 0.33 * 640 = 211.2
    assert classify_position(100.0, frame_width) == "Left"
    assert PositionClassifier.classify(100.0, frame_width) == "Left"

    # Centre: 211.2 <= x <= 422.4
    assert classify_position(320.0, frame_width) == "Centre"

    # Right: x > 422.4
    assert classify_position(550.0, frame_width) == "Right"


def test_position_offset_descriptions():
    frame_width = 640
    # Center
    assert get_position_offset_description(320.0, frame_width) == "directly ahead"
    # Left
    assert "left" in get_position_offset_description(100.0, frame_width)
    # Right
    assert "right" in get_position_offset_description(550.0, frame_width)


def test_bounding_box_center():
    bbox = [100, 200, 300, 400]
    xc, yc = calculate_bounding_box_center(bbox)
    assert xc == 200.0
    assert yc == 300.0


def test_depth_estimation_classes():
    # Large bounding box (near object)
    bbox_close = [100, 100, 500, 460]  # height = 360
    dist_close, prox_close = estimate_distance_and_proximity("chair", bbox_close, 480, 640)

    # Small bounding box (far object)
    bbox_far = [280, 50, 360, 110]  # height = 60
    dist_far, prox_far = estimate_distance_and_proximity("chair", bbox_far, 480, 640)

    assert dist_close < dist_far
    assert prox_close in ["Immediate", "Near"]
    assert prox_far in ["Mid-range", "Far"]


def test_format_distance_speech():
    assert format_distance_speech(0.8) == "less than one metre"
    assert format_distance_speech(1.5) == "approximately 1.5 metres"
    assert "2 metres away" in format_distance_speech(2.0)
