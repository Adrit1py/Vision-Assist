"""
Unit tests for Hardware Adapters
================================
Verifies ESP32-CAM adapter, Ultrasonic sensor node polling, and ESP8266 companion node client.
"""

from unittest.mock import MagicMock, patch
import pytest
import requests

from input.esp32cam_adapter import ESP32CamAdapter
from hardware.ultrasonic_adapter import UltrasonicAdapter, read_sensor_node
from hardware.esp8266_adapter import ESP8266CompanionAdapter


def test_esp32cam_adapter_initialization():
    adapter = ESP32CamAdapter("http://192.168.1.42:81/stream")
    assert adapter.stream_url == "http://192.168.1.42:81/stream"
    assert adapter.resolution == (640, 480)
    assert not adapter.is_opened()


def test_ultrasonic_adapter_mock_response():
    adapter = UltrasonicAdapter("http://192.168.1.43/status")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"distance_cm": 150.0, "button_pressed": True}

    with patch("requests.get", return_value=mock_resp):
        dist_m = adapter.read_distance_meters()
        assert dist_m == 1.50
        assert adapter.is_button_pressed() is True


def test_ultrasonic_adapter_unreachable():
    adapter = UltrasonicAdapter("http://192.168.1.43/status")

    with patch("requests.get", side_effect=requests.RequestException("Timeout")):
        assert adapter.read_distance_meters() is None
        assert adapter.is_button_pressed() is False


def test_esp8266_companion_adapter():
    client = ESP8266CompanionAdapter("http://192.168.1.44")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"button_pressed": True}

    with patch("requests.get", return_value=mock_resp):
        assert client.set_risk_level("high") is True
        assert client.is_button_pressed() is True
