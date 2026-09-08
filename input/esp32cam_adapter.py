"""
ESP32-CAM Wireless Camera Adapter
==================================
Connects to an AI-Thinker ESP32-CAM module streaming MJPEG over Wi-Fi
(typically http://<esp32cam-ip>:81/stream).
Implements the shared BaseCamera interface so it drops into the pipeline
interchangeably with laptop webcam and phone stream.
"""

import logging
import time
from typing import Optional, Tuple
import cv2
import numpy as np

from input.camera_interface import BaseCamera

logger = logging.getLogger("VisionAssist.Input.ESP32Cam")

# Default ESP32-CAM stream endpoint (from Hardware Bring-Up Guide)
DEFAULT_ESP32_CAM_URL = "http://192.168.1.42:81/stream"


class ESP32CamAdapter(BaseCamera):
    """
    Adapter for wireless ESP32-CAM MJPEG video streams.
    Includes auto-reconnect handling for network stability.
    """

    def __init__(self, stream_url: str = DEFAULT_ESP32_CAM_URL, timeout_seconds: float = 5.0):
        self.stream_url = stream_url
        self.timeout_seconds = timeout_seconds
        self.cap: Optional[cv2.VideoCapture] = None
        self._resolution = (640, 480)
        self._last_successful_frame_time = 0.0

    def open(self) -> bool:
        """Connect to the ESP32-CAM MJPEG network stream."""
        logger.info(f"Connecting to ESP32-CAM stream at: {self.stream_url} ...")
        try:
            self.cap = cv2.VideoCapture(self.stream_url)
            if self.cap.isOpened():
                # Probe single frame to confirm valid stream
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    self._resolution = (w, h)
                    self._last_successful_frame_time = time.time()
                    logger.info(f"ESP32-CAM stream connected successfully! Resolution: {w}x{h}")
                    return True
                else:
                    logger.warning(f"Connected to {self.stream_url} but failed to decode initial frame.")
            else:
                logger.warning(f"Failed to open ESP32-CAM stream at {self.stream_url}.")
        except Exception as e:
            logger.error(f"Error connecting to ESP32-CAM stream: {e}")

        return False

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Grab a frame from the ESP32-CAM stream with auto-reconnect fallback.
        """
        if self.cap is None or not self.cap.isOpened():
            if not self.open():
                return False, None

        ret, frame = self.cap.read()
        if ret and frame is not None:
            self._last_successful_frame_time = time.time()
            return True, frame

        # Stream drop detected: attempt reconnect if timeout exceeded
        if time.time() - self._last_successful_frame_time > self.timeout_seconds:
            logger.warning("ESP32-CAM stream stalled. Attempting auto-reconnect...")
            self.release()
            self.open()

        return False, None

    def release(self) -> None:
        """Close stream connection."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                logger.debug(f"Error releasing ESP32-CAM capture: {e}")
            self.cap = None

    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    @property
    def resolution(self) -> Tuple[int, int]:
        return self._resolution


def get_capture(stream_url: str = DEFAULT_ESP32_CAM_URL) -> cv2.VideoCapture:
    """
    Direct drop-in OpenCV VideoCapture helper specified in
    Hardware Bring-Up Guide Section 4.4.
    """
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open ESP32-CAM stream at {stream_url} — check IP/Wi-Fi")
    return cap
