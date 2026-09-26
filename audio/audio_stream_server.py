"""
Audio Stream Server for Phone Audio Endpoint
============================================
Provides an HTTP server for streaming TTS speech to any smartphone
browser or earphones over the local Wi-Fi / hotspot network.
"""

from http.server import HTTPServer
import logging
import threading
import time
from typing import Optional, List, Dict, Any

logger = logging.getLogger("VisionAssist.AudioServer")


class GlobalAudioState:
    """Thread-safe global audio state for wearable clients."""

    def __init__(self):
        self._lock = threading.Lock()
        self.current_utterance: str = ""
        self.is_critical: bool = False
        self.timestamp: float = 0.0
        self.history: List[Dict[str, Any]] = []

    def update_utterance(self, text: str, is_critical: bool = False) -> None:
        with self._lock:
            self.current_utterance = text
            self.is_critical = is_critical
            self.timestamp = time.time()
            self.history.append({
                "text": text,
                "is_critical": is_critical,
                "timestamp": self.timestamp,
            })

    def get_state(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "current_utterance": self.current_utterance,
                "is_critical": self.is_critical,
                "timestamp": self.timestamp,
                "history": list(self.history),
            }


GLOBAL_AUDIO_STATE = GlobalAudioState()


def notify_phone_audio(text: str, is_critical: bool = False) -> None:
    """Publish an utterance to all connected wearable phone clients."""
    GLOBAL_AUDIO_STATE.update_utterance(text, is_critical=is_critical)
    try:
        from audio.tts import _SPEECH_LOCK, _LATEST_SPEECH
        with _SPEECH_LOCK:
            _LATEST_SPEECH["id"] = _LATEST_SPEECH.get("id", 0) + 1
            _LATEST_SPEECH["text"] = text
            _LATEST_SPEECH["timestamp"] = time.time()
            _LATEST_SPEECH["is_urgent"] = is_critical
    except Exception:
        pass


class AudioStreamServer:
    """Manages audio relay HTTP server for wearable phone audio."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8088):  # nosec B104
        self.host = host
        self.port = port
        self.server: Optional[HTTPServer] = None
        self.thread: Optional[threading.Thread] = None
        self.is_running = False

    def start(self) -> None:
        """Starts the audio relay HTTP server on a daemon thread."""
        if self.is_running:
            return
        try:
            from audio.tts import _PhoneAudioHTTPHandler
            self.server = HTTPServer(
                (self.host, self.port),
                _PhoneAudioHTTPHandler
            )
            self.thread = threading.Thread(
                target=self.server.serve_forever,
                daemon=True
            )
            self.thread.start()
            self.is_running = True
            logger.info(
                f"AudioStreamServer running at "
                f"http://{self.host}:{self.port}/phone-audio"
            )
        except Exception as e:
            logger.debug(f"AudioStreamServer port {self.port} notice: {e}")

    def stop(self) -> None:
        """Stops the audio relay HTTP server."""
        if self.server and self.is_running:
            try:
                self.server.shutdown()
                self.server.server_close()
            except Exception as e:
                logger.debug(f"Error stopping AudioStreamServer: {e}")
            self.is_running = False
