"""
Unit tests for Text-To-Speech (TTS) Engine & Audio Stream Server
================================================================
Verifies non-blocking speech queue, emergency interrupt flushing,
listener callbacks, and mobile audio stream endpoint.
"""

import time
import pytest
from audio.tts import TextToSpeechEngine
from audio.audio_stream_server import AudioStreamServer, GLOBAL_AUDIO_STATE


def test_tts_speak_and_listener():
    tts = TextToSpeechEngine(mute=True)

    spoken_items = []

    def on_speak(text, is_critical):
        spoken_items.append((text, is_critical))

    tts.add_listener(on_speak)

    tts.speak("Testing audio guidance.", interrupt=False)
    time.sleep(0.2)

    assert len(spoken_items) == 1
    assert spoken_items[0][0] == "Testing audio guidance."
    assert spoken_items[0][1] is False

    tts.stop()


def test_tts_emergency_interrupt():
    tts = TextToSpeechEngine(mute=True)

    spoken_items = []
    tts.add_listener(lambda text, is_crit: spoken_items.append((text, is_crit)))

    # Queue normal message then immediate interrupt
    tts.speak("Long routine observation about the surroundings.", interrupt=False)
    tts.speak("Warning. Obstacle ahead.", interrupt=True)

    time.sleep(0.2)
    assert len(spoken_items) == 2
    assert spoken_items[1][0] == "Warning. Obstacle ahead."
    assert spoken_items[1][1] is True

    tts.stop()


def test_audio_relay_server_state():
    GLOBAL_AUDIO_STATE.update_utterance("Chair ahead 1.5m", is_critical=False)
    state = GLOBAL_AUDIO_STATE.get_state()
    assert state["current_utterance"] == "Chair ahead 1.5m"
    assert state["is_critical"] is False
    assert len(state["history"]) >= 1
