"""
ESP8266 Companion Node Adapter (Physical Risk LED & Remote Trigger)
===================================================================
Interfaces with an ESP8266 NodeMCU companion node over Wi-Fi
as described in Hardware Bring-Up Guide Section 06.
Features:
  - Updates physical LED state: GET /led?level=high | med | low
  - Reads remote trigger button: GET /status -> {"button_pressed": bool}
"""

import logging
from typing import Optional
import requests

logger = logging.getLogger("VisionAssist.Hardware.ESP8266")

DEFAULT_ESP8266_URL = "http://192.168.1.44"


class ESP8266CompanionAdapter:
    """
    Client for the ESP8266 companion node.
    Mirrors high-risk priority alerts to a physical LED and polls a physical button.
    """

    def __init__(self, base_url: str = DEFAULT_ESP8266_URL, timeout: float = 0.5):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._current_level: str = "low"

    def set_risk_level(self, level: str) -> bool:
        """
        Sends current risk level ('high', 'med', 'low') to the ESP8266 LED endpoint.
        Per Section 6: /led?level=high triggers the physical warning LED.
        """
        level_normalized = level.lower().strip()
        if level_normalized == self._current_level:
            return True  # Avoid unnecessary network spam if level hasn't changed

        url = f"{self.base_url}/led?level={level_normalized}"
        try:
            r = requests.get(url, timeout=self.timeout)
            if r.status_code == 200:
                self._current_level = level_normalized
                return True
        except requests.RequestException as e:
            logger.debug(f"Could not reach ESP8266 companion node at {url}: {e}")
        return False

    def is_button_pressed(self) -> bool:
        """
        Polls /status endpoint to check if remote companion button is pressed.
        """
        url = f"{self.base_url}/status"
        try:
            r = requests.get(url, timeout=self.timeout)
            if r.status_code == 200:
                data = r.json()
                return bool(data.get("button_pressed", False))
        except requests.RequestException:
            pass
        return False
