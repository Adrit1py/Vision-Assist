/**
 * VisionAssist - Frontend Application & Intelligence Client
 * ==========================================================
 * Connects browser video capture and speech synthesis to the
 * Python VisionAssist processing pipeline via WebSockets.
 */

// Application State
const state = {
  activeView: 'landing', // 'landing' | 'app'
  activeMode: 'obstacle', // 'quick_look' | 'obstacle' | 'read' | 'ask' | 'safety'
  cameraActive: false,
  stream: null,
  availableDevices: [],
  selectedDeviceId: null,
  ws: null,
  connected: false,
  lastSpokenText: 'Stairs ahead, approximately 2 metres.',
  lastSpeechTime: 0,
  isProcessing: false,
  useSimulation: false,
  speechSynth: window.speechSynthesis || null,
  fps: 0,
};

// DOM References
const elements = {
  landingView: document.getElementById('landingView'),
  appView: document.getElementById('appView'),
  permScreen: document.getElementById('permScreen'),
  btnStartAssist: document.getElementById('btnStartAssist'),
  btnTryHero: document.getElementById('btnTryHero'),
  btnTryNav: document.getElementById('btnTryNav'),
  btnLaunchBottom: document.getElementById('btnLaunchBottom'),
  btnStartDemo: document.getElementById('btnStartDemo'),
  videoFeed: document.getElementById('videoFeed'),
  canvasOverlay: document.getElementById('canvasOverlay'),
  simulatedFeed: document.getElementById('simulatedFeed'),
  bracketBox: document.getElementById('bracketBox'),
  bracketLabel: document.getElementById('bracketLabel'),
  bracketDist: document.getElementById('bracketDist'),
  readoutPanel: document.getElementById('readoutPanel'),
  readoutMessage: document.getElementById('readoutMessage'),
  readoutMeta: document.getElementById('readoutMeta'),
  readoutSpeaker: document.getElementById('readoutSpeaker'),
  waveform: document.getElementById('waveform'),
  btnReplay: document.getElementById('btnReplay'),
  btnCameraToggle: document.getElementById('btnCameraToggle'),
  cameraLabelText: document.getElementById('cameraLabelText'),
  btnHelp: document.getElementById('btnHelp'),
  btnHelpRail: document.getElementById('btnHelpRail'),
  helpModal: document.getElementById('helpModal'),
  btnModalClose: document.getElementById('btnModalClose'),
  btnModalGotIt: document.getElementById('btnModalGotIt'),
  currentModeIcon: document.getElementById('currentModeIcon'),
  currentModeTitle: document.getElementById('currentModeTitle'),
  currentModeDesc: document.getElementById('currentModeDesc'),
  statusValCamera: document.getElementById('statusValCamera'),
  statusValProcessing: document.getElementById('statusValProcessing'),
  statusValConnection: document.getElementById('statusValConnection'),
  appActiveDot: document.getElementById('appActiveDot'),
  appActiveText: document.getElementById('appActiveText'),
  cameraSelect: document.getElementById('cameraSelect'),
  modeButtons: document.querySelectorAll('[data-mode-btn]'),
};

// Offscreen canvas for frame capture
const captureCanvas = document.createElement('canvas');
const captureCtx = captureCanvas.getContext('2d');

/* ==========================================================================
   VIEW NAVIGATION & INITIALIZATION
   ========================================================================== */
function showAppView() {
  state.activeView = 'app';
  if (elements.landingView) elements.landingView.style.display = 'none';
  if (elements.appView) elements.appView.classList.add('active');
  window.scrollTo({ top: 0, behavior: 'smooth' });

  // If camera not yet requested, show permission screen
  if (!state.cameraActive) {
    if (elements.permScreen) elements.permScreen.classList.remove('hidden');
  } else {
    if (elements.permScreen) elements.permScreen.classList.add('hidden');
  }
}

function showLandingView() {
  state.activeView = 'landing';
  if (elements.appView) elements.appView.classList.remove('active');
  if (elements.landingView) elements.landingView.style.display = 'block';
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

/* ==========================================================================
   CAMERA ACCESS & STREAM MANAGEMENT
   ========================================================================== */
async function enumerateCameras() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    state.availableDevices = devices.filter((d) => d.kind === 'videoinput');
    if (elements.cameraSelect) {
      if (state.availableDevices.length > 1) {
        elements.cameraSelect.style.display = 'inline-block';
        elements.cameraSelect.innerHTML = '';
        state.availableDevices.forEach((device, idx) => {
          const opt = document.createElement('option');
          opt.value = device.deviceId;
          opt.textContent = device.label || `Camera ${idx + 1}`;
          elements.cameraSelect.appendChild(opt);
        });
      } else {
        elements.cameraSelect.style.display = 'none';
      }
    }
  } catch (err) {
    console.warn('Camera enumeration error:', err);
  }
}

async function startCamera(deviceId = null) {
  try {
    if (state.stream) {
      state.stream.getTracks().forEach((track) => track.stop());
    }

    const constraints = {
      video: deviceId
        ? { deviceId: { exact: deviceId } }
        : { facingMode: { ideal: 'environment' }, width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    };

    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    state.stream = stream;
    state.cameraActive = true;
    state.useSimulation = false;

    if (elements.videoFeed) {
      elements.videoFeed.srcObject = stream;
      elements.videoFeed.style.display = 'block';
      await elements.videoFeed.play();
    }

    if (elements.simulatedFeed) elements.simulatedFeed.style.display = 'none';
    if (elements.permScreen) elements.permScreen.classList.add('hidden');
    if (elements.cameraLabelText) elements.cameraLabelText.textContent = 'Camera On';
    if (elements.statusValCamera) elements.statusValCamera.textContent = 'Active';

    await enumerateCameras();
    speak('VisionAssist online. Obstacle awareness active.');
  } catch (err) {
    console.warn('Camera access error. Falling back to test scene:', err);
    useSimulatedScene();
  }
}

function useSimulatedScene() {
  state.cameraActive = true;
  state.useSimulation = true;
  if (elements.videoFeed) elements.videoFeed.style.display = 'none';
  if (elements.simulatedFeed) elements.simulatedFeed.style.display = 'block';
  if (elements.permScreen) elements.permScreen.classList.add('hidden');
  if (elements.cameraLabelText) elements.cameraLabelText.textContent = 'Camera On (Demo)';
  if (elements.statusValCamera) elements.statusValCamera.textContent = 'Active (Demo)';
  speak('VisionAssist online. Demo test feed active.');
}

function stopCamera() {
  if (state.stream) {
    state.stream.getTracks().forEach((track) => track.stop());
    state.stream = null;
  }
  state.cameraActive = false;
  if (elements.videoFeed) elements.videoFeed.style.display = 'none';
  if (elements.simulatedFeed) elements.simulatedFeed.style.display = 'none';
  if (elements.cameraLabelText) elements.cameraLabelText.textContent = 'Camera Off';
  if (elements.statusValCamera) elements.statusValCamera.textContent = 'Off';
  if (elements.permScreen) elements.permScreen.classList.remove('hidden');
}

function toggleCamera() {
  if (state.cameraActive) {
    stopCamera();
  } else {
    startCamera(state.selectedDeviceId);
  }
}

/* ==========================================================================
   WEBSOCKET BACKEND INTEGRATION
   ========================================================================== */
function initWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/process`;

  try {
    state.ws = new WebSocket(wsUrl);

    state.ws.onopen = () => {
      state.connected = true;
      if (elements.statusValConnection) elements.statusValConnection.textContent = 'Connected';
      console.log('WebSocket connected to VisionAssist processing server.');
    };

    state.ws.onmessage = (event) => {
      state.isProcessing = false;
      if (elements.statusValProcessing) elements.statusValProcessing.textContent = 'Running';
      try {
        const data = JSON.parse(event.data);
        handleServerResponse(data);
      } catch (e) {
        console.error('Error parsing backend response:', e);
      }
    };

    state.ws.onclose = () => {
      state.connected = false;
      if (elements.statusValConnection) elements.statusValConnection.textContent = 'Disconnected';
      setTimeout(initWebSocket, 2000);
    };

    state.ws.onerror = () => {
      state.connected = false;
      if (elements.statusValConnection) elements.statusValConnection.textContent = 'Retry';
    };
  } catch (err) {
    console.warn('WebSocket connection not available. Using HTTP API fallback:', err);
  }
}

function captureFrameBase64() {
  if (!state.cameraActive) return null;

  if (state.useSimulation) {
    // Return sample stairs image from DOM
    if (elements.simulatedFeed && elements.simulatedFeed.complete) {
      captureCanvas.width = elements.simulatedFeed.naturalWidth || 640;
      captureCanvas.height = elements.simulatedFeed.naturalHeight || 480;
      captureCtx.drawImage(elements.simulatedFeed, 0, 0, captureCanvas.width, captureCanvas.height);
      return captureCanvas.toDataURL('image/jpeg', 0.8);
    }
    return null;
  }

  if (elements.videoFeed && elements.videoFeed.readyState >= 2) {
    captureCanvas.width = 480;
    captureCanvas.height = 360;
    captureCtx.drawImage(elements.videoFeed, 0, 0, captureCanvas.width, captureCanvas.height);
    return captureCanvas.toDataURL('image/jpeg', 0.75);
  }
  return null;
}

function sendFrameForProcessing() {
  if (state.isProcessing || !state.cameraActive) return;

  const base64Img = captureFrameBase64();
  if (!base64Img) return;

  state.isProcessing = true;
  if (elements.statusValProcessing) elements.statusValProcessing.textContent = 'Analyzing';

  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(
      JSON.stringify({
        type: 'frame',
        image: base64Img,
        mode: state.activeMode,
        timestamp: Date.now(),
      })
    );
  } else {
    // HTTP API Fallback
    fetch('/api/process_frame', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image: base64Img,
        mode: state.activeMode,
      }),
    })
      .then((res) => res.json())
      .then((data) => {
        state.isProcessing = false;
        handleServerResponse(data);
      })
      .catch((err) => {
        state.isProcessing = false;
        console.warn('HTTP pipeline fallback error:', err);
      });
  }
}

/* ==========================================================================
   RESPONSE HANDLING & VISUAL ANNOTATIONS
   ========================================================================== */
function handleServerResponse(data) {
  if (!data) return;

  const message = data.message || '';
  const risk = data.risk || 'low';
  const label = data.label || 'Stairs';
  const distance = data.distance || 2.0;
  const detections = data.detections || [];

  // Update Readout Box
  if (message && message.trim().length > 0) {
    if (elements.readoutMessage) elements.readoutMessage.textContent = message;
    state.lastSpokenText = message;
    speak(message);
  }

  // Update Danger styling
  const isDanger = risk === 'high' || state.activeMode === 'safety';
  if (elements.readoutPanel) {
    if (isDanger) {
      elements.readoutPanel.classList.add('danger-active');
    } else {
      elements.readoutPanel.classList.remove('danger-active');
    }
  }

  // Update Bracket Bounding Box
  if (elements.bracketBox) {
    if (isDanger) {
      elements.bracketBox.classList.add('danger');
    } else {
      elements.bracketBox.classList.remove('danger');
    }

    // Position bracket based on primary detection if available
    if (detections.length > 0) {
      const topDet = detections[0];
      const bbox = topDet.bbox;
      if (bbox && bbox.length === 4) {
        const [x1, y1, x2, y2] = bbox;
        const w = captureCanvas.width || 640;
        const h = captureCanvas.height || 480;

        const leftPct = (x1 / w) * 100;
        const topPct = (y1 / h) * 100;
        const widthPct = ((x2 - x1) / w) * 100;
        const heightPct = ((y2 - y1) / h) * 100;

        elements.bracketBox.style.left = `${Math.max(5, Math.min(80, leftPct))}%`;
        elements.bracketBox.style.top = `${Math.max(5, Math.min(80, topPct))}%`;
        elements.bracketBox.style.width = `${Math.max(15, Math.min(90, widthPct))}%`;
        elements.bracketBox.style.height = `${Math.max(15, Math.min(90, heightPct))}%`;
      }
    }

    if (elements.bracketLabel) elements.bracketLabel.textContent = capitalize(label || 'Stairs');
    if (elements.bracketDist) elements.bracketDist.textContent = `${distance} m`;
  }

  // Update Metadata Text
  if (elements.readoutMeta) {
    const modeLabels = {
      obstacle: 'Obstacle Awareness',
      quick_look: 'Quick Look',
      read: 'Read Mode',
      ask: 'Visual Q&A',
      safety: 'Safety Alert',
    };
    const modeName = modeLabels[state.activeMode] || 'Obstacle Awareness';
    const riskLabel = isDanger ? 'High Risk' : 'Standard Guidance';
    const dotColor = isDanger ? 'red' : 'green';
    elements.readoutMeta.innerHTML = `
      <span class="meta-risk-indicator">
        <span class="status-dot ${dotColor}"></span> ${modeName} · ${riskLabel}
      </span>
    `;
  }
}

function capitalize(s) {
  if (!s) return '';
  return s.charAt(0).toUpperCase() + s.slice(1);
}

/* ==========================================================================
   TEXT-TO-SPEECH (TTS) & REPLAY
   ========================================================================== */
function speak(text, force = false) {
  if (!state.speechSynth || !text) return;

  const now = Date.now();
  // Avoid audio spam unless forced or cooldown passed
  if (!force && text === state.lastSpokenText && now - state.lastSpeechTime < 4000) {
    return;
  }

  state.speechSynth.cancel(); // Interrupt prior speech
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 1.05;
  utterance.pitch = 1.0;
  utterance.lang = 'en-US';

  // Visual pulse on waveform
  if (elements.waveform) elements.waveform.style.opacity = '1';
  utterance.onend = () => {
    if (elements.waveform) elements.waveform.style.opacity = '0.4';
  };

  state.speechSynth.speak(utterance);
  state.lastSpeechTime = now;
  state.lastSpokenText = text;
}

function replayLastSpoken() {
  if (state.lastSpokenText) {
    speak(state.lastSpokenText, true);
  }
}

/* ==========================================================================
   MODE SWITCHING
   ========================================================================== */
function switchMode(newMode) {
  state.activeMode = newMode;

  // Update mode buttons UI
  elements.modeButtons.forEach((btn) => {
    const btnMode = btn.getAttribute('data-mode-btn');
    btn.classList.remove('active-obstacle', 'active-safety', 'active-mint');

    if (btnMode === newMode) {
      if (newMode === 'obstacle') btn.classList.add('active-obstacle');
      else if (newMode === 'safety') btn.classList.add('active-safety');
      else btn.classList.add('active-mint');
    }
  });

  // Update Right Sidebar widget
  const modeData = {
    obstacle: {
      title: 'Obstacle Awareness',
      desc: 'Continuous guidance for a safer journey.',
      icon: `<path d="M13 4a2 2 0 1 0 0-4 2 2 0 0 0 0 4ZM6 8l3 3v7h2v-6l-2-3 2-2 3 3v8h2v-9l-3-3a2 2 0 0 0-2.8 0L6 8Z"/>`,
    },
    quick_look: {
      title: 'Quick Look',
      desc: 'Get an instant snapshot of your surroundings.',
      icon: `<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8Z"/><circle cx="12" cy="12" r="3"/>`,
    },
    read: {
      title: 'Read Mode',
      desc: 'Reads signs, text, and documents aloud.',
      icon: `<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>`,
    },
    ask: {
      title: 'Visual Q&A',
      desc: 'Ask questions about your surroundings.',
      icon: `<circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/>`,
    },
    safety: {
      title: 'Safety Alert',
      desc: 'Urgent interruption for immediate hazards.',
      icon: `<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>`,
    },
  };

  const info = modeData[newMode] || modeData.obstacle;
  if (elements.currentModeTitle) elements.currentModeTitle.textContent = info.title;
  if (elements.currentModeDesc) elements.currentModeDesc.textContent = info.desc;
  if (elements.currentModeIcon) elements.currentModeIcon.innerHTML = info.icon;

  // Immediate speech announcement for mode switch
  speak(`${info.title} active.`);

  // Reset server cooldown on mode switch
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({ type: 'reset_cooldown' }));
  }

  // Force next frame to analyze with the new mode
  sendFrameForProcessing();
}

/* ==========================================================================
   HELP MODAL DIALOG
   ========================================================================== */
function openHelpModal() {
  if (elements.helpModal) {
    elements.helpModal.classList.add('active');
    elements.btnModalGotIt.focus();
  }
}

function closeHelpModal() {
  if (elements.helpModal) {
    elements.helpModal.classList.remove('active');
    if (elements.btnHelp) elements.btnHelp.focus();
  }
}

/* ==========================================================================
   EVENT LISTENERS & BINDINGS
   ========================================================================== */
function attachEventListeners() {
  // Launch App Buttons
  [
    elements.btnTryHero,
    elements.btnTryNav,
    elements.btnLaunchBottom,
    elements.btnStartDemo,
  ].forEach((btn) => {
    if (btn) btn.addEventListener('click', showAppView);
  });

  // Start VisionAssist & Camera Permission
  if (elements.btnStartAssist) {
    elements.btnStartAssist.addEventListener('click', () => {
      startCamera(state.selectedDeviceId);
    });
  }

  // Mode Selection
  elements.modeButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      const targetMode = btn.getAttribute('data-mode-btn');
      if (targetMode) switchMode(targetMode);
    });
  });

  // Action Buttons
  if (elements.btnReplay) elements.btnReplay.addEventListener('click', replayLastSpoken);
  if (elements.btnCameraToggle) elements.btnCameraToggle.addEventListener('click', toggleCamera);
  if (elements.btnHelp) elements.btnHelp.addEventListener('click', openHelpModal);
  if (elements.btnHelpRail) elements.btnHelpRail.addEventListener('click', openHelpModal);
  if (elements.btnModalClose) elements.btnModalClose.addEventListener('click', closeHelpModal);
  if (elements.btnModalGotIt) elements.btnModalGotIt.addEventListener('click', closeHelpModal);

  // Camera selector change
  if (elements.cameraSelect) {
    elements.cameraSelect.addEventListener('change', (e) => {
      state.selectedDeviceId = e.target.value;
      startCamera(state.selectedDeviceId);
    });
  }

  // Close modal on Escape key
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && elements.helpModal && elements.helpModal.classList.contains('active')) {
      closeHelpModal();
    }
  });

  // Check URL hash for direct /app view
  if (window.location.pathname.endsWith('/app') || window.location.hash === '#app') {
    showAppView();
  }
}

// Processing Loop Interval (~3.5 times per second for smooth real-time response without overwhelming)
setInterval(() => {
  if (state.activeView === 'app' && state.cameraActive) {
    sendFrameForProcessing();
  }
}, 300);

// Initialize on DOM load
window.addEventListener('DOMContentLoaded', () => {
  attachEventListeners();
  initWebSocket();
  enumerateCameras();
});
