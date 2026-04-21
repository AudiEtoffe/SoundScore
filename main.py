#!/usr/bin/env python3
"""
SoundScore — Audio Device Quality Tester
Created by Acid Reign Productions

Run:
    python main.py

Requirements:
    pip install PyQt6 sounddevice numpy scipy matplotlib
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QAction
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QSlider,
    QTabWidget, QFrame, QGroupBox,
    QProgressBar, QMessageBox, QSizePolicy,
    QScrollArea, QTableWidget, QTableWidgetItem,
    QHeaderView, QCheckBox, QDialog, QDialogButtonBox,
    QTextEdit,
)

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from audio_io import AudioDeviceManager, AudioPlayer, AudioRecorder
from signal_detection import SignalDetector
import analysis
import scoring
from generate_reference_wav import (
    generate_reference_wav, SEGMENT_TIMES, TONE_FREQS
)

# ── Paths ──────────────────────────────────────────────────────────────────────
APP_DIR          = Path(__file__).parent
REFERENCE_WAV    = str(APP_DIR / "reference_test.wav")
CALIBRATION_FILE = str(APP_DIR / "calibration.json")
FS_DEFAULT       = 44_100

# ── Colour palette ─────────────────────────────────────────────────────────────
C = {
    "bg":       "#12121e",
    "panel":    "#1a1a2e",
    "panel2":   "#1e1e36",
    "border":   "#2a2a4e",
    "accent":   "#e94560",
    "accent2":  "#0f3460",
    "accent3":  "#533483",
    "text":     "#eaeaea",
    "muted":    "#888899",
    "green":    "#00ff88",
    "yellow":   "#ffd700",
    "orange":   "#ff9900",
    "red":      "#ff4444",
    "blue":     "#44aaff",
    "cyan":     "#00ddff",
    "purple":   "#cc66ff",
    "scope_bg": "#080812",
}

SOURCE_INTERNAL = "Internal (USB / Line)"
SOURCE_EXTERNAL = "External (CDJ / Player)"


# ══════════════════════════════════════════════════════════════════════════════
#  Worker signals
# ══════════════════════════════════════════════════════════════════════════════

class WorkerSignals(QObject):
    status     = pyqtSignal(str)
    progress   = pyqtSignal(int)
    complete   = pyqtSignal(dict)
    error      = pyqtSignal(str)
    calib_done = pyqtSignal(dict)


# ══════════════════════════════════════════════════════════════════════════════
#  Test worker thread
# ══════════════════════════════════════════════════════════════════════════════

class TestWorker(QThread):
    def __init__(self, player, recorder, detector,
                 reference_audio, segment_times, baseline,
                 external_source: bool):
        super().__init__()
        self.player          = player
        self.recorder        = recorder
        self.detector        = detector
        self.reference_audio = reference_audio
        self.segment_times   = segment_times
        self.baseline        = baseline
        self.external_source = external_source
        self.signals         = WorkerSignals()
        self._stop           = False

    def run(self) -> None:
        try:
            self._run()
        except Exception as exc:
            import traceback; traceback.print_exc()
            self.signals.error.emit(str(exc))

    def _run(self) -> None:
        sig = self.signals
        fs  = self.recorder.fs
        total_dur = float(self.segment_times.get("total", 32.1))

        sig.status.emit("Starting…")
        sig.progress.emit(5)

        self.recorder.start()

        if self.external_source:
            sig.status.emit(
                "🎵  Waiting for external source — press PLAY on your CDJ / player now…"
            )
            sig.progress.emit(8)
        else:
            self.player.play_wav(REFERENCE_WAV)
            sig.status.emit("Listening for sync chirp…")
            sig.progress.emit(10)

        # Detection window: 60 s for external, 25 s for internal
        detect_deadline = time.time() + (60.0 if self.external_source else 25.0)
        detected    = False
        det_offset  = 0

        while time.time() < detect_deadline and not self._stop:
            buf = self.recorder.get_recorded_audio()
            buf_mono = buf[:, 0] if buf.ndim > 1 else buf
            if len(buf_mono) > fs:
                found, idx = self.detector.detect(buf_mono)
                if found:
                    detected   = True
                    det_offset = idx
                    sig.status.emit("✓ Signal detected — measuring…")
                    sig.progress.emit(25)
                    break

            # Countdown for external source
            remaining = detect_deadline - time.time()
            if self.external_source and int(remaining) % 5 == 0:
                sig.status.emit(
                    f"🎵  Waiting for external source… ({int(remaining)}s)"
                )
            time.sleep(0.1)

        if self._stop:
            self.recorder.stop()
            return

        if not detected:
            if self.external_source:
                self.recorder.stop()
                self.signals.error.emit(
                    "No sync chirp detected within 60 seconds.\n\n"
                    "Make sure you're playing reference_test.wav from your "
                    "CDJ/player and that the output is connected to the correct "
                    "input on your interface."
                )
                return
            else:
                sig.status.emit("Chirp unclear — proceeding from beginning…")
                det_offset = 0

        # Wait for remainder of the test signal
        elapsed   = time.time() - (detect_deadline - (60.0 if self.external_source else 25.0))
        remaining = max(0.0, total_dur - elapsed + 1.5)
        steps = 14
        for i in range(steps):
            if self._stop:
                break
            time.sleep(remaining / steps)
            pct = 25 + int(65 * (i + 1) / steps)
            secs = int(remaining * (1.0 - (i + 1) / steps))
            sig.progress.emit(pct)
            sig.status.emit(f"Measuring… ({secs}s remaining)")

        recording = self.recorder.stop()
        sig.status.emit("Analysing…")
        sig.progress.emit(93)

        min_needed = int(self.segment_times["tones"][-1]["end"] * fs) + det_offset
        if len(recording) < min_needed:
            self.signals.error.emit(
                "Recording too short — check your input device and gain setting."
            )
            return

        meas   = analysis.analyze_recording(
            recording, self.reference_audio, fs,
            self.segment_times, detection_offset=det_offset,
        )
        if "error" in meas:
            self.signals.error.emit(meas["error"])
            return

        scores = scoring.compute_all_scores(meas, self.baseline)
        sig.progress.emit(100)
        sig.status.emit("Done!")
        sig.complete.emit(scores)

    def stop_test(self) -> None:
        self._stop = True
        if self.player:
            self.player.stop()
        try:
            self.recorder.stop()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
#  Calibration worker
# ══════════════════════════════════════════════════════════════════════════════

class CalibWorker(QThread):
    def __init__(self, player, recorder, reference_audio, segment_times):
        super().__init__()
        self.player          = player
        self.recorder        = recorder
        self.reference_audio = reference_audio
        self.segment_times   = segment_times
        self.signals         = WorkerSignals()

    def run(self) -> None:
        try:
            sig = self.signals
            fs  = self.recorder.fs
            sig.status.emit("Calibrating — loopback recording…")
            sig.progress.emit(5)
            self.recorder.start()
            self.player.play_wav(REFERENCE_WAV)
            total = float(self.segment_times.get("total", 32.1)) + 1.5
            steps = 12
            for i in range(steps):
                time.sleep(total / steps)
                sig.progress.emit(5 + int(75 * (i + 1) / steps))
            recording = self.recorder.stop()
            sig.status.emit("Analysing baseline…")
            sig.progress.emit(85)
            meas = analysis.analyze_recording(
                recording, self.reference_audio, fs, self.segment_times
            )
            baseline = {
                "baseline_thd":   meas.get("thd",          0.001),
                "baseline_sinad": meas.get("sinad",         80.0),
                "baseline_noise": meas.get("noise_floor",  -90.0),
            }
            sig.progress.emit(100)
            sig.calib_done.emit(baseline)
        except Exception as exc:
            self.signals.error.emit(f"Calibration failed: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
#  Oscilloscope
# ══════════════════════════════════════════════════════════════════════════════

class OscilloscopeWidget(QFrame):
    BUFFER = 4096

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(100)
        self.setStyleSheet(
            f"background:{C['scope_bg']};border:1px solid {C['border']};border-radius:4px;"
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        self._buf  = np.zeros(self.BUFFER, dtype=np.float32)
        fig        = Figure(figsize=(6, 1.1), facecolor=C["scope_bg"])
        self._ax   = fig.add_subplot(111)
        self._ax.set_facecolor(C["scope_bg"])
        self._ax.set_ylim(-1.05, 1.05)
        self._ax.set_xlim(0, self.BUFFER)
        self._ax.set_xticks([]); self._ax.set_yticks([-1, 0, 1])
        self._ax.tick_params(colors=C["border"], labelsize=6)
        for sp in self._ax.spines.values(): sp.set_color(C["border"])
        self._ax.axhline(0, color=C["border"], lw=0.4, alpha=0.6)
        fig.tight_layout(pad=0.3)
        self._line, = self._ax.plot(np.arange(self.BUFFER), self._buf,
                                    color=C["green"], lw=0.7, alpha=0.9)
        canvas = FigureCanvas(fig)
        canvas.setStyleSheet(f"background:{C['scope_bg']};")
        lay.addWidget(canvas)
        self._canvas = canvas

    def push(self, chunk: np.ndarray) -> None:
        n = min(len(chunk), self.BUFFER)
        self._buf = np.roll(self._buf, -n)
        self._buf[-n:] = chunk[-n:]
        self._line.set_ydata(self._buf)
        self._canvas.draw_idle()


# ══════════════════════════════════════════════════════════════════════════════
#  Level meter
# ══════════════════════════════════════════════════════════════════════════════

class LevelMeterWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(32)
        self.setStyleSheet(
            f"background:{C['scope_bg']};border:1px solid {C['border']};border-radius:4px;"
        )
        lay = QHBoxLayout(self); lay.setContentsMargins(8, 4, 8, 4); lay.setSpacing(6)
        lbl = QLabel("Level"); lbl.setStyleSheet(f"color:{C['muted']};font-size:10px;border:none;")
        lay.addWidget(lbl)
        self._bar = QProgressBar()
        self._bar.setRange(0, 100); self._bar.setValue(0); self._bar.setTextVisible(False)
        self._bar.setStyleSheet(f"""
            QProgressBar{{background:#111122;border:none;border-radius:3px;height:14px;}}
            QProgressBar::chunk{{border-radius:3px;
                background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 {C['green']},stop:0.7 {C['yellow']},stop:1 {C['red']});}}
        """)
        lay.addWidget(self._bar)
        self._db_lbl = QLabel("−∞ dB")
        self._db_lbl.setStyleSheet(f"color:{C['text']};font-size:10px;min-width:55px;border:none;")
        self._db_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(self._db_lbl)
        self._clip = QLabel("")
        self._clip.setStyleSheet(f"color:{C['red']};font-size:10px;font-weight:bold;min-width:36px;border:none;")
        lay.addWidget(self._clip)

    def update_level(self, rms: float, db: float) -> None:
        pct = int(np.clip((db + 60.0) / 60.0 * 100.0, 0, 100))
        self._bar.setValue(pct)
        self._db_lbl.setText(f"{db:.1f} dB" if db > -100 else "−∞ dB")
        self._clip.setText("⚠ CLIP" if rms > 0.95 else "")


# ══════════════════════════════════════════════════════════════════════════════
#  Metric card
# ══════════════════════════════════════════════════════════════════════════════

class MetricCard(QFrame):
    def __init__(self, title: str, subtitle: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            f"QFrame{{background:{C['panel2']};border:1px solid {C['border']};border-radius:8px;}}"
        )
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        lay = QVBoxLayout(self); lay.setContentsMargins(12, 10, 12, 10); lay.setSpacing(2)
        self._title = QLabel(title)
        self._title.setStyleSheet(f"color:{C['text']};font-size:13px;font-weight:bold;border:none;")
        lay.addWidget(self._title)
        self._sub = QLabel(subtitle)
        self._sub.setStyleSheet(f"color:{C['muted']};font-size:10px;border:none;")
        lay.addWidget(self._sub)
        self._score = QLabel("—")
        self._score.setStyleSheet(f"color:{C['accent']};font-size:30px;font-weight:bold;border:none;")
        lay.addWidget(self._score)
        self._desc = QLabel("")
        self._desc.setStyleSheet(f"color:#aaaaaa;font-size:10px;border:none;")
        self._desc.setWordWrap(True); lay.addWidget(self._desc)
        self._raw = QLabel("")
        self._raw.setStyleSheet(f"color:#555577;font-size:9px;border:none;")
        lay.addWidget(self._raw)

    def set_score(self, score: float, desc: str = "", raw: str = "") -> None:
        self._score.setText(f"{score:.0f}")
        self._desc.setText(desc); self._raw.setText(raw)
        colour = (C["green"] if score >= 90 else C["blue"] if score >= 75
                  else C["yellow"] if score >= 60 else C["red"])
        self._score.setStyleSheet(f"color:{colour};font-size:30px;font-weight:bold;border:none;")

    def reset(self) -> None:
        self._score.setText("—"); self._score.setStyleSheet(
            f"color:{C['accent']};font-size:30px;font-weight:bold;border:none;")
        self._desc.setText(""); self._raw.setText("")


# ══════════════════════════════════════════════════════════════════════════════
#  Main window
# ══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SoundScore — by Acid Reign Productions")
        self.setMinimumSize(1060, 780)

        self.baseline:        Optional[dict]       = None
        self.reference_audio: Optional[np.ndarray] = None
        self.test_worker:     Optional[TestWorker]  = None
        self.calib_worker:    Optional[CalibWorker] = None
        self._last_rms:   float = 0.0
        self._last_db:    float = -100.0
        self._last_chunk: Optional[np.ndarray] = None

        self._load_calibration()
        self._ensure_reference_wav()
        self._build_ui()
        self._apply_theme()

        self._refresh = QTimer(self)
        self._refresh.setInterval(50)
        self._refresh.timeout.connect(self._tick)

    # ── Bootstrap ─────────────────────────────────────────────────────────────

    def _ensure_reference_wav(self) -> None:
        if not os.path.exists(REFERENCE_WAV):
            generate_reference_wav(REFERENCE_WAV, FS_DEFAULT)
        self._load_ref_audio(REFERENCE_WAV)

    def _load_ref_audio(self, path: str) -> None:
        from scipy.io import wavfile
        try:
            rate, data = wavfile.read(path)
            data = data.astype(np.float32) / 32768.0 if data.dtype == np.int16 else data.astype(np.float32)
            if data.ndim > 1: data = data[:, 0]
            self.reference_audio = data
        except Exception as exc:
            print(f"[Main] Reference audio load error: {exc}")

    def _load_calibration(self) -> None:
        if os.path.exists(CALIBRATION_FILE):
            try:
                with open(CALIBRATION_FILE) as f:
                    self.baseline = json.load(f)
            except Exception:
                self.baseline = None

    def _save_calibration(self, data: dict) -> None:
        with open(CALIBRATION_FILE, "w") as f:
            json.dump(data, f, indent=2)
        self.baseline = data

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        vlay = QVBoxLayout(root)
        vlay.setContentsMargins(12, 10, 12, 10)
        vlay.setSpacing(8)

        vlay.addWidget(self._make_header())

        self._tabs = QTabWidget()
        self._tabs.setStyleSheet(self._tab_style())
        self._tabs.addTab(self._make_test_tab(),    "Test")
        self._tabs.addTab(self._make_results_tab(), "Results")
        self._tabs.addTab(self._make_advanced_tab(),"Advanced")
        vlay.addWidget(self._tabs)

        self.statusBar().setStyleSheet(
            f"color:{C['muted']};background:{C['panel']};font-size:11px;"
        )
        self.statusBar().showMessage(
            "SoundScore ready — select your input and press  ▶  Start Test"
        )

    # ── Header ────────────────────────────────────────────────────────────────

    def _make_header(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet(
            f"QFrame{{background:{C['panel']};border-radius:8px;}}"
        )
        lay = QHBoxLayout(frame)
        lay.setContentsMargins(16, 8, 16, 8)

        logo = QLabel("🎚  SoundScore")
        logo.setStyleSheet(
            f"color:{C['text']};font-size:21px;font-weight:bold;"
        )
        lay.addWidget(logo)

        credit = QLabel("by Acid Reign Productions")
        credit.setStyleSheet(f"color:{C['accent']};font-size:12px;")
        lay.addWidget(credit)

        lay.addStretch()

        self._calib_badge = QLabel(
            "✓ Calibrated" if self.baseline else "No calibration"
        )
        colour = C["green"] if self.baseline else C["muted"]
        self._calib_badge.setStyleSheet(f"color:{colour};font-size:12px;")
        lay.addWidget(self._calib_badge)

        return frame

    # ── Test tab ──────────────────────────────────────────────────────────────

    def _make_test_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)
        lay.addWidget(self._make_controls())
        lay.addWidget(self._make_scope_section())
        lay.addWidget(self._make_score_row())
        lay.addWidget(self._make_metrics_section())
        lay.addWidget(self._make_inline_freq_response())
        return w

    def _make_controls(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet(
            f"QFrame{{background:{C['panel']};border:1px solid {C['border']};border-radius:8px;}}"
        )
        g = QGridLayout(frame)
        g.setContentsMargins(14, 10, 14, 10)
        g.setSpacing(8)

        # Source selector
        g.addWidget(self._mlabel("Source:"), 0, 0)
        self._source_combo = QComboBox()
        self._source_combo.addItems([SOURCE_INTERNAL, SOURCE_EXTERNAL])
        self._source_combo.setStyleSheet(self._combo_style())
        self._source_combo.currentTextChanged.connect(self._on_source_changed)
        g.addWidget(self._source_combo, 0, 1)

        # Input device
        g.addWidget(self._mlabel("Input Device:"), 0, 2)
        self._input_combo = QComboBox()
        self._input_combo.setStyleSheet(self._combo_style())
        self._populate_inputs()
        g.addWidget(self._input_combo, 0, 3)

        # Output device (hidden for external source)
        self._output_lbl = self._mlabel("Output Device:")
        g.addWidget(self._output_lbl, 1, 0)
        self._output_combo = QComboBox()
        self._output_combo.setStyleSheet(self._combo_style())
        self._populate_outputs()
        g.addWidget(self._output_combo, 1, 1)

        # Stereo checkbox
        self._stereo_chk = QCheckBox("Stereo (L/R analysis)")
        self._stereo_chk.setStyleSheet(
            f"color:{C['text']};font-size:11px;"
        )
        g.addWidget(self._stereo_chk, 1, 2)

        # Sample rate
        g.addWidget(self._mlabel("Sample Rate:"), 1, 3)
        # (put SR inside same cell via inner layout)
        sr_w = QWidget()
        sr_lay = QHBoxLayout(sr_w); sr_lay.setContentsMargins(0,0,0,0)
        self._sr_combo = QComboBox()
        self._sr_combo.addItems(["44100 Hz", "48000 Hz"])
        self._sr_combo.setStyleSheet(self._combo_style())
        sr_lay.addWidget(self._sr_combo)
        g.addWidget(sr_w, 1, 3)

        # Gain
        g.addWidget(self._mlabel("Input Gain:"), 2, 0)
        gain_w = QWidget()
        gain_lay = QHBoxLayout(gain_w); gain_lay.setContentsMargins(0,0,0,0)
        self._gain_slider = QSlider(Qt.Orientation.Horizontal)
        self._gain_slider.setRange(10, 500)
        self._gain_slider.setValue(100)
        self._gain_slider.setToolTip("Adjust input gain. Watch oscilloscope — avoid clipping.")
        self._gain_slider.valueChanged.connect(self._on_gain_change)
        gain_lay.addWidget(self._gain_slider)
        self._gain_lbl = QLabel("1.0×")
        self._gain_lbl.setStyleSheet(f"color:{C['text']};font-size:12px;min-width:38px;")
        gain_lay.addWidget(self._gain_lbl)
        g.addWidget(gain_w, 2, 1)

        # External source hint
        self._ext_hint = QLabel(
            "ℹ  Copy reference_test.wav to a USB drive and play it on your CDJ/player. "
            "Connect its output to your interface input, then press Start Test and press PLAY."
        )
        self._ext_hint.setStyleSheet(
            f"color:{C['cyan']};font-size:10px;background:{C['accent2']};"
            f"border-radius:4px;padding:4px 8px;border:none;"
        )
        self._ext_hint.setWordWrap(True)
        self._ext_hint.hide()
        g.addWidget(self._ext_hint, 3, 0, 1, 4)

        # Buttons
        btn_lay = QHBoxLayout()

        self._start_btn = QPushButton("▶  Start Test")
        self._start_btn.setStyleSheet(self._btn_style(C["accent"]))
        self._start_btn.setMinimumHeight(40)
        self._start_btn.clicked.connect(self._on_start)
        btn_lay.addWidget(self._start_btn)

        self._stop_btn = QPushButton("■  Stop")
        self._stop_btn.setStyleSheet(self._btn_style("#444455"))
        self._stop_btn.setMinimumHeight(40)
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        btn_lay.addWidget(self._stop_btn)

        self._calib_btn = QPushButton("⚙  Calibrate (Loopback)")
        self._calib_btn.setStyleSheet(self._btn_style(C["accent2"]))
        self._calib_btn.setMinimumHeight(40)
        self._calib_btn.clicked.connect(self._on_calibrate)
        btn_lay.addWidget(self._calib_btn)

        self._regen_btn = QPushButton("⟳  Regenerate Test WAV")
        self._regen_btn.setStyleSheet(self._btn_style("#2a2a44"))
        self._regen_btn.setMinimumHeight(40)
        self._regen_btn.setToolTip("Regenerate reference_test.wav")
        self._regen_btn.clicked.connect(self._on_regen_wav)
        btn_lay.addWidget(self._regen_btn)

        g.addLayout(btn_lay, 4, 0, 1, 4)

        return frame

    def _make_scope_section(self) -> QGroupBox:
        grp = QGroupBox("Input Signal")
        grp.setStyleSheet(self._grp_style())
        lay = QVBoxLayout(grp)
        lay.setContentsMargins(8, 8, 8, 6)
        lay.setSpacing(4)
        self._scope   = OscilloscopeWidget()
        self._lvl_mtr = LevelMeterWidget()
        lay.addWidget(self._scope)
        lay.addWidget(self._lvl_mtr)
        return grp

    def _make_score_row(self) -> QWidget:
        container = QWidget()
        lay = QHBoxLayout(container)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        # Big score
        sf = QFrame()
        sf.setStyleSheet(
            f"QFrame{{background:{C['panel']};border:2px solid {C['accent']};border-radius:12px;}}"
        )
        sf.setFixedWidth(210)
        sf_lay = QVBoxLayout(sf)
        sf_lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sf_lay.setContentsMargins(12, 14, 12, 14)
        lbl = QLabel("Sound Score"); lbl.setStyleSheet(f"color:{C['muted']};font-size:12px;border:none;")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter); sf_lay.addWidget(lbl)
        self._main_score = QLabel("—")
        self._main_score.setStyleSheet(f"color:{C['accent']};font-size:56px;font-weight:bold;border:none;")
        self._main_score.setAlignment(Qt.AlignmentFlag.AlignCenter); sf_lay.addWidget(self._main_score)
        self._grade_lbl = QLabel("")
        self._grade_lbl.setStyleSheet(f"color:{C['text']};font-size:15px;font-weight:bold;border:none;")
        self._grade_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter); sf_lay.addWidget(self._grade_lbl)
        arp = QLabel("Acid Reign Productions")
        arp.setStyleSheet(f"color:{C['accent']};font-size:9px;border:none;")
        arp.setAlignment(Qt.AlignmentFlag.AlignCenter); sf_lay.addWidget(arp)
        lay.addWidget(sf)

        # Status + progress
        status_frame = QFrame()
        status_frame.setStyleSheet(
            f"QFrame{{background:{C['panel']};border:1px solid {C['border']};border-radius:8px;}}"
        )
        sf2 = QVBoxLayout(status_frame)
        sf2.setContentsMargins(14, 14, 14, 14); sf2.setSpacing(6)
        s_lbl = QLabel("Status"); s_lbl.setStyleSheet(f"color:{C['muted']};font-size:11px;border:none;")
        sf2.addWidget(s_lbl)
        self._status_lbl = QLabel("Ready")
        self._status_lbl.setStyleSheet(f"color:{C['text']};font-size:17px;font-weight:bold;border:none;")
        self._status_lbl.setWordWrap(True); sf2.addWidget(self._status_lbl)
        sf2.addStretch()
        self._progress = QProgressBar()
        self._progress.setRange(0, 100); self._progress.setValue(0); self._progress.setTextVisible(True)
        self._progress.setStyleSheet(f"""
            QProgressBar{{background:#111122;border:none;border-radius:5px;height:12px;
                         color:{C['text']};font-size:9px;}}
            QProgressBar::chunk{{border-radius:5px;
                background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 {C['accent2']},stop:1 {C['accent']});}}
        """)
        sf2.addWidget(self._progress)
        lay.addWidget(status_frame)

        return container

    def _make_metrics_section(self) -> QGroupBox:
        grp = QGroupBox("Breakdown")
        grp.setStyleSheet(self._grp_style())
        lay = QHBoxLayout(grp)
        lay.setContentsMargins(10, 10, 10, 10); lay.setSpacing(8)
        self._card_clarity    = MetricCard("Clarity",    "Signal quality  (SINAD)")
        self._card_distortion = MetricCard("Distortion", "Harmonics  (THD)")
        self._card_noise      = MetricCard("Noise",      "Background noise floor")
        self._card_balance    = MetricCard("Balance",    "Frequency response")
        for c in (self._card_clarity, self._card_distortion,
                  self._card_noise, self._card_balance):
            lay.addWidget(c)
        return grp

    def _make_inline_freq_response(self) -> QGroupBox:
        """Frequency-response chart shown on the main Test tab after a test completes."""
        grp = QGroupBox("Frequency Response  (20 Hz – 20 kHz)")
        grp.setStyleSheet(self._grp_style())
        lay = QVBoxLayout(grp)
        lay.setContentsMargins(5, 5, 5, 5)

        self._inline_fr_fig    = Figure(figsize=(8, 2.2), facecolor=C["panel"])
        self._inline_fr_ax     = self._inline_fr_fig.add_subplot(111)
        self._style_ax(self._inline_fr_ax, "Frequency (Hz)", "Response (dB)", log_x=True)
        self._inline_fr_canvas = FigureCanvas(self._inline_fr_fig)
        self._inline_fr_canvas.setStyleSheet(f"background:{C['panel']};")
        self._inline_fr_canvas.setFixedHeight(180)
        lay.addWidget(self._inline_fr_canvas)

        placeholder = QLabel("Run a test to see the frequency response here.")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet(f"color:{C['muted']};font-size:12px;border:none;")
        self._inline_fr_placeholder = placeholder
        lay.addWidget(placeholder)

        grp.hide()           # hidden until first test result
        self._inline_fr_grp = grp
        return grp

    # ── Results tab ───────────────────────────────────────────────────────────

    def _make_results_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        # Scroll area wraps everything
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"QScrollArea{{border:none;background:{C['bg']};}}")
        content = QWidget()
        self._results_inner = QVBoxLayout(content)
        self._results_inner.setContentsMargins(4, 4, 4, 4)
        self._results_inner.setSpacing(10)

        # Placeholder label
        self._results_placeholder = QLabel(
            "Run a test to see detailed results here."
        )
        self._results_placeholder.setStyleSheet(
            f"color:{C['muted']};font-size:15px;"
        )
        self._results_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._results_inner.addWidget(self._results_placeholder)
        self._results_inner.addStretch()

        scroll.setWidget(content)
        lay.addWidget(scroll)
        self._results_scroll = scroll
        self._results_content = content
        return w

    def _populate_results(self, scores: dict) -> None:
        """Fill the Results tab after a test completes."""
        # Clear old content
        for i in reversed(range(self._results_inner.count())):
            item = self._results_inner.itemAt(i)
            if item.widget():
                item.widget().deleteLater()
            elif item.spacerItem():
                self._results_inner.removeItem(item)

        is_stereo = scores.get("is_stereo", False)
        final     = scores.get("final", 0.0)
        grade, emoji = scoring.get_grade(final)

        # ── Title ──────────────────────────────────────────────────────────────
        title_frame = QFrame()
        title_frame.setStyleSheet(
            f"QFrame{{background:{C['panel']};border:2px solid {C['accent']};border-radius:10px;}}"
        )
        tlay = QHBoxLayout(title_frame)
        tlay.setContentsMargins(20, 12, 20, 12)

        colour = (C["green"] if final >= 90 else C["blue"] if final >= 75
                  else C["yellow"] if final >= 60 else C["red"])
        score_big = QLabel(f"{emoji}  Sound Score:  {final:.0f} / 100  — {grade}")
        score_big.setStyleSheet(f"color:{colour};font-size:20px;font-weight:bold;border:none;")
        tlay.addWidget(score_big)
        tlay.addStretch()
        arp_lbl = QLabel("Acid Reign Productions")
        arp_lbl.setStyleSheet(f"color:{C['accent']};font-size:11px;border:none;")
        tlay.addWidget(arp_lbl)

        self._results_inner.addWidget(title_frame)

        # ── Summary metrics ────────────────────────────────────────────────────
        sum_grp = QGroupBox("Measurement Summary")
        sum_grp.setStyleSheet(self._grp_style())
        sum_lay = QGridLayout(sum_grp)
        sum_lay.setContentsMargins(12, 12, 12, 12); sum_lay.setSpacing(6)

        def _add_metric(row, label, value, unit, score_val):
            c2 = (C["green"] if score_val >= 90 else C["blue"] if score_val >= 75
                  else C["yellow"] if score_val >= 60 else C["red"])
            sum_lay.addWidget(self._rlabel(label, bold=True),  row, 0)
            sum_lay.addWidget(self._rlabel(f"{value}  {unit}"), row, 1)
            bar = QProgressBar(); bar.setRange(0,100); bar.setValue(int(score_val))
            bar.setTextVisible(False); bar.setFixedHeight(10)
            bar.setStyleSheet(f"""
                QProgressBar{{background:#111122;border:none;border-radius:5px;}}
                QProgressBar::chunk{{background:{c2};border-radius:5px;}}
            """)
            sum_lay.addWidget(bar, row, 2)
            sum_lay.addWidget(self._rlabel(f"{score_val:.0f}/100", colour=c2), row, 3)

        _add_metric(0, "Clarity (SINAD)", f"{scores.get('raw_sinad',0):.1f}", "dB",
                    scores.get("sinad", 0))
        _add_metric(1, "Distortion (THD)", f"{scores.get('raw_thd_db',0):.1f}", "dB",
                    scores.get("thd", 0))
        _add_metric(2, "Noise Floor", f"{scores.get('raw_noise_db',0):.1f}", "dBFS",
                    scores.get("noise", 0))
        _add_metric(3, "Frequency Balance", "", "",
                    scores.get("flatness", 0))

        sum_lay.setColumnStretch(2, 1)
        self._results_inner.addWidget(sum_grp)

        # ── Per-frequency table ────────────────────────────────────────────────
        freq_grp = QGroupBox("Per-Frequency Analysis")
        freq_grp.setStyleSheet(self._grp_style())
        flay = QVBoxLayout(freq_grp)
        flay.setContentsMargins(8, 8, 8, 8)

        if is_stereo:
            headers = ["Frequency", "L SINAD", "R SINAD", "L THD", "R THD",
                       "L Score", "R Score", "Winner"]
            cols    = len(headers)
        else:
            headers = ["Frequency", "SINAD", "THD", "Score", "Rating"]
            cols    = len(headers)

        tbl = QTableWidget(len(scores.get("tone_scores", [])), cols)
        tbl.setHorizontalHeaderLabels(headers)
        tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        tbl.verticalHeader().setVisible(False)
        tbl.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        tbl.setAlternatingRowColors(True)
        tbl.setStyleSheet(f"""
            QTableWidget{{background:{C['panel']};color:{C['text']};
                gridline-color:{C['border']};border:none;font-size:11px;}}
            QTableWidget::item{{padding:4px 8px;}}
            QHeaderView::section{{background:{C['accent2']};color:{C['text']};
                font-weight:bold;padding:4px;border:none;font-size:11px;}}
            QTableWidget::item:alternate{{background:{C['panel2']};}}
        """)

        def _ci(text, colour=None, align=Qt.AlignmentFlag.AlignCenter):
            item = QTableWidgetItem(text)
            item.setTextAlignment(align)
            if colour:
                from PyQt6.QtGui import QColor as QC
                item.setForeground(QC(colour))
            return item

        for row, ts in enumerate(scores.get("tone_scores", [])):
            freq = ts["freq"]
            freq_str = (f"{int(freq)} Hz" if freq < 1000
                        else f"{freq/1000:.0f} kHz" if freq % 1000 == 0
                        else f"{freq/1000:.1f} kHz")
            if is_stereo:
                sc = scores.get("stereo_scores", [{}])[row]
                l_s = sc.get("L", 0); r_s = sc.get("R", 0)
                winner = sc.get("winner", "=")
                winner_str = ("← L" if winner == "L" else "R →"
                              if winner == "R" else "Tied")
                w_colour = (C["cyan"] if winner == "L" else C["purple"]
                            if winner == "R" else C["green"])
                tbl.setItem(row, 0, _ci(freq_str))
                tbl.setItem(row, 1, _ci(f"{ts.get('sinad_L', 0):.1f} dB"))
                tbl.setItem(row, 2, _ci(f"{ts.get('sinad_R', 0):.1f} dB"))
                tbl.setItem(row, 3, _ci(f"{ts.get('thd_L', 0):.1f} dB"))
                tbl.setItem(row, 4, _ci(f"{ts.get('thd_R', 0):.1f} dB"))
                lc = (C["green"] if l_s >= 75 else C["yellow"] if l_s >= 60 else C["red"])
                rc = (C["green"] if r_s >= 75 else C["yellow"] if r_s >= 60 else C["red"])
                tbl.setItem(row, 5, _ci(f"{l_s:.0f}", lc))
                tbl.setItem(row, 6, _ci(f"{r_s:.0f}", rc))
                tbl.setItem(row, 7, _ci(winner_str, w_colour))
            else:
                s = ts.get("score", 0)
                sc = (C["green"] if s >= 75 else C["yellow"] if s >= 60 else C["red"])
                grade_s, _ = scoring.get_grade(s)
                tbl.setItem(row, 0, _ci(freq_str))
                tbl.setItem(row, 1, _ci(f"{ts.get('sinad', 0):.1f} dB"))
                tbl.setItem(row, 2, _ci(f"{ts.get('thd_db', 0):.1f} dB"))
                tbl.setItem(row, 3, _ci(f"{s:.0f}", sc))
                tbl.setItem(row, 4, _ci(grade_s, sc))

        tbl.setFixedHeight(28 + 28 * len(scores.get("tone_scores", [])))
        flay.addWidget(tbl)
        self._results_inner.addWidget(freq_grp)

        # ── Channel comparison (stereo only) ──────────────────────────────────
        if is_stereo:
            ch_summary = scores.get("channel_summary", {})
            ch_grp = QGroupBox("L / R Channel Comparison")
            ch_grp.setStyleSheet(self._grp_style())
            ch_lay = QGridLayout(ch_grp)
            ch_lay.setContentsMargins(12, 12, 12, 12); ch_lay.setSpacing(8)

            l_score = ch_summary.get("L_score", 0)
            r_score = ch_summary.get("R_score", 0)
            winner  = ch_summary.get("winner", "=")

            def _ch_col(label, score_val, noise_db, side):
                cf = QFrame()
                border_c = (C["cyan"] if winner == side else
                            C["purple"] if winner != "=" and winner != side
                            else C["green"])
                cf.setStyleSheet(
                    f"QFrame{{background:{C['panel2']};border:2px solid {border_c};"
                    f"border-radius:8px;}}"
                )
                cl = QVBoxLayout(cf); cl.setContentsMargins(16, 12, 16, 12); cl.setSpacing(4)
                ch_lbl = QLabel(label)
                ch_colour = C["cyan"] if side == "L" else C["purple"]
                ch_lbl.setStyleSheet(f"color:{ch_colour};font-size:16px;font-weight:bold;border:none;")
                ch_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter); cl.addWidget(ch_lbl)
                sc_c = (C["green"] if score_val >= 75 else C["yellow"] if score_val >= 60 else C["red"])
                sc_lbl = QLabel(f"{score_val:.0f}")
                sc_lbl.setStyleSheet(f"color:{sc_c};font-size:40px;font-weight:bold;border:none;")
                sc_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter); cl.addWidget(sc_lbl)
                noise_s = scoring.score_noise(noise_db)
                cl.addWidget(self._rlabel(f"Noise: {noise_db:.1f} dBFS  ({noise_s:.0f}/100)",
                                         colour=sc_c))
                if winner == side:
                    w_lbl = QLabel("★  Better channel")
                    w_lbl.setStyleSheet(f"color:{C['green']};font-size:11px;font-weight:bold;border:none;")
                    w_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter); cl.addWidget(w_lbl)
                return cf

            ch_lay.addWidget(_ch_col("Left (L)",  l_score,
                                     ch_summary.get("L_noise", -60), "L"), 0, 0)
            diff_lbl = QLabel("vs")
            diff_lbl.setStyleSheet(f"color:{C['muted']};font-size:18px;font-weight:bold;")
            diff_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            ch_lay.addWidget(diff_lbl, 0, 1)
            ch_lay.addWidget(_ch_col("Right (R)", r_score,
                                     ch_summary.get("R_noise", -60), "R"), 0, 2)
            ch_lay.setColumnStretch(0, 1); ch_lay.setColumnStretch(2, 1)

            diff = abs(l_score - r_score)
            if diff < 2:
                verdict = "✓  Both channels are performing equally well."
            elif winner == "L":
                verdict = f"⚠  Left channel scores {diff:.0f} points higher than Right."
            else:
                verdict = f"⚠  Right channel scores {diff:.0f} points higher than Left."

            v_lbl = QLabel(verdict)
            v_lbl.setStyleSheet(
                f"color:{C['text'] if diff < 5 else C['orange']};"
                f"font-size:12px;padding:6px;border:none;"
            )
            v_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            ch_lay.addWidget(v_lbl, 1, 0, 1, 3)
            self._results_inner.addWidget(ch_grp)

        # ── Per-frequency bar chart ────────────────────────────────────────────
        chart_grp = QGroupBox("Frequency Score Chart")
        chart_grp.setStyleSheet(self._grp_style())
        chart_lay = QVBoxLayout(chart_grp)
        chart_lay.setContentsMargins(5, 5, 5, 5)

        tone_scores = scores.get("tone_scores", [])
        if tone_scores:
            fig2 = Figure(figsize=(9, 2.8), facecolor=C["panel"])
            ax2  = fig2.add_subplot(111)
            ax2.set_facecolor(C["scope_bg"])
            freqs_hz = [ts["freq"] for ts in tone_scores]
            x        = np.arange(len(freqs_hz))
            freq_labels = []
            for f in freqs_hz:
                freq_labels.append(f"{int(f)} Hz" if f < 1000
                                   else f"{f/1000:.0f} kHz")

            if is_stereo:
                ss = scores.get("stereo_scores", [])
                l_vals = [s.get("L", 0) for s in ss]
                r_vals = [s.get("R", 0) for s in ss]
                bw = 0.35
                bars_l = ax2.bar(x - bw/2, l_vals, bw, label="Left",
                                 color=C["cyan"], alpha=0.85)
                bars_r = ax2.bar(x + bw/2, r_vals, bw, label="Right",
                                 color=C["purple"], alpha=0.85)
                ax2.legend(facecolor=C["panel"], edgecolor=C["border"],
                           labelcolor=C["text"], fontsize=8)
            else:
                vals     = [ts.get("score", 0) for ts in tone_scores]
                bar_cols = [C["green"] if v >= 90 else C["blue"] if v >= 75
                            else C["yellow"] if v >= 60 else C["red"]
                            for v in vals]
                ax2.bar(x, vals, color=bar_cols, alpha=0.85)

            ax2.set_xticks(x); ax2.set_xticklabels(freq_labels, fontsize=8,
                                                    color=C["text"], rotation=30)
            ax2.set_ylim(0, 105)
            ax2.set_ylabel("Score", color=C["muted"], fontsize=9)
            ax2.tick_params(colors=C["text"], labelsize=8)
            for sp in ax2.spines.values(): sp.set_color(C["border"])
            ax2.axhline(75, color=C["blue"], lw=0.7, ls="--", alpha=0.6, label="Good")
            ax2.axhline(90, color=C["green"], lw=0.7, ls="--", alpha=0.6, label="Excellent")
            ax2.grid(axis="y", color=C["border"], alpha=0.4, lw=0.5)
            fig2.tight_layout(pad=0.4)
            canvas2 = FigureCanvas(fig2)
            canvas2.setStyleSheet(f"background:{C['panel']};")
            canvas2.setFixedHeight(220)
            chart_lay.addWidget(canvas2)

        self._results_inner.addWidget(chart_grp)

        # ── Recommendations ────────────────────────────────────────────────────
        rec_grp = QGroupBox("Recommendations")
        rec_grp.setStyleSheet(self._grp_style())
        rec_lay = QVBoxLayout(rec_grp)
        rec_lay.setContentsMargins(12, 8, 12, 8)

        recs = self._build_recommendations(scores)
        for rec in recs:
            r_lbl = QLabel(f"• {rec}")
            r_lbl.setStyleSheet(f"color:{C['text']};font-size:11px;border:none;padding:2px;")
            r_lbl.setWordWrap(True)
            rec_lay.addWidget(r_lbl)

        self._results_inner.addWidget(rec_grp)
        self._results_inner.addStretch()

    def _build_recommendations(self, scores: dict) -> list[str]:
        recs = []
        noise_db  = scores.get("raw_noise_db", -90.0)
        sinad_db  = scores.get("raw_sinad",    60.0)
        thd_db    = scores.get("raw_thd_db",   -40.0)
        noise_s   = scores.get("noise",        50.0)
        sinad_s   = scores.get("sinad",        50.0)
        thd_s     = scores.get("thd",          50.0)
        flat_s    = scores.get("flatness",     50.0)
        final     = scores.get("final",        50.0)
        is_stereo = scores.get("is_stereo",    False)
        baseline  = self.baseline

        # ── 1. No calibration reminder ─────────────────────────────────────────
        if not baseline:
            recs.append(
                "⚙  No calibration found. Run 'Calibrate (Loopback)' with a loopback cable "
                "connected at HIGH gain to establish your interface's baseline. This improves "
                "scoring accuracy significantly."
            )

        # ── 2. Gain staging — the #1 cause of a high noise floor ──────────────
        if noise_db > -80.0:
            recs.append(
                f"🔊  Noise floor is {noise_db:.1f} dBFS — this is the most common issue and is "
                "almost always caused by gain staging, NOT the cable type. Even a perfect "
                "balanced cable can't overcome insufficient gain. Make sure your interface's "
                "INPUT GAIN knob is turned up before testing so the signal is well above "
                "the noise floor. A rule of thumb: peaks should reach −12 dBFS or higher."
            )

        # ── 3. Recalibrate at correct gain level ──────────────────────────────
        if noise_db > -80.0 and baseline:
            recs.append(
                "🔁  If you recently changed your gain setting, recalibrate. Calibration "
                "captures the noise floor at the gain level it was set when you calibrated. "
                "If your test gain is different, the comparison will be inaccurate. "
                "Match your gain exactly, or recalibrate at the gain you plan to use for testing."
            )

        # ── 4. Ground loop / hum (low-frequency noise) ────────────────────────
        tone_scores = scores.get("tone_scores", [])
        low_freq_poor = any(
            ts.get("freq", 0) <= 100 and ts.get("score", 100) < 60
            for ts in tone_scores
        )
        if low_freq_poor or (sinad_s < 60 and noise_db > -70.0):
            recs.append(
                "🔌  Low-frequency noise detected. This usually indicates a ground loop — "
                "common when mixing gear from different power circuits. Try plugging all "
                "equipment into the same power strip/board. A DI box or ground-lift adapter "
                "between the CDJ and interface can eliminate hum without affecting audio quality."
            )

        # ── 5. High-frequency rolloff ─────────────────────────────────────────
        high_freq_poor = any(
            ts.get("freq", 0) >= 5000 and ts.get("score", 100) < 55
            for ts in tone_scores
        )
        if high_freq_poor:
            recs.append(
                "📉  High-frequency performance drops above 5 kHz. This is expected in longer "
                "unbalanced cable runs (capacitance causes treble rolloff above ~10 m). "
                "If using balanced XLR cables under 10 m this may indicate your CDJ or "
                "player has a built-in high-frequency limiter, or EQ has been applied. "
                "Check that EQ/ISOLATOR knobs on the CDJ and mixer are centred/flat."
            )

        # ── 6. Clarity / SINAD low ────────────────────────────────────────────
        if sinad_s < 60:
            recs.append(
                f"📻  Clarity (SINAD) is {sinad_db:.1f} dB — signal quality is below average. "
                "Common causes: (1) gain too low so noise dominates, "
                "(2) USB power supply noise coupling into the interface (try a different USB port "
                "or a powered USB hub), (3) WiFi/Bluetooth interference — disable nearby wireless "
                "devices when testing, (4) ground loop (see above)."
            )

        # ── 7. THD / distortion ───────────────────────────────────────────────
        if thd_s < 60:
            recs.append(
                f"📊  Distortion (THD) is {thd_db:.1f} dB — higher than expected. "
                "Main causes: (1) input gain is too high causing ADC saturation — reduce "
                "until ⚠ CLIP disappears from the level meter, "
                "(2) source device (CDJ/player) is itself clipping — check its output level, "
                "(3) cables with loose/dirty connectors can introduce harmonic distortion."
            )

        # ── 8. Frequency balance / EQ ─────────────────────────────────────────
        if flat_s < 60:
            recs.append(
                "🎛  Frequency balance is uneven. Check: (1) EQ on your CDJ or mixer is "
                "completely flat/centred — even small boosts heavily affect the sweep test, "
                "(2) isolator knobs are at 12 o'clock, (3) no DSP effects (reverb, echo, "
                "filters) are active on the channel. For best results, connect your "
                "source directly to the interface, bypassing any mixer."
            )

        # ── 9. USB noise ──────────────────────────────────────────────────────
        if noise_db > -85.0 and sinad_s < 70:
            recs.append(
                "🖥  USB noise can degrade audio interfaces significantly. Try: "
                "(1) connect the interface to a rear USB port on the PC/laptop (they often "
                "have cleaner power), (2) use a powered USB hub with its own supply, "
                "(3) avoid USB3 ports (the 5 GHz switching can couple into audio), "
                "(4) ensure no USB hard drives or high-draw devices share the same USB hub."
            )

        # ── 10. Stereo imbalance ──────────────────────────────────────────────
        if is_stereo and scores.get("channel_summary"):
            ch   = scores["channel_summary"]
            diff = abs(ch.get("L_score", 0) - ch.get("R_score", 0))
            if diff >= 10:
                winner = ch.get("winner", "=")
                worse  = "Right" if winner == "L" else "Left"
                better = "Left" if winner == "L" else "Right"
                recs.append(
                    f"↔  {worse} channel scores {diff:.0f} points below {better}. "
                    "Swap the cables on your interface inputs to determine if the problem "
                    "follows the cable (cable issue) or stays on the same input (interface "
                    "channel issue). A difference > 3 dB in noise or SINAD between channels "
                    "warrants replacing the cable or cleaning the jack contacts."
                )
            elif diff >= 4:
                recs.append(
                    f"↔  Minor channel imbalance ({diff:.0f} pts). This is often due to "
                    "slightly different cable lengths, or one jack being slightly worn. "
                    "Check both connectors are fully seated."
                )

        # ── 11. Balanced cable note ───────────────────────────────────────────
        if noise_db > -75.0:
            recs.append(
                "💡  Note on balanced cables: balanced (XLR/TRS) connections reduce "
                "electromagnetic interference but they cannot fix a noise floor caused by "
                "insufficient gain staging or USB interference. If you're already using "
                "balanced cables and still seeing high noise, focus on gain level and USB "
                "power quality first — those have far more impact on measured noise floor."
            )

        # ── 12. Excellent result ──────────────────────────────────────────────
        if final >= 90:
            recs.append(
                "🏆  Outstanding result! Your audio path is performing at a professional "
                "level. SINAD, THD, noise floor, and frequency balance are all excellent. "
                "This setup is ready for recording and broadcast."
            )
        elif final >= 75 and not recs:
            recs.append(
                "✅  Good result overall. Minor improvements may be possible through "
                "optimal gain staging and USB power quality, but your setup is performing "
                "well for live and studio use."
            )
        elif not recs:
            recs.append(
                "All measurements are within acceptable ranges for this signal chain. "
                "Run with calibration enabled for more accurate relative scoring."
            )

        return recs

    # ── Advanced tab ──────────────────────────────────────────────────────────

    def _make_advanced_tab(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        fft_grp = QGroupBox("FFT Spectrum  (1 kHz tone segment)")
        fft_grp.setStyleSheet(self._grp_style())
        fl = QVBoxLayout(fft_grp); fl.setContentsMargins(5, 5, 5, 5)
        self._fft_fig    = Figure(figsize=(8, 2.6), facecolor=C["panel"])
        self._fft_ax     = self._fft_fig.add_subplot(111)
        self._style_ax(self._fft_ax, "Frequency (Hz)", "Magnitude (dB)", log_x=True)
        self._fft_canvas = FigureCanvas(self._fft_fig)
        self._fft_canvas.setStyleSheet(f"background:{C['panel']};")
        fl.addWidget(self._fft_canvas)
        lay.addWidget(fft_grp)

        fr_grp = QGroupBox("Frequency Response  (20 Hz – 20 kHz)")
        fr_grp.setStyleSheet(self._grp_style())
        rl = QVBoxLayout(fr_grp); rl.setContentsMargins(5, 5, 5, 5)
        self._fr_fig    = Figure(figsize=(8, 2.6), facecolor=C["panel"])
        self._fr_ax     = self._fr_fig.add_subplot(111)
        self._style_ax(self._fr_ax, "Frequency (Hz)", "Response (dB)", log_x=True)
        self._fr_canvas = FigureCanvas(self._fr_fig)
        self._fr_canvas.setStyleSheet(f"background:{C['panel']};")
        rl.addWidget(self._fr_canvas)
        lay.addWidget(fr_grp)

        return w

    def _style_ax(self, ax, xlabel, ylabel, log_x=False) -> None:
        ax.set_facecolor(C["scope_bg"])
        ax.tick_params(colors=C["text"], labelsize=8)
        for sp in ax.spines.values(): sp.set_color(C["border"])
        ax.set_xlabel(xlabel, color=C["muted"], fontsize=9)
        ax.set_ylabel(ylabel, color=C["muted"], fontsize=9)
        ax.grid(True, color=C["border"], alpha=0.5, lw=0.5)
        if log_x:
            ax.set_xscale("log"); ax.set_xlim(20, 20_000)
        try:
            ax.figure.tight_layout(pad=0.4)
        except Exception:
            pass

    # ── Device population ─────────────────────────────────────────────────────

    def _populate_inputs(self) -> None:
        self._input_combo.clear()
        default = AudioDeviceManager.get_default_input()
        for dev in AudioDeviceManager.get_input_devices():
            self._input_combo.addItem(dev["name"], dev["index"])
        for i in range(self._input_combo.count()):
            if self._input_combo.itemData(i) == default:
                self._input_combo.setCurrentIndex(i); break

    def _populate_outputs(self) -> None:
        self._output_combo.clear()
        default = AudioDeviceManager.get_default_output()
        for dev in AudioDeviceManager.get_output_devices():
            self._output_combo.addItem(dev["name"], dev["index"])
        for i in range(self._output_combo.count()):
            if self._output_combo.itemData(i) == default:
                self._output_combo.setCurrentIndex(i); break

    # ── Event handlers ────────────────────────────────────────────────────────

    def _on_source_changed(self, text: str) -> None:
        external = (text == SOURCE_EXTERNAL)
        self._ext_hint.setVisible(external)
        self._output_lbl.setEnabled(not external)
        self._output_combo.setEnabled(not external)

    def _on_gain_change(self, value: int) -> None:
        self._gain_lbl.setText(f"{value/100.0:.1f}×")

    def _on_start(self) -> None:
        if self.reference_audio is None:
            QMessageBox.warning(self, "Missing Reference",
                                "Reference test WAV could not be loaded.")
            return

        external   = (self._source_combo.currentText() == SOURCE_EXTERNAL)
        input_dev  = self._input_combo.currentData()
        output_dev = self._output_combo.currentData()
        fs         = 44_100 if "44100" in self._sr_combo.currentText() else 48_000
        gain       = self._gain_slider.value() / 100.0
        stereo     = self._stereo_chk.isChecked()

        player   = AudioPlayer(output_device=output_dev) if not external else None
        recorder = AudioRecorder(device=input_dev, fs=fs,
                                 channels=2 if stereo else 1, gain=gain)
        recorder.level_callback = self._level_cb

        detector = SignalDetector(
            reference_wav_path=REFERENCE_WAV, fs=fs, threshold=0.45
        )

        self._reset_results()
        self._progress.setValue(0)
        self._set_running(True)
        self._refresh.start()

        self.test_worker = TestWorker(
            player, recorder, detector,
            self.reference_audio, SEGMENT_TIMES, self.baseline, external
        )
        self.test_worker.signals.status.connect(self._on_status)
        self.test_worker.signals.progress.connect(self._progress.setValue)
        self.test_worker.signals.complete.connect(self._on_complete)
        self.test_worker.signals.error.connect(self._on_error)
        self.test_worker.start()

    def _on_stop(self) -> None:
        if self.test_worker:
            self.test_worker.stop_test()
        self._refresh.stop()
        self._set_running(False)
        self._on_status("Stopped")

    def _on_calibrate(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("⚙  Calibrate — Loopback Setup")
        dlg.setMinimumWidth(540)
        dlg.setStyleSheet(
            f"QDialog{{background:{C['panel']};color:{C['text']};}}"
            f"QLabel{{color:{C['text']};font-size:11px;background:transparent;}}"
            f"QPushButton{{background:{C['accent2']};color:white;padding:6px 14px;"
            f"border-radius:4px;font-weight:bold;font-size:12px;}}"
        )
        dlg_lay = QVBoxLayout(dlg)
        dlg_lay.setContentsMargins(20, 16, 20, 16)
        dlg_lay.setSpacing(10)

        # Warning banner
        warn = QLabel("⚠  GAIN CALIBRATION — READ BEFORE PROCEEDING")
        warn.setStyleSheet(
            f"background:{C['accent']};color:white;font-size:13px;"
            f"font-weight:bold;padding:8px 12px;border-radius:6px;"
        )
        warn.setAlignment(Qt.AlignmentFlag.AlignCenter)
        dlg_lay.addWidget(warn)

        steps_txt = (
            "<ol style='margin:0;padding-left:20px;line-height:1.8;'>"
            f"<li><b style='color:{C['green']};font-size:12px;'>Connect loopback</b>"
            " — run a cable from your interface's <b>OUTPUT</b> back into its <b>INPUT</b>.</li>"

            f"<li><b style='color:{C['green']};font-size:12px;'>Watch the oscilloscope above</b>"
            " — the green waveform shows your live input level.</li>"

            f"<li><b style='color:{C['yellow']};font-size:12px;'>Turn your INPUT GAIN UP</b>"
            " — use the hardware gain knob on your interface.  Turn it <b>HIGH</b>."
            " You want the oscilloscope waveform to fill most of the window."
            " The level meter should reach <b>GREEN or YELLOW</b>.</li>"

            f"<li><b style='color:{C['orange']};font-size:12px;'>Avoid clipping!</b>"
            " If you see <b style='color:{C['red']};'>⚠ CLIP</b> in the level meter,"
            " or the waveform flatlines at the top/bottom, reduce gain slightly"
            " until clipping disappears.</li>"

            f"<li><b style='color:{C['cyan']};font-size:12px;'>Click OK</b>"
            " — calibration will play the full test signal (~32 s) and record"
            " the loopback as your baseline.  Keep the cable connected throughout.</li>"
            "</ol>"
        )
        steps_lbl = QLabel(steps_txt)
        steps_lbl.setWordWrap(True)
        steps_lbl.setTextFormat(Qt.TextFormat.RichText)
        steps_lbl.setStyleSheet(f"color:{C['text']};font-size:11px;background:transparent;")
        dlg_lay.addWidget(steps_lbl)

        tip = QLabel(
            "💡  Tip: Calibration saves your interface's own noise & distortion floor."
            "  Tests will subtract this baseline so only the device-under-test is measured."
        )
        tip.setWordWrap(True)
        tip.setStyleSheet(
            f"color:{C['cyan']};font-size:10px;"
            f"background:{C['accent2']};border-radius:4px;padding:6px 10px;"
        )
        dlg_lay.addWidget(tip)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.button(QDialogButtonBox.StandardButton.Ok).setText("▶  Start Calibration")
        btns.button(QDialogButtonBox.StandardButton.Cancel).setText("Cancel")
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        btns.setStyleSheet(
            f"QPushButton{{background:{C['accent2']};color:white;padding:6px 16px;"
            f"border-radius:4px;font-weight:bold;font-size:12px;}}"
            f"QPushButton[text='▶  Start Calibration']{{background:{C['accent']};}}"
        )
        dlg_lay.addWidget(btns)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        if self.reference_audio is None:
            QMessageBox.warning(self, "Missing Reference",
                                "Reference test WAV not loaded.")
            return

        input_dev  = self._input_combo.currentData()
        output_dev = self._output_combo.currentData()
        fs         = 44_100 if "44100" in self._sr_combo.currentText() else 48_000
        gain       = self._gain_slider.value() / 100.0

        player   = AudioPlayer(output_device=output_dev)
        recorder = AudioRecorder(device=input_dev, fs=fs, channels=1, gain=gain)
        recorder.level_callback = self._level_cb

        self._set_running(True)
        self._refresh.start()

        self.calib_worker = CalibWorker(
            player, recorder, self.reference_audio, SEGMENT_TIMES
        )
        self.calib_worker.signals.status.connect(self._on_status)
        self.calib_worker.signals.progress.connect(self._progress.setValue)
        self.calib_worker.signals.calib_done.connect(self._on_calib_done)
        self.calib_worker.signals.error.connect(self._on_error)
        self.calib_worker.start()

    def _on_regen_wav(self) -> None:
        fs = 44_100 if "44100" in self._sr_combo.currentText() else 48_000
        generate_reference_wav(REFERENCE_WAV, fs)
        self._load_ref_audio(REFERENCE_WAV)
        self.statusBar().showMessage(
            f"Reference test signal regenerated  ({fs} Hz, "
            f"{SEGMENT_TIMES['total']:.1f}s)."
        )

    # ── Worker result handlers ────────────────────────────────────────────────

    def _on_status(self, text: str) -> None:
        self._status_lbl.setText(text)
        self.statusBar().showMessage(text)

    def _on_complete(self, scores: dict) -> None:
        self._refresh.stop()
        self._set_running(False)

        final = scores.get("final", 0.0)
        self._main_score.setText(f"{final:.0f}")
        grade, emoji = scoring.get_grade(final)
        self._grade_lbl.setText(f"{emoji}  {grade}")

        colour = (C["green"] if final >= 90 else C["blue"] if final >= 75
                  else C["yellow"] if final >= 60 else C["red"])
        self._main_score.setStyleSheet(
            f"color:{colour};font-size:56px;font-weight:bold;border:none;"
        )

        s = scores.get("sinad", 0.0)
        self._card_clarity.set_score(s, scoring.get_metric_description("sinad", s),
                                     f"SINAD: {scores.get('raw_sinad',0):.1f} dB")
        t = scores.get("thd", 0.0)
        self._card_distortion.set_score(t, scoring.get_metric_description("thd", t),
                                        f"THD: {scores.get('raw_thd_db',0):.1f} dB")
        n = scores.get("noise", 0.0)
        self._card_noise.set_score(n, scoring.get_metric_description("noise", n),
                                   f"Floor: {scores.get('raw_noise_db',0):.1f} dBFS")
        f = scores.get("flatness", 0.0)
        self._card_balance.set_score(f, scoring.get_metric_description("flatness", f))

        # ── Freq response — update BOTH the inline (Test tab) and Advanced tab plots ──
        freqs = scores.get("freq_response_freqs")
        resp  = scores.get("freq_response_db")
        is_st = scores.get("is_stereo", False)

        def _draw_fr(ax, fig, canvas):
            ax.clear()
            self._style_ax(ax, "Frequency (Hz)", "Response (dB)", log_x=True)
            if freqs is not None and resp is not None:
                ax.plot(freqs, resp, color=C["accent"], lw=1.0, alpha=0.9, label="Avg")
                if is_st:
                    rL = scores.get("freq_response_db_L")
                    rR = scores.get("freq_response_db_R")
                    if rL is not None:
                        ax.plot(freqs, rL, color=C["cyan"], lw=0.8, alpha=0.7,
                                ls="--", label="L")
                    if rR is not None:
                        ax.plot(freqs, rR, color=C["purple"], lw=0.8, alpha=0.7,
                                ls="--", label="R")
                    ax.legend(facecolor=C["panel"], edgecolor=C["border"],
                              labelcolor=C["text"], fontsize=8)
                ax.axhline(0, color=C["muted"], lw=0.5, ls="--")
                ax.set_ylim(-30, 30)
                fig.tight_layout(pad=0.4)
                canvas.draw()

        _draw_fr(self._fr_ax,        self._fr_fig,        self._fr_canvas)
        _draw_fr(self._inline_fr_ax, self._inline_fr_fig, self._inline_fr_canvas)

        # Show/hide the inline chart placeholder vs plot
        if freqs is not None and resp is not None:
            self._inline_fr_placeholder.hide()
            self._inline_fr_canvas.show()
        else:
            self._inline_fr_placeholder.show()
            self._inline_fr_canvas.hide()
        self._inline_fr_grp.show()

        # Populate Results tab and switch to it
        self._populate_results(scores)
        self._tabs.setCurrentIndex(1)   # switch to Results

    def _on_calib_done(self, baseline: dict) -> None:
        self._refresh.stop()
        self._set_running(False)
        self._save_calibration(baseline)
        self._calib_badge.setText("✓ Calibrated")
        self._calib_badge.setStyleSheet(f"color:{C['green']};font-size:12px;")
        thd_db = 20 * np.log10(baseline["baseline_thd"] + 1e-12)
        QMessageBox.information(
            self, "Calibration Complete",
            f"Baseline saved:\n"
            f"  SINAD : {baseline['baseline_sinad']:.1f} dB\n"
            f"  THD   : {thd_db:.1f} dB\n"
            f"  Noise : {baseline['baseline_noise']:.1f} dBFS\n\n"
            "Future tests will subtract this baseline automatically."
        )
        self.statusBar().showMessage("Calibration complete.")

    def _on_error(self, msg: str) -> None:
        self._refresh.stop()
        self._set_running(False)
        self._on_status("Error")
        QMessageBox.critical(self, "Test Error", msg)

    # ── Audio level callback (audio thread — no Qt calls) ─────────────────────

    def _level_cb(self, rms: float, db: float, chunk: np.ndarray) -> None:
        self._last_rms   = rms
        self._last_db    = db
        self._last_chunk = chunk.copy()

    def _tick(self) -> None:
        if self._last_chunk is not None and len(self._last_chunk) > 0:
            self._scope.push(self._last_chunk)
            self._lvl_mtr.update_level(self._last_rms, self._last_db)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _set_running(self, running: bool) -> None:
        self._start_btn.setEnabled(not running)
        self._stop_btn.setEnabled(running)
        self._calib_btn.setEnabled(not running)
        self._regen_btn.setEnabled(not running)
        self._source_combo.setEnabled(not running)

    def _reset_results(self) -> None:
        self._main_score.setText("—")
        self._main_score.setStyleSheet(
            f"color:{C['accent']};font-size:56px;font-weight:bold;border:none;"
        )
        self._grade_lbl.setText("")
        for card in (self._card_clarity, self._card_distortion,
                     self._card_noise, self._card_balance):
            card.reset()

    def closeEvent(self, event) -> None:
        if self.test_worker and self.test_worker.isRunning():
            self.test_worker.stop_test()
            self.test_worker.wait(2000)
        event.accept()

    # ── Label helpers ─────────────────────────────────────────────────────────

    def _mlabel(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color:{C['muted']};font-size:11px;")
        return lbl

    def _rlabel(self, text: str, bold: bool = False,
                colour: Optional[str] = None) -> QLabel:
        lbl = QLabel(text)
        c   = colour or C["text"]
        w   = "bold" if bold else "normal"
        lbl.setStyleSheet(f"color:{c};font-size:11px;font-weight:{w};border:none;")
        return lbl

    # ── Style sheets ──────────────────────────────────────────────────────────

    @staticmethod
    def _btn_style(colour: str) -> str:
        return (f"QPushButton{{background:{colour};color:white;border:none;"
                f"border-radius:6px;padding:8px 14px;font-size:13px;font-weight:bold;}}"
                f"QPushButton:hover{{background:{colour}cc;}}"
                f"QPushButton:disabled{{background:#2a2a3a;color:#555566;}}")

    @staticmethod
    def _combo_style() -> str:
        return (f"QComboBox{{background:{C['accent2']};color:{C['text']};"
                f"border:1px solid {C['border']};border-radius:4px;"
                f"padding:4px 8px;min-width:140px;font-size:11px;}}"
                f"QComboBox::drop-down{{border:none;}}"
                f"QComboBox QAbstractItemView{{background:{C['panel']};color:{C['text']};"
                f"border:1px solid {C['border']};"
                f"selection-background-color:{C['accent2']};}}")

    @staticmethod
    def _grp_style() -> str:
        return (f"QGroupBox{{color:{C['muted']};font-size:11px;font-weight:bold;"
                f"border:1px solid {C['border']};border-radius:6px;"
                f"margin-top:8px;padding-top:6px;}}"
                f"QGroupBox::title{{subcontrol-origin:margin;"
                f"subcontrol-position:top left;left:10px;padding:0 4px;}}")

    @staticmethod
    def _tab_style() -> str:
        return (f"QTabWidget::pane{{border:1px solid {C['border']};"
                f"border-radius:4px;background:{C['panel']};}}"
                f"QTabBar::tab{{background:{C['accent2']};color:{C['text']};"
                f"padding:6px 18px;border-radius:4px 4px 0 0;"
                f"margin-right:2px;font-size:12px;}}"
                f"QTabBar::tab:selected{{background:{C['accent']};"
                f"color:white;font-weight:bold;}}")

    def _apply_theme(self) -> None:
        self.setStyleSheet(f"""
            QMainWindow {{ background:{C['bg']}; }}
            QWidget      {{ background:{C['bg']};color:{C['text']};
                            font-family:'Segoe UI',Arial,sans-serif; }}
            QScrollArea  {{ background:{C['bg']};border:none; }}
            QSlider::groove:horizontal {{
                background:#111122;height:6px;border-radius:3px;
            }}
            QSlider::handle:horizontal {{
                background:{C['accent']};width:15px;height:15px;
                margin:-5px 0;border-radius:8px;
            }}
            QSlider::sub-page:horizontal {{
                background:{C['accent2']};border-radius:3px;
            }}
            QCheckBox::indicator {{
                width:14px;height:14px;border:1px solid {C['border']};
                border-radius:3px;background:{C['panel']};
            }}
            QCheckBox::indicator:checked {{
                background:{C['accent']};
            }}
        """)


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("SoundScore")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
