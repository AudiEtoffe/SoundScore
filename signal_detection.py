#!/usr/bin/env python3
"""
Signal Detection Module
=======================
Detects the start of the test signal inside a recording by performing
normalised cross-correlation against the known sync-chirp template.

How it works
------------
1. A chirp template (500 Hz → 5 kHz, 0.5 s) is either loaded from the
   first 0.5 s of the reference WAV, or synthesised on-the-fly.
2. The detector searches the captured audio buffer for a region whose
   normalised correlation with the template exceeds ``threshold``.
3. The returned index is the sample offset of the chirp start, which can
   be used to align the recording with the reference WAV.
"""

from typing import Optional, Tuple

import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate


class SignalDetector:
    """
    Chirp-based test-signal detector.

    Parameters
    ----------
    reference_wav_path : path to the reference WAV file.  If supplied the
                         first ``template_duration`` seconds are used as the
                         template.  Otherwise a synthetic chirp is generated.
    fs                 : sample rate expected in the audio being tested.
    threshold          : normalised correlation coefficient that triggers
                         detection (0–1; lower = more sensitive, 0.45 is
                         a reasonable starting point).
    template_duration  : seconds to use as the chirp template (default 0.45 s,
                         slightly shorter than the full 0.5 s chirp to give
                         the correlator room to slide).
    """

    def __init__(
        self,
        reference_wav_path: Optional[str] = None,
        fs: int = 44100,
        threshold: float = 0.45,
        template_duration: float = 0.45,
    ):
        self.fs = fs
        self.threshold = threshold
        self.template: Optional[np.ndarray] = None

        if reference_wav_path:
            self._load_template(reference_wav_path, template_duration)
        else:
            self._generate_template(template_duration)

    # ── public API ────────────────────────────────────────────────────────────

    def detect(self, audio_buffer: np.ndarray) -> Tuple[bool, int]:
        """
        Search ``audio_buffer`` for the sync chirp.

        Returns
        -------
        (detected, index)
          detected : True if chirp was found
          index    : sample index of the best match inside ``audio_buffer``
                     (-1 if not detected)
        """
        if self.template is None or len(audio_buffer) < len(self.template):
            return False, -1

        buf  = audio_buffer.astype(np.float32)
        tmpl = self.template.astype(np.float32)

        # Normalise both signals to unit peak
        buf_peak  = np.max(np.abs(buf))  + 1e-9
        tmpl_peak = np.max(np.abs(tmpl)) + 1e-9
        buf_n  = buf  / buf_peak
        tmpl_n = tmpl / tmpl_peak

        # Full cross-correlation (valid mode: slide template over buffer)
        corr = correlate(buf_n, tmpl_n, mode="valid")

        # Normalise: divide by the template energy so a perfect match → ~1.0
        energy = float(np.sum(tmpl_n ** 2))
        if energy < 1e-9:
            return False, -1
        corr_norm = corr / energy

        best_idx  = int(np.argmax(np.abs(corr_norm)))
        best_val  = float(np.abs(corr_norm[best_idx]))

        if best_val >= self.threshold:
            return True, best_idx
        return False, -1

    def reset(self) -> None:
        """Clear any streaming state (call before a new test)."""
        # No persistent streaming state in this implementation,
        # but kept for API compatibility.
        pass

    # ── internal helpers ──────────────────────────────────────────────────────

    def _generate_template(self, duration: float) -> None:
        """Synthesise a linear chirp matching the reference WAV marker."""
        n = int(self.fs * duration)
        t = np.linspace(0, duration, n, endpoint=False)
        f_start, f_end = 500.0, 5000.0
        k = (f_end - f_start) / duration
        phase = 2 * np.pi * (f_start * t + 0.5 * k * t ** 2)
        sig = np.sin(phase).astype(np.float32)
        # Normalise
        self.template = sig / (np.max(np.abs(sig)) + 1e-9)

    def _load_template(self, wav_path: str, duration: float) -> None:
        """Load the first N seconds of the reference WAV as the template."""
        try:
            rate, data = wavfile.read(wav_path)
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            else:
                data = data.astype(np.float32)
            if data.ndim > 1:
                data = data[:, 0]

            n = int(rate * duration)
            self.template = data[:n]
            self.template = self.template / (np.max(np.abs(self.template)) + 1e-9)

            # Update fs to match the WAV (in case caller didn't match)
            self.fs = rate
        except Exception as exc:
            print(f"[SignalDetector] Could not load template from {wav_path}: {exc}")
            print("[SignalDetector] Falling back to synthetic chirp template.")
            self._generate_template(duration)
