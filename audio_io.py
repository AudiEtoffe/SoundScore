#!/usr/bin/env python3
"""
Audio I/O Module — SoundScore
Created by Acid Reign Productions

Handles device enumeration, WAV playback, and stream recording.

Stereo support
--------------
Set channels=2 on AudioRecorder to capture both channels.
stop() and get_recorded_audio() return:
  * shape (N,)   when channels=1  (mono float32)
  * shape (N, 2) when channels=2  (stereo float32, col 0=L, col 1=R)
"""

import threading
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
from scipy.io import wavfile


# ── Device management ─────────────────────────────────────────────────────────

class AudioDeviceManager:
    """Query and select system audio devices."""

    @staticmethod
    def get_input_devices() -> list[dict]:
        return [
            {
                "index": i,
                "name": dev["name"],
                "channels": dev["max_input_channels"],
                "default_samplerate": int(dev["default_samplerate"]),
            }
            for i, dev in enumerate(sd.query_devices())
            if dev["max_input_channels"] > 0
        ]

    @staticmethod
    def get_output_devices() -> list[dict]:
        return [
            {
                "index": i,
                "name": dev["name"],
                "channels": dev["max_output_channels"],
                "default_samplerate": int(dev["default_samplerate"]),
            }
            for i, dev in enumerate(sd.query_devices())
            if dev["max_output_channels"] > 0
        ]

    @staticmethod
    def get_default_input() -> int:
        idx = sd.default.device[0]
        return int(idx) if idx is not None else 0

    @staticmethod
    def get_default_output() -> int:
        idx = sd.default.device[1]
        return int(idx) if idx is not None else 0


# ── Playback ──────────────────────────────────────────────────────────────────

class AudioPlayer:
    """Non-blocking WAV file player."""

    def __init__(self, output_device: Optional[int] = None):
        self.output_device = output_device
        self.is_playing = False
        self._thread: Optional[threading.Thread] = None

    def play_wav(self, wav_path: str,
                 on_done: Optional[Callable] = None) -> None:
        """Play a WAV file asynchronously."""
        def _worker():
            self.is_playing = True
            try:
                rate, data = wavfile.read(wav_path)
                audio = self._to_float32(data)
                if audio.ndim > 1:
                    audio = audio[:, 0]
                sd.play(audio, samplerate=rate, device=self.output_device)
                sd.wait()
            except Exception as exc:
                print(f"[AudioPlayer] {exc}")
            finally:
                self.is_playing = False
                if on_done:
                    on_done()

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        sd.stop()
        self.is_playing = False

    @staticmethod
    def _to_float32(data: np.ndarray) -> np.ndarray:
        if data.dtype == np.int16:
            return data.astype(np.float32) / 32768.0
        if data.dtype == np.int32:
            return data.astype(np.float32) / 2_147_483_648.0
        return data.astype(np.float32)


# ── Recording ─────────────────────────────────────────────────────────────────

LevelCallback = Callable[[float, float, np.ndarray], None]
"""(rms, dBFS, mono_chunk) → None   — called from the audio thread."""


class AudioRecorder:
    """
    Streaming audio recorder built on sounddevice.InputStream.

    Parameters
    ----------
    device    : sounddevice device index
    fs        : sample rate
    channels  : 1 = mono, 2 = stereo
    gain      : linear amplitude applied to every block (can be changed live)
    blocksize : samples per callback

    Returns
    -------
    stop() / get_recorded_audio() return:
      (N,)   float32 array when channels == 1
      (N, 2) float32 array when channels == 2
    """

    def __init__(
        self,
        device:    Optional[int] = None,
        fs:        int           = 44_100,
        channels:  int           = 1,
        gain:      float         = 1.0,
        blocksize: int           = 1024,
    ):
        self.device    = device
        self.fs        = fs
        self.channels  = max(1, channels)
        self.gain      = gain
        self.blocksize = blocksize
        self.is_recording = False

        self.level_callback: Optional[LevelCallback] = None

        self._chunks: list[np.ndarray] = []
        self._stream: Optional[sd.InputStream] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            self._chunks = []
        self.is_recording = True
        self._stream = sd.InputStream(
            device=self.device,
            channels=self.channels,
            samplerate=self.fs,
            callback=self._callback,
            blocksize=self.blocksize,
            dtype="float32",
        )
        self._stream.start()

    def stop(self) -> np.ndarray:
        """Stop and return full recording."""
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self.is_recording = False
        return self._assemble()

    def get_recorded_audio(self) -> np.ndarray:
        """Snapshot of what's been captured so far (non-destructive)."""
        with self._lock:
            chunks = list(self._chunks)
        if not chunks:
            return np.array([], dtype=np.float32)
        return self._assemble(chunks)

    # ── internal ──────────────────────────────────────────────────────────────

    def _callback(self, indata: np.ndarray, frames: int,
                  time_info, status) -> None:
        if status:
            print(f"[AudioRecorder] {status}")
        chunk = indata.copy() * self.gain
        with self._lock:
            self._chunks.append(chunk)

        if self.level_callback is not None:
            mono = chunk[:, 0] if chunk.ndim > 1 else chunk.flatten()
            rms  = float(np.sqrt(np.mean(mono ** 2)))
            db   = float(20.0 * np.log10(rms + 1e-12))
            try:
                self.level_callback(rms, db, mono)
            except Exception:
                pass

    def _assemble(self, chunks: Optional[list] = None) -> np.ndarray:
        with self._lock:
            if chunks is None:
                chunks = list(self._chunks)
        if not chunks:
            return np.array([], dtype=np.float32)

        audio = np.concatenate(chunks, axis=0).astype(np.float32)

        if self.channels == 1:
            # Flatten to 1-D mono
            return audio[:, 0] if audio.ndim > 1 else audio
        else:
            # Return (N, channels); pad to exactly 2 cols if needed
            if audio.ndim == 1:
                audio = audio[:, np.newaxis]
            if audio.shape[1] == 1:
                audio = np.concatenate([audio, audio], axis=1)
            return audio[:, :2]   # (N, 2)
