#!/usr/bin/env python3
"""
SoundScore Reference Test Signal Generator
===========================================
Created by Acid Reign Productions

Generates the master WAV file used for all SoundScore measurements.
Play this file from any source — internal playback, CDJ, or MP3 player.

Signal layout  (total ≈ 32.1 seconds)
--------------------------------------
  0.00 – 0.50 s  : Sync chirp 500 → 5 000 Hz  (auto-detection trigger)
  0.50 – 0.60 s  : Gap / silence
  0.60 – 3.10 s  : 50 Hz tone  @ −12 dBFS  (2.5 s)
  3.10 – 5.60 s  : 100 Hz tone  @ −12 dBFS
  5.60 – 8.10 s  : 250 Hz tone  @ −12 dBFS
  8.10 – 10.60 s : 500 Hz tone  @ −12 dBFS
 10.60 – 13.10 s : 1 000 Hz tone @ −12 dBFS
 13.10 – 15.60 s : 2 000 Hz tone @ −12 dBFS
 15.60 – 18.10 s : 5 000 Hz tone @ −12 dBFS
 18.10 – 20.60 s : 10 000 Hz tone @ −12 dBFS
 20.60 – 23.10 s : 16 000 Hz tone @ −12 dBFS
 23.10 – 26.10 s : Silence  (noise floor measurement)
 26.10 – 31.10 s : Log sweep 20 → 20 000 Hz  (frequency response)
 31.10 – 32.10 s : End silence

Run standalone to regenerate the master WAV:
    python generate_reference_wav.py [output_path] [sample_rate]
"""

import sys
import numpy as np
from scipy.io import wavfile

# ── Test-tone frequencies ──────────────────────────────────────────────────────
TONE_FREQS = [50, 100, 250, 500, 1_000, 2_000, 5_000, 10_000, 16_000]
TONE_DURATION = 2.5  # seconds per tone


# ── Build segment timing dict ─────────────────────────────────────────────────

def _build_segment_times() -> dict:
    t = 0.0
    times: dict = {}

    # Sync chirp
    times["chirp_start"] = t;  t += 0.5
    times["chirp_end"]   = t

    # Gap
    t += 0.1
    times["gap_end"] = t

    # Tone segments
    times["tones"] = []
    for freq in TONE_FREQS:
        seg = {"freq": freq, "start": round(t, 6), "end": round(t + TONE_DURATION, 6)}
        times["tones"].append(seg)
        t += TONE_DURATION

    # Silence for noise floor
    times["silence_start"] = round(t, 6);  t += 3.0
    times["silence_end"]   = round(t, 6)

    # Logarithmic sweep
    times["sweep_start"] = round(t, 6);  t += 5.0
    times["sweep_end"]   = round(t, 6)

    # End silence
    t += 1.0
    times["total"] = round(t, 6)

    return times


SEGMENT_TIMES = _build_segment_times()


# ── Signal generators ─────────────────────────────────────────────────────────

def _linear_chirp(dur: float, f0: float, f1: float,
                  fs: int, amp: float = 0.70) -> np.ndarray:
    t = np.linspace(0, dur, int(fs * dur), endpoint=False)
    k = (f1 - f0) / dur
    return (amp * np.sin(2 * np.pi * (f0 * t + 0.5 * k * t ** 2))).astype(np.float32)


def _sine_tone(dur: float, freq: float,
               fs: int, dbfs: float = -12.0) -> np.ndarray:
    amp = 10 ** (dbfs / 20.0)
    t   = np.linspace(0, dur, int(fs * dur), endpoint=False)
    # Fade in/out 10 ms to prevent clicks
    fade_n = min(int(0.01 * fs), int(0.1 * fs * dur))
    sig = amp * np.sin(2 * np.pi * freq * t)
    win = np.ones(len(sig))
    win[:fade_n]  = np.linspace(0, 1, fade_n)
    win[-fade_n:] = np.linspace(1, 0, fade_n)
    return (sig * win).astype(np.float32)


def _silence(dur: float, fs: int) -> np.ndarray:
    return np.zeros(int(fs * dur), dtype=np.float32)


def _log_sweep(dur: float, f0: float, f1: float,
               fs: int, amp: float = 0.50) -> np.ndarray:
    t   = np.linspace(0, dur, int(fs * dur), endpoint=False)
    k   = dur / np.log(f1 / f0)
    sig = amp * np.sin(2 * np.pi * k * f0 * (np.exp(t / k) - 1))
    # Fade in/out
    fade_n = min(int(0.02 * fs), 1000)
    sig[:fade_n]  *= np.linspace(0, 1, fade_n)
    sig[-fade_n:] *= np.linspace(1, 0, fade_n)
    return sig.astype(np.float32)


# ── Main generation function ──────────────────────────────────────────────────

def generate_reference_wav(output_path: str = "reference_test.wav",
                           fs: int = 44_100) -> dict:
    """
    Generate and save the SoundScore master reference WAV.

    Parameters
    ----------
    output_path : destination file path
    fs          : sample rate (44100 or 48000)

    Returns
    -------
    SEGMENT_TIMES dict
    """
    print("SoundScore — Reference Test Signal Generator")
    print("Created by Acid Reign Productions")
    print(f"Generating {SEGMENT_TIMES['total']:.1f}s test signal at {fs} Hz → {output_path}")

    parts = [
        _linear_chirp(0.50, 500, 5_000, fs, amp=0.70),   # sync chirp
        _silence(0.10, fs),                                # gap
    ]
    for seg in SEGMENT_TIMES["tones"]:
        parts.append(_sine_tone(TONE_DURATION, seg["freq"], fs, dbfs=-12.0))

    parts += [
        _silence(3.00, fs),                                # noise floor
        _log_sweep(5.00, 20, 20_000, fs, amp=0.50),       # freq response
        _silence(1.00, fs),                                # end pad
    ]

    signal = np.concatenate(parts)
    signal = np.clip(signal, -0.99, 0.99)
    wavfile.write(output_path, fs, (signal * 32767).astype(np.int16))

    dur = len(signal) / fs
    print(f"  Saved: {dur:.2f}s  ({len(signal):,} samples)")
    print("\n  Segments:")
    print(f"    Sync chirp   : 0.00 – 0.50 s")
    for seg in SEGMENT_TIMES["tones"]:
        print(f"    {seg['freq']:>6} Hz    : {seg['start']:.2f} – {seg['end']:.2f} s")
    times = SEGMENT_TIMES
    print(f"    Silence      : {times['silence_start']:.2f} – {times['silence_end']:.2f} s")
    print(f"    Log sweep    : {times['sweep_start']:.2f} – {times['sweep_end']:.2f} s")
    return dict(SEGMENT_TIMES)


if __name__ == "__main__":
    out  = sys.argv[1] if len(sys.argv) > 1 else "reference_test.wav"
    rate = int(sys.argv[2]) if len(sys.argv) > 2 else 44_100
    generate_reference_wav(out, rate)
