#!/usr/bin/env python3
"""
Audio Analysis Module — SoundScore
Created by Acid Reign Productions

Core DSP: per-frequency THD/SINAD, stereo L/R comparison,
noise floor, and frequency response.

Key improvements over v1
------------------------
* SINAD uses a ±10 Hz window around the fundamental (not a single bin)
  so Hanning-window leakage doesn't falsely inflate noise estimates.
* Tone segments are trimmed 150 ms at each end to skip transients.
* analyze_recording() handles both mono (N,) and stereo (N,2) arrays.
* Per-frequency results are returned for the detailed results page.
"""

from typing import Optional, Tuple

import numpy as np


# ── Low-level DSP ─────────────────────────────────────────────────────────────

def compute_fft(signal: np.ndarray, fs: int) -> Tuple[np.ndarray, np.ndarray]:
    """Windowed real FFT → (freqs, magnitude)."""
    n      = len(signal)
    window = np.hanning(n)
    spec   = np.fft.rfft(signal * window)
    freqs  = np.fft.rfftfreq(n, 1.0 / fs)
    return freqs, np.abs(spec)


def _peak_near(mag: np.ndarray, freqs: np.ndarray,
               target_hz: float, window_hz: float = 50.0) -> Tuple[float, int]:
    """Return (peak_magnitude, bin_index) near target_hz."""
    freq_res   = freqs[1] - freqs[0]
    w          = max(5, int(window_hz / freq_res))
    idx        = int(np.argmin(np.abs(freqs - target_hz)))
    lo, hi     = max(0, idx - w), min(len(mag), idx + w + 1)
    peak_idx   = lo + int(np.argmax(mag[lo:hi]))
    return float(mag[peak_idx]), peak_idx


def compute_thd(signal: np.ndarray, fs: int,
                fundamental_freq: float = 1_000.0) -> float:
    """THD as a linear ratio (0 = perfect)."""
    freqs, mag = compute_fft(signal, fs)
    fund, _    = _peak_near(mag, freqs, fundamental_freq)
    if fund < 1e-10:
        return 0.0
    harmonics = []
    for n in range(2, 7):
        hf = fundamental_freq * n
        if hf > fs / 2.0:
            break
        h, _ = _peak_near(mag, freqs, hf, window_hz=30.0)
        harmonics.append(h)
    if not harmonics:
        return 0.0
    return float(np.sqrt(np.sum(np.square(harmonics))) / fund)


def compute_thd_db(signal: np.ndarray, fs: int,
                   fundamental_freq: float = 1_000.0) -> float:
    return float(20.0 * np.log10(compute_thd(signal, fs, fundamental_freq) + 1e-12))


def compute_sinad(signal: np.ndarray, fs: int,
                  fundamental_freq: float = 1_000.0) -> float:
    """
    SINAD (dB).  Uses ±10 Hz window around fundamental to avoid
    counting Hanning-window leakage as noise.
    """
    freqs, mag = compute_fft(signal, fs)
    freq_res   = max(freqs[1] - freqs[0], 1e-6)

    # Signal energy: window around fundamental
    window_hz  = max(10.0, freq_res * 4)
    lo  = max(0, int((fundamental_freq - window_hz) / freq_res))
    hi  = min(len(mag), int((fundamental_freq + window_hz) / freq_res) + 1)
    sig_power  = float(np.sum(mag[lo:hi] ** 2))

    total_power = float(np.sum(mag ** 2))
    nd_power    = total_power - sig_power

    if nd_power <= 0.0:
        return 120.0
    return float(10.0 * np.log10(sig_power / (nd_power + 1e-30)))


def compute_noise_floor(signal: np.ndarray) -> float:
    """RMS noise floor in dBFS from a silence segment."""
    rms = float(np.sqrt(np.mean(signal ** 2)))
    return float(20.0 * np.log10(rms + 1e-12))


def compute_frequency_response(
    recorded:  np.ndarray,
    reference: np.ndarray,
    fs:        int,
    freq_range: Tuple[float, float] = (20.0, 20_000.0),
) -> Tuple[np.ndarray, np.ndarray]:
    """Frequency response: recorded/reference magnitude in dB."""
    n   = min(len(recorded), len(reference))
    f_r, m_r = compute_fft(recorded[:n],  fs)
    _,   m_ref = compute_fft(reference[:n], fs)
    resp = 20.0 * np.log10((m_r + 1e-9) / (m_ref + 1e-9))
    mask = (f_r >= freq_range[0]) & (f_r <= freq_range[1])
    return f_r[mask], resp[mask]


# ── Per-tone analysis ─────────────────────────────────────────────────────────

def _trim_edges(audio: np.ndarray, fs: int, ms: int = 150) -> np.ndarray:
    """Drop `ms` milliseconds from each end to skip click transients."""
    n = int(fs * ms / 1_000)
    if len(audio) <= 2 * n:
        return audio
    return audio[n:-n]


def analyze_tone(mono: np.ndarray, fs: int, freq: float) -> dict:
    """
    Analyse a single mono tone segment.
    Returns dict with thd, thd_db, sinad, level_db.
    """
    seg = _trim_edges(mono, fs)
    if len(seg) < fs * 0.5:
        return {"freq": freq, "thd": 0.0, "thd_db": 0.0,
                "sinad": 0.0, "level_db": -99.0}
    return {
        "freq":     freq,
        "thd":      compute_thd(seg, fs, freq),
        "thd_db":   compute_thd_db(seg, fs, freq),
        "sinad":    compute_sinad(seg, fs, freq),
        "level_db": compute_noise_floor(seg),   # signal RMS level
    }


def analyze_tone_stereo(stereo: np.ndarray, fs: int, freq: float) -> dict:
    """
    Analyse a stereo tone segment (shape N×2).
    Returns dict with 'left', 'right' sub-dicts.
    """
    left  = stereo[:, 0] if stereo.ndim > 1 else stereo
    right = stereo[:, 1] if stereo.ndim > 1 and stereo.shape[1] > 1 else left
    return {
        "freq":  freq,
        "left":  analyze_tone(left,  fs, freq),
        "right": analyze_tone(right, fs, freq),
    }


# ── Full recording pipeline ───────────────────────────────────────────────────

def analyze_recording(
    recording:       np.ndarray,
    reference_audio: np.ndarray,
    fs:              int,
    segment_times:   dict,
    detection_offset: int = 0,
) -> dict:
    """
    Run all measurements over a full aligned recording.

    Parameters
    ----------
    recording        : 1-D (mono) or 2-D shape (N,2) (stereo) float32 array
    reference_audio  : 1-D float32 reference signal
    fs               : sample rate
    segment_times    : SEGMENT_TIMES dict from generate_reference_wav
    detection_offset : sample index of the detected chirp start

    Returns
    -------
    dict with keys:
        is_stereo, tone_results, noise_floor,
        freq_response_freqs, freq_response_db,
        error (only on failure)
    """
    # Align to detection offset
    if detection_offset > 0:
        recording = recording[detection_offset:] if recording.ndim == 1 \
                    else recording[detection_offset:]

    is_stereo = recording.ndim == 2 and recording.shape[1] >= 2

    def t2s(t: float) -> int:
        return int(round(t * fs))

    # Validate length
    tone_list = segment_times.get("tones", [])
    if not tone_list:
        return {"error": "SEGMENT_TIMES contains no tone definitions."}

    last_tone_end = t2s(tone_list[-1]["end"])
    if len(recording) < last_tone_end:
        return {
            "error": (
                f"Recording is too short ({len(recording)/fs:.1f}s). "
                "Check input device and gain."
            )
        }

    results: dict = {
        "is_stereo":    is_stereo,
        "tone_results": [],
    }

    # ── Per-frequency tones ────────────────────────────────────────────────────
    for seg in tone_list:
        s, e, freq = t2s(seg["start"]), t2s(seg["end"]), seg["freq"]
        e = min(e, len(recording))
        chunk = recording[s:e]

        if is_stereo:
            results["tone_results"].append(
                analyze_tone_stereo(chunk, fs, float(freq))
            )
        else:
            mono  = chunk[:, 0] if chunk.ndim > 1 else chunk
            results["tone_results"].append(
                analyze_tone(mono, fs, float(freq))
            )

    # Keep 1 kHz result at top level for backward-compat with scoring
    for tr in results["tone_results"]:
        if abs(tr["freq"] - 1_000) < 1:
            if is_stereo:
                avg_thd   = (tr["left"]["thd"]   + tr["right"]["thd"])   / 2
                avg_sinad = (tr["left"]["sinad"]  + tr["right"]["sinad"]) / 2
            else:
                avg_thd   = tr["thd"]
                avg_sinad = tr["sinad"]
            results["thd"]    = avg_thd
            results["thd_db"] = float(20.0 * np.log10(avg_thd + 1e-12))
            results["sinad"]  = avg_sinad
            break

    # ── Silence / noise floor ─────────────────────────────────────────────────
    sil_s = t2s(segment_times.get("silence_start", 23.1))
    sil_e = t2s(segment_times.get("silence_end",   26.1))
    if len(recording) >= sil_e:
        sil = recording[sil_s:sil_e]
        if is_stereo:
            nf_L = compute_noise_floor(sil[:, 0])
            nf_R = compute_noise_floor(sil[:, 1])
            results["noise_floor"]   = (nf_L + nf_R) / 2.0
            results["noise_floor_L"] = nf_L
            results["noise_floor_R"] = nf_R
        else:
            m = sil[:, 0] if sil.ndim > 1 else sil
            results["noise_floor"] = compute_noise_floor(m)
    else:
        results["noise_floor"] = -60.0

    # ── Frequency response (sweep) ────────────────────────────────────────────
    sw_s = t2s(segment_times.get("sweep_start", 26.1))
    sw_e = t2s(segment_times.get("sweep_end",   31.1))
    if len(recording) >= sw_e and len(reference_audio) >= sw_e:
        rec_sw = recording[sw_s:sw_e]
        ref_sw = reference_audio[sw_s:sw_e]
        if is_stereo:
            fq_L, rsp_L = compute_frequency_response(rec_sw[:, 0], ref_sw, fs)
            fq_R, rsp_R = compute_frequency_response(rec_sw[:, 1], ref_sw, fs)
            results["freq_response_freqs"]   = fq_L
            results["freq_response_db"]      = (rsp_L + rsp_R) / 2.0
            results["freq_response_db_L"]    = rsp_L
            results["freq_response_db_R"]    = rsp_R
        else:
            m = rec_sw[:, 0] if rec_sw.ndim > 1 else rec_sw
            fq, rsp = compute_frequency_response(m, ref_sw, fs)
            results["freq_response_freqs"] = fq
            results["freq_response_db"]    = rsp
    else:
        results["freq_response_freqs"] = None
        results["freq_response_db"]    = None

    # Ensure mandatory keys always present
    results.setdefault("thd",    0.01)
    results.setdefault("thd_db", -40.0)
    results.setdefault("sinad",  60.0)

    return results
