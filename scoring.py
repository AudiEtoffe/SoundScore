#!/usr/bin/env python3
"""
Scoring Module — SoundScore
Created by Acid Reign Productions

Converts raw DSP measurements into 0–100 scores.

Bug fix vs v1
-------------
score_noise was INVERTED — a noise floor of -124 dBFS (excellent hardware)
was scoring 0 instead of 100.  Now: lower dBFS = quieter = higher score.

Scoring ranges
--------------
  SINAD  : 40 dB → 0,  100 dB → 100   (range chosen for pro & consumer gear)
  THD    : −40 dB → 0, −100 dB → 100  (log scale)
  Noise  : −40 dBFS → 0, −120 dBFS → 100  (FIXED: quieter = better)
  Flatness: σ = 10 → 0, σ = 0 → 100

Weights
-------
  SINAD    40 %   overall clarity / signal quality
  THD      25 %   harmonic distortion
  Noise    20 %   background noise floor
  Flatness 15 %   frequency balance
"""

from typing import Optional
import numpy as np


# ── Individual metric scorers ─────────────────────────────────────────────────

def score_sinad(sinad: float) -> float:
    """40 dB → 0,  100 dB → 100.  (Widened range so real-world CDJ chains score fairly.)"""
    return float(np.clip((sinad - 40.0) / 60.0 * 100.0, 0.0, 100.0))


def score_thd(thd: float) -> float:
    """THD ratio → 0-100.  −40 dB = 0,  −100 dB = 100."""
    thd_db = 20.0 * np.log10(thd + 1e-12)
    return float(np.clip((-thd_db - 40.0) / 60.0 * 100.0, 0.0, 100.0))


def score_noise(noise_db: float) -> float:
    """
    Noise floor (dBFS) → 0-100.
    QUIETER IS BETTER: -40 dBFS → 0,  -120 dBFS → 100.

    v1 Bug fix: the old formula gave score 0 for an excellent -120 dBFS
    reading (Presonus Studio 24c measured -124.8 dBFS).
    """
    return float(np.clip((-noise_db - 40.0) / 80.0 * 100.0, 0.0, 100.0))


def score_flatness(freq_response_db: Optional[np.ndarray]) -> float:
    """σ of response curve → 0-100.  Flat = 100."""
    if freq_response_db is None or len(freq_response_db) == 0:
        return 50.0
    valid = freq_response_db[np.abs(freq_response_db) < 30.0]
    if len(valid) == 0:
        return 0.0
    return float(np.clip(100.0 - np.std(valid) * 10.0, 0.0, 100.0))


def final_score(sinad_s: float, thd_s: float,
                noise_s: float, flatness_s: float) -> float:
    return float(sinad_s * 0.40 + thd_s * 0.25
                 + noise_s * 0.20 + flatness_s * 0.15)


# ── Per-frequency scorer ──────────────────────────────────────────────────────

def score_tone_result(tone: dict) -> float:
    """
    Score a single tone-analysis dict (from analysis.analyze_tone).
    Blends SINAD (60 %) and THD (40 %) for that frequency.
    """
    sinad_s = score_sinad(tone.get("sinad", 60.0))
    thd_s   = score_thd(tone.get("thd",    0.01))
    return float(sinad_s * 0.60 + thd_s * 0.40)


def score_stereo_tone(tone_stereo: dict) -> dict:
    """Score L and R channels of a stereo tone result."""
    l_score = score_tone_result(tone_stereo["left"])
    r_score = score_tone_result(tone_stereo["right"])
    winner  = "L" if l_score > r_score + 0.5 else \
              "R" if r_score > l_score + 0.5 else "="
    return {"freq": tone_stereo["freq"], "L": l_score, "R": r_score,
            "winner": winner}


# ── Qualitative labels ────────────────────────────────────────────────────────

def get_grade(score: float) -> tuple[str, str]:
    if score >= 90: return "Excellent", "🟢"
    if score >= 75: return "Good",      "🔵"
    if score >= 60: return "Average",   "🟡"
    return "Poor", "🔴"


_LABELS: dict[str, dict[str, str]] = {
    "sinad": {
        "excellent": "Crystal clear audio",
        "good":      "Clear with minor noise",
        "average":   "Noticeable noise",
        "poor":      "Heavy noise / distortion",
    },
    "thd": {
        "excellent": "Virtually zero distortion",
        "good":      "Very low distortion",
        "average":   "Moderate distortion",
        "poor":      "High distortion",
    },
    "noise": {
        "excellent": "Extremely quiet",
        "good":      "Very low noise floor",
        "average":   "Moderate noise",
        "poor":      "Loud background noise",
    },
    "flatness": {
        "excellent": "Perfectly balanced",
        "good":      "Well balanced",
        "average":   "Some frequency peaks",
        "poor":      "Uneven frequency response",
    },
}


def get_metric_description(metric: str, score: float) -> str:
    level = ("excellent" if score >= 90 else "good" if score >= 75
             else "average" if score >= 60 else "poor")
    return _LABELS.get(metric, {}).get(level, "")


# ── Top-level aggregator ──────────────────────────────────────────────────────

def compute_all_scores(measurements: dict,
                       baseline: Optional[dict] = None) -> dict:
    """
    Convert analysis.analyze_recording() output into final scores dict.

    Returns
    -------
    dict with:
        sinad, thd, noise, flatness, final  (0–100)
        raw_sinad, raw_thd_db, raw_noise_db  (physical values)
        tone_scores  list[dict]   per-frequency scores
        stereo_scores list[dict]  per-frequency L/R scores (if stereo)
        channel_summary  dict     overall L vs R (if stereo)
        freq_response_*  pass-through for plots
    """
    sinad    = float(measurements.get("sinad",       60.0))
    thd      = float(measurements.get("thd",          0.01))
    noise_db = float(measurements.get("noise_floor", -60.0))
    fr_db    = measurements.get("freq_response_db")

    # Baseline correction
    # NOTE: SINAD correction has been intentionally removed.
    # The formula (sinad - base_sinad + 80) could produce values like 124 dB
    # when the loopback baseline is low (e.g. 29 dB) — wildly unrealistic.
    # THD correction is kept: subtracting baseline THD removes the interface's
    # own harmonic fingerprint so you're measuring the device under test.
    if baseline:
        base_thd = float(baseline.get("baseline_thd", 0.001))
        thd      = max(thd - base_thd, 1e-12)

    scores: dict = {
        "sinad":    score_sinad(sinad),
        "thd":      score_thd(thd),
        "noise":    score_noise(noise_db),
        "flatness": score_flatness(fr_db),
    }
    scores["final"] = final_score(
        scores["sinad"], scores["thd"], scores["noise"], scores["flatness"]
    )

    # Raw values
    scores["raw_sinad"]    = sinad
    scores["raw_thd_db"]   = float(20.0 * np.log10(thd + 1e-12))
    scores["raw_noise_db"] = noise_db

    # Pass freq-response arrays through
    scores["freq_response_freqs"] = measurements.get("freq_response_freqs")
    scores["freq_response_db"]    = fr_db
    scores["freq_response_db_L"]  = measurements.get("freq_response_db_L")
    scores["freq_response_db_R"]  = measurements.get("freq_response_db_R")
    scores["is_stereo"]           = measurements.get("is_stereo", False)

    # Per-frequency scores
    tone_results = measurements.get("tone_results", [])
    is_stereo    = measurements.get("is_stereo", False)
    scores["tone_scores"]   = []
    scores["stereo_scores"] = []

    if is_stereo:
        for tr in tone_results:
            ss = score_stereo_tone(tr)
            scores["stereo_scores"].append(ss)
            avg = (ss["L"] + ss["R"]) / 2.0
            scores["tone_scores"].append({"freq": tr["freq"], "score": avg,
                                          "sinad_L": tr["left"]["sinad"],
                                          "sinad_R": tr["right"]["sinad"],
                                          "thd_L":   tr["left"]["thd_db"],
                                          "thd_R":   tr["right"]["thd_db"]})
        if scores["stereo_scores"]:
            l_mean = float(np.mean([s["L"] for s in scores["stereo_scores"]]))
            r_mean = float(np.mean([s["R"] for s in scores["stereo_scores"]]))
            l_nf   = float(measurements.get("noise_floor_L", noise_db))
            r_nf   = float(measurements.get("noise_floor_R", noise_db))
            scores["channel_summary"] = {
                "L_score":   l_mean,
                "R_score":   r_mean,
                "L_noise":   l_nf,
                "R_noise":   r_nf,
                "L_noise_s": score_noise(l_nf),
                "R_noise_s": score_noise(r_nf),
                "winner":    ("L" if l_mean > r_mean + 1.0
                              else "R" if r_mean > l_mean + 1.0 else "="),
            }
    else:
        for tr in tone_results:
            scores["tone_scores"].append({
                "freq":  tr.get("freq", 0),
                "score": score_tone_result(tr),
                "sinad": tr.get("sinad", 60.0),
                "thd_db": tr.get("thd_db", -40.0),
            })

    return scores
