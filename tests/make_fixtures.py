"""
Generate synthetic but musically-structured audio fixtures.

Not white noise with a label on it. The beat has a real 4/4 kick/snare/hat
pattern with an 808 following a chord progression, and the vocal is a
phrased melody sung in the same key with breaths between phrases and
realistic intonation error. That matters because the engine's behaviour is
entirely driven by what it measures: a fixture without downbeats, phrases,
notes or harmony exercises only the error paths.

Used by the end-to-end smoke test, which is the thing the project has never
had -- its only recorded real run produced zero renders and left no
explanation.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import soundfile as sf

SR = 44100


def _adsr(n: int, sr: int, a=0.005, d=0.05, s=0.7, r=0.1) -> np.ndarray:
    env = np.ones(n)
    ai, di, ri = int(a * sr), int(d * sr), int(r * sr)
    ai, di, ri = min(ai, n), min(di, max(n - ai, 0)), min(ri, n)
    if ai:
        env[:ai] = np.linspace(0, 1, ai)
    if di:
        env[ai:ai + di] = np.linspace(1, s, di)
    env[ai + di:] = s
    if ri:
        env[-ri:] *= np.linspace(1, 0, ri)
    return env


def _kick(sr: int, dur=0.25) -> np.ndarray:
    n = int(dur * sr)
    t = np.arange(n) / sr
    f = 120.0 * np.exp(-t * 28.0) + 45.0          # pitch drop
    y = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t * 11.0)
    return (y * 0.9).astype(np.float32)


def _snare(sr: int, dur=0.2, seed=0) -> np.ndarray:
    n = int(dur * sr)
    t = np.arange(n) / sr
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, n) * np.exp(-t * 22.0)
    tone = np.sin(2 * np.pi * 190.0 * t) * np.exp(-t * 30.0) * 0.5
    return ((noise * 0.7 + tone) * 0.55).astype(np.float32)


def _hat(sr: int, dur=0.06, seed=0) -> np.ndarray:
    n = int(dur * sr)
    t = np.arange(n) / sr
    rng = np.random.default_rng(seed)
    y = rng.normal(0, 1, n) * np.exp(-t * 90.0)
    # Crude high-pass so it reads as a hat rather than broadband noise.
    y = np.diff(np.concatenate([[0.0], y]))
    return (y * 0.22).astype(np.float32)


def _sine_stack(f0: float, n: int, sr: int, partials=(1, 2, 3, 4),
                gains=(1.0, 0.5, 0.3, 0.15)) -> np.ndarray:
    t = np.arange(n) / sr
    y = np.zeros(n)
    for p, g in zip(partials, gains):
        y += g * np.sin(2 * np.pi * f0 * p * t)
    return y / max(np.max(np.abs(y)), 1e-9)


def midi_hz(m: float) -> float:
    return 440.0 * (2.0 ** ((m - 69.0) / 12.0))


def make_beat(path: str, bpm: float = 140.0, bars: int = 16,
              key_pc: int = 9, mode: str = "minor",
              progression: Tuple[int, ...] = (0, 5, 3, 4),
              swing: float = 0.5, sr: int = SR,
              write_stems_dir: str = None) -> dict:
    """A trap-ish instrumental with a real grid, chords and an 808."""
    beat = 60.0 / bpm
    bar = beat * 4
    n = int(bars * bar * sr) + sr
    drums = np.zeros(n)
    bass = np.zeros(n)
    other = np.zeros(n)

    scale = (0, 2, 3, 5, 7, 8, 10) if mode == "minor" else (0, 2, 4, 5, 7, 9, 11)

    def place(buf, sample, at):
        i = int(at * sr)
        j = min(i + len(sample), len(buf))
        if i < len(buf):
            buf[i:j] += sample[:j - i]

    downbeats: List[float] = []
    for b in range(bars):
        t0 = b * bar
        downbeats.append(t0)
        deg = progression[b % len(progression)]
        root_pc = (key_pc + scale[deg % 7]) % 12
        root_midi = 33 + root_pc                   # low 808 register

        # Drums: kick on 1 and 3-and, snare on 2 and 4, hats in 8ths.
        for k in (0.0, 2.5):
            place(drums, _kick(sr), t0 + k * beat)
        for s in (1.0, 3.0):
            place(drums, _snare(sr, seed=b), t0 + s * beat)
        for h in range(8):
            frac = h / 2.0
            if h % 2 == 1:                         # swing the off-beats
                frac += (swing - 0.5) * 1.0
            place(drums, _hat(sr, seed=b * 8 + h), t0 + frac * beat)

        # 808: root note held for the bar.
        nb = int(bar * sr)
        sub = _sine_stack(midi_hz(root_midi), nb, sr, (1, 2), (1.0, 0.25))
        place(bass, (sub * _adsr(nb, sr, 0.004, 0.25, 0.55, 0.15) * 0.8), t0)

        # Chord pad: triad in a mid register, so there is real harmony to detect.
        third = 4 if mode == "major" else 3
        for iv in (0, third, 7):
            pad = _sine_stack(midi_hz(57 + root_pc + iv), nb, sr, (1, 2, 3),
                              (1.0, 0.35, 0.15))
            place(other, pad * _adsr(nb, sr, 0.03, 0.2, 0.5, 0.3) * 0.16, t0)

    mix = drums + bass + other
    mix = mix / max(np.max(np.abs(mix)), 1e-9) * 0.7
    sf.write(path, mix.astype(np.float32), sr, subtype="PCM_24")

    stems = {}
    if write_stems_dir:
        os.makedirs(write_stems_dir, exist_ok=True)
        for name, buf in (("drums", drums), ("bass", bass), ("other", other)):
            p = os.path.join(write_stems_dir, "%s.wav" % name)
            sf.write(p, (buf / max(np.max(np.abs(mix)), 1e-9) * 0.7).astype(np.float32),
                     sr, subtype="PCM_16")
            stems[name] = p

    return {"path": path, "bpm": bpm, "bars": bars, "downbeats": downbeats,
            "key_pc": key_pc, "mode": mode, "stems": stems, "sr": sr}


def make_vocal(path: str, bpm: float = 140.0, bars: int = 16,
               key_pc: int = 9, mode: str = "minor",
               phrases_per_4bars: int = 1, sr: int = SR,
               detune_cents: float = 18.0, seed: int = 3) -> dict:
    """A phrased melody in the same key, with breaths and intonation error."""
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    bar = beat * 4
    n = int(bars * bar * sr) + sr
    y = np.zeros(n)
    scale = (0, 2, 3, 5, 7, 8, 10) if mode == "minor" else (0, 2, 4, 5, 7, 9, 11)

    phrase_spans: List[Tuple[float, float]] = []
    notes: List[dict] = []

    for group in range(bars // 4):
        # Sing for three bars, breathe for one -- so phrase detection has a
        # real gap to find rather than a continuous drone.
        t0 = group * 4 * bar
        t_end = t0 + 3 * bar
        phrase_spans.append((t0, t_end))
        t = t0
        while t < t_end - 0.05:
            dur = float(rng.choice([0.5, 0.5, 1.0, 0.25, 0.75])) * beat
            dur = min(dur, t_end - t)
            deg = int(rng.integers(0, 7))
            midi = 57 + key_pc + scale[deg]        # comfortable vocal register
            midi += float(rng.normal(0, detune_cents / 100.0))
            nn = int(dur * 0.85 * sr)
            if nn > 32:
                seg = _sine_stack(midi_hz(midi), nn, sr)
                seg *= _adsr(nn, sr, 0.02, 0.08, 0.75, 0.06)
                # Light vibrato so the analyser sees real expression.
                vt = np.arange(nn) / sr
                seg *= 1.0 + 0.02 * np.sin(2 * np.pi * 5.2 * vt)
                i = int(t * sr)
                y[i:i + nn] += seg * 0.55
                notes.append({"start": t, "end": t + nn / sr, "midi": midi})
            t += dur

    y += rng.normal(0, 10 ** (-62 / 20.0), n)      # realistic noise floor
    y = y / max(np.max(np.abs(y)), 1e-9) * 0.6
    sf.write(path, y.astype(np.float32), sr, subtype="PCM_24")
    return {"path": path, "bpm": bpm, "key_pc": key_pc, "mode": mode,
            "phrases": phrase_spans, "notes": notes, "sr": sr}


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "fixtures")
    os.makedirs(out, exist_ok=True)
    b = make_beat(os.path.join(out, "beat.wav"),
                  write_stems_dir=os.path.join(out, "stems"))
    v = make_vocal(os.path.join(out, "vocal.wav"))
    print("beat :", b["path"], b["bpm"], "BPM", len(b["downbeats"]), "bars")
    print("vocal:", v["path"], len(v["phrases"]), "phrases", len(v["notes"]), "notes")
