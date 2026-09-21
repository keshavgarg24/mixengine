"""
Transition effects: the moves that make a section change sound decided.

The plan says *where* the energy steps and *which* devices the step
justifies (`musical/energy.transition_before`). This module makes the
audio. Each effect is built from the material already in the render where
that is possible -- the vocal's own tail, the beat's own last beat --
because a device made from the song belongs to it in a way a sample never
quite does, and because there is no sample library here to draw on.

  **riser**             filtered noise swelling into the boundary, plus a
                        sine sweeping up an octave underneath it; cut dead
                        on the downbeat so the arrival is the silence that
                        follows it
  **impact**            a sub-frequency sine dropping in pitch as it
                        decays, with a short noise burst on the front, on
                        the downbeat itself
  **reverse_vocal_tail** the last half-second of vocal before the boundary,
                        reversed and faded in so it ends exactly on the
                        downbeat -- the oldest vocal transition there is
  **drum_fill**         delivered as a *reverse swell* of the beat's own
                        last beat. A real fill needs drum samples or a
                        drum stem to write into; neither is guaranteed
                        here, and the report says which was delivered
  **beat_dropout_1bar** the beat muted for the bar before the boundary, so
                        the vocal (and the reversed tail) carries the drop
                        alone

Levels are conservative on purpose. A transition that announces itself is a
transition that was noticed, and the point of all of these is that the
listener feels the section arrive rather than hears the device.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..audio import dsp

log = logging.getLogger("mixengine.arrange.transitions")

# Peak levels, dBFS, before the balance stage.
RISER_DB = -20.0
IMPACT_DB = -11.0
REVERSE_TAIL_DB = -8.0
REVERSE_SWELL_DB = -10.0

REVERSE_TAIL_S = 0.55
IMPACT_S = 0.45

# The dropout gets short fades so the cut is a decision rather than a click.
DROPOUT_FADE_S = 0.008


@dataclass
class TransitionReport:
    planned: int = 0
    built: List[Dict] = field(default_factory=list)
    skipped: List[Dict] = field(default_factory=list)
    beat_muted_s: float = 0.0

    def to_dict(self) -> dict:
        return {"planned": self.planned, "built": self.built,
                "skipped": self.skipped,
                "beat_muted_s": round(self.beat_muted_s, 3)}


def build(vocal: np.ndarray, beat: np.ndarray, sr: int,
          transitions: Sequence[Tuple[float, Sequence[str]]], *,
          bar_s: float, beat_s: float, seed: int = 0
          ) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Render every planned transition.

    Returns `(fx_bus, beat_gain, report)`: a stereo bus of generated audio
    at the vocal's length, a per-sample gain curve for the beat (1.0
    everywhere except inside a dropout), and what was built.
    """
    v = dsp.as_2d(vocal)
    b = dsp.as_2d(beat)
    n = len(v)
    fx = np.zeros((n, 2), dtype=np.float32)
    beat_gain = np.ones(n, dtype=np.float32)
    rep = TransitionReport(planned=sum(len(e) for _, e in transitions))
    rng = np.random.default_rng(seed)

    if bar_s <= 0 or beat_s <= 0 or n < sr:
        rep.skipped.append({"reason": "no usable bar length"})
        return fx, beat_gain, rep.to_dict()

    for t, effects in transitions:
        at = int(round(float(t) * sr))
        if at <= int(sr * 0.25) or at >= n - int(sr * 0.1):
            rep.skipped.append({"at": round(float(t), 3),
                                "reason": "boundary too close to an edge"})
            continue
        for name in effects:
            try:
                built = _one(name, v, b, sr, at, bar_s, beat_s, rng, fx, beat_gain)
            except Exception as exc:                        # pragma: no cover
                log.warning("transition %s at %.2fs failed: %s", name, t, exc)
                rep.skipped.append({"at": round(float(t), 3), "effect": name,
                                    "reason": str(exc)})
                continue
            if built is None:
                rep.skipped.append({"at": round(float(t), 3), "effect": name,
                                    "reason": "nothing to build from"})
                continue
            built["at"] = round(float(t), 3)
            rep.built.append(built)
            if name == "beat_dropout_1bar":
                rep.beat_muted_s += built.get("seconds", 0.0)

    if rep.built:
        log.info("  transitions: %s", ", ".join(
            f"{x['effect']}@{x['at']:.1f}s" for x in rep.built))
    return fx, beat_gain, rep.to_dict()


def _one(name: str, v: np.ndarray, b: np.ndarray, sr: int, at: int,
         bar_s: float, beat_s: float, rng, fx: np.ndarray,
         beat_gain: np.ndarray) -> Optional[Dict]:
    if name == "riser":
        return _riser(sr, at, bar_s, rng, fx)
    if name == "impact":
        return _impact(sr, at, rng, fx)
    if name == "reverse_vocal_tail":
        return _reverse_tail(v, sr, at, fx)
    if name == "drum_fill":
        return _reverse_swell(b, sr, at, beat_s, fx)
    if name == "beat_dropout_1bar":
        return _dropout(sr, at, bar_s, beat_gain)
    return None


# ─────────────────────────────────────────────────────────────────────────────

def _place(fx: np.ndarray, start: int, sig: np.ndarray, gain_db: float) -> None:
    """Sum a mono signal into the stereo bus at `start`, clipped to bounds."""
    n = fx.shape[0]
    s = max(0, start)
    e = min(n, start + len(sig))
    if e <= s:
        return
    seg = sig[s - start:e - start]
    peak = float(np.max(np.abs(seg))) or 1.0
    seg = seg / peak * dsp.db_to_lin(gain_db)
    fx[s:e, 0] += seg
    fx[s:e, 1] += seg


def _riser(sr: int, at: int, bar_s: float, rng, fx: np.ndarray) -> Dict:
    """Noise swelling into the downbeat with a sine sweeping underneath.

    The bandpass centre rises with the level, which is what makes it read
    as motion rather than as a fade: a static filter merely gets louder.
    Done in short crossfaded blocks rather than with a time-varying filter,
    because a biquad whose coefficients change per sample is not guaranteed
    stable and the block seams are inaudible under a rising noise floor.
    """
    length = int(round(min(bar_s, 3.0) * sr))
    length = max(length, int(sr * 0.5))
    t = np.arange(length) / float(sr)
    noise = rng.normal(0.0, 1.0, length).astype(np.float32)

    blocks = 12
    edges = np.linspace(0, length, blocks + 1).astype(int)
    out = np.zeros(length, dtype=np.float32)
    fade = max(8, int(sr * 0.01))
    for i in range(blocks):
        s, e = edges[i], edges[i + 1]
        if e - s < fade * 2:
            continue
        frac = (i + 0.5) / blocks
        centre = 300.0 * (20.0 ** frac)            # 300 Hz -> 6 kHz
        lo, hi = centre / 1.6, min(centre * 1.6, sr * 0.45)
        seg = dsp.to_mono(dsp.bandpass(noise[s:e][:, None], sr, lo, hi, order=2))
        w = np.ones(e - s, dtype=np.float32)
        ramp = np.linspace(0, 1, fade, dtype=np.float32)
        w[:fade] = ramp
        w[-fade:] = ramp[::-1]
        out[s:e] += seg * w

    # An octave of sine underneath, quiet, so the rise has a pitch to it.
    f0 = 110.0
    phase = 2 * np.pi * f0 * (t + (t ** 2) / (2 * t[-1]))   # 110 -> 220 Hz
    out += 0.25 * np.sin(phase).astype(np.float32)

    # Exponential swell: quiet for most of the bar, loud for the last part.
    env = (np.exp(3.2 * t / t[-1]) - 1.0) / (np.exp(3.2) - 1.0)
    out *= env.astype(np.float32)
    # Hard stop on the downbeat with a 4 ms fade -- the arrival is the cut.
    stop = max(4, int(sr * 0.004))
    out[-stop:] *= np.linspace(1, 0, stop, dtype=np.float32)

    _place(fx, at - length, out, RISER_DB)
    return {"effect": "riser", "seconds": round(length / sr, 3)}


def _impact(sr: int, at: int, rng, fx: np.ndarray) -> Dict:
    """A sub-frequency hit on the downbeat: pitch falls as it decays."""
    length = int(IMPACT_S * sr)
    t = np.arange(length) / float(sr)
    f_start, f_end = 62.0, 30.0
    # Exponential pitch glide, integrated to phase.
    k = np.log(f_end / f_start) / t[-1]
    phase = 2 * np.pi * f_start * (np.exp(k * t) - 1.0) / k
    body = np.sin(phase) * np.exp(-t / 0.16)

    click_n = int(sr * 0.018)
    click = rng.normal(0.0, 1.0, click_n).astype(np.float32)
    click = dsp.to_mono(dsp.lowpass(click[:, None], sr, 5000.0, order=2))
    click *= np.exp(-np.arange(click_n) / (sr * 0.004)).astype(np.float32)

    out = body.astype(np.float32)
    out[:click_n] += 0.5 * click
    out = dsp.to_mono(dsp.highpass(out[:, None], sr, 24.0, order=2))
    _place(fx, at, out, IMPACT_DB)
    return {"effect": "impact", "seconds": round(length / sr, 3)}


def _reverse_tail(v: np.ndarray, sr: int, at: int, fx: np.ndarray
                  ) -> Optional[Dict]:
    """The last voiced half-second before the boundary, reversed.

    Searches back from the boundary for the last stretch of voice rather
    than blindly taking the audio at the boundary, because the last phrase
    of a section usually ends a beat or two before the next one starts and
    what sits right at the boundary is silence.
    """
    mono = dsp.to_mono(v)
    win = int(sr * 0.05)
    limit = max(0, at - int(sr * 2.5))
    end = at
    # Walk back in 50 ms steps until we find voice.
    while end - win > limit:
        if dsp.rms_db(mono[end - win:end]) > -45.0:
            break
        end -= win
    else:
        return None
    length = int(REVERSE_TAIL_S * sr)
    start = max(0, end - length)
    seg = mono[start:end]
    if seg.size < int(sr * 0.15) or float(np.max(np.abs(seg))) < 1e-4:
        return None
    rev = seg[::-1].copy()
    rev *= np.linspace(0.0, 1.0, rev.size, dtype=np.float32) ** 1.5
    rev = dsp.to_mono(dsp.highpass(rev[:, None], sr, 200.0, order=2))
    _place(fx, at - rev.size, rev, REVERSE_TAIL_DB)
    return {"effect": "reverse_vocal_tail", "seconds": round(rev.size / sr, 3),
            "source_end_s": round(end / sr, 3)}


def _reverse_swell(b: np.ndarray, sr: int, at: int, beat_s: float,
                   fx: np.ndarray) -> Optional[Dict]:
    """The beat's own last beat before the boundary, reversed and swelled.

    Stands in for a drum fill. The report names it honestly: without drum
    samples or a drum stem there is nothing to write a fill *with*, and a
    reversed swell of the track's own material is the device a producer
    reaches for in exactly that situation.
    """
    mono = dsp.to_mono(b)
    length = int(round(beat_s * sr))
    start = at - length
    if start < 0 or at > mono.size:
        return None
    seg = mono[start:at]
    if float(np.max(np.abs(seg))) < 1e-4:
        return None
    rev = seg[::-1].copy()
    rev *= np.linspace(0.0, 1.0, rev.size, dtype=np.float32) ** 2
    rev = dsp.to_mono(dsp.highpass(rev[:, None], sr, 150.0, order=2))
    _place(fx, at - rev.size, rev, REVERSE_SWELL_DB)
    return {"effect": "drum_fill", "delivered_as": "reverse_swell",
            "seconds": round(rev.size / sr, 3),
            "note": "no drum samples or drum stem; used the beat's own last beat"}


def _dropout(sr: int, at: int, bar_s: float, beat_gain: np.ndarray) -> Dict:
    """Mute the beat for the bar before the boundary."""
    length = int(round(bar_s * sr))
    start = max(0, at - length)
    fade = max(4, int(DROPOUT_FADE_S * sr))
    beat_gain[start:at] = 0.0
    if start - fade >= 0:
        beat_gain[start - fade:start] = np.linspace(1, 0, fade, dtype=np.float32)
    if at + fade <= beat_gain.size:
        beat_gain[at:at + fade] = np.linspace(0, 1, fade, dtype=np.float32)
    return {"effect": "beat_dropout_1bar", "seconds": round((at - start) / sr, 3)}
