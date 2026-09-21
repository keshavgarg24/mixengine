"""
Genre from the audio, not from a filename.

The engine carries a mix profile per genre -- loudness target, vocal-to-
instrumental ratio, ducking depth, how much air, how hard to tune -- and
until now none of it engaged unless a producer had tagged the file. An
untagged beat fell back to the default profile, so every genre-specific
decision in the mixer was dead code for most inputs.

The classifier is a set of explicit templates rather than a learned model,
for three reasons that all point the same way. A template can be read and
argued with, which matters when the output is "this is drill, so duck the
beat 3 dB harder". It needs no training data, no download, and no torch.
And it *abstains*: the profile it selects is only used when the winner
clearly beats the runner-up, because the cost of confidently choosing the
wrong profile -- mastering an R&B ballad to a trap target -- is much higher
than the cost of the neutral default.

Accuracy is not competitive with a trained model and is not meant to be.
The question here is narrow: which of nine mix profiles fits, with a real
option to answer "I don't know".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.types import FloatSeq

log = logging.getLogger("mixengine.genre")

# How far ahead of the runner-up the winner must be before its profile is
# used. Below this the two genres disagree about the mix in ways the
# evidence does not settle, and the default profile is the honest answer.
MIN_MARGIN = 0.10
MIN_SCORE = 0.42

# ...unless the runner-up is a near neighbour. Trap and drill routinely
# land within a few hundredths of each other because they *are* close --
# same tempo, same 808s, different hi-hat pattern -- and their mix profiles
# differ by half a dB of loudness and half a dB of vocal ratio. Abstaining
# to the neutral default in that case is the worse answer by a wide margin:
# it gives up 1.5 dB of loudness target and 1 dB of vocal ratio to avoid an
# error that costs a tenth of that. So a tie inside a family resolves to
# the winner, and only a tie *across* families abstains.
FAMILIES: Dict[str, str] = {
    "trap": "rap_808", "drill": "rap_808", "hip_hop": "rap_808",
    "pop": "song", "rnb": "song", "afrobeats": "song",
    "drum_and_bass": "electronic",
    "lofi": "mellow",
}


@dataclass
class GenreTemplate:
    """What one genre looks like in the features the engine already has.

    Ranges are `(low, high)` and score 1.0 inside, falling off linearly
    outside over the width given by `tol`. `weight` is how much the feature
    counts: tempo is the strongest single signal in this feature set, sub
    energy separates the 808 genres from everything else, and the rest
    refine.
    """
    name: str
    bpm: Tuple[float, float] = (0.0, 999.0)
    bpm_halftime_ok: bool = False
    sub: Tuple[float, float] = (0.0, 1.0)
    air: Tuple[float, float] = (0.0, 1.0)
    centroid: Tuple[float, float] = (0.0, 22050.0)
    flatness: Tuple[float, float] = (0.0, 1.0)
    dynamic_range: Tuple[float, float] = (0.0, 40.0)
    hat_density: Tuple[float, float] = (0.0, 40.0)
    swing: Tuple[float, float] = (0.0, 1.0)
    onset_rate: Tuple[float, float] = (0.0, 40.0)
    prior: float = 1.0


# Tempo ranges are the "felt" tempo where a genre has one. Trap is written
# at 130-150 but felt at half that, and drill's 138-145 overlaps it exactly
# -- which is why the two are separated by their hi-hat patterns and sub
# weight rather than by tempo.
TEMPLATES: List[GenreTemplate] = [
    GenreTemplate("trap", bpm=(128, 155), bpm_halftime_ok=True,
                  sub=(0.12, 0.45), air=(0.15, 0.70), centroid=(1800, 6000),
                  flatness=(0.0, 0.06), dynamic_range=(6, 16),
                  hat_density=(4.0, 22.0), swing=(0.44, 0.56),
                  onset_rate=(2.0, 9.0), prior=1.05),
    GenreTemplate("drill", bpm=(134, 150), bpm_halftime_ok=True,
                  sub=(0.15, 0.50), air=(0.10, 0.55), centroid=(1500, 5000),
                  flatness=(0.0, 0.05), dynamic_range=(6, 15),
                  hat_density=(3.0, 18.0), swing=(0.40, 0.50),
                  onset_rate=(2.0, 8.0), prior=0.95),
    GenreTemplate("hip_hop", bpm=(82, 102), sub=(0.08, 0.32),
                  air=(0.05, 0.45), centroid=(1700, 4200),
                  flatness=(0.0, 0.06), dynamic_range=(8, 20),
                  hat_density=(1.0, 8.0), swing=(0.50, 0.66),
                  onset_rate=(1.5, 6.0), prior=1.0),
    GenreTemplate("rnb", bpm=(58, 100), sub=(0.05, 0.28),
                  air=(0.05, 0.40), centroid=(900, 3600),
                  flatness=(0.0, 0.06), dynamic_range=(10, 24),
                  hat_density=(0.5, 7.0), swing=(0.48, 0.62),
                  onset_rate=(1.0, 5.0), prior=1.0),
    GenreTemplate("pop", bpm=(96, 132), sub=(0.04, 0.24),
                  air=(0.10, 0.55), centroid=(1600, 5200),
                  flatness=(0.0, 0.08), dynamic_range=(6, 16),
                  hat_density=(1.0, 9.0), swing=(0.47, 0.56),
                  onset_rate=(1.5, 7.0), prior=1.0),
    GenreTemplate("afrobeats", bpm=(98, 118), sub=(0.05, 0.28),
                  air=(0.08, 0.45), centroid=(1400, 4600),
                  flatness=(0.0, 0.08), dynamic_range=(8, 18),
                  hat_density=(1.5, 10.0), swing=(0.52, 0.66),
                  onset_rate=(2.5, 9.0), prior=0.9),
    GenreTemplate("drum_and_bass", bpm=(160, 180), sub=(0.10, 0.40),
                  air=(0.10, 0.55), centroid=(1800, 6000),
                  flatness=(0.0, 0.10), dynamic_range=(5, 14),
                  hat_density=(4.0, 20.0), swing=(0.46, 0.56),
                  onset_rate=(5.0, 16.0), prior=0.85),
    GenreTemplate("lofi", bpm=(66, 94), sub=(0.03, 0.22),
                  air=(0.0, 0.20), centroid=(600, 2400),
                  flatness=(0.02, 0.40), dynamic_range=(10, 26),
                  hat_density=(0.5, 6.0), swing=(0.52, 0.68),
                  onset_rate=(1.0, 5.0), prior=0.9),
]


@dataclass
class GenreResult:
    genre: Optional[str] = None
    confidence: float = 0.0
    margin: float = 0.0
    scores: Dict[str, float] = field(default_factory=dict)
    features: Dict[str, float] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)
    source: str = "detected"
    runner_up: Optional[str] = None
    note: str = ""

    @property
    def family(self) -> Optional[str]:
        return FAMILIES.get(self.genre or "")

    @property
    def usable(self) -> bool:
        """Whether the caller should apply this genre's mix profile."""
        if not self.genre or self.confidence < MIN_SCORE:
            return False
        if self.margin >= MIN_MARGIN:
            return True
        return bool(self.runner_up
                    and FAMILIES.get(self.runner_up) == self.family)

    def to_dict(self) -> dict:
        return {
            "genre": self.genre, "confidence": round(self.confidence, 3),
            "margin": round(self.margin, 3), "usable": self.usable,
            "family": self.family, "runner_up": self.runner_up,
            "source": self.source,
            "scores": {k: round(v, 3) for k, v in
                       sorted(self.scores.items(), key=lambda kv: -kv[1])[:4]},
            "features": {k: round(v, 4) for k, v in self.features.items()},
            "evidence": self.evidence, "note": self.note,
        }


def detect(y: Optional[np.ndarray], sr: int, *,
           bpm: float = 0.0,
           spectral: Optional[dict] = None,
           dynamic_range_db: float = 0.0,
           swing_ratio: float = 0.5,
           onsets: Optional[FloatSeq] = None,
           beats: Optional[FloatSeq] = None,
           tagged: Optional[str] = None) -> GenreResult:
    """Classify a beat. Pass whatever the DNA already measured.

    A producer's own tag wins outright when present -- they know what they
    made, and second-guessing them on weaker evidence than they have would
    be the wrong trade.
    """
    res = GenreResult()
    prefix = ""
    if tagged:
        key = str(tagged).strip().lower().replace(" ", "_").replace("-", "_")
        known = {t.name for t in TEMPLATES}
        if key in known:
            res.genre, res.confidence, res.margin = key, 1.0, 1.0
            res.source = "tagged"
            res.evidence.append(f"tagged as {key} by the producer")
            return res
        prefix = f"tag '{tagged}' is not a known profile; classifying instead"

    feats = _features(y, sr, bpm=bpm, spectral=spectral,
                      dynamic_range_db=dynamic_range_db,
                      swing_ratio=swing_ratio, onsets=onsets, beats=beats)
    res.features = feats
    if feats.get("bpm", 0.0) <= 0:
        res.note = _join(prefix, "no tempo available; cannot classify")
        return res

    for t in TEMPLATES:
        res.scores[t.name] = _score(t, feats)

    ranked = sorted(res.scores.items(), key=lambda kv: -kv[1])
    res.genre = ranked[0][0]
    res.confidence = float(ranked[0][1])
    res.runner_up = ranked[1][0] if len(ranked) > 1 else None
    res.margin = float(ranked[0][1] - ranked[1][1]) if len(ranked) > 1 else 1.0
    res.evidence = _evidence(
        next(t for t in TEMPLATES if t.name == res.genre), feats)
    if not res.usable:
        res.note = _join(prefix,
                         f"{res.genre} scores {res.confidence:.2f} but only "
                         f"{res.margin:.2f} ahead of {res.runner_up}, which is "
                         f"a different kind of record; using the default mix "
                         f"profile rather than guessing")
    elif res.margin < MIN_MARGIN:
        res.note = _join(prefix,
                         f"{res.genre} and {res.runner_up} are close "
                         f"({res.margin:.2f} apart) but both are {res.family}, "
                         f"so the profiles barely differ")
    else:
        res.note = prefix
    return res


def _join(*parts: str) -> str:
    return "; ".join(p for p in parts if p)


# ─────────────────────────────────────────────────────────────────────────────

def _features(y, sr, *, bpm, spectral, dynamic_range_db, swing_ratio,
              onsets, beats) -> Dict[str, float]:
    sp = spectral or {}
    bands = sp.get("band_energy") or {}
    out = {
        "bpm": float(bpm or 0.0),
        "sub": float(bands.get("sub", 0.0)),
        "air": float(bands.get("air", 0.0)),
        "centroid": float(sp.get("centroid_hz", 0.0)),
        "flatness": float(sp.get("flatness", 0.0)),
        "dynamic_range": float(dynamic_range_db or 0.0),
        "swing": float(swing_ratio if swing_ratio else 0.5),
        "onset_rate": 0.0,
        "hat_density": 0.0,
    }
    b = np.asarray(beats if beats is not None else [], dtype=np.float64)
    o = np.asarray(onsets if onsets is not None else [], dtype=np.float64)
    if o.size >= 4 and b.size >= 2:
        span = float(b[-1] - b[0]) or 1.0
        beat_s = float(np.median(np.diff(b))) or 0.5
        out["onset_rate"] = float(o.size / span * beat_s)

    if y is not None:
        out["hat_density"] = _hat_density(y, sr, b)
    return out


def _hat_density(y: np.ndarray, sr: int, beats: np.ndarray) -> float:
    """Hi-hat events per beat, measured in the band hats actually occupy.

    This is the feature that separates trap from every other genre at the
    same tempo: the rolls. Counting onsets across the full spectrum does not
    see them, because the kick and the 808 dominate the detection function
    and the hats sit inside their decay. Band-limiting to 7-14 kHz leaves
    almost nothing but hats and the top of the snare.
    """
    try:
        import librosa
        from ..audio import dsp
        mono = dsp.to_mono(dsp.as_2d(y)).astype(np.float32)
        if mono.size < sr:
            return 0.0
        hi = dsp.to_mono(dsp.bandpass(mono[:, None], sr, 7000.0,
                                      min(14000.0, sr * 0.45), order=2))
        env = librosa.onset.onset_strength(y=np.ascontiguousarray(hi), sr=sr,
                                           hop_length=256)
        peaks = librosa.util.peak_pick(
            env, pre_max=3, post_max=3, pre_avg=5, post_avg=5,
            delta=float(np.percentile(env, 70)) * 0.4, wait=2)
        dur = mono.size / float(sr)
        if beats.size >= 2:
            beat_s = float(np.median(np.diff(beats))) or 0.5
        else:
            beat_s = 0.5
        return float(len(peaks) / max(dur, 1e-6) * beat_s)
    except Exception as exc:                                # pragma: no cover
        log.debug("hat density failed: %s", exc)
        return 0.0


def _range_score(value: float, lo: float, hi: float, tol: float) -> float:
    if lo <= value <= hi:
        return 1.0
    d = (lo - value) if value < lo else (value - hi)
    return float(max(0.0, 1.0 - d / max(tol, 1e-9)))


def _bpm_score(t: GenreTemplate, bpm: float) -> float:
    """Tempo, allowing for the halftime ambiguity where a genre has one.

    Trap tracked at 140 and trap tracked at 70 are the same music. A
    tracker that picks the half or the double is not wrong about the genre,
    so both are scored and the better one is taken.
    """
    lo, hi = t.bpm
    tol = (hi - lo) * 0.5 + 6.0
    best = _range_score(bpm, lo, hi, tol)
    if t.bpm_halftime_ok:
        best = max(best,
                   _range_score(bpm * 2.0, lo, hi, tol),
                   _range_score(bpm * 0.5, lo, hi, tol))
    return best


def _score(t: GenreTemplate, f: Dict[str, float]) -> float:
    parts: List[Tuple[float, float]] = [
        (_bpm_score(t, f["bpm"]), 3.0),
        (_range_score(f["sub"], *t.sub, tol=0.18), 1.6),
        (_range_score(f["centroid"], *t.centroid, tol=1800.0), 1.2),
        (_range_score(f["air"], *t.air, tol=0.25), 1.0),
        # Flatness carries more weight than its dynamic range suggests:
        # it is the only feature that separates lo-fi from the boom bap it
        # is otherwise identical to. Tape hiss, vinyl and saturation raise
        # it; a clean record does not.
        (_range_score(f["flatness"], *t.flatness, tol=0.05), 1.3),
        (_range_score(f["dynamic_range"], *t.dynamic_range, tol=8.0), 0.8),
        (_range_score(f["swing"], *t.swing, tol=0.09), 0.9),
    ]
    # The two rhythm features are only informative when they were actually
    # measured. Scoring a zero as "outside the range" would systematically
    # punish every genre that expects hats whenever the audio was absent.
    if f.get("hat_density", 0.0) > 0:
        parts.append((_range_score(f["hat_density"], *t.hat_density, tol=8.0), 1.4))
    if f.get("onset_rate", 0.0) > 0:
        parts.append((_range_score(f["onset_rate"], *t.onset_rate, tol=4.0), 1.0))

    total_w = sum(w for _, w in parts)
    raw = sum(v * w for v, w in parts) / total_w
    return float(np.clip(raw * t.prior, 0.0, 1.0))


def _evidence(t: GenreTemplate, f: Dict[str, float]) -> List[str]:
    """Plain-language reasons, so a wrong answer can be argued with."""
    out: List[str] = []
    bpm = f["bpm"]
    if _bpm_score(t, bpm) > 0.8:
        if t.bpm_halftime_ok and not (t.bpm[0] <= bpm <= t.bpm[1]):
            out.append(f"{bpm:.0f} BPM reads as {t.bpm[0]:.0f}-{t.bpm[1]:.0f} "
                       f"half-time")
        else:
            out.append(f"{bpm:.0f} BPM sits in the {t.name} range")
    if f["sub"] > 0.14:
        out.append(f"{f['sub'] * 100:.0f}% of the energy is below 120 Hz - "
                   f"an 808-led low end")
    if f.get("hat_density", 0) > 6:
        out.append(f"{f['hat_density']:.1f} hi-hat events per beat - rolls")
    if f["flatness"] > 0.05:
        out.append("noisy spectrum - tape, vinyl or heavy saturation")
    if f["swing"] > 0.56:
        out.append(f"swung {f['swing']:.2f} rather than straight")
    if f["centroid"] and f["centroid"] < 1800:
        out.append("dark spectrum - filtered or low-passed")
    return out
