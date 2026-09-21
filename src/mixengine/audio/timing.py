"""
Musical timing correction.

Quantisation here targets the beat's own groove, not a mathematically
exact grid, and it corrects in proportion to how much each moment matters.

Why not a plain grid
--------------------
The obvious implementation snaps every syllable to the nearest subdivision.
That produces timing that is arithmetically perfect and rhythmically dead,
because no record was ever made against a perfect grid. A track's feel
lives in *systematic* offsets -- a snare consistently behind the beat, a hat
consistently ahead -- and those offsets are measurable and reproducible.

The opposite mistake is just as common and worse: adding random jitter to
"humanise" the result. The evidence is against it. Scaling expert
performers' microtiming showed groove ratings high at or below the
magnitude originally performed and *falling* when deviations were
exaggerated, with fully quantised versions rating as highly as the human
originals. Across commercial tracks, groove correlated with beat salience
and event density and with neither microtiming measure.

So the engine measures the beat's groove template and uses it as the
target. Landing exactly on those positions *is* landing in the pocket, and
no separate humanisation pass is needed or wanted.

Why correction strength varies
------------------------------
A syllable 15 ms late on a downbeat is audible. The same 15 ms on an
off-beat sixteenth in a fast bar is not, and correcting it spends stretch
artifacts on something nobody can hear while flattening the flow. Every
move is therefore weighted by metrical position, phrase position and
lyrical stress, and anything below the perceptual threshold is left alone.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from ..core.types import FloatSeq

from . import dsp
from ..core.types import GrooveTemplate, Phrase, Section
from ..musical import groove as groove_mod
from ..musical import salience

log = logging.getLogger("mixengine.timing")


@dataclass
class TimingContext:
    """The grid a vocal is being placed against."""
    beats: np.ndarray                       # beat times, seconds
    downbeats: np.ndarray
    bar_duration_s: float = 0.0
    beats_per_bar: int = 4
    groove: GrooveTemplate = field(default_factory=GrooveTemplate)
    sections: Sequence[Section] = ()
    phrases: Sequence[Phrase] = ()
    grid_stability: float = 1.0

    def __post_init__(self):
        self.beats = np.asarray(self.beats, dtype=np.float64)
        self.downbeats = np.asarray(self.downbeats, dtype=np.float64)
        if self.bar_duration_s <= 0 and self.downbeats.size >= 2:
            self.bar_duration_s = float(np.median(np.diff(self.downbeats)))

    @staticmethod
    def from_beat_dna(bdna: dict) -> "TimingContext":
        beats = np.asarray(bdna.get("beats") or [], dtype=np.float64)
        downbeats = np.asarray(bdna.get("downbeats") or [], dtype=np.float64)
        bpm = float(bdna.get("bpm") or 0.0)
        bpb = int(bdna.get("beats_per_bar") or 4)
        bar = (60.0 / bpm) * bpb if bpm > 0 else 0.0
        return TimingContext(
            beats=beats, downbeats=downbeats, bar_duration_s=bar,
            beats_per_bar=bpb,
            groove=GrooveTemplate.from_dict(bdna.get("groove")),
            sections=[Section.from_dict(s) for s in (bdna.get("sections") or [])],
            grid_stability=float(bdna.get("grid_stability") or 0.0))

    def target_grid(self, subdivision: int = 16,
                    apply_groove: bool = True) -> np.ndarray:
        """Grid positions to align onto, carrying the track's feel."""
        if self.downbeats.size >= 2 and self.bar_duration_s > 0:
            return groove_mod.grid_with_groove(
                self.downbeats, self.bar_duration_s, self.groove,
                subdivision=subdivision,
                amount=1.0 if apply_groove else 0.0)
        # No bar information: subdivide the beats evenly instead.
        if self.beats.size < 2:
            return np.zeros(0)
        step = float(np.median(np.diff(self.beats)))
        per_beat = max(1, subdivision // max(self.beats_per_bar, 1))
        out: List[float] = []
        for i in range(self.beats.size - 1):
            for k in range(per_beat):
                out.append(float(self.beats[i]) + step * k / per_beat)
        out.append(float(self.beats[-1]))
        return np.asarray(out, dtype=np.float64)

    def section_at(self, t: float) -> Optional[Section]:
        for s in self.sections:
            if s.start <= t < s.end:
                return s
        return None

    def phrase_at(self, t: float) -> Optional[Phrase]:
        for p in self.phrases:
            if p.start <= t < p.end:
                return p
        return None


@dataclass
class TimingReport:
    enabled: bool = True
    method: str = "groove_aware"
    subdivision: int = 16
    groove_applied: bool = False
    swing_ratio: float = 0.5
    strength: float = 0.0
    onsets_considered: int = 0
    onsets_moved: int = 0
    onsets_below_threshold: int = 0
    mean_move_ms: float = 0.0
    max_move_ms: float = 0.0
    residual_error_ms: float = 0.0

    def to_dict(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


def quantize_musical(y: np.ndarray, sr: int, onsets_s: FloatSeq,
                     context: TimingContext, *,
                     strength: float = 0.4,
                     subdivision: int = 16,
                     max_move_s: float = 0.06,
                     word_stress: Optional[FloatSeq] = None,
                     time_stretch_fn: Optional[Callable] = None
                     ) -> Tuple[np.ndarray, dict]:
    """Nudge syllable onsets toward the beat's groove. Returns `(audio, report)`.

    Correction is applied by elastically stretching the gap *before* each
    onset rather than by slicing, so nothing clicks and the audio stays
    continuous.
    """
    v = dsp.as_2d(y)
    rep = TimingReport(enabled=strength > 0.0, subdivision=subdivision,
                       strength=float(strength),
                       onsets_considered=len(onsets_s))

    grid = context.target_grid(subdivision, apply_groove=True)
    if strength <= 0 or grid.size < 2 or len(onsets_s) < 2:
        rep.enabled = False
        return v, rep.to_dict()

    rep.groove_applied = bool(context.groove.is_meaningful)
    rep.swing_ratio = float(context.groove.swing_ratio)

    if time_stretch_fn is None:
        from .transform import time_stretch as _ts
        time_stretch_fn = _ts

    # Below this, a deviation is not audible in context and correcting it
    # buys nothing while costing stretch artifacts.
    slot_ms = (context.bar_duration_s / max(subdivision, 1)) * 1000.0 \
        if context.bar_duration_s > 0 else 125.0
    threshold_ms = salience.perceptible_timing_threshold_ms(slot_ms)

    weights = groove_mod.slot_weights(subdivision, context.beats_per_bar,
                                      context.groove)
    stress = list(word_stress or [])

    segments: List[np.ndarray] = []
    cursor = 0
    moves: List[float] = []
    residuals: List[float] = []
    res_weights: List[float] = []

    for i, o in enumerate(onsets_s):
        idx = int(float(o) * sr)
        if idx <= cursor or idx >= len(v):
            continue

        nearest = int(np.argmin(np.abs(grid - float(o))))
        target = float(grid[nearest])
        raw_delta = target - float(o)
        residuals.append(abs(raw_delta) * 1000.0)

        sal = salience.timing_salience(
            float(o), downbeats=context.downbeats,
            bar_duration_s=context.bar_duration_s, subdivision=subdivision,
            beats_per_bar=context.beats_per_bar,
            phrase=context.phrase_at(float(o)),
            section=context.section_at(float(o)),
            word_stress=(stress[i] if i < len(stress) else 0.5))
        res_weights.append(sal)

        # Grid positions do not attract equally: a downbeat pulls firmly, an
        # off-beat 32nd barely at all. Without this the vocal gets flattened
        # onto every available subdivision.
        slot_w = float(weights[nearest % weights.size]) if weights.size else 1.0

        # The audibility gate asks whether the *error* is audible, not
        # whether the correction is large. Testing the scaled correction
        # instead inverts the logic: a clearly audible 36 ms error scaled by
        # strength and slot weight becomes a 3 ms move, falls under the
        # threshold, and is skipped -- so the errors that most need fixing
        # are precisely the ones that get ignored.
        if abs(raw_delta) * 1000.0 < threshold_ms:
            rep.onsets_below_threshold += 1
            continue
        if abs(raw_delta) > max_move_s:
            # Too far from the grid to be a timing error against *this*
            # subdivision. Moving it would land it somewhere musically
            # arbitrary.
            continue

        applied = raw_delta * float(np.clip(strength * sal * slot_w, 0.0, 0.95))
        if abs(applied) < 1e-4:
            continue

        gap = v[cursor:idx]
        if len(gap) < 64:
            continue
        new_len = int(len(gap) + applied * sr)
        if new_len < 32:
            continue
        ratio = new_len / len(gap)
        if not (0.5 < ratio < 2.0):
            continue
        try:
            segments.append(time_stretch_fn(gap, sr, ratio,
                                            preserve_formants=False))
        except Exception:
            continue
        cursor = idx
        moves.append(applied * 1000.0)

    if cursor < len(v):
        segments.append(v[cursor:])
    if not segments:
        return v, rep.to_dict()

    rep.onsets_moved = len(moves)
    if moves:
        rep.mean_move_ms = float(np.mean(np.abs(moves)))
        rep.max_move_ms = float(np.max(np.abs(moves)))
    if residuals:
        rep.residual_error_ms = salience.weighted_error(residuals, res_weights)

    if moves:
        log.info("  timing: moved %d/%d onsets (mean %.0f ms, groove=%s, "
                 "swing %.2f), %d left below the %.0f ms audibility threshold",
                 rep.onsets_moved, rep.onsets_considered, rep.mean_move_ms,
                 rep.groove_applied, rep.swing_ratio,
                 rep.onsets_below_threshold, threshold_ms)
    return np.vstack(segments).astype(np.float32), rep.to_dict()


def extract_beat_groove(onsets_s: FloatSeq, beats: FloatSeq,
                        downbeats: FloatSeq, bpm: float,
                        beats_per_bar: int = 4,
                        strengths: Optional[FloatSeq] = None
                        ) -> GrooveTemplate:
    """Measure a beat's groove template from its own onsets.

    Called once per beat during catalog analysis and cached with the DNA,
    because it is the target every vocal placed on that beat will be
    aligned to.
    """
    if bpm <= 0:
        return GrooveTemplate()
    bar = (60.0 / bpm) * max(beats_per_bar, 1)
    return groove_mod.extract(onsets_s, downbeats, bar, subdivision=None,
                              strengths=strengths, beats=beats,
                              beats_per_bar=beats_per_bar, source="measured")
