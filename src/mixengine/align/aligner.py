"""
The policy layer: deciding how much of a timing correction to actually apply.

`dtw.py` answers *where each onset belongs*. This module answers *how far to
move it*, which is a musical question rather than a numerical one, and then
hands the result to `warp.py`.

Four rules shape the answer:

**Below the audibility threshold, do nothing.** A deviation that no
listener can hear costs stretch artifacts to correct and buys exactly
nothing. The threshold scales with the subdivision, because what reads as
"late" depends on how long a slot is.

**Correct in proportion to how much the moment matters.** A downbeat
landing late is the most audible timing error in music; the fourth
sixteenth of a bar landing late is a performance. Correction strength is
scaled by metrical weight, so the grid is enforced where the ear is
checking it and the performance survives everywhere else.

**Never let a correction cross or crowd a neighbour.** Two onsets 90 ms
apart cannot both be pulled 60 ms in opposite directions without the
syllable between them being destroyed. The constraint is expressed as a
bound on how much each inter-onset gap may stretch, because that is
exactly what the warp enforces downstream -- a bound stated any other way
is a different bound, and the warp then discards the anchors that violate
its own.

**Fix drift globally, not onset by onset.** A take whose tempo differs from
the beat's needs one ratio applied to everything, not an ever-growing
series of nudges. The ratio is measured and removed *first*, before the
correspondence is computed, because an assignment made on drifted onsets
reproduces the drift faithfully.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ..core.types import FloatSeq

from ..audio import dsp
from ..core.types import Phrase
from ..musical import groove as groove_mod
from ..musical import salience
from . import dtw, warp as warp_mod

log = logging.getLogger("mixengine.align")

# Below this, nothing downstream is worth doing.
MIN_ONSETS = 6

# How near a phrase's start an onset must fall to count as its entry, and
# how much more the entry is worth than a syllable inside the line. The
# window matches `salience.timing_salience`, which uses the same idea for
# every other stage; the multiplier is smaller than its 1.6 because here
# it scales a correction that is already capped at the full error.
PHRASE_ENTRY_S = 0.12
PHRASE_ENTRY_PULL = 1.35

# A tempo difference smaller than this is performance, not drift. 0.0008 is
# 0.08%: about 140 ms over a three-minute song, which is roughly one
# sixteenth at 110 BPM -- the point at which the error stops being a feel
# and starts being an alignment mistake.
MIN_DRIFT_SLOPE = 0.0008

# ...and it must actually help. A ratio search always returns *some*
# minimum; requiring it to remove a tenth of the grid-fit error stops the
# engine stretching a take to chase search noise on material that simply
# does not sit on a grid.
MIN_DRIFT_GAIN = 0.10


@dataclass
class AlignReport:
    enabled: bool = True
    method: str = "dtw_grid"
    subdivision: int = 16
    strength: float = 0.0
    groove_applied: bool = False
    onsets: int = 0
    assigned: int = 0
    skipped_unassigned: int = 0
    below_threshold: int = 0
    moved: int = 0
    threshold_ms: float = 0.0
    error_before_ms: float = 0.0
    error_after_ms: float = 0.0
    max_move_ms: float = 0.0
    phrase_entries: int = 0
    drift_slope: float = 0.0
    drift_corrected: bool = False
    drift_removed_ms: float = 0.0
    drift_source: str = ""
    warp: Dict = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict:
        out = {}
        for k, v in self.__dict__.items():
            out[k] = round(v, 4) if isinstance(v, float) else v
        return out


def align_to_grid(y: np.ndarray, sr: int, onsets_s: FloatSeq,
                  context, *,
                  strength: float = 0.5,
                  subdivision: int = 16,
                  max_move_s: float = 0.08,
                  correct_drift: bool = True,
                  drift_ratio: Optional[float] = None,
                  phrases: Optional[Sequence[Phrase]] = None
                  ) -> Tuple[np.ndarray, dict]:
    """Variable-rate alignment of a vocal onto a beat's groove grid.

    `context` is a `timing.TimingContext`: it carries the beat's measured
    downbeats and its groove template, so the target is the track's own
    pocket rather than a mathematically exact grid.

    `drift_ratio`, when given, is the take's tempo relative to the beat's
    (vocal bar / beat bar; > 1 = slower), measured upstream from the
    take's own bar-ones. It replaces the onset-based tempo search, whose
    noise floor on dense material is wider than the drift it looks for.

    Returns `(audio, report)`. On any condition that makes alignment
    meaningless -- too few onsets, no grid, zero strength -- the audio is
    returned untouched and the report says which condition it was.
    """
    v = dsp.as_2d(y)
    rep = AlignReport(subdivision=subdivision, strength=float(strength),
                      onsets=len(onsets_s))

    o = np.asarray(sorted(float(t) for t in onsets_s), dtype=np.float64)
    grid = context.target_grid(subdivision, apply_groove=True)

    if strength <= 0.02:
        rep.enabled, rep.note = False, "strength below threshold"
        return v, rep.to_dict()
    if o.size < MIN_ONSETS:
        rep.enabled, rep.note = False, f"only {o.size} onsets detected"
        return v, rep.to_dict()
    if grid.size < 4:
        rep.enabled, rep.note = False, "beat has no usable grid"
        return v, rep.to_dict()

    rep.groove_applied = bool(context.groove.is_meaningful)

    slot_s = float(np.median(np.diff(grid)))
    threshold_ms = salience.perceptible_timing_threshold_ms(slot_s * 1000.0)
    rep.threshold_ms = threshold_ms

    weights = groove_mod.slot_weights(subdivision, context.beats_per_bar,
                                      context.groove)

    # ── 1. Drift ─────────────────────────────────────────────────────────
    # Measured and removed *before* the correspondence is computed, in that
    # order for a reason. Drift is a property of the whole take; if it is
    # left in, every later onset is closer to the wrong slot than the right
    # one, and the assignment faithfully reproduces the mistake. Measuring
    # it afterwards does not work either -- see `dtw.residual_drift`.
    rep.error_before_ms = float(dtw.grid_fit_error(o, grid) * 1000)
    t0 = float(o[0])
    o_work = o
    drift = np.zeros_like(o)
    if correct_drift:
        if drift_ratio is not None:
            # The take's bar grid has already measured its tempo; a search
            # over these onsets would only rediscover it, plus noise.
            ratio = float(drift_ratio)
            fit = dtw.grid_fit_error(t0 + (o - t0) / ratio, grid)
            gain = 1.0
            rep.drift_source = "vocal_grid"
        else:
            ratio, fit, gain = dtw.estimate_tempo_ratio(o, grid)
            rep.drift_source = "onset_search"
        rep.drift_slope = float(ratio - 1.0)
        if abs(ratio - 1.0) >= MIN_DRIFT_SLOPE and gain >= MIN_DRIFT_GAIN:
            o_work = t0 + (o - t0) / ratio
            drift = o_work - o
            rep.drift_corrected = True
            rep.drift_removed_ms = float(abs(drift[-1] - drift[0]) * 1000)
            log.info("  align: take runs %.2f%% %s the beat (%s); removing "
                     "%.0f ms of drift (grid fit %.0f -> %.0f ms)",
                     abs(ratio - 1.0) * 100,
                     "slower than" if ratio > 1 else "faster than",
                     rep.drift_source, rep.drift_removed_ms,
                     rep.error_before_ms, fit * 1000)

    # ── 2. Correspondence ────────────────────────────────────────────────
    assign = dtw.assign_onsets_to_grid(
        o_work, grid, max_move_s=max_move_s, slot_weights=weights,
        beats_per_bar=context.beats_per_bar)
    rep.assigned = assign.n_assigned
    rep.skipped_unassigned = int(o.size - assign.n_assigned)
    if assign.n_assigned < MIN_ONSETS:
        rep.enabled = False
        rep.note = (f"only {assign.n_assigned} onsets could be matched to the "
                    f"grid within {max_move_s * 1000:.0f} ms")
        return v, rep.to_dict()

    # Residual is measured from the de-drifted positions: what is left after
    # the tempo difference is accounted for is the performance's own timing.
    residual = assign.targets - o_work
    residual[~assign.assigned] = 0.0

    # ── 3. How much of each correction to apply ──────────────────────────
    downbeats = np.asarray(context.downbeats, dtype=np.float64)
    bar = float(context.bar_duration_s or 0.0)
    # `phrases` was a parameter this function accepted and never read, so
    # the entry weighting it exists for was never applied.
    entries = np.asarray([float(p.start) for p in (phrases or ())
                          if p is not None], dtype=np.float64)
    move = np.zeros_like(o)
    for i in range(o.size):
        if not assign.assigned[i]:
            continue
        err = residual[i]
        if abs(err) * 1000.0 < threshold_ms:
            rep.below_threshold += 1
            continue
        w = _slot_weight(assign.slot_index[i], weights)
        if downbeats.size >= 2 and bar > 0:
            w = max(w, salience.metrical_weight(float(assign.targets[i]),
                                                downbeats, bar,
                                                context.beats_per_bar))
        # A phrase entry is where the listener locks onto the pocket, so
        # it is worth more than a syllable in the middle of a line.
        if entries.size and np.min(np.abs(entries - assign.targets[i])) < PHRASE_ENTRY_S:
            w *= PHRASE_ENTRY_PULL
            rep.phrase_entries += 1
        # The weight decides *which* onsets are worth moving, never how
        # far past the grid to move them. Uncapped it did both: the
        # metrical weight reaches 2.0 on a downbeat, so at full strength a
        # syllable 30 ms late was moved 60 ms and arrived 30 ms early --
        # the error mirrored rather than corrected, on precisely the
        # positions a listener locks onto.
        move[i] = err * float(np.clip(strength * w, 0.0, 1.0))

    # ── 4. Neighbour safety ──────────────────────────────────────────────
    # Limited against the de-drifted spacing, because that is the spacing
    # the residual moves actually act on. Drift itself needs no limiting: a
    # uniform scaling is monotonic by construction and cannot make two
    # onsets cross, however large it is.
    move = _limit_by_neighbours(o_work, move)
    targets = o_work + move

    displacement = targets - o
    rep.moved = int(np.count_nonzero(np.abs(displacement) > 1e-4))
    rep.max_move_ms = float(np.max(np.abs(displacement)) * 1000) \
        if displacement.size else 0.0

    if rep.moved == 0:
        rep.enabled = False
        rep.note = "every deviation was already below the audibility threshold"
        return v, rep.to_dict()

    after = assign.targets - targets
    rep.error_after_ms = float(np.mean(np.abs(after[assign.assigned])) * 1000)

    # ── 5. Warp ──────────────────────────────────────────────────────────
    out, warp_report = warp_mod.warp(v, sr, o, targets, preserve_formants=True)
    rep.warp = warp_report
    if warp_report.get("method", "").startswith("skipped"):
        rep.enabled = False
        rep.note = "warp rejected the anchor set"
        return v, rep.to_dict()
    return out, rep.to_dict()


def _slot_weight(slot_index: int, weights: FloatSeq) -> float:
    if slot_index < 0 or len(weights) == 0:
        return 0.5
    return float(weights[slot_index % len(weights)])


def _limit_by_neighbours(onsets: np.ndarray, move: np.ndarray,
                         *, min_ratio: float = warp_mod.MIN_RATIO,
                         max_ratio: float = warp_mod.MAX_RATIO,
                         iterations: int = 8) -> np.ndarray:
    """Shrink moves until every inter-onset gap stretches by a legal ratio.

    The constraint the warp actually enforces is on *ratios between
    consecutive anchors*, so that is what gets enforced here. A margin
    expressed as a fraction of the gap is not the same thing and was the
    looser of the two: allowing each endpoint to move 35% of the gap lets a
    gap change by 70%, which is a local ratio of 1.7 against a limit of
    1.25. The warp then had to discard those anchors, and an onset whose
    anchor is discarded does not move at all.

    Violating pairs have both moves scaled back together, so neither onset
    is arbitrarily privileged, and the passes are repeated because fixing
    one pair can disturb its neighbour. That relaxation converges quickly
    but is not guaranteed to converge *completely* -- adjacent violations
    can trade the excess back and forth -- so a final left-to-right pass
    hard-enforces the bound. The forward pass alone would load all of the
    correction onto later onsets; the relaxation before it is what keeps
    the distribution even, and leaves the forward pass with almost nothing
    to do.
    """
    out = move.astype(np.float64).copy()
    n = onsets.size
    if n < 2:
        return out
    gaps = np.diff(onsets)

    for _ in range(iterations):
        worst = 0.0
        for i in range(n - 1):
            g = gaps[i]
            if g <= 1e-9:
                out[i + 1] = out[i]
                continue
            delta = out[i + 1] - out[i]
            ratio = 1.0 + delta / g
            if ratio > max_ratio:
                allowed = (max_ratio - 1.0) * g
            elif ratio < min_ratio:
                allowed = (min_ratio - 1.0) * g
            else:
                continue
            worst = max(worst, abs(ratio - 1.0))
            excess = delta - allowed
            out[i] += excess * 0.5
            out[i + 1] -= excess * 0.5
        if worst == 0.0:
            break

    for i in range(n - 1):
        g = gaps[i]
        if g <= 1e-9:
            out[i + 1] = out[i]
            continue
        lo = out[i] + (min_ratio - 1.0) * g
        hi = out[i] + (max_ratio - 1.0) * g
        out[i + 1] = float(np.clip(out[i + 1], lo, hi))
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Aligning to a reference recording
# ═════════════════════════════════════════════════════════════════════════════

def align_to_reference(y: np.ndarray, sr: int, reference: np.ndarray,
                       *, strength: float = 1.0,
                       max_move_s: float = 0.25) -> Tuple[np.ndarray, dict]:
    """Warp a take onto a reference recording of the same material.

    The case this exists for: the singer recorded over a beat whose playback
    was not sample-locked to the recording -- a phone speaker, a browser
    that resampled, a take started by hand. The offset between them is then
    not constant, and no single delay corrects it.

    Not used for matching a vocal to an *unrelated* beat: there is no
    correspondence to find there, and DTW will invent one.
    """
    rep: Dict = {"method": "dtw_reference", "enabled": True}
    v = dsp.as_2d(y)
    res = dtw.align_audio(v, dsp.as_2d(reference), sr)
    if res is None:
        rep.update(enabled=False, note="CQT features unavailable")
        return v, rep
    src, dst, cost = res
    rep["path_cost"] = round(float(cost), 4)

    # Thin the path to anchors: a full frame-by-frame map is both enormous
    # and far noisier than the underlying correspondence.
    step = max(1, int(round(0.25 / dtw.DTW_FRAME_S)))
    src_a, dst_a = src[::step], dst[::step]
    shift = dst_a - src_a
    shift = np.clip(shift, -max_move_s, max_move_s) * float(np.clip(strength, 0, 1))
    targets = src_a + shift
    rep["anchors"] = int(src_a.size)
    rep["mean_shift_ms"] = round(float(np.mean(np.abs(shift)) * 1000), 2)
    rep["max_shift_ms"] = round(float(np.max(np.abs(shift)) * 1000), 2) \
        if shift.size else 0.0

    out, warp_report = warp_mod.warp(v, sr, src_a, targets)
    rep["warp"] = warp_report
    return out, rep
