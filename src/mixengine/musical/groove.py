"""
Groove: measuring feel, and transferring it.

The premise
-----------
The obvious way to make an automatically placed vocal sound human is to
add timing randomness. That intuition is wrong, and the evidence against
it is unusually clear.

When researchers took expert bass-and-drums performances and scaled their
microtiming deviations up and down, perceived groove was high at or below
the magnitude originally performed and *fell* when deviations were
exaggerated -- and fully quantised versions of the same performances rated
just as high as the human originals. Across a hundred commercial tracks in
five genres, groove ratings correlated with beat salience and event
density but with neither systematic nor unsystematic microtiming. In jazz,
quantised versions have been found to elicit the strongest groove, because
they make the pulse more predictable.

So randomness does not buy feel. It just makes timing worse.

What does carry feel is *systematic, directional* offset. Producers do not
describe scattering notes; they describe a snare that sits consistently
behind the beat, a hat pushed consistently ahead of it, a swing ratio
applied to every second eighth. Those are reproducible properties of a
track, they repeat bar after bar, and they can be measured.

That is what this module does. It extracts a track's own timing signature
into a `GrooveTemplate`, and it builds grids that carry that signature so
alignment can place a vocal *in the pocket* rather than on a
mathematically exact grid that no record was ever made against.

The rule the engine follows everywhere:

    Never quantise to a dead grid. Never inject random jitter.
    Measure the beat's groove, and align the vocal to that.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.types import FloatSeq

from ..core.types import GrooveTemplate

log = logging.getLogger("mixengine.groove")

# An onset further than this fraction of a slot from its nearest slot is not
# a late or early hit -- it belongs to a different slot, or it is not on the
# grid at all. Including it would corrupt the average.
_SLOT_CAPTURE = 0.40

# Deviations beyond this are not feel, they are tracking errors.
_MAX_PLAUSIBLE_OFFSET_MS = 90.0

# A slot needs this many observations before its average means anything.
_MIN_OBSERVATIONS = 3


# ═════════════════════════════════════════════════════════════════════════════
# Extraction
# ═════════════════════════════════════════════════════════════════════════════

def beats_from_downbeats(downbeats: FloatSeq,
                         beats_per_bar: int = 4) -> np.ndarray:
    """Interpolate beat positions from bar lines.

    Used when only downbeats are available. Assumes beats divide the bar
    evenly, which holds for the programmed material this is mostly applied
    to; genuinely rubato material will not have reliable downbeats either,
    so nothing is lost by the assumption.
    """
    dbs = np.asarray(downbeats, dtype=np.float64)
    if dbs.size < 2 or beats_per_bar < 1:
        return dbs
    out: List[float] = []
    for i in range(dbs.size - 1):
        span = float(dbs[i + 1] - dbs[i])
        for k in range(beats_per_bar):
            out.append(float(dbs[i]) + span * k / beats_per_bar)
    out.append(float(dbs[-1]))
    return np.asarray(out, dtype=np.float64)


def choose_subdivision(swing_ratio: float, is_compound: bool = False) -> int:
    """Pick the grid resolution that matches the track's feel.

    A swung track must not be measured against a straight sixteenth grid.
    At a triplet feel the off-beat eighth lands between sixteenth slots, so
    every swung onset is either rejected as implausible or mis-assigned to
    its neighbour -- which corrupts that neighbour's measurement as well as
    losing the swung one. Switching to a triplet grid puts the slots where
    the music actually is.
    """
    if is_compound:
        return 12
    return 12 if abs(float(swing_ratio) - 0.5) > 0.055 else 16


def extract(onsets: FloatSeq,
            downbeats: FloatSeq,
            bar_duration_s: float,
            subdivision: Optional[int] = None,
            strengths: Optional[FloatSeq] = None,
            beats: Optional[FloatSeq] = None,
            beats_per_bar: int = 4,
            source: str = "measured") -> GrooveTemplate:
    """Measure a track's microtiming signature.

    For every onset, find the subdivision slot it belongs to and record how
    far it sits from that slot's mathematically exact position. Averaging
    those deviations per slot over the whole track isolates the systematic
    component -- the part that repeats and therefore carries feel -- while
    the unsystematic part averages toward zero, which is exactly what we
    want since the unsystematic part is not what listeners respond to.

    Swing is measured first and separately, from raw onset phase rather
    than from the slot grid, because the grid resolution has to be chosen
    to match the feel before any slot assignment can be trusted. Passing
    `subdivision=None` (the default) lets that choice happen automatically.

    `consistency` reports how tightly the per-slot deviations cluster. A
    low value means the pattern does not repeat, and callers must not
    apply the template: imposing a non-repeating pattern is the random
    jitter this module exists to avoid.
    """
    ons = np.asarray(onsets, dtype=np.float64)
    dbs = np.asarray(downbeats, dtype=np.float64)

    if ons.size < 8 or dbs.size < 2 or bar_duration_s <= 0:
        return GrooveTemplate(subdivision=int(subdivision or 16), source="none")

    beat_times = (np.asarray(beats, dtype=np.float64) if beats is not None
                  else beats_from_downbeats(dbs, beats_per_bar))
    swing = detect_swing(ons, beat_times)
    n_slots = max(2, int(subdivision) if subdivision else choose_subdivision(swing))

    if ons.size < n_slots:
        return GrooveTemplate(subdivision=n_slots, swing_ratio=swing, source="none")

    w = (np.ones(ons.size) if strengths is None
         else np.asarray(strengths, dtype=np.float64))
    if w.size != ons.size:
        w = np.ones(ons.size)

    slot_dev: List[List[float]] = [[] for _ in range(n_slots)]
    slot_w: List[List[float]] = [[] for _ in range(n_slots)]

    for bar_i in range(dbs.size - 1):
        start, nxt = float(dbs[bar_i]), float(dbs[bar_i + 1])
        span = nxt - start
        # Guard against dropped or spurious downbeats: a bar that is not
        # roughly the nominal length would smear every slot in it.
        if not (0.5 * bar_duration_s < span < 2.0 * bar_duration_s):
            continue
        slot_dur = span / n_slots
        capture = slot_dur * _SLOT_CAPTURE

        # Indices, not values. Looking the weight up by float timestamp in a
        # dict silently dropped onsets that shared a timestamp, and returned
        # the *first* onset's weight for any lookup that missed -- a wrong
        # answer dressed as a valid one. Carrying the index through keeps
        # every onset paired with its own strength.
        in_bar = np.flatnonzero((ons >= start - capture) & (ons < nxt - capture))
        if in_bar.size == 0:
            continue

        for i in in_bar:
            t = float(ons[i])
            rel = (t - start) / slot_dur
            slot = int(round(rel)) % n_slots
            exact = start + slot * slot_dur
            dev_ms = (t - exact) * 1000.0
            if abs(t - exact) > capture:
                continue
            if abs(dev_ms) > _MAX_PLAUSIBLE_OFFSET_MS:
                continue
            slot_dev[slot].append(dev_ms)
            slot_w[slot].append(float(w[i]))

    offsets = np.zeros(n_slots)
    velocities = np.zeros(n_slots)
    spreads: List[float] = []
    observed_slots = 0

    for s in range(n_slots):
        devs = np.asarray(slot_dev[s], dtype=np.float64)
        wts = np.asarray(slot_w[s], dtype=np.float64)
        if devs.size < _MIN_OBSERVATIONS:
            continue
        # Median rather than mean: a couple of mis-assigned onsets should
        # not drag a slot's measured feel.
        centre = float(np.median(devs))
        # Reject outliers around the median, then take a weighted mean.
        keep = np.abs(devs - centre) < max(15.0, 2.5 * _mad(devs))
        if np.count_nonzero(keep) >= _MIN_OBSERVATIONS:
            offsets[s] = float(np.average(devs[keep], weights=np.maximum(wts[keep], 1e-6)))
            spreads.append(float(np.std(devs[keep])))
        else:
            offsets[s] = centre
            spreads.append(float(np.std(devs)))
        velocities[s] = float(np.mean(wts)) if wts.size else 0.0
        observed_slots += 1

    if observed_slots < max(2, n_slots // 4):
        return GrooveTemplate(subdivision=n_slots, source="none")

    if velocities.max() > 0:
        velocities = velocities / velocities.max()
    else:
        velocities = np.ones(n_slots)

    slot_dur_ms = (bar_duration_s / n_slots) * 1000.0
    mean_spread = float(np.mean(spreads)) if spreads else slot_dur_ms
    # Consistency compares how tightly hits cluster at their slot against
    # how wide the slot itself is. Spread approaching a quarter-slot means
    # the pattern is not reproducible.
    consistency = float(np.clip(1.0 - mean_spread / max(slot_dur_ms * 0.25, 1e-6), 0.0, 1.0))

    n_bars = int(dbs.size - 1)
    tmpl = GrooveTemplate(
        subdivision=n_slots, offsets_ms=offsets, velocities=velocities,
        swing_ratio=swing, consistency=round(consistency, 4),
        n_bars_observed=n_bars, source=source)
    log.debug("groove: %d slots, %d bars, consistency %.2f, swing %.3f, max %.1f ms",
              n_slots, n_bars, consistency, tmpl.swing_ratio, tmpl.max_offset_ms)
    return tmpl


def _mad(x: np.ndarray) -> float:
    """Median absolute deviation -- an outlier-resistant spread estimate."""
    if x.size == 0:
        return 0.0
    return float(np.median(np.abs(x - np.median(x)))) * 1.4826


def detect_swing(onsets: FloatSeq, beats: FloatSeq) -> float:
    """Swing ratio, measured from where off-beat eighths actually fall.

    Straight time is 0.5: the second eighth sits exactly halfway between
    beats. Triplet swing is 0.667. The number is reported on the scale a
    DAW uses, so it is directly comparable to what a producer would dial in.

    This deliberately works on raw onset *phase within the beat* rather
    than on a quantised slot grid, because a slot grid cannot measure
    swing at all. At a triplet feel the off-beat eighth moves by a third
    of an eighth-note -- well over 100 ms at typical tempos -- which is
    further than any sane sixteenth-grid capture window. A grid-based
    measurement would either reject those onsets as implausible or assign
    them to the neighbouring slot, and in both cases report the swung
    track as straight.

    Method: histogram every onset by its phase within the enclosing beat,
    then find the dominant cluster in the window where an off-beat eighth
    can legitimately live. Sixteenth-note material sits at phase 0.25 and
    0.75 and is excluded by that window, so the presence of sixteenths
    does not bias the estimate.
    """
    o = np.asarray(onsets, dtype=np.float64)
    b = np.asarray(beats, dtype=np.float64)
    if o.size < 8 or b.size < 3:
        return 0.5

    phases: List[float] = []
    for i in range(b.size - 1):
        span = float(b[i + 1] - b[i])
        if span <= 1e-6:
            continue
        inside = o[(o >= b[i]) & (o < b[i + 1])]
        for t in inside:
            phases.append(float((t - b[i]) / span))
    if len(phases) < 6:
        return 0.5

    ph = np.asarray(phases)
    # The off-beat eighth lives between a straight 0.5 and roughly a
    # dotted feel. The upper bound stops the sixteenth cluster at 0.75
    # from being mistaken for an extremely swung eighth.
    window = ph[(ph >= 0.42) & (ph <= 0.73)]
    if window.size < 3:
        return 0.5

    # Dominant cluster rather than the mean: a mean would be dragged by
    # any sixteenth material that leaks into the window.
    hist, edges = np.histogram(window, bins=24, range=(0.42, 0.73))
    if hist.max() < 2:
        return 0.5
    centres = (edges[:-1] + edges[1:]) / 2.0
    peak = int(np.argmax(hist))
    # Refine with the weighted centroid of the peak and its neighbours, so
    # the estimate is not quantised to the histogram bin width.
    lo, hi = max(0, peak - 1), min(hist.size, peak + 2)
    w = hist[lo:hi].astype(np.float64)
    if w.sum() <= 0:
        return 0.5
    return float(np.clip(float(np.average(centres[lo:hi], weights=w)), 0.30, 0.78))


def extract_per_element(element_onsets: Dict[str, FloatSeq],
                        downbeats: FloatSeq,
                        bar_duration_s: float,
                        subdivision: int = 16) -> Dict[str, np.ndarray]:
    """Measure each drum element's feel separately.

    Elements do not share a groove -- that is the whole point. Producers
    describe keeping kick and snare tight while letting hats breathe, or
    pushing a snare ahead for forward momentum. Collapsing them into one
    average destroys precisely the relationship that creates the feel.
    """
    out: Dict[str, np.ndarray] = {}
    for name, ons in (element_onsets or {}).items():
        t = extract(ons, downbeats, bar_duration_s, subdivision,
                    source="measured_%s" % name)
        if t.offsets_ms.size:
            out[name] = t.offsets_ms
    return out


def timing_bias(onsets: FloatSeq, grid: FloatSeq) -> Tuple[float, float]:
    """A performer's habitual placement against a grid.

    Returns `(bias_ms, spread_ms)`, where negative bias means consistently
    ahead of the beat. Bias is a stylistic fingerprint worth preserving --
    rappers who sit behind the beat are doing it on purpose -- while spread
    is the part that reads as sloppiness and is what correction should
    reduce.
    """
    o = np.asarray(onsets, dtype=np.float64)
    g = np.asarray(grid, dtype=np.float64)
    if o.size == 0 or g.size == 0:
        return 0.0, 0.0
    devs = np.array([float(g[np.argmin(np.abs(g - t))] - t) for t in o])
    devs = -devs * 1000.0                       # negative = ahead of the grid
    keep = np.abs(devs) <= _MAX_PLAUSIBLE_OFFSET_MS
    if not np.any(keep):
        return 0.0, 0.0
    return float(np.median(devs[keep])), float(np.std(devs[keep]))


# ═════════════════════════════════════════════════════════════════════════════
# Application
# ═════════════════════════════════════════════════════════════════════════════

def grid_with_groove(downbeats: FloatSeq, bar_duration_s: float,
                     groove: GrooveTemplate,
                     subdivision: Optional[int] = None,
                     amount: float = 1.0) -> np.ndarray:
    """Build the alignment target grid, carrying the track's own feel.

    This is the array a vocal gets warped onto. Because the slot positions
    already include the beat's measured microtiming, landing exactly on
    them puts the vocal in the pocket -- no separate humanisation pass is
    needed, and none should be added.
    """
    dbs = np.asarray(downbeats, dtype=np.float64)
    n_slots = int(subdivision or groove.subdivision or 16)
    if dbs.size < 2 or bar_duration_s <= 0:
        return np.zeros(0)

    use = groove.is_meaningful and amount > 0.0
    out: List[float] = []
    for i in range(dbs.size - 1):
        start, nxt = float(dbs[i]), float(dbs[i + 1])
        span = nxt - start
        if not (0.5 * bar_duration_s < span < 2.0 * bar_duration_s):
            span = bar_duration_s
        for slot in range(n_slots):
            t = start + span * slot / n_slots
            if use:
                scale = n_slots / max(groove.subdivision, 1)
                src_slot = int(round(slot / scale)) if scale != 1 else slot
                t += groove.offset_at_slot(src_slot) * amount / 1000.0
            out.append(t)
    out.append(float(dbs[-1]))
    return np.asarray(out, dtype=np.float64)


def slot_weights(subdivision: int = 16, beats_per_bar: int = 4,
                 groove: Optional[GrooveTemplate] = None) -> np.ndarray:
    """Per-slot attraction strength for grid snapping.

    Not every grid position pulls equally. A syllable near a downbeat
    should be drawn to it firmly; one near an off-beat 32nd barely at all.
    Weighting the target grid this way is what keeps correction from
    flattening a performance onto every available subdivision.
    """
    n = max(1, int(subdivision))
    per_beat = max(1, n // max(beats_per_bar, 1))
    w = np.full(n, 0.25)
    for s in range(n):
        if s % n == 0:
            w[s] = 1.0                       # downbeat
        elif s % per_beat == 0:
            w[s] = 0.75                      # beat
        elif s % max(per_beat // 2, 1) == 0:
            w[s] = 0.50                      # eighth
    if groove is not None and groove.velocities.size == n:
        # Slots the track actually accents deserve more pull than silent ones.
        w = w * (0.6 + 0.4 * groove.velocities)
    return w


def quantize_strength(genre: Optional[str], performance_type: str,
                      grid_consistency: float = 1.0) -> float:
    """How hard to pull a performance toward the grid, in [0, 1].

    Partial strength is the correct tool: at 50%, a note 40 ms late ends up
    20 ms late, so the performer's direction of feel survives while the
    sloppiness halves. Full strength is almost never right for a voice.

    The genre values follow production practice -- looser for laid-back
    hip-hop and soul, tighter for programmed dance music, near-zero for
    jazz and spoken word where rigid timing destroys the point.
    """
    g = (genre or "").lower().replace(" ", "_")
    base = {
        "drill": 0.55, "trap": 0.45, "melodic_trap": 0.40,
        "hip_hop": 0.38, "boom_bap": 0.32,
        "drum_and_bass": 0.62, "house": 0.65, "edm": 0.70, "pop": 0.45,
        "rnb": 0.28, "soul": 0.22, "afrobeats": 0.35, "dancehall": 0.38,
        "lofi": 0.18, "jazz": 0.10, "blues": 0.12,
    }.get(g, 0.35)

    # Rap lives or dies on its relationship to the grid, so it tolerates --
    # and benefits from -- more correction than sung material, where rigid
    # placement reads as mechanical.
    if performance_type == "rap":
        base *= 1.25
    elif performance_type == "melodic_rap":
        base *= 1.10
    elif performance_type == "spoken":
        base = 0.0
    elif performance_type == "sung":
        base *= 0.75

    # There is no point quantising to a grid that is itself unreliable.
    base *= float(np.clip(grid_consistency, 0.0, 1.0))
    return float(np.clip(base, 0.0, 0.85))


# ═════════════════════════════════════════════════════════════════════════════
# Comparison
# ═════════════════════════════════════════════════════════════════════════════

def compatibility(a: GrooveTemplate, b: GrooveTemplate) -> float:
    """How well two grooves sit together, in [0, 1].

    Used by the matcher. A straight-eighths vocal over a heavily swung beat
    fights the track no matter how well the key and tempo line up, and that
    mismatch is invisible to harmonic and tempo scoring -- which is exactly
    why it deserves its own term.
    """
    if not a.is_meaningful or not b.is_meaningful:
        return 0.6                     # unknown: neither reward nor punish

    swing_gap = abs(a.swing_ratio - b.swing_ratio)
    swing_score = float(np.clip(1.0 - swing_gap / 0.18, 0.0, 1.0))

    # Different grid resolutions mean the two tracks were measured against
    # different feels -- one straight, one triplet. Their offset arrays are
    # not comparable slot for slot, and correlating them anyway would
    # produce a confident meaningless number. Swing alone is the honest
    # answer here, and it already captures the mismatch.
    if a.subdivision != b.subdivision:
        return float(np.clip(swing_score, 0.0, 1.0))

    n = min(a.offsets_ms.size, b.offsets_ms.size)
    if n == 0:
        return swing_score

    oa, ob = a.offsets_ms[:n], b.offsets_ms[:n]
    if np.std(oa) < 1e-6 or np.std(ob) < 1e-6:
        # One or both are perfectly quantised, so there is no shape to
        # correlate. Defer to swing rather than scoring a flat pattern as
        # a perfect match against anything.
        shape_score = swing_score
    else:
        corr = float(np.corrcoef(oa, ob)[0, 1])
        shape_score = float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))

    return float(np.clip(0.6 * swing_score + 0.4 * shape_score, 0.0, 1.0))


def backbeat_offset_ms(groove: GrooveTemplate, beats_per_bar: int = 4) -> float:
    """Average placement of beats 2 and 4, where the snare lives.

    Reported separately from the whole-bar average because the whole-bar
    average hides it. A track with a snare 14 ms behind and everything else
    dead on has a mean offset near 2 ms -- which describes nothing a
    listener hears, while the backbeat placement describes the feel exactly.
    """
    n = groove.offsets_ms.size
    if n == 0 or beats_per_bar < 2:
        return 0.0
    per_beat = max(1, n // beats_per_bar)
    slots = [i * per_beat for i in range(1, beats_per_bar, 2) if i * per_beat < n]
    if not slots:
        return 0.0
    return float(np.mean([groove.offsets_ms[s] for s in slots]))


def describe(groove: GrooveTemplate, beats_per_bar: int = 4) -> str:
    """Human-readable feel description, for match explanations."""
    if not groove.is_meaningful:
        return "straight / unmeasured"
    bits: List[str] = []
    if groove.swing_ratio > 0.58:
        bits.append("swung (%.2f)" % groove.swing_ratio)
    elif groove.swing_ratio < 0.46:
        bits.append("rushed eighths")
    else:
        bits.append("straight")

    back = backbeat_offset_ms(groove, beats_per_bar)
    if back > 5.0:
        bits.append("backbeat laid back %.0f ms" % back)
    elif back < -5.0:
        bits.append("backbeat pushed %.0f ms" % abs(back))

    m = float(np.mean(groove.offsets_ms))
    if m > 6.0:
        bits.append("sits behind %.0f ms" % m)
    elif m < -6.0:
        bits.append("sits ahead %.0f ms" % abs(m))

    bits.append("%.0f%% consistent" % (groove.consistency * 100))
    return ", ".join(bits)
