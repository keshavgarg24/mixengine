"""
Monotonic alignment: dynamic programming over onsets and audio frames.

Two problems live here, and they are genuinely different.

**Onsets against a grid.** A vocal has ~200 onsets; a sixteenth grid over
three minutes has ~3000 slots. Almost every slot is empty. Choosing each
onset's nearest slot independently -- what the per-onset quantiser does --
produces three failures that only show up over a whole take:

  * *collision*: two syllables snap onto one slot and the gap between them
    is stretched to nothing;
  * *crossing*: onset B lands on an earlier slot than onset A that preceded
    it, which is a negative time interval and cannot be rendered at all;
  * *drift*: if the take runs 0.4% fast, nearest-neighbour follows the drift
    happily until it is a full slot out, then jumps a whole sixteenth.

A monotonic DP fixes all three, because monotonicity is a property of the
*sequence*, not of any single onset. Each onset may also be skipped -- kept
where it is -- so a melisma run or a breath is not forced onto a grid that
was never meant to hold it.

**Audio against audio.** Subsequence DTW over log-magnitude CQT frames,
following Raffel & Ellis's large-scale parameter search (*Large-Scale
Content-Based Matching of MIDI and Audio Files*, ISMIR 2015, and the
follow-up systematic evaluation): ~46 ms frames, cosine distance on
L2-normalised log-magnitude CQT, an additive penalty equal to the median
distance over all frame pairs, and a `gully` of 0.96 so the path may start
and end away from the corners. The step constraint -- at most one
consecutive horizontal or vertical move -- is what keeps a path from
parking on a single frame and producing a stutter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from ..core.types import FloatSeq

log = logging.getLogger("mixengine.align.dtw")

# Cost, in units of "one slot of distance", charged for leaving an onset
# where it is. Set below 1.0 so that moving an onset more than a full slot
# is never preferred to leaving it alone: a correction that large is
# always a misassignment rather than a sloppy performance.
DEFAULT_SKIP_COST = 0.85

# Raffel & Ellis, tuned on a large synthetic corpus of MIDI/audio pairs.
DTW_GULLY = 0.96
DTW_FRAME_S = 0.04644  # 2048 samples at 44.1 kHz; their hop, not ours
MAX_CONSECUTIVE_STEPS = 1


# ═════════════════════════════════════════════════════════════════════════════
# Onsets against a grid
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Assignment:
    """The result of matching onsets to grid slots.

    `targets[i]` is where onset `i` should land. For a skipped onset that is
    its own original time, so the caller can build a warp path from
    `(onsets, targets)` without special-casing anything.
    """
    onsets: np.ndarray = field(default_factory=lambda: np.zeros(0))
    slot_index: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    targets: np.ndarray = field(default_factory=lambda: np.zeros(0))
    assigned: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    cost: float = 0.0

    @property
    def n_assigned(self) -> int:
        return int(self.assigned.sum())

    @property
    def errors_s(self) -> np.ndarray:
        """Signed distance each assigned onset must move, in seconds."""
        if not self.assigned.any():
            return np.zeros(0)
        return self.targets[self.assigned] - self.onsets[self.assigned]

    def summary(self) -> dict:
        err = np.abs(self.errors_s)
        return {
            "onsets": int(self.onsets.size),
            "assigned": self.n_assigned,
            "skipped": int(self.onsets.size - self.n_assigned),
            "mean_error_ms": round(float(err.mean() * 1000), 2) if err.size else 0.0,
            "max_error_ms": round(float(err.max() * 1000), 2) if err.size else 0.0,
            "cost": round(float(self.cost), 3),
        }


def assign_onsets_to_grid(onsets: FloatSeq,
                          grid: FloatSeq,
                          *,
                          max_move_s: float = 0.08,
                          skip_cost: float = DEFAULT_SKIP_COST,
                          slot_weights: Optional[FloatSeq] = None,
                          beats_per_bar: int = 4) -> Assignment:
    """Assign each onset to at most one grid slot, monotonically.

    `slot_weights` is optional and indexed by *position within the bar*, not
    by absolute slot: a downbeat is a more plausible landing place than an
    off-beat sixteenth, so a small discount is applied to strong positions.
    This breaks ties toward musically sensible answers without ever letting
    an implausible slot win on weight alone.

    Complexity is O(len(onsets) x len(grid)) with a running prefix minimum,
    which is a few million operations for a three-minute take -- well under
    a second, and vectorised per onset.
    """
    o = np.asarray(onsets, dtype=np.float64)
    g = np.asarray(grid, dtype=np.float64)
    n, m = o.size, g.size
    if n == 0 or m == 0:
        return Assignment(onsets=o, slot_index=np.full(n, -1, dtype=int),
                          targets=o.copy(), assigned=np.zeros(n, dtype=bool))

    slot_s = float(np.median(np.diff(g))) if m >= 2 else max_move_s
    if slot_s <= 0:
        slot_s = max_move_s

    # Per-slot discount from metrical position. A downbeat costs slightly
    # less to land on than the fourth sixteenth of a beat.
    discount = np.zeros(m, dtype=np.float64)
    if slot_weights is not None and len(slot_weights) > 0:
        w = np.asarray(slot_weights, dtype=np.float64)
        idx = np.arange(m) % w.size
        # At most a fifth of a slot of advantage -- enough to break ties,
        # never enough to pull an onset past a closer slot.
        discount = -0.2 * w[idx]

    INF = np.float64(1e18)

    # D[j] = best total cost so far with the last *assigned* slot equal to j.
    # Column m is the "nothing assigned yet" state, kept at the end so that
    # prefix minima over 0..j-1 stay contiguous.
    prev = np.full(m + 1, INF, dtype=np.float64)
    prev[m] = 0.0  # before any onset, nothing has been assigned

    # Backtracking needs, for every (onset, slot), which previous state was
    # used. Stored as int32 to keep a 300 x 3000 table small.
    back = np.zeros((n, m + 1), dtype=np.int32)
    took = np.zeros((n, m + 1), dtype=bool)   # True = onset assigned here

    for i in range(n):
        # Exclusive prefix minimum over slots strictly below j, including
        # the virtual "nothing assigned" state which must be reachable from
        # any slot. np.minimum.accumulate gives the inclusive prefix; shift
        # it by one for the exclusive form.
        base = np.minimum(prev[:m], prev[m])
        inc_min = np.minimum.accumulate(base)
        inc_arg = _accumulate_argmin(base)
        pre_min = np.empty(m, dtype=np.float64)
        pre_arg = np.empty(m, dtype=np.int32)
        pre_min[0] = prev[m]
        pre_arg[0] = m
        if m > 1:
            pre_min[1:] = inc_min[:-1]
            pre_arg[1:] = inc_arg[:-1]
        # A state reached via the "nothing yet" path must record that.
        use_virtual = prev[m] <= pre_min
        pre_min = np.where(use_virtual, prev[m], pre_min)
        pre_arg = np.where(use_virtual, np.int32(m), pre_arg)

        dist = np.abs(g - o[i])
        cost = dist / slot_s + discount
        cost[dist > max_move_s] = INF

        assign = cost + pre_min
        assign[pre_min >= INF] = INF

        skip = prev + skip_cost          # length m+1: last slot unchanged

        cur = np.empty(m + 1, dtype=np.float64)
        cur[:m] = np.minimum(assign, skip[:m])
        cur[m] = skip[m]                 # still nothing assigned

        chose_assign = assign < skip[:m]
        took[i, :m] = chose_assign
        back[i, :m] = np.where(chose_assign, pre_arg, np.arange(m, dtype=np.int32))
        back[i, m] = m
        prev = cur

    end = int(np.argmin(prev))
    total = float(prev[end])
    if not np.isfinite(total) or total >= INF:
        # Nothing was reachable -- every onset stays where it is.
        return Assignment(onsets=o, slot_index=np.full(n, -1, dtype=int),
                          targets=o.copy(), assigned=np.zeros(n, dtype=bool),
                          cost=0.0)

    slot_index = np.full(n, -1, dtype=int)
    state = end
    for i in range(n - 1, -1, -1):
        if state < m and took[i, state]:
            slot_index[i] = state
        state = int(back[i, state])

    assigned = slot_index >= 0
    targets = o.copy()
    targets[assigned] = g[slot_index[assigned]]
    return Assignment(onsets=o, slot_index=slot_index, targets=targets,
                      assigned=assigned, cost=total)


def _accumulate_argmin(x: np.ndarray) -> np.ndarray:
    """Index of the running minimum of `x`, element by element.

    numpy has `minimum.accumulate` but no argmin equivalent, and the loop
    here is over slots rather than over (onset, slot) pairs, so it costs a
    few thousand iterations per take rather than a few million.
    """
    n = x.size
    out = np.empty(n, dtype=np.int32)
    best_i = 0
    best_v = x[0]
    for j in range(n):
        if x[j] < best_v:
            best_v = x[j]
            best_i = j
        out[j] = best_i
    return out


def residual_drift(assignment: Assignment) -> Tuple[float, float]:
    """Linear trend through the assignment errors: `(slope, intercept_s)`.

    Useful only for *residual* drift, after the take's tempo has already
    been matched. It cannot be used to measure drift in the first place,
    and the reason is worth stating because the failure is silent:

    once a take has drifted by more than half a slot, the assignment moves
    it to the next slot and the error resets to near zero. The drift is
    still there in the audio but has been aliased out of the error signal.
    Measured on a take drifting 0.3%, this returned a slope of +0.0005
    against a true -0.0040 -- wrong by a factor of eight, and wrong in
    sign. Use `estimate_tempo_ratio`, which measures the drift directly
    and does not alias.
    """
    if assignment.n_assigned < 4:
        return 0.0, 0.0
    t = assignment.onsets[assignment.assigned]
    e = assignment.errors_s
    if float(t.max() - t.min()) < 1e-6:
        return 0.0, float(np.mean(e))
    slope, intercept = np.polyfit(t, e, 1)
    return float(slope), float(intercept)


def grid_fit_error(onsets: FloatSeq, grid: FloatSeq) -> float:
    """Mean distance from each onset to the nearest grid slot, in seconds."""
    o = np.asarray(onsets, dtype=np.float64)
    g = np.asarray(grid, dtype=np.float64)
    if o.size == 0 or g.size < 2:
        return float("inf")
    i = np.clip(np.searchsorted(g, o), 1, g.size - 1)
    return float(np.mean(np.minimum(np.abs(o - g[i]), np.abs(o - g[i - 1]))))


def estimate_tempo_ratio(onsets: FloatSeq, grid: FloatSeq, *,
                         lo: float = 0.96, hi: float = 1.04,
                         steps: int = 801) -> Tuple[float, float, float]:
    """The tempo ratio that best lands `onsets` on `grid`.

    Returns `(ratio, fit_error_s, improvement)`. Dividing onset times by
    `ratio` -- equivalently, stretching the audio by `ratio` -- removes the
    drift. `improvement` is the fraction of the original fit error removed,
    so a caller can decline a correction that did not actually help.

    A direct search rather than a fit, because a fit needs the errors and
    the errors are the thing that aliases. Scoring candidate ratios by how
    well the whole onset set lands on the grid has no such failure: the
    measurement is made on the onsets themselves.

    Ties near 1.0 are broken toward 1.0 by a tilt four orders of magnitude
    below a real difference in fit, so a take that is genuinely on tempo is
    never stretched by search noise.
    """
    o = np.asarray(onsets, dtype=np.float64)
    g = np.asarray(grid, dtype=np.float64)
    if o.size < 8 or g.size < 4:
        return 1.0, grid_fit_error(o, g), 0.0

    t0 = float(o[0])
    cands = np.linspace(lo, hi, steps)
    errs = np.empty(steps)
    for k, r in enumerate(cands):
        errs[k] = grid_fit_error(t0 + (o - t0) / r, g)
    errs = errs + 1e-4 * np.abs(cands - 1.0)

    k = int(np.argmin(errs))
    ratio = float(cands[k])
    best = float(grid_fit_error(t0 + (o - t0) / ratio, g))
    base = float(grid_fit_error(o, g))
    improvement = (base - best) / base if base > 1e-9 else 0.0
    return ratio, best, float(improvement)


# ═════════════════════════════════════════════════════════════════════════════
# Audio against audio
# ═════════════════════════════════════════════════════════════════════════════

def cqt_features(y: np.ndarray, sr: int, *,
                 hop_s: float = DTW_FRAME_S,
                 fmin_hz: float = 65.4,        # C2
                 n_octaves: int = 6,
                 bins_per_octave: int = 12) -> Optional[np.ndarray]:
    """Log-magnitude CQT, L2-normalised per frame. `(n_frames, n_bins)`.

    Log magnitude rather than linear: the search that produced these
    parameters found log compression to matter more than any other single
    choice, because linear magnitude lets one loud frame dominate the
    distance between two otherwise similar frames.
    """
    try:
        import librosa
    except ImportError:
        return None
    mono = y if y.ndim == 1 else y.mean(axis=1)
    if mono.size < int(sr * hop_s * 4):
        return None
    hop = max(1, int(round(sr * hop_s)))
    try:
        C = np.abs(librosa.cqt(mono, sr=sr, hop_length=hop, fmin=fmin_hz,
                               n_bins=n_octaves * bins_per_octave,
                               bins_per_octave=bins_per_octave))
    except Exception as exc:                       # pragma: no cover
        log.debug("cqt failed: %s", exc)
        return None
    C = np.log1p(C).T.astype(np.float64)
    norm = np.linalg.norm(C, axis=1, keepdims=True)
    norm[norm < 1e-9] = 1.0
    return C / norm


def cosine_cost(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine distance between every pair of (already L2-normalised) frames."""
    return np.clip(1.0 - a @ b.T, 0.0, 2.0)


def dtw_path(cost: np.ndarray, *,
             penalty: Optional[float] = None,
             gully: float = DTW_GULLY,
             max_consecutive: int = MAX_CONSECUTIVE_STEPS
             ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Subsequence DTW. Returns `(rows, cols, normalised_cost)`.

    `penalty` is added to every non-diagonal step. Setting it to the median
    of the whole cost matrix -- the default -- makes the path prefer moving
    diagonally unless the off-diagonal cell is better than a typical frame
    pair, which is what stops a path from wandering through a silent or
    unvoiced passage.

    `gully` relaxes the requirement that the path reach the far corner: it
    may end once it has covered that fraction of the shorter sequence. Real
    recordings have lead-in and tail that correspond to nothing.

    `max_consecutive` caps runs of horizontal or vertical steps. Without it
    a path can sit on one source frame for a whole phrase, which on playback
    is a held phoneme -- an audible stutter rather than a small timing error.
    """
    C = np.asarray(cost, dtype=np.float64)
    n, m = C.shape
    if n < 2 or m < 2:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int), float("inf")

    pen = float(np.median(C)) if penalty is None else float(penalty)
    INF = np.float64(1e18)

    # Three accumulated-cost planes, one per "how many consecutive steps of
    # this kind have just been taken". Tracking the run length in the state
    # is what makes the constraint exact rather than a post-hoc repair.
    K = max(1, int(max_consecutive))
    D = np.full((n, m), INF)
    # run[i, j] encodes the move that reached (i, j): 0 diagonal,
    # 1..K horizontal run length, K+1..2K vertical run length.
    move = np.zeros((n, m), dtype=np.int8)
    hrun = np.zeros((n, m), dtype=np.int8)
    vrun = np.zeros((n, m), dtype=np.int8)

    D[0, 0] = C[0, 0]
    for j in range(1, m):
        # The first row is free to start anywhere: subsequence alignment.
        D[0, j] = C[0, j]
    for i in range(1, n):
        prev = D[i - 1]
        prev_v = vrun[i - 1]
        cur = D[i]
        for j in range(m):
            best = INF
            best_move = 0
            # Diagonal
            if j > 0 and prev[j - 1] < INF:
                cand = prev[j - 1]
                if cand < best:
                    best, best_move = cand, 0
            # Vertical (advance source, hold target)
            if prev[j] < INF and prev_v[j] < K:
                cand = prev[j] + pen
                if cand < best:
                    best, best_move = cand, 2
            # Horizontal (hold source, advance target)
            if j > 0 and cur[j - 1] < INF and hrun[i, j - 1] < K:
                cand = cur[j - 1] + pen
                if cand < best:
                    best, best_move = cand, 1
            if best >= INF:
                continue
            cur[j] = best + C[i, j]
            move[i, j] = best_move
            if best_move == 1:
                hrun[i, j] = hrun[i, j - 1] + 1
                vrun[i, j] = 0
            elif best_move == 2:
                vrun[i, j] = prev_v[j] + 1
                hrun[i, j] = 0

    # End anywhere past the gully, in either dimension.
    i_end = n - 1
    j_min = int(gully * (m - 1))
    tail = D[i_end, j_min:]
    if not np.isfinite(tail).any() or tail.min() >= INF:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int), float("inf")
    j_end = j_min + int(np.argmin(tail))

    rows, cols = [], []
    i, j = i_end, j_end
    while i > 0 or j > 0:
        rows.append(i)
        cols.append(j)
        mv = move[i, j]
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        elif mv == 0:
            i -= 1
            j -= 1
        elif mv == 1:
            j -= 1
        else:
            i -= 1
    rows.append(0)
    cols.append(0)
    rows.reverse()
    cols.reverse()
    path_cost = float(D[i_end, j_end] / max(len(rows), 1))
    return np.asarray(rows, dtype=int), np.asarray(cols, dtype=int), path_cost


def align_audio(source: np.ndarray, reference: np.ndarray, sr: int, *,
                hop_s: float = DTW_FRAME_S,
                gully: float = DTW_GULLY
                ) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """DTW a take against a reference recording.

    Returns `(source_times_s, reference_times_s, cost)`, or None when the
    features could not be computed. The returned pairs are the warp anchors
    the caller feeds to `warp`; they are already monotonic.
    """
    a = cqt_features(source, sr, hop_s=hop_s)
    b = cqt_features(reference, sr, hop_s=hop_s)
    if a is None or b is None:
        return None
    cost = cosine_cost(a, b)
    rows, cols, c = dtw_path(cost, gully=gully)
    if rows.size < 2:
        return None
    return rows * hop_s, cols * hop_s, c
