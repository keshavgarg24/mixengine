"""
Audio transformation: stretching, pitch shifting, alignment, tuning.

The corrections to the original engine live here:

  * Stretch by the **tempo ratio**, never the duration ratio. Making a
    vocal's total length equal the beat's length is musically meaningless.
  * Align phrases to **downbeats** from the beat grid, not to the first
    detected onset -- the first detected beat is rarely bar 1.
  * Search-and-score the pitch shift, and refuse when it is too large. A
    shift capped at the maximum lands on a key matching neither source,
    which is strictly worse than not shifting.
  * Shift only the **tonal stems**, leaving drums untouched, whenever stems
    are available.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import dsp
from ..core.capabilities import CAPS

log = logging.getLogger("mixengine.transform")


# ─────────────────────────────────────────────────────────────────────────────
# Time stretching
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Rubber Band
#
# The CLI is driven directly rather than through pyrubberband, because
# pyrubberband cannot express boolean flags. It emits every rbargs entry as
# a `key value` pair, so a flag like `--formant` -- which takes no argument
# -- becomes `--formant ""`. Rubber Band reads the empty string as a third
# positional filename, prints its usage banner and exits 2.
#
# That mattered more than it looks. The previous code passed exactly that,
# caught the resulting exception, and silently fell back to the librosa
# phase vocoder. Formant preservation therefore never ran once, while the
# capability probe and the docs both reported it as active -- and formant
# preservation is the difference between a stretched vocal that still
# sounds like the singer and one that sounds synthetic.
#
# Failures are logged at warning level here for the same reason: a silent
# fallback to a worse algorithm is indistinguishable from success right up
# until someone listens.
# ─────────────────────────────────────────────────────────────────────────────

_RB_FORMANT_OK: Optional[bool] = None


def _rubberband_supports_formant() -> bool:
    """Probe once whether this build accepts `--formant`.

    The flag arrived in Rubber Band v1.8.2 and the R3 engine in v3.0, and
    the engine must work against whatever the host has installed rather
    than assuming a version.
    """
    global _RB_FORMANT_OK
    if _RB_FORMANT_OK is not None:
        return _RB_FORMANT_OK
    try:
        proc = subprocess.run(["rubberband", "--help"], capture_output=True,
                              text=True, timeout=10)
        text = (proc.stdout or "") + (proc.stderr or "")
        _RB_FORMANT_OK = "--formant" in text
    except Exception:
        _RB_FORMANT_OK = False
    return _RB_FORMANT_OK


def _rubberband_supports_r3() -> bool:
    try:
        proc = subprocess.run(["rubberband", "--help"], capture_output=True,
                              text=True, timeout=10)
        return "--fine" in ((proc.stdout or "") + (proc.stderr or ""))
    except Exception:
        return False


def _rubberband(y2: np.ndarray, sr: int, flags: List[str],
                preserve_formants: bool, label: str) -> Optional[np.ndarray]:
    """Run the Rubber Band CLI. Returns None on failure so callers fall back.

    Boolean flags are passed as bare tokens, which is the only form the CLI
    accepts. The R3 engine is requested when available: it almost always
    produces better results than R2, and is markedly better on vocals,
    soft onsets and smooth pitch changes -- at roughly three times the CPU,
    which is the right trade for offline rendering.
    """
    import soundfile as sf

    args = ["rubberband", "-q"]
    if _rubberband_supports_r3():
        args.append("--fine")
    if preserve_formants and _rubberband_supports_formant():
        args.append("--formant")
    args += flags

    tmpdir = tempfile.mkdtemp(prefix="mixengine_rb_")
    src = os.path.join(tmpdir, "in.wav")
    dst = os.path.join(tmpdir, "out.wav")
    try:
        data = y2[:, 0] if y2.shape[1] == 1 else y2
        sf.write(src, data, sr, subtype="FLOAT")
        proc = subprocess.run(args + [src, dst], capture_output=True,
                              text=True, timeout=900)
        if proc.returncode != 0 or not os.path.exists(dst):
            log.warning("rubberband %s failed (rc=%d): %s", label,
                        proc.returncode,
                        ((proc.stderr or proc.stdout or "")[:200]).replace("\n", " "))
            return None
        out, _ = sf.read(dst, dtype="float32", always_2d=True)
        if out.size == 0:
            log.warning("rubberband %s produced empty output", label)
            return None
        return dsp.as_2d(out)
    except Exception as e:
        log.warning("rubberband %s unavailable (%s); falling back", label, e)
        return None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def time_stretch(y: np.ndarray, sr: int, ratio: float,
                 preserve_formants: bool = True) -> np.ndarray:
    """Stretch by `ratio` (>1 = longer/slower).

    Rubber Band with formant preservation is strongly preferred for vocals;
    without it, stretched voices acquire the characteristic hollow, phasey
    quality. librosa's phase vocoder is the fallback.
    """
    if abs(ratio - 1.0) < 1e-4:
        return dsp.as_2d(y)

    y2 = dsp.as_2d(y)
    ratio = float(np.clip(ratio, 0.25, 4.0))

    if CAPS.can_stretch_well:
        # `--time X` stretches to X times the original duration, which is
        # exactly our ratio -- no reciprocal, and no chance of inverting it.
        out = _rubberband(y2, sr, ["--time", "%.9f" % ratio],
                          preserve_formants, "stretch")
        if out is not None:
            return out.astype(np.float32)

    return _stretch_librosa(y2, ratio)


def _stretch_librosa(y2: np.ndarray, ratio: float) -> np.ndarray:
    import librosa
    chans = []
    for c in range(y2.shape[1]):
        chans.append(librosa.effects.time_stretch(
            np.ascontiguousarray(y2[:, c]), rate=1.0 / ratio))
    n = max(len(c) for c in chans)
    return np.stack([np.pad(c, (0, n - len(c))) for c in chans],
                    axis=1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Pitch shifting
# ─────────────────────────────────────────────────────────────────────────────

def pitch_shift(y: np.ndarray, sr: int, semitones: float,
                preserve_formants: bool = True) -> np.ndarray:
    """Shift pitch by `semitones`, preserving formants where possible.

    Formant preservation is what separates a shifted vocal that still
    sounds like the same singer from one that sounds like a cartoon.
    """
    if abs(semitones) < 1e-4:
        return dsp.as_2d(y)

    y2 = dsp.as_2d(y)
    if CAPS.can_stretch_well:
        out = _rubberband(y2, sr, ["--pitch", "%.6f" % float(semitones)],
                          preserve_formants, "pitch shift")
        if out is not None:
            return out.astype(np.float32)

    import librosa
    chans = [librosa.effects.pitch_shift(np.ascontiguousarray(y2[:, c]),
                                         sr=sr, n_steps=float(semitones))
             for c in range(y2.shape[1])]
    n = max(len(c) for c in chans)
    return np.stack([np.pad(c, (0, n - len(c))) for c in chans],
                    axis=1).astype(np.float32)


def shift_beat_stems(stems: Dict[str, np.ndarray], sr: int,
                     semitones: int) -> Dict[str, np.ndarray]:
    """Pitch-shift the tonal stems only, leaving drums untouched.

    This resolves the classic dilemma -- shift the vocal and it sounds
    robotic, shift the whole beat and the drums detune and smear. Drums are
    broadband and largely pitch-neutral, so moving the harmonic content
    while leaving the groove alone is both musically correct and
    artifact-free. It is only possible because the catalog stores stems.
    """
    if abs(semitones) < 1 or not stems:
        return stems
    out = dict(stems)
    for name in ("bass", "other", "vocals"):
        if name in out and out[name] is not None:
            log.info("  shifting '%s' stem by %+d semitones", name, semitones)
            out[name] = pitch_shift(out[name], sr, semitones,
                                    preserve_formants=(name == "vocals"))
    return out


def plan_pitch_shift(semitones: int, has_stems: bool) -> dict:
    """Decide which side to shift, and whether to shift at all.

    Priority: shift the beat's tonal stems (best), then split the shift
    between beat and vocal, then refuse. Refusing is a legitimate outcome
    and much better than forcing a bad render.
    """
    s = int(semitones)
    if s == 0:
        return {"beat_shift": 0, "vocal_shift": 0, "strategy": "none",
                "quality": "optimal"}
    if abs(s) <= 2 and has_stems:
        return {"beat_shift": s, "vocal_shift": 0, "strategy": "beat_stems",
                "quality": "optimal",
                "note": "tonal stems shifted; drums untouched"}
    if abs(s) <= 2:
        return {"beat_shift": 0, "vocal_shift": -s, "strategy": "vocal",
                "quality": "good",
                "note": "vocal shifted with formant preservation"}
    if abs(s) <= 4:
        half = int(np.sign(s) * (abs(s) // 2))
        return {"beat_shift": half, "vocal_shift": -(s - half),
                "strategy": "split", "quality": "degraded",
                "note": "shift split between beat and vocal to limit artifacts"}
    return {"beat_shift": 0, "vocal_shift": 0, "strategy": "refused",
            "quality": "incompatible",
            "note": f"{abs(s)} semitones is too far - rendering without shift"}


# ─────────────────────────────────────────────────────────────────────────────
# Alignment
# ─────────────────────────────────────────────────────────────────────────────

# A measured tempo within this of the beat's is the same tempo. Stretching
# by less than this trades nothing audible for phase-vocoder artefacts.
GRID_TEMPO_TOLERANCE = 0.001


def grid_tempo_ratio(vocal_ones: Optional[np.ndarray], beat_bar_s: float
                     ) -> Optional[float]:
    """The vocal's tempo relative to the beat's, from its own bar-ones.

    A line fitted through forty bar-ones measures the bar to a fraction
    of a millisecond; a tempo histogram over the same take read 198.77
    against a true 200, and the 0.6% stretch built on it sped up a vocal
    that was already exactly in tempo, then drifted 641 ms that a later
    stage had to remove. Returns vocal_bar / beat_bar -- the factor by
    which the vocal is *longer* than the beat per bar -- or None when
    there are too few bars to fit.
    """
    if vocal_ones is None or vocal_ones.size < 8 or beat_bar_s <= 0:
        return None
    k = np.arange(vocal_ones.size, dtype=np.float64)
    slope = float(np.polyfit(k, np.asarray(vocal_ones, dtype=np.float64),
                             1)[0])
    if slope <= 0:
        return None
    return slope / beat_bar_s


# Below this share of the vocal's bar-ones landing on the beat's, the
# vocal's own grid is not trusted for placement and the phrase-start
# scoring decides instead.
VOCAL_GRID_CONFIDENCE = 0.6
VOCAL_GRID_TEMPO_WINDOW = 0.08     # how far a take may run from the beat's tempo


def _tempo_window(bar_s: float, beats_per_bar: int
                  ) -> Optional[Tuple[float, float]]:
    """The tracker's tempo range for counting `beats_per_bar` to a bar.

    Free-running, the tracker keeps whichever pulse is strongest, and on
    a rap take with nothing under it that is often the half-time feel:
    it counted a 150 BPM take at 75 and reported a bar twice the beat's,
    so the grid was thrown away and the take was never stretched. The
    beat's bar is known, so the tempo is too, to within how far a
    performer drifts. None when that tempo is outside what the
    activations can resolve.
    """
    if bar_s <= 0 or beats_per_bar <= 0:
        return None
    bpm = 60.0 * beats_per_bar / bar_s
    lo, hi = bpm * (1 - VOCAL_GRID_TEMPO_WINDOW), bpm * (1 + VOCAL_GRID_TEMPO_WINDOW)
    if lo < 40.0 or hi > 300.0:
        return None
    return lo, hi


def vocal_downbeats(vocal: np.ndarray, sr: int, bar_s: float,
                    max_seconds: float = 150.0,
                    beats_per_bar: int = 4) -> Optional[np.ndarray]:
    """The vocal's own bar-ones, from a downbeat tracker run on it alone.

    A rapper's bar structure is audible without the drums -- couplets,
    rhyme endings, breath -- and a tracker trained to find metrical
    downbeats hears it. Measured on a take with no beat under it, the
    onset autocorrelation peaked at two bars, exactly where couplets
    fall. This is the signal every cross-correlation against the *beat*
    was missing: the two signals share no content, so correlating them
    peaks at chance, but each carries its own grid, and two grids can be
    laid over one another.

    The tracker is told the beat's tempo, within a drift window, and is
    asked to count the beat's meter and its double: a 200 BPM reading of
    a 100 BPM take is the same music counted twice as fast. The answer
    whose bar matches the beat's is kept.
    """
    if not CAPS.madmom or bar_s <= 0:
        return None
    try:
        from madmom.features.downbeats import (DBNDownBeatTrackingProcessor,
                                               RNNDownBeatProcessor)
    except Exception:                                    # noqa: BLE001
        return None

    import warnings
    mono = np.ascontiguousarray(dsp.to_mono(vocal).astype(np.float32))
    if sr != 44100:
        import librosa
        mono = librosa.resample(mono, orig_sr=sr, target_sr=44100)
    mono = mono[:int(max_seconds * 44100)]
    if mono.size < 44100 * 4:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        activations = RNNDownBeatProcessor()(mono)
        best, best_err = None, np.inf
        for bpb in ([beats_per_bar], [2 * beats_per_bar]):
            window = _tempo_window(bar_s, bpb[0])
            if window is None:
                continue
            try:
                out = DBNDownBeatTrackingProcessor(
                    beats_per_bar=bpb, fps=100, min_bpm=window[0],
                    max_bpm=window[1])(activations)
            except Exception:                            # noqa: BLE001
                continue
            ones = out[out[:, 1] == 1][:, 0]
            if ones.size < 3:
                continue
            bar = float(np.median(np.diff(ones)))
            err = abs(bar - bar_s) / bar_s
            if err < best_err:
                best, best_err = ones, err
    if best is None or best_err > 0.08:
        return None
    return np.asarray(best, dtype=np.float64)


def _grid_alignment(vocal_ones: np.ndarray, beat_ones: np.ndarray,
                    bar_s: float) -> Tuple[float, float]:
    """Shift laying the vocal's bar-ones onto the beat's, and how well.

    Each vocal downbeat's offset from the nearest beat downbeat is a
    phase within the bar. The circular mean of those phases is the shift;
    their concentration is the confidence -- a vocal whose ones scatter
    across the bar has no grid to align, and says so.
    """
    if vocal_ones.size == 0 or beat_ones.size == 0 or bar_s <= 0:
        return 0.0, 0.0
    idx = np.searchsorted(beat_ones, vocal_ones)
    lo = beat_ones[np.clip(idx - 1, 0, beat_ones.size - 1)]
    hi = beat_ones[np.clip(idx, 0, beat_ones.size - 1)]
    nearest = np.where(np.abs(vocal_ones - lo) < np.abs(vocal_ones - hi),
                       lo, hi)
    residual = (vocal_ones - nearest)          # vocal one minus beat one
    angle = residual / bar_s * 2.0 * np.pi
    vec = np.mean(np.exp(1j * angle))
    confidence = float(np.abs(vec))
    mean_phase = float(np.angle(vec)) / (2.0 * np.pi) * bar_s
    # A positive residual means the vocal's one falls after the beat's,
    # so the vocal must move earlier by that much.
    return float(-mean_phase), confidence


def _placement_candidates(downbeats: np.ndarray,
                          beats_s: Optional[np.ndarray],
                          first_onset_s: float,
                          allow_beat_level: bool) -> List[Tuple[float, str]]:
    """Bar-line placements worth scoring for the first phrase.

    The two downbeats either side of the first onset, plus -- when a
    pickup is plausible -- the beats around it, since a take that leads
    into bar one starts fractionally before a downbeat rather than on it.
    """
    out: List[Tuple[float, str]] = []
    if downbeats.size:
        idx = int(np.searchsorted(downbeats, first_onset_s))
        for i in (idx - 1, idx, idx + 1):
            if 0 <= i < downbeats.size:
                out.append((float(downbeats[i]), "downbeat"))
    if allow_beat_level and beats_s is not None and len(beats_s):
        b = np.asarray(beats_s, dtype=np.float64)
        idx = int(np.searchsorted(b, first_onset_s))
        for i in (idx - 1, idx):
            if 0 <= i < b.size:
                out.append((float(b[i]), "beat_pickup"))
    return out or [(first_onset_s, "none")]


# A rotation off the tracker's own phase wins only on an unambiguous
# surface: it must halve the phrase-start cost *and* put the lines on the
# bar lines outright. The tracker's phase proved the more consistent
# signal -- identical across re-encodings of one take to 0.01 s -- while
# a rotation that merely edged the cost down by a quarter fired on four
# of eight variants and scattered their landings by 2 s. Whole beats
# only, for the same reason.
PHASE_ROTATION_RATIO = 0.5
PHASE_ROTATION_MAX_COST = 0.12


def choose_bar_phase(shift: float, beat_s: float, starts: np.ndarray,
                     downbeats: np.ndarray) -> Tuple[float, dict]:
    """Which of the vocal's beats is its bar-one.

    The grid alignment lays the vocal's tracked bar-ones on the beat's and
    reports how regular that lattice is -- not whether the tracker chose
    the right beat as "one". On a rap take it often does not: the same
    performance, re-encoded or with one channel flipped, came back with
    its first line anywhere from 1.8 s before the drop to 1.6 s after it,
    every time with a grid confidence over 0.99. The lattice was right;
    the phase was a beat or a half-beat off, and nothing checked.

    Lines of a verse begin on bar lines, or just before them. So the
    shift the grid found is tried at every beat and half-beat rotation
    within the bar, each scored by how far *all* the phrase starts then
    sit from a bar line, and the lowest cost wins -- with the tracker's
    own phase kept unless another rotation beats it clearly. The fine,
    sub-beat part of the shift is never touched: the grid measured that
    well, and the lattice keeps its tempo.
    """
    info = {"rotation_beats": 0.0, "cost": None, "tracker_cost": None}
    if beat_s <= 0 or starts.size == 0 or downbeats.size < 2:
        return shift, info
    bar_s = float(np.median(np.diff(downbeats)))
    if bar_s <= 0:
        return shift, info
    n_beats = max(1, int(round(bar_s / beat_s)))
    tracker_cost = _bar_phase_cost(starts + shift, downbeats)
    best_shift, best_cost, best_rot = shift, tracker_cost, 0.0
    surface = {"0": round(tracker_cost, 4)}
    for rot in range(1, n_beats):
        for sign in (1.0, -1.0):
            cand = shift + sign * rot * beat_s
            # Keep the total shift within the bar the grid chose.
            cand = ((cand + bar_s / 2.0) % bar_s) - bar_s / 2.0
            cost = _bar_phase_cost(starts + cand, downbeats)
            surface["%+d" % int(sign * rot)] = round(cost, 4)
            if (cost < best_cost and cost <= PHASE_ROTATION_MAX_COST
                    and cost <= PHASE_ROTATION_RATIO * tracker_cost):
                best_shift, best_cost, best_rot = cand, cost, sign * rot
    info.update({"rotation_beats": best_rot, "cost": round(best_cost, 4),
                 "tracker_cost": round(tracker_cost, 4), "surface": surface})
    return float(best_shift), info


def _bar_phase_cost(starts: np.ndarray, downbeats: np.ndarray) -> float:
    """Mean distance from each phrase start to the nearest bar line.

    Normalised by the bar so the number means "fraction of a bar out",
    and capped at half a bar because a phrase that genuinely begins
    mid-bar should not dominate the vote for every other phrase.
    """
    if starts.size == 0 or downbeats.size < 2:
        return float("inf")
    bar = float(np.median(np.diff(downbeats)))
    if bar <= 0:
        return float("inf")
    phase = np.abs((starts[:, None] - downbeats[None, :]))
    nearest = phase.min(axis=1)
    return float(np.mean(np.minimum(nearest, bar / 2.0)) / bar)


# Fewer stable line starts than this and the phrase starts are kept: the
# phase is a vote, and three lines is the smallest number that can carry
# one against a phrase list that usually has more members.
MIN_PHASE_LINES = 3


def _phase_starts(phrase_starts: np.ndarray,
                  line_starts_s: Optional[np.ndarray]) -> Tuple[np.ndarray, str]:
    """Which set of starts decides the bar phase, and where it came from."""
    if line_starts_s is None:
        return phrase_starts, "phrases"
    lines = np.asarray(line_starts_s, dtype=np.float64)
    lines = lines[np.isfinite(lines)]
    if lines.size < MIN_PHASE_LINES:
        return phrase_starts, "phrases"
    return lines, "lyrics"


def align_to_downbeat(vocal: np.ndarray, sr: int,
                      phrases_samples: Sequence[Tuple[int, int]],
                      downbeats_s: np.ndarray,
                      beats_s: Optional[np.ndarray] = None,
                      allow_beat_level: bool = True,
                      vocal_ones: Optional[np.ndarray] = None,
                      line_starts_s: Optional[np.ndarray] = None
                      ) -> Tuple[np.ndarray, dict]:
    """Shift the vocal so its first phrase begins on a bar line.

    `line_starts_s` are where the transcriber says the sung lines begin.
    When they are given they settle the bar phase in place of the phrase
    starts, because they are the same measurement taken a stabler way: a
    phrase start is an energy threshold and moves when the take is
    clipped or re-encoded, and a word start comes from the model's
    attention alignment and does not.

    The original engine aligned the first detected vocal onset to the first
    detected beat transient. Two problems: the first detected beat is
    usually not bar 1, and aligning one point does nothing for the rest of
    the track. This aligns to an actual downbeat, and reports the offset so
    later stages can warp against the same grid.
    """
    v = dsp.as_2d(vocal)
    info: Dict[str, Any] = {"offset_s": 0.0, "target": None, "method": "none"}

    if len(downbeats_s) == 0 or not len(phrases_samples):
        return v, info

    first_onset_s = phrases_samples[0][0] / sr
    grid = np.asarray(downbeats_s, dtype=np.float64)

    # Where the bar line falls is a question about the whole vocal, not
    # about its first syllable. Deciding it from one onset let a take
    # whose first breath happened to sit 0.12s from beat 3 pull every
    # phrase after it onto beat 3 -- perfectly on the sixteenth grid, and
    # half a bar out of the music.
    #
    # So score each candidate placement by how well *all* the phrase
    # starts land on bar lines, and take the best. A genuine pickup still
    # wins when it is one: shifting so the pickup's own phrase begins a
    # bar leaves every later phrase off the grid, and scores badly.
    bar_s = float(np.median(np.diff(grid))) if grid.size >= 2 else 0.0

    # Preferred: lay the vocal's own bar grid over the beat's. Each signal
    # is tracked on its own, so the two never have to share content --
    # which is why every cross-correlation between them found nothing.
    ones = (vocal_ones if vocal_ones is not None
            else (vocal_downbeats(vocal, sr, bar_s) if bar_s > 0 else None))
    shift, conf = (_grid_alignment(ones, grid, bar_s)
                   if ones is not None else (0.0, 0.0))
    if ones is not None and conf >= VOCAL_GRID_CONFIDENCE:
        starts = np.array([s / sr for s, _ in phrases_samples],
                          dtype=np.float64)
        phase_starts, phase_src = _phase_starts(starts, line_starts_s)
        beat_s = (float(np.median(np.diff(np.asarray(beats_s, dtype=np.float64))))
                  if beats_s is not None and len(beats_s) >= 2 else bar_s / 4.0)
        offset_s, phase = choose_bar_phase(shift, beat_s, phase_starts, grid)
        phase["starts_from"] = phase_src
        method = "vocal_grid"
        info.update({"vocal_bars": int(ones.size),
                     "grid_confidence": round(conf, 3),
                     "grid_shift_s": round(float(shift), 4),
                     "bar_phase": phase})
        if phase.get("rotation_beats"):
            log.info("  bar phase: the tracker's one was %+.1f beats off "
                     "the lines (cost %.3f -> %.3f)", -phase["rotation_beats"],
                     phase["tracker_cost"], phase["cost"])
    else:
        if ones is not None:
            info["grid_confidence"] = round(conf, 3)
        starts = np.array([s / sr for s, _ in phrases_samples],
                          dtype=np.float64)
        starts, phase_src = _phase_starts(starts, line_starts_s)
        info["starts_from"] = phase_src
        candidates = _placement_candidates(grid, beats_s, first_onset_s,
                                           allow_beat_level)
        best_offset, best_cost, method = 0.0, np.inf, "none"
        for cand, label in candidates:
            offset = float(cand - first_onset_s)
            cost = _bar_phase_cost(starts + offset, grid)
            if cost < best_cost:
                best_offset, best_cost, method = offset, cost, label
        offset_s = best_offset

    target = first_onset_s + offset_s

    # Cap the move so a misdetected phrase start cannot displace the whole
    # vocal by many seconds.
    max_shift = 4.0
    offset_s = float(np.clip(offset_s, -max_shift, max_shift))
    offset = int(round(offset_s * sr))

    if offset > 0:
        v = np.vstack([np.zeros((offset, v.shape[1]), dtype=np.float32), v])
    elif offset < 0:
        v = v[min(-offset, len(v) - 1):]

    info.update({"offset_s": round(offset_s, 4),
                 "target": round(float(target), 4),
                 "first_onset_s": round(first_onset_s, 4),
                 "method": method})
    log.info("  aligned first phrase to %s at %.3fs (moved %+.3fs)",
             method, target, offset_s)
    return v.astype(np.float32), info


MAX_LEAD_IN_BARS = 8
# A first line this far before the entry's bar line is a pickup into it.
PICKUP_BEATS = 1.0


def place_at_section(vocal: np.ndarray, sr: int,
                     phrases_samples: Sequence[Tuple[int, int]],
                     downbeats_s: np.ndarray,
                     sections: Optional[List[dict]],
                     beats_per_bar: int = 4
                     ) -> Tuple[np.ndarray, dict]:
    """Move the vocal, by whole bars, to where the beat arrives.

    Grid alignment settles which bar *line* the vocal sits on, not which
    bar. Left there, a take begins at the top of the beat -- inside its
    intro -- and the drop lands a few bars into the first verse. A
    producer starts the vocal where the beat opens up: the first section
    after the intro (or a break) that carries the track's energy. That
    entry is snapped to a tracked downbeat and the vocal's own first
    bar-one is moved onto it, a whole number of bars, so the bar phase
    the alignment found is kept exactly.

    An opening longer than MAX_LEAD_IN_BARS is left as placed: a long
    instrumental intro is an arrangement decision for the listener, not
    half a minute of silence for the engine to impose.
    """
    v = dsp.as_2d(vocal)
    info: Dict[str, Any] = {"method": "none", "moved_s": 0.0}
    grid = np.asarray(downbeats_s, dtype=np.float64)
    if not sections or grid.size < 2 or not len(phrases_samples):
        return v, info
    bar_s = float(np.median(np.diff(grid)))
    entry = next((s for s in sorted(sections, key=lambda s: float(s["start"]))
                  if s.get("label") not in ("intro", "break", "outro")), None)
    if entry is None or bar_s <= 0:
        return v, info

    entry_s = float(entry["start"])
    lead_in_bars = entry_s / bar_s
    if lead_in_bars > MAX_LEAD_IN_BARS + 0.5:
        info.update({"method": "kept", "entry_s": round(entry_s, 3),
                     "reason": "the beat opens for %.0f bars before its %s; "
                               "left where the grid put it"
                               % (lead_in_bars, entry.get("label"))})
        return v, info

    target_s, _ = _nearest(grid, entry_s)
    first_s = phrases_samples[0][0] / sr
    # The first line lands inside the entry's first bar, or leads into it
    # by up to a beat. Snapping the first line's *nearest* bar line onto
    # the entry was a coin flip whenever the line sat mid-bar: the same
    # take, re-encoded, moved a whole bar on a 0.1 s difference, and half
    # the renders opened with the drop landing mid-line.
    beat_s = bar_s / max(1, beats_per_bar)
    window_start = target_s - PICKUP_BEATS * beat_s
    bars = int(np.ceil((window_start - first_s) / bar_s - 1e-9))
    move_s = bars * bar_s
    if bars == 0:
        info.update({"method": "section_entry", "section": entry.get("label"),
                     "entry_s": round(entry_s, 3), "target_s": round(target_s, 3),
                     "moved_bars": 0})
        return v, info
    if move_s < 0 and -move_s > first_s:
        info.update({"method": "kept", "entry_s": round(entry_s, 3),
                     "reason": "moving to the %s would cut into the first phrase"
                               % entry.get("label")})
        return v, info

    n = int(round(move_s * sr))
    if n > 0:
        v = np.vstack([np.zeros((n, v.shape[1]), dtype=np.float32), v])
    else:
        v = v[-n:]
    info.update({"method": "section_entry", "section": entry.get("label"),
                 "entry_s": round(entry_s, 3), "target_s": round(target_s, 3),
                 "moved_s": round(move_s, 4),
                 "moved_bars": int(round(move_s / bar_s))})
    log.info("  placed the vocal's first bar at the beat's %s (%.2fs; moved "
             "%+d bars)", entry.get("label"), target_s, info["moved_bars"])
    return v.astype(np.float32), info


def _extend_lattice(grid: np.ndarray, bar_s: float, t: float) -> np.ndarray:
    """The tracked bar lines, continued at the bar length to cover `t`."""
    if grid.size == 0 or bar_s <= 0:
        return grid
    out = grid
    if t < grid[0]:
        n = int(np.ceil((grid[0] - t) / bar_s)) + 1
        out = np.concatenate([grid[0] - bar_s * np.arange(n, 0, -1), out])
    if t > grid[-1]:
        n = int(np.ceil((t - grid[-1]) / bar_s)) + 1
        out = np.concatenate([out, grid[-1] + bar_s * np.arange(1, n + 1)])
    return out


def _nearest(grid: np.ndarray, t: float) -> Tuple[float, float]:
    if len(grid) == 0:
        return t, 0.0
    i = int(np.argmin(np.abs(grid - t)))
    return float(grid[i]), float(abs(grid[i] - t))


def warp_phrases_to_grid(vocal: np.ndarray, sr: int,
                         phrases_samples: Sequence[Tuple[int, int]],
                         downbeats_s: np.ndarray,
                         max_correction_s: float = 0.12,
                         enabled: bool = True) -> Tuple[np.ndarray, dict]:
    """Correct per-phrase drift by micro-stretching each phrase.

    A single global stretch cannot keep a human vocal locked to a grid for
    three minutes -- singers drift, breathe, and push or lay back. This
    nudges each phrase start toward the nearest grid position and stretches
    the gap before it to absorb the change, so the audio stays continuous.

    This is the algorithmic equivalent of Elastic Audio / Beat Detective,
    and it is what keeps a take in the pocket rather than merely starting
    in it.
    """
    v = dsp.as_2d(vocal)
    report: Dict[str, Any] = {"enabled": enabled, "phrases_corrected": 0,
              "max_correction_s": 0.0, "corrections": []}
    if not enabled or len(downbeats_s) == 0 or len(phrases_samples) < 2:
        return v, report

    grid = np.asarray(downbeats_s, dtype=np.float64)
    segments: List[np.ndarray] = []
    cursor = 0
    corrections = 0
    max_applied = 0.0

    for (start, end) in phrases_samples:
        start = int(np.clip(start, 0, len(v)))
        end = int(np.clip(end, start, len(v)))
        if start < cursor:
            continue

        gap = v[cursor:start]
        phrase = v[start:end]

        target, dist = _nearest(grid, start / sr)
        delta = float(target - start / sr)

        if abs(delta) > max_correction_s or abs(delta) < 0.004 or len(gap) < 32:
            segments.append(gap)
            segments.append(phrase)
            cursor = end
            continue

        # Absorb the correction in the silence before the phrase.
        new_gap_len = int(len(gap) + delta * sr)
        if new_gap_len < 16:
            segments.append(gap)
            segments.append(phrase)
            cursor = end
            continue

        stretch = new_gap_len / len(gap)
        if 0.4 < stretch < 2.5:
            try:
                segments.append(time_stretch(gap, sr, stretch,
                                             preserve_formants=False))
                corrections += 1
                max_applied = max(max_applied, abs(delta))
                report["corrections"].append(round(delta, 4))
            except Exception:
                segments.append(gap)
        else:
            segments.append(gap)
        segments.append(phrase)
        cursor = end

    if cursor < len(v):
        segments.append(v[cursor:])

    segments = [s for s in segments if len(s) > 0]
    if not segments:
        return v, report

    out = np.vstack(segments).astype(np.float32)
    report["phrases_corrected"] = corrections
    report["max_correction_s"] = round(max_applied, 4)
    if corrections:
        log.info("  warped %d phrases to the grid (max %.0f ms)",
                 corrections, max_applied * 1000)
    return out, report




# ─────────────────────────────────────────────────────────────────────────────
# Arrangement helpers
# ─────────────────────────────────────────────────────────────────────────────

def fit_beat_to_vocal(beat: np.ndarray, sr: int, target_len: int,
                      downbeats_s: np.ndarray,
                      sections: Optional[List[dict]] = None,
                      plan: Optional[Any] = None
                      ) -> Tuple[np.ndarray, dict]:
    """Trim or loop the beat to cover the vocal, always on bar boundaries.

    Cutting mid-bar is the most obvious tell of an automated edit, so every
    boundary here snaps to a downbeat. When the beat is too short, a
    musically coherent section is looped rather than the whole file
    repeated -- and when the plan says the beat already covers the vocal,
    the shortfall is the render's own reverb tail and is padded with
    silence rather than filled with a repeat of the intro.
    """
    b = dsp.as_2d(beat)
    report = {"action": "none", "original_len_s": round(len(b) / sr, 2),
              "target_len_s": round(target_len / sr, 2)}

    if target_len <= 0:
        return b, report
    if abs(len(b) - target_len) < sr * 0.5:
        return dsp.pad_to(b, target_len), report

    if len(b) > target_len:
        grid = np.asarray(downbeats_s, dtype=np.float64) * sr
        grid = grid[(grid > target_len * 0.55) & (grid <= len(b))]
        cut = int(grid[np.argmin(np.abs(grid - target_len))]) if grid.size else target_len
        out = b[:cut]
        report.update({"action": "trimmed_to_bar",
                       "result_len_s": round(cut / sr, 2)})
        return dsp.fade(out, sr, 0.001, 0.05), report

    # Too short. If the plan says only a tail is missing, pad it: the
    # shortfall is the render's own reverb decay, not missing music.
    # Looping here is what discarded 133 of a beat's 147 seconds and
    # repeated its first 14 ten times under a vocal it already covered.
    if plan is not None and getattr(plan, "beat_fit", None) is not None \
            and plan.beat_fit.method == "pad":
        report.update({"action": "padded",
                       "result_len_s": round(target_len / sr, 2),
                       "reason": plan.beat_fit.reason})
        return dsp.pad_to(b, target_len), report

    # Too short: loop a section.
    loop_start, loop_end = _choose_loop(b, sr, downbeats_s, sections)
    if loop_end <= loop_start:
        return dsp.pad_to(b, target_len), report

    out = b[:loop_end]
    loop = b[loop_start:loop_end]
    xf = min(int(0.02 * sr), len(loop) // 8)
    while len(out) < target_len:
        if xf > 2:
            ramp = np.linspace(0, 1, xf)[:, None]
            head = loop[:xf] * ramp + out[-xf:] * (1 - ramp)
            out = np.vstack([out[:-xf], head, loop[xf:]])
        else:
            out = np.vstack([out, loop])
    out = out[:target_len]
    report.update({"action": "looped",
                   "loop_s": [round(loop_start / sr, 2), round(loop_end / sr, 2)],
                   "result_len_s": round(len(out) / sr, 2)})
    return out, report


def _choose_loop(b: np.ndarray, sr: int, downbeats_s: np.ndarray,
                 sections: Optional[List[dict]]) -> Tuple[int, int]:
    """Pick a bar-aligned, musically sensible loop region.

    Search from the end. A beat's last full section is written to sit
    under a final chorus and loops without announcing itself; its intro
    is written to arrive once, and repeating it is the most audible edit
    the engine can make.
    """
    if sections:
        for label in ("chorus", "verse"):
            for s in reversed(sections):
                if s.get("label") == label:
                    a, z = int(s["start"] * sr), int(s["end"] * sr)
                    if z - a > sr * 4:
                        return a, min(z, len(b))
    grid = np.asarray(downbeats_s, dtype=np.float64) * sr
    grid = grid[(grid >= 0) & (grid < len(b))]
    if len(grid) >= 9:
        return int(grid[max(0, len(grid) - 9)]), int(grid[len(grid) - 1])
    return 0, len(b)
