"""
Applying a monotonic time map to audio.

A time map is a set of `(source_time, target_time)` anchors. Between
anchors the audio is stretched by whatever local ratio carries one to the
next, so a take can be pulled onto a grid without a single global ratio
being imposed on the whole performance. This is what a DAW calls elastic
audio, and it is the difference between "the first verse is in time" and
"the whole song is in time".

Two implementations. The preference between them was measured, not assumed,
and the measurement inverted the obvious answer.

**Segment-wise stretching with equal-power crossfades.** The default. Each
inter-anchor segment is stretched by its own fixed ratio and written at its
absolute target position. Because each segment is an ordinary fixed-ratio
stretch, it goes through Rubber Band's R3 engine -- which has markedly
better transient handling than R2 and is the reason `transform.py` asks for
it everywhere else.

**Rubber Band's R2 engine with `--timemap`.** The fallback, despite being
the purpose-built tool. Two findings pushed it there:

*R3 does not honour time maps.* Asking `--fine -t 1.0 -M map` for a
two-second file returns 1.58 seconds: the map is read but never reconciled
with the overall ratio. R2 returns exactly the requested length. A time map
therefore forces R2, and forfeits R3's transient handling.

*R2 smears onsets by more than the errors being corrected.* An identity
map -- source equal to target, a guaranteed no-op -- moved detected onsets
a median 11.6 ms and lost 6 of 92 of them. On a real correction its onsets
landed a median 19-23 ms from the positions they were given. The segment
path under the same test is exactly transparent on an identity map (0.0 ms,
all 92 onsets intact) and lands real corrections within 4.5-8 ms.

Anchors should sit *on onsets*. The elastic stretching then happens in the
gaps between syllables rather than in the middle of a vowel, where the ear
is far less able to hear it.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.types import FloatSeq

from ..audio import dsp

log = logging.getLogger("mixengine.align.warp")

# Local stretch ratios outside this band are audible as warble on sustained
# material regardless of the algorithm. A vocal that needs more than this to
# reach the grid was not performing the same tempo, and forcing it there
# trades a timing error for a worse artifact.
MIN_RATIO = 0.80
MAX_RATIO = 1.25

# Keep anchors at least this far apart. Below roughly a syllable, the
# stretcher has no room to distribute the change and the segment boundary
# itself becomes the artifact.
MIN_ANCHOR_GAP_S = 0.045

_RB_TIMEMAP_OK: Optional[bool] = None


def rubberband_timemap_available() -> bool:
    global _RB_TIMEMAP_OK
    if _RB_TIMEMAP_OK is not None:
        return _RB_TIMEMAP_OK
    try:
        proc = subprocess.run(["rubberband", "--full-help"], capture_output=True,
                              text=True, timeout=10)
        text = (proc.stdout or "") + (proc.stderr or "")
        _RB_TIMEMAP_OK = "--timemap" in text
    except Exception:
        _RB_TIMEMAP_OK = False
    return _RB_TIMEMAP_OK


# ═════════════════════════════════════════════════════════════════════════════
# Anchor conditioning
# ═════════════════════════════════════════════════════════════════════════════

def sanitize_anchors(src: FloatSeq, dst: FloatSeq,
                     duration_s: float, *,
                     min_ratio: float = MIN_RATIO,
                     max_ratio: float = MAX_RATIO,
                     min_gap_s: float = MIN_ANCHOR_GAP_S
                     ) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Make an anchor list safe to hand to a stretcher.

    Enforces, in order: sorted by source, endpoints pinned, minimum spacing,
    strict monotonicity in both axes, and bounded local ratios. The ratio
    limiter runs last and propagates forward, because clamping one segment
    moves every later target -- clamping them independently would produce a
    map whose targets no longer increase.
    """
    s = np.asarray(src, dtype=np.float64)
    d = np.asarray(dst, dtype=np.float64)
    info: Dict[str, Any] = {"input": int(s.size), "dropped_close": 0,
                            "ratio_clamped": 0}
    if s.size != d.size or s.size == 0:
        return (np.array([0.0, duration_s]), np.array([0.0, duration_s]), info)

    order = np.argsort(s, kind="stable")
    s, d = s[order], d[order]

    # Pin both endpoints. Without a start anchor the stretcher has no
    # reference for the lead-in; without an end anchor the tail drifts by
    # whatever the last segment's ratio happened to be.
    keep_s: List[float] = [0.0]
    keep_d: List[float] = [0.0]
    for i in range(s.size):
        if s[i] <= min_gap_s or s[i] >= duration_s - min_gap_s:
            continue
        if s[i] - keep_s[-1] < min_gap_s or d[i] - keep_d[-1] < min_gap_s:
            info["dropped_close"] += 1
            continue
        keep_s.append(float(s[i]))
        keep_d.append(float(d[i]))

    # Bound every local ratio by *dropping* the anchor that cannot be
    # reached, never by clamping it and carrying the difference forward.
    #
    # Carrying it forward was the first implementation and it was wrong in a
    # way worth recording: each clamp displaced every later anchor by the
    # unabsorbed remainder, so a take with twelve clamped segments had its
    # onsets land a median 24.5 ms from the grid positions they were
    # explicitly told to hit -- worse than not aligning at all. Anchors are
    # absolute targets. An anchor that cannot be honoured must be given up,
    # not approximated, because approximating it corrupts the ones that
    # could have been.
    out_s = [keep_s[0]]
    out_d = [keep_d[0]]
    for i in range(1, len(keep_s)):
        ds = keep_s[i] - out_s[-1]
        dd = keep_d[i] - out_d[-1]
        if ds <= 0 or dd <= 0:
            info["dropped_close"] += 1
            continue
        if not (min_ratio <= dd / ds <= max_ratio):
            info["ratio_clamped"] += 1
            continue
        out_s.append(keep_s[i])
        out_d.append(keep_d[i])

    # Final segment to the end of the file, at whatever ratio the tail needs
    # -- clamped like the rest so an outlying last anchor cannot squash it.
    tail_ds = duration_s - out_s[-1]
    if tail_ds > 1e-6:
        tail_ratio = float(np.clip(1.0, min_ratio, max_ratio))
        out_s.append(duration_s)
        out_d.append(out_d[-1] + tail_ds * tail_ratio)

    info["output"] = len(out_s)
    info["total_ratio"] = round(out_d[-1] / max(out_s[-1], 1e-9), 6)
    return np.asarray(out_s), np.asarray(out_d), info


def anchor_ratios(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    ds = np.diff(src)
    dd = np.diff(dst)
    ok = ds > 1e-9
    out = np.ones(ds.size)
    out[ok] = dd[ok] / ds[ok]
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Applying the map
# ═════════════════════════════════════════════════════════════════════════════

def warp(y: np.ndarray, sr: int, src_s: FloatSeq, dst_s: FloatSeq,
         *, preserve_formants: bool = True) -> Tuple[np.ndarray, dict]:
    """Warp `y` so that each `src_s[i]` lands at `dst_s[i]`.

    Returns `(audio, report)`. The report always records which backend ran,
    because a silent fallback from Rubber Band to the segment stretcher is a
    real change in output quality and needs to be visible afterwards.
    """
    y2 = dsp.as_2d(y)
    dur = len(y2) / float(sr)
    s, d, info = sanitize_anchors(src_s, dst_s, dur)
    report = {"method": "none", "anchors": int(s.size), **info}

    if s.size < 3:
        report["method"] = "skipped_too_few_anchors"
        return y2, report

    ratios = anchor_ratios(s, d)
    report["min_ratio"] = round(float(ratios.min()), 4)
    report["max_ratio"] = round(float(ratios.max()), 4)
    report["mean_abs_deviation_pct"] = round(
        float(np.mean(np.abs(ratios - 1.0)) * 100), 3)

    if float(np.max(np.abs(ratios - 1.0))) < 1e-4:
        report["method"] = "identity"
        return y2, report

    try:
        out = _warp_segments(y2, sr, s, d, preserve_formants)
        report["method"] = "segment_r3"
        report["out_seconds"] = round(len(out) / sr, 3)
        return out, report
    except Exception as exc:
        log.warning("segment warp failed (%s); trying the time map", exc)

    if rubberband_timemap_available():
        rb_out = _warp_rubberband(y2, sr, s, d, preserve_formants)
        if rb_out is not None:
            report["method"] = "rubberband_r2_timemap"
            report["out_seconds"] = round(len(rb_out) / sr, 3)
            return rb_out, report

    report["method"] = "failed"
    return y2, report


def _warp_rubberband(y2: np.ndarray, sr: int, s: np.ndarray, d: np.ndarray,
                     preserve_formants: bool) -> Optional[np.ndarray]:
    """Rubber Band R2 with a key-frame map.

    `--fine` is deliberately absent: the R3 engine reads a time map but does
    not reconcile it with the overall ratio, and returns an output of the
    wrong length. R2 is correct here, and correctness outranks R3's better
    transient handling when the alternative is a file that is 20% short.
    """
    import soundfile as sf

    total = float(d[-1] / max(s[-1], 1e-9))
    tmpdir = tempfile.mkdtemp(prefix="mixengine_warp_")
    src_p = os.path.join(tmpdir, "in.wav")
    dst_p = os.path.join(tmpdir, "out.wav")
    map_p = os.path.join(tmpdir, "map.txt")
    try:
        data = y2[:, 0] if y2.shape[1] == 1 else y2
        sf.write(src_p, data, sr, subtype="FLOAT")

        # Frame numbers, strictly increasing in both columns. Rubber Band
        # rejects a map that repeats or reverses, and a rounding collision
        # between two close anchors is the usual way that happens.
        lines: List[str] = []
        last_i = last_o = -1
        for si, di in zip(s, d):
            fi, fo = int(round(si * sr)), int(round(di * sr))
            if fi <= last_i or fo <= last_o:
                continue
            lines.append(f"{fi} {fo}")
            last_i, last_o = fi, fo
        if len(lines) < 2:
            return None
        with open(map_p, "w") as f:
            f.write("\n".join(lines) + "\n")

        args = ["rubberband", "-q", "--time", "%.9f" % total, "--timemap", map_p]
        if preserve_formants:
            from ..audio.transform import _rubberband_supports_formant
            if _rubberband_supports_formant():
                args.append("--formant")

        proc = subprocess.run(args + [src_p, dst_p], capture_output=True,
                              text=True, timeout=900)
        if proc.returncode != 0 or not os.path.exists(dst_p):
            log.warning("rubberband timemap failed (rc=%d): %s", proc.returncode,
                        ((proc.stderr or proc.stdout or "")[:200]).replace("\n", " "))
            return None
        out, _ = sf.read(dst_p, dtype="float32", always_2d=True)
        if out.size == 0:
            return None
        return dsp.as_2d(out)
    except Exception as exc:
        log.warning("rubberband timemap unavailable (%s); using segment warp", exc)
        return None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _warp_segments(y2: np.ndarray, sr: int, s: np.ndarray, d: np.ndarray,
                   preserve_formants: bool = True) -> np.ndarray:
    """Stretch each inter-anchor segment independently and crossfade joins.

    The crossfade is equal-power over a window short enough to stay inside
    the gap between syllables. Segments are stretched with the same routine
    the rest of the engine uses, so whatever quality is available for a
    fixed ratio -- R3, formant preservation -- is available here too. That
    is the whole advantage over a time map.
    """
    from ..audio.transform import time_stretch

    fade_n = max(8, int(sr * 0.006))
    total = int(round(d[-1] * sr)) + fade_n
    acc = np.zeros((total, y2.shape[1]), dtype=np.float32)
    filled = np.zeros(total, dtype=bool)

    for i in range(s.size - 1):
        a = max(0, min(int(round(s[i] * sr)), len(y2)))
        b = max(a, min(int(round(s[i + 1] * sr)), len(y2)))
        seg = y2[a:b]
        if seg.shape[0] < 4:
            continue
        want = max(4, int(round((d[i + 1] - d[i]) * sr)))
        ratio = want / float(seg.shape[0])
        out = time_stretch(seg, sr, ratio, preserve_formants=preserve_formants) \
            if abs(ratio - 1.0) > 1e-3 else seg
        out = _fit_length(out, want).astype(np.float32)

        # Each segment is written at its own *absolute* target position.
        # Laying them end to end instead makes every crossfade consume
        # length: an identity map through 93 segments with a 6 ms fade lost
        # half a second, which read as 70 ms of onset displacement from a
        # warp that was supposed to change nothing.
        pos = int(round(d[i] * sr))
        n = out.shape[0]
        if pos + n > total:
            n = total - pos
            out = out[:n]
        if n <= 0:
            continue
        f = min(fade_n, n // 2)
        if f > 0 and filled[pos:pos + f].any():
            t = np.linspace(0.0, 1.0, f, dtype=np.float32)[:, None]
            acc[pos:pos + f] *= np.sqrt(1.0 - t)
            acc[pos:pos + f] += out[:f] * np.sqrt(t)
            acc[pos + f:pos + n] = out[f:]
        else:
            acc[pos:pos + n] = out
        filled[pos:pos + n] = True
    return acc


def _fit_length(x: np.ndarray, n: int) -> np.ndarray:
    if x.shape[0] == n:
        return x
    if x.shape[0] > n:
        return x[:n]
    return np.pad(x, ((0, n - x.shape[0]), (0, 0)))


def map_times(src_s: FloatSeq, dst_s: FloatSeq,
              times: FloatSeq) -> np.ndarray:
    """Where `times` end up after the same warp. Linear between anchors.

    Callers need this for anything measured before the warp -- phrase
    boundaries, note events, onsets -- because re-detecting them afterwards
    costs an analysis pass and introduces a second set of estimation errors
    on top of the first.
    """
    s = np.asarray(src_s, dtype=np.float64)
    d = np.asarray(dst_s, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64)
    if s.size < 2:
        return t.copy()
    return np.interp(t, s, d)
