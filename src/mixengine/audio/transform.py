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

def align_to_downbeat(vocal: np.ndarray, sr: int,
                      phrases_samples: Sequence[Tuple[int, int]],
                      downbeats_s: np.ndarray,
                      beats_s: Optional[np.ndarray] = None,
                      allow_beat_level: bool = True) -> Tuple[np.ndarray, dict]:
    """Shift the vocal so its first phrase begins on a bar line.

    The original engine aligned the first detected vocal onset to the first
    detected beat transient. Two problems: the first detected beat is
    usually not bar 1, and aligning one point does nothing for the rest of
    the track. This aligns to an actual downbeat, and reports the offset so
    later stages can warp against the same grid.
    """
    v = dsp.as_2d(vocal)
    info = {"offset_s": 0.0, "target": None, "method": "none"}

    if len(downbeats_s) == 0 or not len(phrases_samples):
        return v, info

    first_onset_s = phrases_samples[0][0] / sr

    grid = np.asarray(downbeats_s, dtype=np.float64)
    method = "downbeat"
    if allow_beat_level and beats_s is not None and len(beats_s) > 0:
        # Prefer a downbeat, but if a beat-level position is dramatically
        # closer, the vocal probably starts on a pickup.
        db_t, db_d = _nearest(grid, first_onset_s)
        b_t, b_d = _nearest(np.asarray(beats_s, dtype=np.float64), first_onset_s)
        if b_d < db_d * 0.4:
            grid, method = np.asarray(beats_s, dtype=np.float64), "beat_pickup"

    target, _ = _nearest(grid, first_onset_s)
    offset_s = float(target - first_onset_s)

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
                      sections: Optional[List[dict]] = None
                      ) -> Tuple[np.ndarray, dict]:
    """Trim or loop the beat to cover the vocal, always on bar boundaries.

    Cutting mid-bar is the most obvious tell of an automated edit, so every
    boundary here snaps to a downbeat. When the beat is too short, a
    musically coherent section is looped rather than the whole file
    repeated.
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
    """Pick a bar-aligned, musically sensible loop region."""
    if sections:
        for label in ("chorus", "verse"):
            for s in sections:
                if s.get("label") == label:
                    a, z = int(s["start"] * sr), int(s["end"] * sr)
                    if z - a > sr * 4:
                        return a, min(z, len(b))
    grid = np.asarray(downbeats_s, dtype=np.float64) * sr
    grid = grid[(grid >= 0) & (grid < len(b))]
    if len(grid) >= 9:
        return int(grid[max(0, len(grid) // 4)]), int(grid[min(len(grid) - 1,
                                                              len(grid) // 4 + 8)])
    return 0, len(b)
