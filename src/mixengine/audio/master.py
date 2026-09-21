"""
Mastering.

Glue compression -> tonal balance toward the beat's own reference curve ->
limiting -> loudness targeting -> true-peak safety.

Two deliberate choices:

  * The loudness target is genre-dependent, not a universal -14 LUFS.
    Modern rap and pop masters sit far hotter; delivering a drill record at
    -14 makes it sound weak next to everything around it.
  * The tonal match is coarse and partial. Matching a reference curve
    precisely makes every output sound identical and strips the beat's
    character, so the match strength is deliberately limited.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np

from ..core import audio_io
from . import dsp
from ..core.capabilities import CAPS
from ..config import CFG, GenreProfile

log = logging.getLogger("mixengine.master")


def master(mix: np.ndarray, sr: int, profile: GenreProfile,
           reference_curve: Optional[dict] = None,
           overrides: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
    """Run the master chain. Returns `(audio, report)`."""
    o = overrides or {}
    target_lufs = profile.lufs_target + float(o.get("lufs_target", 0.0))
    report: Dict = {"target_lufs": round(target_lufs, 2)}

    y = dsp.as_2d(mix)
    if y.shape[1] == 1:
        y = np.repeat(y, 2, axis=1)

    report["input_lufs"] = _r(audio_io.integrated_lufs(y, sr))
    report["input_true_peak_db"] = _r(dsp.true_peak_db(y))

    # ── 1. Remove any DC and subsonic content ─────────────────────────────
    y = dsp.highpass(y, sr, 24.0, order=2)

    # ── 2. Glue compression ───────────────────────────────────────────────
    y, glue = dsp.auto_compressor(
        y, sr, target_gr_db=CFG.mix.glue_target_gr_db,
        ratio=CFG.mix.glue_ratio, attack_ms=28.0, release_ms=180.0, knee_db=8.0)
    report["glue"] = glue

    # ── 3. Tonal balance toward the reference ─────────────────────────────
    strength = CFG.mix.tonal_match_strength
    if reference_curve and reference_curve.get("freqs") and strength > 0:
        try:
            rf = np.asarray(reference_curve["freqs"], dtype=np.float64)
            rd = np.asarray(reference_curve["db"], dtype=np.float64)
            y = dsp.spectral_tilt_match(y, rf, rd, sr, strength=strength)
            report["tonal_match"] = {"applied": True, "strength": strength}
        except Exception as e:
            log.debug("tonal match skipped (%s)", e)
            report["tonal_match"] = {"applied": False}
    else:
        report["tonal_match"] = {"applied": False}

    # ── 4-6. Converge on loudness through the limiter ─────────────────────
    #
    # A single linear gain cannot satisfy both a LUFS target and a true-peak
    # ceiling. The previous chain tried: it gained up to hit the target, then
    # trimmed down to respect the ceiling, and that trim silently undid the
    # loudness targeting with nothing re-checking afterwards. On a dense trap
    # mix the shortfall was exactly the size of the peak trim -- a 5.05 dB
    # trim produced a 5.05 dB loudness miss, delivering -13.6 LUFS against a
    # -8.5 target.
    #
    # The limiter is what absorbs the difference; that is what a limiter is
    # for. So gain is pushed *into* the limiter and the result re-measured,
    # repeating until the target is met. Convergence is quick because each
    # pass overshoots predictably.
    #
    # `max_limiting_db` is a musical budget, not a safety rail. Loudness is
    # always reachable given enough limiting, but past a few dB the cost is
    # transient snap -- and a squashed master stays squashed after the
    # platform normalises it back down. Stopping short and saying so is the
    # honest outcome.
    y, loud = _converge_loudness(y, sr, target_lufs,
                                 ceiling_db=CFG.mix.limiter_ceiling_db,
                                 tolerance_db=0.3,
                                 max_limiting_db=CFG.mix.max_limiting_db)
    report.update(loud)

    # ── 7. True-peak safety ───────────────────────────────────────────────
    # The limiter is true-peak aware, so this should now be a no-op. Anything
    # more than a hair means the limiter let an inter-sample peak through,
    # which is worth surfacing rather than quietly correcting.
    tp = dsp.true_peak_db(y)
    if tp > CFG.mix.true_peak_db:
        trim = CFG.mix.true_peak_db - tp
        y = y * dsp.db_to_lin(trim)
        report["true_peak_trim_db"] = round(trim, 2)
        if trim < -0.5:
            log.warning("true-peak trim of %.2f dB after limiting -- this costs "
                        "loudness the limiter should have absorbed", trim)

    y = dsp.fade(y, sr, 0.003, 0.05)

    report["output_lufs"] = _r(audio_io.integrated_lufs(y, sr))
    report["output_true_peak_db"] = _r(dsp.true_peak_db(y))
    report["output_peak_db"] = _r(dsp.peak_db(y))
    report["mono_loss_db"] = _r(dsp.mono_compatibility_loss_db(y))
    report["dynamic_range_db"] = _r(dsp.peak_db(y) - dsp.rms_db(y))

    log.info("  master: %.1f LUFS, %.2f dBTP, DR %.1f dB",
             report["output_lufs"] or 0.0, report["output_true_peak_db"] or 0.0,
             report["dynamic_range_db"] or 0.0)
    return y.astype(np.float32), report


def _converge_loudness(y: np.ndarray, sr: int, target_lufs: float,
                       ceiling_db: float, tolerance_db: float = 0.3,
                       max_limiting_db: float = 6.0,
                       max_passes: int = 6) -> Tuple[np.ndarray, dict]:
    """Drive the mix to `target_lufs` by pushing gain through the limiter.

    Returns `(audio, report)`. The report records how much gain was pushed
    in, how many passes it took, and -- importantly -- whether the target
    was actually reached, so the critic and the user see an honest number
    rather than a silent shortfall.

    Each pass applies the measured deficit as gain and re-limits. Loudness
    rises by less than the applied gain because the limiter is removing
    peaks, so the loop converges from below rather than oscillating.
    """
    report: Dict = {"limiter_ceiling_db": ceiling_db}
    src = dsp.as_2d(y).copy()
    current = audio_io.integrated_lufs(src, sr)
    if not np.isfinite(current):
        return _limit(src, sr, ceiling_db), {**report, "loudness_converged": False,
                                             "loudness_note": "loudness not measurable"}
    report["input_lufs_pre_limit"] = _r(current)

    # Gain that merely lifts the true peak up to the ceiling is free: the
    # limiter never engages, so nothing is squashed. Only gain applied
    # *beyond* that point costs dynamics, so the budget is charged against
    # that excess rather than against total gain -- otherwise a quiet mix
    # would exhaust its entire budget on makeup gain before the limiter had
    # done any work at all.
    tp = dsp.true_peak_db(src)
    free_gain = float(np.clip(ceiling_db - tp, -24.0, 24.0)) if np.isfinite(tp) else 0.0
    report["free_gain_db"] = round(free_gain, 2)

    limiting = 0.0
    best, best_err = None, float("inf")

    for _ in range(max_passes):
        out = _limit(src * dsp.db_to_lin(free_gain + limiting), sr, ceiling_db)
        lufs = audio_io.integrated_lufs(out, sr)
        if not np.isfinite(lufs):
            break
        err = abs(target_lufs - lufs)
        if err < best_err:
            best, best_err = out, err
        if err <= tolerance_db:
            break
        deficit = target_lufs - lufs
        if deficit > 0 and limiting >= max_limiting_db:
            break
        limiting = float(np.clip(limiting + deficit, -24.0, max_limiting_db))

    out = best if best is not None else _limit(src * dsp.db_to_lin(free_gain), sr, ceiling_db)
    final_err = best_err if best is not None else float("inf")

    report["gain_into_limiter_db"] = round(limiting, 2)
    report["total_gain_db"] = round(free_gain + limiting, 2)
    report["loudness_error_db"] = round(float(final_err), 2)
    report["loudness_converged"] = bool(final_err <= tolerance_db + 0.2)
    if not report["loudness_converged"]:
        report["loudness_note"] = (
            "target not reachable within a %.1f dB limiting budget; delivering "
            "%.1f dB under rather than crushing the dynamics"
            % (max_limiting_db, final_err))
        log.info("  loudness: %.1f dB short of target after %.1f dB of limiting",
                 final_err, limiting)
    return out, report


def _limit(y: np.ndarray, sr: int, ceiling_db: float,
           true_peak: bool = True) -> np.ndarray:
    """Limit to `ceiling_db`.

    The in-house look-ahead limiter is preferred when true-peak behaviour is
    required, because pedalboard's limiter works in the sample domain and
    lets inter-sample peaks through -- which is what forced a corrective
    trim afterwards and cost the master its loudness.
    """
    if true_peak:
        return _lookahead_limiter(y, sr, ceiling_db)
    if CAPS.pedalboard:
        try:
            from pedalboard import Pedalboard, Limiter
            board = Pedalboard([Limiter(threshold_db=float(ceiling_db),
                                        release_ms=120.0)])
            return dsp.as_2d(board(dsp.as_2d(y).T.astype(np.float32), sr).T)
        except Exception as e:
            log.debug("pedalboard limiter unavailable (%s)", e)
    return _lookahead_limiter(y, sr, ceiling_db)


def _true_peak_envelope(y2: np.ndarray, oversample: int = 4) -> np.ndarray:
    """Per-sample inter-sample peak magnitude.

    Sample peak under-reads by 1-3 dB on limited material, because the
    reconstructed waveform overshoots between samples. Encoders and D/A
    converters see that overshoot, which is why -1 dBTP is the delivery
    standard rather than -1 dBFS.

    Detecting on the oversampled signal while applying gain at base rate
    makes this a true-peak limiter. Without it the limiter satisfies its
    ceiling in the sample domain, a later true-peak check finds the ceiling
    breached anyway, and the resulting corrective trim eats exactly the
    loudness the limiter was asked to deliver.
    """
    from scipy.signal import resample_poly
    n = len(y2)
    if n < 8:
        return np.max(np.abs(y2), axis=1)

    # Processed in blocks. Oversampling a whole track at once costs roughly
    # 4x its float64 size -- around 500 MB for three minutes of stereo --
    # and the limiter runs this several times while converging on loudness.
    # Blocking bounds peak memory to a few MB at no accuracy cost, because
    # the result is a per-sample maximum and therefore has no cross-block
    # dependency. The overlap only exists to keep the resampler's filter
    # from ringing at block edges; those samples are then discarded.
    block = 1 << 18                     # ~6 s at 44.1 kHz
    overlap = 64
    out = np.empty(n, dtype=np.float64)

    for start in range(0, n, block):
        stop = min(start + block, n)
        a = max(0, start - overlap)
        b = min(n, stop + overlap)
        seg = y2[a:b]
        up = resample_poly(seg, oversample, 1, axis=0)
        peak_up = np.max(np.abs(up), axis=1)
        want = (b - a) * oversample
        if peak_up.size < want:
            peak_up = np.pad(peak_up, (0, want - peak_up.size), mode="edge")
        folded = peak_up[:want].reshape(b - a, oversample).max(axis=1)
        out[start:stop] = folded[start - a: start - a + (stop - start)]
    return out


def _lookahead_limiter(y: np.ndarray, sr: int, ceiling_db: float,
                       lookahead_ms: float = 5.0,
                       release_ms: float = 120.0) -> np.ndarray:
    """Smooth true-peak limiter with look-ahead.

    Computes the required reduction, smooths it with a release curve, then
    delays the signal to match. Smoothing before application is what stops
    a limiter from adding the distortion it exists to prevent.
    """
    y2 = dsp.as_2d(y).astype(np.float64)
    ceiling = dsp.db_to_lin(ceiling_db)
    la = max(1, int(lookahead_ms * 0.001 * sr))

    peak = _true_peak_envelope(y2)
    # Running maximum over the look-ahead window.
    kernel = la * 2 + 1
    padded = np.pad(peak, (la, la), mode="edge")
    from scipy.ndimage import maximum_filter1d
    running = maximum_filter1d(padded, size=kernel)[la:la + len(peak)]

    required = np.minimum(ceiling / np.maximum(running, 1e-9), 1.0)
    # Release smoothing: gain may drop fast, must recover slowly.
    from scipy import signal as sps
    a = float(np.exp(-1.0 / (release_ms * 0.001 * sr)))
    smoothed = sps.lfilter([1 - a], [1, -a], required)
    gain = np.minimum(required, smoothed)

    # Cosmetic smoothing to take the corners off the gain curve -- but it is
    # a zero-phase filter, so it can lift the curve *above* `required` at a
    # dip. Measured overshoot on transient material was 0.068 in linear
    # gain, which let true peaks through at +0.08 dBTP against a -1.0 dBTP
    # ceiling. A limiter that breaches its own ceiling is a delivery defect,
    # so the result is clamped back: smoothing may only ever lower gain.
    gain = np.minimum(dsp.smooth(gain, sr, 0.002), required)

    # No delay is applied to the signal. `running` is a *centred* maximum
    # over +/-la, so gain[i] already accounts for peaks ahead of sample i.
    # Delaying the audio as well shifted the gain curve off the peaks it was
    # computed for, which was the other half of the ceiling breach.
    out = y2 * gain[:, None]
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def _r(v, nd: int = 2):
    try:
        f = float(v)
        return round(f, nd) if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None
