"""
The automatic critic.

This is what makes generate-and-verify work. Without a reliable judge, the
architecture is just "generate". The critic runs three layers:

  1. Hard gates      -- objective defects: true peak, loudness, phase,
                        dropouts. Binary pass/fail with a deterministic fix.
  2. Musical gates   -- sync error, tuning error, harmonic clash, vocal
                        presence. Measured by re-analysing the *rendered*
                        audio rather than trusting the render parameters.
  3. Perceptual      -- optional learned scorers (Audiobox-Aesthetics,
                        SongEval). Used for ranking, never as ground truth:
                        the literature is clear that objective metrics
                        correlate imperfectly with human preference.

Every failure maps to a specific parameter change, so the repair loop is
deterministic rather than a retry-and-hope.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..analysis import analysis
from ..core import audio_io
from . import dsp, timing
from ..config import CFG, GenreProfile
from ..core.keys import Key

log = logging.getLogger("mixengine.critic")


@dataclass
class Gate:
    name: str
    passed: bool
    value: Optional[float] = None
    limit: Optional[float] = None
    severity: str = "error"          # error | warning | info
    message: str = ""
    repair: Optional[dict] = None    # parameter overrides that would fix it

    def to_dict(self) -> dict:
        d = asdict(self)
        # `message` explains a failure. For a gate that passed it would
        # read as one -- "true peak exceeds the limit" beside a green
        # chip -- so serialise a neutral statement instead. An info gate
        # passed by design and its message *is* the information.
        if self.passed and self.severity not in ("info", "skipped"):
            d["message"] = ("%s within limit" % self.name
                            if self.value is None or self.limit is None else
                            "%s %g within limit %g" % (self.name, self.value,
                                                       self.limit))
        return d


@dataclass
class CriticReport:
    variant: str = ""
    score: float = 0.0
    passed: bool = False
    gates: List[Gate] = field(default_factory=list)
    sub_scores: Dict[str, float] = field(default_factory=dict)
    perceptual: Dict[str, float] = field(default_factory=dict)
    repairs: Dict[str, float] = field(default_factory=dict)

    @property
    def errors(self) -> List[Gate]:
        return [g for g in self.gates if not g.passed and g.severity == "error"]

    @property
    def warnings(self) -> List[Gate]:
        return [g for g in self.gates if not g.passed and g.severity == "warning"]

    @property
    def infos(self) -> List[Gate]:
        return [g for g in self.gates if g.severity == "info"]

    def to_dict(self) -> dict:
        return {
            "variant": self.variant,
            "score": round(self.score, 4),
            "score_pct": round(self.score * 100),
            "passed": self.passed,
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
            "n_infos": len(self.infos),
            "gates": [g.to_dict() for g in self.gates],
            "sub_scores": self.sub_scores,
            "perceptual": self.perceptual,
            "repairs": self.repairs,
        }

    def summary(self) -> str:
        lines = [f"{self.variant}: {self.score*100:.0f}% "
                 f"({'PASS' if self.passed else 'FAIL'})"]
        for g in self.errors:
            lines.append(f"  FAIL {g.name}: {g.message}")
        for g in self.warnings:
            lines.append(f"  warn {g.name}: {g.message}")
        for g in self.infos:
            lines.append(f"  info {g.name}: {g.message}")
        return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═════════════════════════════════════════════════════════════════════════════

def evaluate(mix: np.ndarray, sr: int, variant: str,
             profile: GenreProfile,
             vocal_dna: Optional[dict] = None,
             beat_dna: Optional[dict] = None,
             rendered_vocal: Optional[np.ndarray] = None,
             master_report: Optional[dict] = None,
             semitone_shift: int = 0,
             tuning_report: Optional[dict] = None) -> CriticReport:
    """Evaluate a rendered mix.

    `tuning_report` is what the tuner actually did. Given it, the tuning
    gate is scored from that record rather than by re-tracking pitch over
    the finished vocal -- a second full CREPE pass that cost 197 seconds
    on a 36-second render, and that measured the singer's own intonation
    against an assumed key whenever the engine had rightly left the
    pitch alone.
    """
    r = CriticReport(variant=variant)
    c = CFG.critic
    y = dsp.as_2d(mix)

    if len(y) < sr:
        r.gates.append(Gate("length", False, len(y) / sr, 1.0, "error",
                            "output is shorter than one second"))
        return r

    # ── Layer 1: hard gates ───────────────────────────────────────────────
    tp = dsp.true_peak_db(y)
    r.gates.append(Gate(
        "true_peak", tp <= c.max_true_peak_db, round(tp, 2), c.max_true_peak_db,
        "error", f"true peak {tp:.2f} dBTP exceeds {c.max_true_peak_db} dBTP",
        repair={"lufs_target": -1.0}))

    lufs = audio_io.integrated_lufs(y, sr)
    if np.isfinite(lufs):
        target = profile.lufs_target
        delta = abs(lufs - target)
        shortfall = target - lufs
        # A master that stops short of its target with the true peak
        # already on the ceiling is following the engine's own policy
        # (`MasterConfig.max_limiting_db`): the last decibels were there
        # only by crushing the dynamics, and it declined. That is
        # information, not a defect, and it is recognisable from the audio
        # alone. Short of target with headroom left is a real miss.
        on_ceiling = tp >= c.max_true_peak_db - c.lufs_ceiling_window_db
        within_grace = (0.0 < shortfall
                        <= c.lufs_tolerance_db + c.lufs_ceiling_grace_db)
        if delta <= c.lufs_tolerance_db:
            r.gates.append(Gate(
                "loudness", True, round(lufs, 2), target, "warning",
                f"{lufs:.1f} LUFS is {delta:.1f} dB from the {target:.1f} target"))
        elif on_ceiling and within_grace:
            r.gates.append(Gate(
                "loudness", True, round(lufs, 2), target, "info",
                f"{lufs:.1f} LUFS, {shortfall:.1f} dB under the {target:.1f} "
                f"target with the true peak on the ceiling: the master kept "
                f"the dynamics rather than limit harder"))
        else:
            r.gates.append(Gate(
                "loudness", False, round(lufs, 2), target, "warning",
                f"{lufs:.1f} LUFS is {delta:.1f} dB from the {target:.1f} target"))

    mono_loss = dsp.mono_compatibility_loss_db(y)
    r.gates.append(Gate(
        "mono_compatibility", mono_loss <= c.max_mono_loss_db,
        round(mono_loss, 2), c.max_mono_loss_db, "error",
        f"{mono_loss:.1f} dB lost in mono - phase cancellation from widening",
        repair={"double_gain_db": -4.0}))

    clip_pct = 100.0 * float(np.mean(np.abs(y) >= 0.999))
    r.gates.append(Gate(
        "clipping", clip_pct < 0.01, round(clip_pct, 4), 0.01, "error",
        f"{clip_pct:.3f}% of samples are clipped", repair={"lufs_target": -1.5}))

    gap = _longest_silence(y, sr)
    r.gates.append(Gate(
        "continuity", gap <= c.max_silence_gap_s, round(gap, 2),
        c.max_silence_gap_s, "warning",
        f"{gap:.1f}s of silence - possible dropout or arrangement gap"))

    # ── Layer 2: musical gates ────────────────────────────────────────────
    harmonic = 0.7
    if beat_dna and beat_dna.get("beats"):
        sync_err = _sync_error(y, sr, beat_dna)
        if sync_err is not None:
            r.gates.append(Gate(
                "sync", sync_err <= c.max_sync_error_ms, round(sync_err, 1),
                c.max_sync_error_ms, "warning",
                f"vocal onsets are {sync_err:.0f} ms off the beat grid"))

    if tuning_report is not None:
        # The tuner already measured every note it considered. Asking a
        # second pitch tracker to grade the result adds no information
        # the engine does not already hold.
        if tuning_report.get("enabled"):
            considered = float(tuning_report.get("notes_considered") or 0)
            corrected = float(tuning_report.get("notes_corrected") or 0)
            mean_cents = float(tuning_report.get("mean_correction_cents") or 0)
            moved = (corrected / considered) if considered else 0.0
            value = mean_cents * moved
            r.gates.append(Gate(
                "tuning", value <= c.max_tuning_error_cents, round(value, 1),
                c.max_tuning_error_cents, "warning",
                f"moved {corrected:.0f} of {considered:.0f} notes, "
                f"mean {mean_cents:.0f} cents",
                repair={"tune_strength": 0.25}))
            harmonic = float(np.clip(1.0 - value / 100.0, 0.0, 1.0))
        else:
            # Tuning was deliberately skipped. The take's own intonation
            # is the artist's, not a defect this render introduced, and
            # scoring it would penalise the engine for restraint.
            r.gates.append(Gate(
                "tuning", True, None, c.max_tuning_error_cents, "skipped",
                "tuning was not applied to this render"))
            harmonic = 0.85
    elif rendered_vocal is not None and vocal_dna is not None:
        tuning = _tuning_error(rendered_vocal, sr, beat_dna, semitone_shift)
        if tuning is not None:
            r.gates.append(Gate(
                "tuning", tuning <= c.max_tuning_error_cents, round(tuning, 1),
                c.max_tuning_error_cents, "warning",
                f"median pitch deviation {tuning:.0f} cents from the scale",
                repair={"tune_strength": 0.25}))
            harmonic = float(np.clip(1.0 - tuning / 100.0, 0.0, 1.0))

    presence = _vocal_presence(y, sr, rendered_vocal)
    r.gates.append(Gate(
        "vocal_presence", presence >= c.min_vocal_presence_db,
        round(presence, 1), c.min_vocal_presence_db, "error",
        f"vocal is only {presence:.0f} dB in the mix - likely buried",
        repair={"vir_db": 1.5, "duck_depth_db": 1.0}))

    clarity = _clarity(y, sr)
    r.gates.append(Gate(
        "clarity", clarity > 0.25, round(clarity, 3), 0.25, "warning",
        f"midrange clarity {clarity:.2f} - vocal may be masked",
        repair={"mask_strength": 0.15, "duck_depth_db": 1.0}))

    # ── Layer 3: perceptual (optional) ────────────────────────────────────
    r.perceptual = _perceptual_scores(y, sr)

    # ── Combine ───────────────────────────────────────────────────────────
    n_err = len([g for g in r.gates if not g.passed and g.severity == "error"])
    n_warn = len([g for g in r.gates if not g.passed and g.severity == "warning"])
    gate_score = float(np.clip(1.0 - n_err * 0.35 - n_warn * 0.08, 0.0, 1.0))

    loudness_score = 1.0
    if np.isfinite(lufs):
        loudness_score = float(np.clip(
            1.0 - abs(lufs - profile.lufs_target) / 6.0, 0.0, 1.0))

    r.sub_scores = {
        "gates": round(gate_score, 4),
        "harmonic": round(harmonic, 4),
        "clarity": round(float(np.clip(clarity / 0.6, 0.0, 1.0)), 4),
        "loudness": round(loudness_score, 4),
    }
    r.score = float(
        c.w_gates * r.sub_scores["gates"]
        + c.w_harmonic * r.sub_scores["harmonic"]
        + c.w_clarity * r.sub_scores["clarity"]
        + c.w_loudness * r.sub_scores["loudness"])

    if r.perceptual.get("production_quality") is not None:
        # Nudge, don't dominate -- objective metrics are proxies.
        r.score = float(np.clip(
            r.score * 0.85 + r.perceptual["production_quality"] * 0.15, 0.0, 1.0))

    r.passed = n_err == 0
    r.repairs = _collect_repairs(r)
    return r


def _collect_repairs(r: CriticReport) -> Dict[str, float]:
    """Merge the repair hints from every failed gate into one override set."""
    out: Dict[str, float] = {}
    for g in r.gates:
        if g.passed or not g.repair:
            continue
        for k, v in g.repair.items():
            out[k] = out.get(k, 0.0) + float(v)
    return {k: round(v, 3) for k, v in out.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Measurements
# ─────────────────────────────────────────────────────────────────────────────

def _longest_silence(y: np.ndarray, sr: int) -> float:
    r = dsp.frame_rms(y, int(0.05 * sr), int(0.025 * sr))
    if r.size == 0:
        return 0.0
    db = dsp.lin_to_db(r)
    quiet = db < (np.percentile(db, 95) - 45.0)
    longest, run = 0, 0
    for q in quiet:
        run = run + 1 if q else 0
        longest = max(longest, run)
    return longest * 0.025


def _sync_error(mix: np.ndarray, sr: int,
                beat_dna: Optional[dict]) -> Optional[float]:
    """Re-detect onsets in the render and measure deviation from the grid.

    Measuring the *output* rather than trusting the render parameters is
    the whole point: it catches drift that accumulated during processing,
    which parameter inspection cannot.

    The grid must be the one the beat actually follows, and establishing
    that took a measurement: running this gate on the beat *by itself* --
    which should score near zero, since a track is by definition in time
    with itself -- returned 36.0 ms. Two separate errors, each worth
    naming because each produced a plausible-looking number:

    *Ideal grid instead of measured beats.* Generating slots by stepping
    from the first beat by the median beat length accumulates the tracker's
    fractional-BPM error. On this 28-second beat the ideal grid had walked
    60 ms away from the measured beats by the end -- over half a sixteenth,
    from an error far too small to notice in the BPM figure itself.
    Subdividing the measured beats instead: 36.0 -> 23.2 ms.

    *No groove.* The remaining error was the beat's own pocket. This track
    pulls its odd sixteenths about 22 ms early, consistently, across all 16
    bars. Measuring against a mathematically even grid scores that feel as
    an error -- so the gate punished a vocal for landing exactly where the
    drums land, and a vocal that passed it would have been *off* the beat.
    Adding the groove: 23.2 -> 5.3 ms.

    Both bugs pushed the same way, which is why the gate sat just above its
    own 35 ms limit and looked like a marginal sync problem rather than a
    broken measurement.
    """
    if not beat_dna:
        return None
    beats = beat_dna.get("beats") or []
    if len(beats) < 4:
        return None
    try:
        onsets = analysis.detect_onsets(mix, sr)
        if len(onsets) < 4:
            return None
        ctx = timing.TimingContext.from_beat_dna(beat_dna)
        fine = ctx.target_grid(16, apply_groove=True)
        if fine.size < 4:
            fine = analysis.subdivide(np.asarray(beats, dtype=np.float64), 4)
        if fine.size < 4:
            return None
        errors = [float(np.min(np.abs(fine - o))) for o in onsets[:200]]
        return float(np.median(errors)) * 1000.0
    except Exception:
        return None


def _tuning_error(vocal: np.ndarray, sr: int, beat_dna: Optional[dict],
                  shift: int) -> Optional[float]:
    """Median cents deviation of rendered notes from the target scale."""
    if beat_dna is None:
        return None
    key = Key.from_dict(beat_dna.get("key"))
    if key is None:
        return None
    try:
        target = key.transposed(shift)
        pitch = analysis.track_pitch(vocal, sr)
        notes = pitch.notes
        if not notes:
            return None
        scale = set(target.scale_pcs)
        devs = []
        for n in notes:
            if n.get("duration", 0) < 0.15:
                continue
            midi = float(n["midi"])
            best = min((abs(midi - cand) for cand in
                        range(int(midi) - 2, int(midi) + 3) if cand % 12 in scale),
                       default=None)
            if best is not None:
                devs.append(best * 100.0)
        return float(np.median(devs)) if devs else None
    except Exception:
        return None


def _vocal_presence(mix: np.ndarray, sr: int,
                    vocal: Optional[np.ndarray]) -> float:
    """How present the vocal is in the mix, in dB relative to the full mix."""
    if vocal is None:
        # Fall back to midrange energy as a proxy.
        band = dsp.bandpass(mix, sr, 400.0, 3500.0)
        return float(dsp.rms_db(band) - dsp.rms_db(mix))
    n = min(len(dsp.as_2d(mix)), len(dsp.as_2d(vocal)))
    if n < sr:
        return -99.0
    return float(dsp.rms_db(dsp.as_2d(vocal)[:n]) - dsp.rms_db(dsp.as_2d(mix)[:n]))


def _clarity(mix: np.ndarray, sr: int) -> float:
    """Midrange definition: how much the 1-4 kHz band stands out.

    A crowded mix buries the intelligibility band under low-mid energy;
    this ratio drops when that happens.
    """
    f, mag = dsp.long_term_spectrum(mix, sr)
    if f.size == 0:
        return 0.5
    lin = 10.0 ** (mag / 20.0)
    total = float(np.sum(lin)) + 1e-12
    speech = float(np.sum(lin[(f >= 1000) & (f <= 4000)])) / total
    mud = float(np.sum(lin[(f >= 150) & (f <= 500)])) / total
    return float(np.clip(speech / (mud + 1e-6), 0.0, 2.0))


def _perceptual_scores(y: np.ndarray, sr: int) -> Dict[str, float]:
    """Optional learned quality scorers.

    Returns an empty dict when unavailable. Weighted lightly in the final
    score even when present -- these metrics are useful as a ranker and a
    regression alarm, not as ground truth. The real signal comes from
    logging which variant users actually pick.
    """
    out: Dict[str, float] = {}
    try:
        import audiobox_aesthetics  # noqa: F401
        from audiobox_aesthetics.infer import initialize_predictor
        predictor = initialize_predictor()
        mono = dsp.to_mono(y)
        res = predictor.forward([{"path": mono, "sample_rate": sr}])
        if res:
            d = res[0]
            out["production_quality"] = float(d.get("PQ", 0)) / 10.0
            out["content_enjoyment"] = float(d.get("CE", 0)) / 10.0
            out["production_complexity"] = float(d.get("PC", 0)) / 10.0
    except Exception:
        pass
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Ranking
# ═════════════════════════════════════════════════════════════════════════════

def rank(reports: Sequence[CriticReport],
         enforce_diversity: bool = True) -> List[CriticReport]:
    """Order variants best-first, preferring passing renders.

    A variant that passes every hard gate always outranks one that fails,
    regardless of perceptual score -- a defective master is not a stylistic
    choice.
    """
    ordered = sorted(reports, key=lambda r: (r.passed, r.score), reverse=True)
    return list(ordered)


def should_repair(report, attempt: int) -> bool:
    """Whether another repair pass is worth attempting.

    Accepts a `CriticReport` or its serialised form, because the pipeline
    holds the dict. It had its own inline copy of this condition, which is
    the kind of duplication that stays correct right up until one of the
    two is changed.
    """
    if isinstance(report, CriticReport):
        passed, repairs = report.passed, report.repairs
    else:
        passed = bool((report or {}).get("passed"))
        repairs = (report or {}).get("repairs") or {}
    return (not passed
            and attempt < CFG.critic.max_repair_attempts
            and bool(repairs))
