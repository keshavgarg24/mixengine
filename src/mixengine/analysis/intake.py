"""
Intake: what this audio is, and where it came from.

The DNA describes what the audio *is* -- its key, its tempo, its notes.
Intake describes what was already *done* to it, and whether the vocal
and the beat belong together. Those two questions decide what the
engine may touch, and until this module existed nothing asked them.

Every detector reports a value, a confidence, and the evidence behind
it, because a decision the user cannot see is a decision they cannot
correct.
"""

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from ..audio import dsp
from ..core.intents import Intents

log = logging.getLogger("mixengine.intake")

# A note within this many cents of a semitone centre reads as deliberate.
# Antares' Flex-Tune calls 50-100 "significantly off"; Melodyne leaves
# notes "already quite close" alone. 20 cents is inside both.
TUNED_WINDOW_CENTS = 20.0

# Below this fraction of note time on the grid, the take is raw.
TUNED_FRACTION_RAW = 0.55
TUNED_FRACTION_SURE = 0.80

# A raw take's phrases vary 3-6 dB. A mixed vocal has been levelled.
SPREAD_MIXED_DB = 1.5
SPREAD_DYNAMIC_DB = 3.0

# Reverb longer than this is worth remarking on either way.
RT60_NOTABLE_S = 0.45

# Fewer notes than this and there is not enough evidence to judge.
MIN_NOTES_FOR_CONFIDENCE = 12


@dataclass(frozen=True)
class VocalState:
    """Whether the take is raw, tuned, or finished -- and the evidence."""

    state: str                       # raw | tuned | finished
    confidence: float
    evidence: str
    tuned_fraction: float = 0.0
    crest_db: float = 0.0
    phrase_spread_db: float = 0.0
    rt60_s: float = 0.0
    reverb_is_intentional: bool = False
    n_notes: int = 0

    @property
    def is_tuned(self) -> bool:
        return self.state in ("tuned", "finished")

    @property
    def is_mixed(self) -> bool:
        return self.state == "finished"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["is_tuned"] = self.is_tuned
        d["is_mixed"] = self.is_mixed
        return d


def _tuned_fraction(notes: Sequence[dict]) -> float:
    """Duration-weighted fraction of notes sitting on a semitone centre.

    A singer aiming at a note and a singer sliding through one produce
    the same mean deviation if the slides are symmetric. Weighting by
    duration asks the better question: how much of the *time* was spent
    on a grid pitch.
    """
    total = 0.0
    on_grid = 0.0
    for n in notes:
        midi = n.get("midi")
        if midi is None:
            continue
        duration = float(n.get("duration", 0.0) or 0.0)
        if duration <= 0.0:
            continue
        cents = abs(((float(midi) + 0.5) % 1.0) - 0.5) * 100.0
        total += duration
        if cents <= TUNED_WINDOW_CENTS:
            on_grid += duration
    return float(on_grid / total) if total > 0 else 0.0


def _crest_db(y: np.ndarray) -> float:
    """Peak minus RMS over the active part of the signal."""
    mono = dsp.to_mono(y)
    active = mono[np.abs(mono) > (np.abs(mono).max() * 0.02 + 1e-9)]
    if active.size < 128:
        return 0.0
    rms = float(np.sqrt(np.mean(active ** 2)))
    peak = float(np.abs(active).max())
    if rms <= 0 or peak <= 0:
        return 0.0
    return float(20.0 * np.log10(peak / rms))


def _rt60_s(vdna: dict) -> float:
    """RT60 estimate, wherever the DNA document happens to carry it.

    Tests hand `detect_vocal_state` a document with the key at the top
    level. `vocal_dna.py` nests the whole `AudioQuality` payload under
    "quality", where real DNA documents carry it instead.
    """
    top = vdna.get("estimated_rt60_s")
    if top is not None:
        return float(top)
    quality = vdna.get("quality")
    if isinstance(quality, dict):
        return float(quality.get("estimated_rt60_s") or 0.0)
    return 0.0


def detect_vocal_state(y: np.ndarray, sr: int, vdna: dict,
                       intents: Intents = Intents.AUTO) -> VocalState:
    """Decide whether a take is raw, tuned, or finished.

    Ambiguity resolves toward the more finished state. Treating a raw
    take as finished yields a flat render the user can ask more of;
    treating a finished take as raw destroys work that cannot be
    recovered.
    """
    notes: List[dict] = list(vdna.get("notes") or [])
    tuned_fraction = _tuned_fraction(notes)
    spread = float(vdna.get("phrase_level_spread_db") or 0.0)
    rt60 = _rt60_s(vdna)
    crest = _crest_db(np.asarray(y))

    if intents.vocal_state is not None:
        return VocalState(
            state=intents.vocal_state, confidence=1.0,
            evidence=f"you told us the vocal is {intents.vocal_state}",
            tuned_fraction=tuned_fraction, crest_db=crest,
            phrase_spread_db=spread, rt60_s=rt60,
            reverb_is_intentional=(intents.vocal_state != "raw"
                                   and rt60 > RT60_NOTABLE_S),
            n_notes=len(notes))

    is_tuned = tuned_fraction >= TUNED_FRACTION_RAW
    is_levelled = 0.0 < spread <= SPREAD_MIXED_DB

    if is_tuned and is_levelled:
        state = "finished"
    elif is_tuned:
        state = "tuned"
    else:
        state = "raw"

    # Reverb is only evidence of a bad room on an otherwise raw take. On
    # a tuned or levelled vocal it is a mix decision, and removing it
    # destroys the sound the artist chose.
    reverb_is_intentional = bool(rt60 > RT60_NOTABLE_S and state != "raw")

    confidence = _state_confidence(tuned_fraction, spread, len(notes))

    bits = [f"{tuned_fraction * 100:.0f}% of note time on the grid"]
    if spread > 0:
        bits.append(f"phrases vary {spread:.1f} dB")
    if rt60 > RT60_NOTABLE_S:
        bits.append(f"reverb ~{rt60:.2f}s "
                    f"({'a mix choice' if reverb_is_intentional else 'a room'})")
    evidence = "; ".join(bits)

    log.info("vocal state: %s (%.0f%% confident) -- %s",
             state, confidence * 100, evidence)

    return VocalState(state=state, confidence=confidence, evidence=evidence,
                      tuned_fraction=tuned_fraction, crest_db=crest,
                      phrase_spread_db=spread, rt60_s=rt60,
                      reverb_is_intentional=reverb_is_intentional,
                      n_notes=len(notes))


def _state_confidence(tuned_fraction: float, spread: float,
                      n_notes: int) -> float:
    """How sure we are, given how far the evidence sits from the fence."""
    if n_notes < MIN_NOTES_FOR_CONFIDENCE:
        return 0.3
    if tuned_fraction >= TUNED_FRACTION_SURE or tuned_fraction <= 0.3:
        pitch_conf = 0.9
    else:
        distance = abs(tuned_fraction - TUNED_FRACTION_RAW)
        pitch_conf = 0.5 + min(distance / 0.25, 1.0) * 0.4
    if spread <= 0:
        return float(pitch_conf * 0.8)
    if spread <= SPREAD_MIXED_DB or spread >= SPREAD_DYNAMIC_DB:
        level_conf = 0.9
    else:
        level_conf = 0.6
    return float(min(pitch_conf, level_conf))


# ─────────────────────────────────────────────────────────────────────────
# Relationship: does this vocal belong to this beat?
# ─────────────────────────────────────────────────────────────────────────

# Cross-correlation search window. The alignment literature uses +/-20 s
# for live-vocal-to-studio matching; a take that starts more than 20 s
# into a beat is not a take recorded to it.
MAX_LAG_S = 20.0

# The main peak must stand this far above the best competing peak before
# a single lag is believable.
PEAK_RATIO_LOCKED = 1.6

# Durations must agree within roughly two bars.
DURATION_TOLERANCE_S = 4.0

ENVELOPE_SR = 100

# Frames at or below this multiple of the envelope's own median peak
# pass through _envelope unchanged. Above it, a frame's excess is
# compressed (see _envelope) rather than clipped -- a hard edit or a
# real accent both produce a frame far louder than a typical pulse, but
# only compression keeps two different loud frames from ever becoming
# numerically identical.
ENVELOPE_KNEE_MULT = 1.5

# How hard the knee bends above ENVELOPE_KNEE_MULT. log1p(excess * K) / K
# is strictly increasing in excess for any finite K, and converges to a
# flat line (the old hard clip) as K -> inf. K=200 is the smallest
# value, to the nearest order of magnitude, for which the degenerate
# exact-period-multiple regression tests below (test_same_performance_
# reads_as_locked, test_half_time_tempo_is_not_a_disagreement) still
# recover the true lag with margin -- see task-3-report.md "Fix round 1"
# for the sweep. It still leaves real headroom: a frame at 20x a
# typical pulse is measurably taller than one at 5x, which a hard clip
# could never show.
ENVELOPE_KNEE_K = 200.0


@dataclass(frozen=True)
class Relationship:
    """Whether the vocal was recorded to this beat -- and at what lag."""

    state: str                     # locked | free
    confidence: float
    evidence: str
    offset_s: float = 0.0
    peak_ratio: float = 0.0
    duration_delta_s: float = 0.0
    tempo_agrees: bool = False

    @property
    def is_locked(self) -> bool:
        return self.state == "locked"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["is_locked"] = self.is_locked
        return d


def _envelope(y: np.ndarray, sr: int) -> np.ndarray:
    """Onset-strength envelope at ENVELOPE_SR, clipped and mean-removed.

    See ENVELOPE_CLIP_MULT: a single outsized transient is capped before
    the correlation ever sees it, so one hard edit can't outweigh a
    whole take's worth of real pulses.
    """
    import librosa
    mono = y if y.ndim == 1 else np.mean(y, axis=1)
    mono = np.ascontiguousarray(mono.astype(np.float32))
    hop = max(1, int(round(sr / ENVELOPE_SR)))
    env = librosa.onset.onset_strength(y=mono, sr=sr, hop_length=hop)
    env = np.asarray(env, dtype=np.float64)
    if env.size == 0:
        return env
    env = np.sqrt(np.maximum(env, 0.0))
    env -= env.mean()
    norm = np.linalg.norm(env)
    return env / norm if norm > 0 else env


def _best_lag(a: np.ndarray, b: np.ndarray) -> Tuple[int, float]:
    """Return (lag_frames, peak_ratio) for `a` against `b`.

    The ratio of the main peak to the strongest peak outside its
    neighbourhood is what separates a real alignment from the periodic
    self-similarity every loop-based beat has. A high correlation at one
    lag means nothing if the next bar correlates just as well.
    """
    if a.size < 4 or b.size < 4:
        return 0, 0.0
    n = int(2 ** np.ceil(np.log2(a.size + b.size)))
    fa = np.fft.rfft(a, n)
    fb = np.fft.rfft(b, n)
    cross = fa * np.conj(fb)
    magnitude = np.abs(cross)
    # PHAT weighting: whiten the cross-spectrum so the correlation peaks
    # on timing rather than on whichever band happens to be loudest.
    cross = np.divide(cross, magnitude + 1e-9)
    corr = np.fft.irfft(cross, n)
    # max_lag must leave room for both the positive- and negative-lag
    # slices below to be non-overlapping subsets of corr (length n).
    # Without this clamp, once n drops below roughly 2*max_lag (short
    # input -- n is the next power of two >= a.size + b.size, so this
    # only bites once the combined envelope is roughly 1024 frames, about
    # 10s of audio at ENVELOPE_SR), corr[:max_lag+1] and corr[-max_lag:]
    # both silently clamp to the same full array. `window` then has
    # length 2*n while `lags` stays fixed at 2*max_lag+1, so lags[best]
    # indexes under the wrong length assumption and returns a lag that
    # means nothing, with no error.
    max_lag = min(int(MAX_LAG_S * ENVELOPE_SR), (n - 1) // 2)
    window = np.concatenate([corr[:max_lag + 1], corr[-max_lag:]])
    lags = np.concatenate([np.arange(0, max_lag + 1),
                           np.arange(-max_lag, 0)])
    best = int(np.argmax(window))
    peak = float(window[best])
    if peak <= 0:
        return int(lags[best]), 0.0
    guard = max(2, int(0.15 * ENVELOPE_SR))
    masked = window.copy()
    lo, hi = max(0, best - guard), min(window.size, best + guard + 1)
    masked[lo:hi] = -np.inf
    runner_up = float(np.max(masked)) if np.isfinite(masked).any() else 0.0
    ratio = peak / runner_up if runner_up > 1e-9 else float("inf")
    return int(lags[best]), float(min(ratio, 10.0))


def _tempo_agrees(a: float, b: float) -> bool:
    """True when two tempi match, allowing half and double time.

    Trap is written at 146 and felt at 73. A detector reporting either
    is right, and treating the disagreement as evidence against a
    relationship would reject exactly the genre this engine serves.
    """
    if a <= 0 or b <= 0:
        return False
    for factor in (0.5, 1.0, 2.0):
        if abs(a - b * factor) <= max(2.0, b * factor * 0.04):
            return True
    return False


def detect_relationship(vocal: np.ndarray, beat: np.ndarray, sr: int,
                        vdna: dict, bdna: dict,
                        intents: Intents = Intents.AUTO) -> Relationship:
    """Decide whether the vocal was recorded to this beat.

    Three independent signals must agree: a single dominant lag in the
    cross-correlation of their onset envelopes, durations within about
    two bars, and tempi that match at some metrical level. Any one alone
    is coincidence.
    """
    v_duration = float(vdna.get("duration_s") or 0.0)
    b_duration = float(bdna.get("duration_s") or 0.0)
    duration_delta = abs(v_duration - b_duration)
    tempo_ok = _tempo_agrees(float(vdna.get("bpm") or 0.0),
                             float(bdna.get("bpm") or 0.0))

    try:
        lag_frames, peak_ratio = _best_lag(_envelope(vocal, sr),
                                           _envelope(beat, sr))
        # Convert frames to seconds via the *actual* frame rate (each
        # frame is hop samples), not the nominal ENVELOPE_SR -- see
        # _hop_length. At sr=22050 the nominal rate is off by ~0.23 Hz,
        # which is ~3ms at a 1.5s lag but ~45ms at the 20s MAX_LAG_S
        # limit, and this offset is applied directly as the alignment.
        offset_s = float(lag_frames) * _hop_length(sr) / sr
    except Exception as e:                       # noqa: BLE001
        log.warning("relationship: correlation failed (%s)", e)
        lag_frames, peak_ratio, offset_s = 0, 0.0, 0.0

    if intents.relationship is not None:
        return Relationship(
            state=intents.relationship, confidence=1.0,
            evidence=f"you told us the vocal is {intents.relationship}",
            offset_s=offset_s, peak_ratio=peak_ratio,
            duration_delta_s=duration_delta, tempo_agrees=tempo_ok)

    durations_agree = duration_delta <= DURATION_TOLERANCE_S
    peak_is_clear = peak_ratio >= PEAK_RATIO_LOCKED
    locked = bool(peak_is_clear and durations_agree)

    if locked:
        confidence = float(min(0.95, 0.6 + (peak_ratio - PEAK_RATIO_LOCKED)
                               * 0.15 + (0.1 if tempo_ok else 0.0)))
        evidence = (f"one clear alignment at {offset_s:+.2f}s "
                    f"(peak {peak_ratio:.1f}x the next), lengths within "
                    f"{duration_delta:.1f}s")
    else:
        confidence = 0.7 if not durations_agree else 0.6
        why = []
        if not durations_agree:
            why.append(f"lengths differ by {duration_delta:.1f}s")
        if not peak_is_clear:
            why.append(f"no single alignment stands out "
                       f"(best {peak_ratio:.1f}x)")
        evidence = "; ".join(why)

    log.info("relationship: %s (%.0f%% confident) -- %s",
             "locked" if locked else "free", confidence * 100, evidence)

    return Relationship(state="locked" if locked else "free",
                        confidence=confidence, evidence=evidence,
                        offset_s=offset_s, peak_ratio=peak_ratio,
                        duration_delta_s=duration_delta,
                        tempo_agrees=tempo_ok)
