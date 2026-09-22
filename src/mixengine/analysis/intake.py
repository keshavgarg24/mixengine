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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..audio import dsp
from ..core.intents import Intents
from ..core.keys import Key, best_shift, parse_key

log = logging.getLogger("mixengine.intake")

# A note within this many cents of a semitone centre reads as deliberate.
# Antares' Flex-Tune calls 50-100 "significantly off"; Melodyne leaves
# notes "already quite close" alone. 20 cents is inside both.
TUNED_WINDOW_CENTS = 20.0

# Below this fraction of note time on the grid, the take is raw.
TUNED_FRACTION_RAW = 0.55

# A mean deviation at or under this reads as tuned whatever the fraction
# says. Both references put deliberate intonation well inside it: Antares
# calls 50-100 cents "significantly off", Melodyne leaves notes "already
# quite close" alone. No untuned singer averages 15 cents across a take.
TUNED_DEVIATION_CENTS = 22.0
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

    # Two independent readings of the same question, because either can
    # sit just the wrong side of a threshold. The fraction asks how much
    # note *time* landed on a semitone centre; the mean deviation asks
    # how far off the misses were. A take can spend 54% of its time on
    # the grid -- one point under the line -- while averaging 15 cents,
    # which no untuned singer does. Trusting the fraction alone sent a
    # mixed vocal down the full production chain and retuned 221 of its
    # 315 notes.
    deviation = float(vdna.get("tuning_deviation_cents") or 0.0)
    is_tuned = (tuned_fraction >= TUNED_FRACTION_RAW
                or (0.0 < deviation <= TUNED_DEVIATION_CENTS
                    and len(notes) >= MIN_NOTES_FOR_CONFIDENCE))
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
    if deviation > 0:
        bits.append(f"averaging {deviation:.0f} cents off")
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
# a single lag is believable on its own.
PEAK_RATIO_LOCKED = 1.6

# A lower bar for *using* the measured lag once the pair is known to be
# related by other evidence. Declaring two files related on a marginal
# peak would be a guess; reading the lag off that same peak, when they are
# already known to belong together, is just a measurement. Below this the
# correlation is noise and zero is the better answer.
PEAK_RATIO_USABLE = 1.35

# Durations must agree within roughly two bars.
DURATION_TOLERANCE_S = 4.0

# Tighter than a performance ever matches by chance: two files this close
# in length came out of the same bounce, not the same taste in tempo.
SAME_BOUNCE_DURATION_S = 0.25

# Detected tempo agreement at this precision is not a coincidence either.
# Trackers reading two unrelated files land on round-ish numbers that
# differ by tenths; the same timeline gives the same hundredths.
SAME_BOUNCE_BPM = 0.5

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


def _hop_length(sr: int) -> int:
    """Samples per envelope frame, rounded.

    sr / ENVELOPE_SR is rarely exact (220 at sr=22050, not 220.5), so
    the envelope's *actual* frame rate is sr / hop, not ENVELOPE_SR.
    _envelope and the frame-to-seconds conversion in detect_relationship
    both need this same value, or their notions of "one frame" drift
    apart.
    """
    return max(1, int(round(sr / ENVELOPE_SR)))


def _envelope(y: np.ndarray, sr: int) -> np.ndarray:
    """Onset-strength envelope at ENVELOPE_SR, compressed and mean-removed.

    Onset strength is unbounded above: a hard digital edit in test audio
    and a genuine accent -- a hard consonant, a slapped snare -- both
    produce a frame far louder than a typical pulse, often several times
    the envelope's own median. A hard ceiling treats every frame above it
    identically, which is fine for an artifact but wrong for a transient:
    it is precisely the tallest, cleanest transients that give a
    cross-correlation its sharpest peak, and flattening them to one
    shared value throws that away.

    Frames at or below ENVELOPE_KNEE_MULT times the envelope's own
    median peak pass through unchanged. Above it, the excess is
    compressed with log1p (scaled by ENVELOPE_KNEE_K) instead of
    clipped -- log1p is strictly increasing, so a taller transient always
    produces a taller, if compressed, value. Two different loud frames
    are never tied the way a hard clip ties them.
    """
    import librosa
    mono = y if y.ndim == 1 else np.mean(y, axis=1)
    mono = np.ascontiguousarray(mono.astype(np.float32))
    hop = _hop_length(sr)
    env = librosa.onset.onset_strength(y=mono, sr=sr, hop_length=hop)
    env = np.asarray(env, dtype=np.float64)
    if env.size == 0:
        return env
    positive = env[env > 0]
    if positive.size > 0:
        knee = float(np.median(positive)) * ENVELOPE_KNEE_MULT
        if knee > 0:
            excess = np.maximum(env - knee, 0.0)
            env = np.minimum(env, knee) + \
                np.log1p(excess * ENVELOPE_KNEE_K) / ENVELOPE_KNEE_K
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


def _vocal_tempo(vdna: dict, beat_bpm: float) -> Tuple[float, str]:
    """The vocal's tempo, including readings it declined to commit to.

    A take whose estimators disagree reports `bpm: 0` and "no stable
    tempo" -- the analyser is right to refuse a single answer, but the
    individual readings are still evidence and it keeps them. On the take
    that prompted this, the headline was 0 while the autocorrelation said
    146.34 against a 146.34 BPM beat, and throwing that away is what left
    the pair looking unrelated.

    Candidates are preferred in order of how directly they measure
    period, and a candidate matching the beat wins: among several
    readings of an ambiguous take, the one that agrees with the beat it
    was cut over is the one describing the same timeline.
    """
    est = vdna.get("bpm_estimators") or {}
    candidates = [
        (float(vdna.get("bpm") or 0.0), "bpm"),
        (float(vdna.get("bpm_detected") or 0.0), "detected"),
        (float(est.get("autocorrelation") or 0.0), "autocorrelation"),
        (float(est.get("histogram") or 0.0), "histogram"),
    ]
    for alt in (vdna.get("bpm_alternates") or []):
        try:
            candidates.append((float(alt), "alternate"))
        except (TypeError, ValueError):
            continue

    live = [(v, s) for v, s in candidates if v > 0]
    for value, source in live:
        if _tempo_matches_exactly(value, beat_bpm):
            return value, source
    return live[0] if live else (0.0, "none")


def _tempo_matches_exactly(a: float, b: float) -> bool:
    """True when two tempi agree to within `SAME_BOUNCE_BPM`.

    Half and double time count: trap is written at 146 and felt at 73,
    and a vocal tracker reading the syllable rate rather than the kick
    lands on the other one. Either reading describes the same timeline.
    """
    if a <= 0 or b <= 0:
        return False
    return any(abs(a - b * f) <= SAME_BOUNCE_BPM for f in (0.5, 1.0, 2.0))


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

    # Two stems exported from the same session agree on length and tempo to
    # a precision nothing else reaches. A vocal and a beat that share a
    # duration to a hundredth of a second and a tempo to a hundredth of a
    # BPM were bounced from one timeline, whatever the onset correlation
    # says -- and for a rap take it often says very little, because a
    # sparse syllabic envelope against a dense 808 pattern has no dominant
    # peak to find. Requiring the correlation to agree is what made the
    # engine call a genuine pair "free" and then quantise a performance
    # that was already in time.
    b_bpm = float(bdna.get("bpm") or 0.0)
    v_bpm, v_bpm_src = _vocal_tempo(vdna, b_bpm)
    same_bounce = bool(duration_delta <= SAME_BOUNCE_DURATION_S
                       and _tempo_matches_exactly(v_bpm, b_bpm))

    locked = bool(same_bounce or (peak_is_clear and durations_agree))

    if same_bounce and peak_ratio < PEAK_RATIO_USABLE:
        # Bounced together, and the envelopes give no trustworthy lag: a
        # sparse syllabic envelope against a dense 808 pattern correlates
        # at roughly chance however it is weighted. Stems from one
        # timeline start together, so zero is the honest answer -- and on
        # the take that prompted this it is also the one the beat's own
        # grid agrees with.
        offset_s = 0.0

    if locked:
        if same_bounce:
            confidence = 0.9
            evidence = (f"same length to {duration_delta * 1000:.0f} ms and "
                        f"same tempo ({v_bpm:.2f} vs {b_bpm:.2f} BPM, "
                        f"vocal read by {v_bpm_src}) -- bounced from "
                        f"one session")
            if peak_is_clear:
                evidence += f", aligned at {offset_s:+.2f}s"
        else:
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


# ─────────────────────────────────────────────────────────────────────────
# Key: a distribution, not a label
# ─────────────────────────────────────────────────────────────────────────

# Below this, a key estimate is not strong enough to transpose against.
KEY_CONFIDENCE_TO_TRANSPOSE = 0.55

# Below this, the tuner gets the union scale rather than a forced third.
KEY_CONFIDENCE_TO_NARROW = 0.45


@dataclass(frozen=True)
class KeyDecision:
    """Which key to work in, and whether to move the vocal at all."""

    key: Optional["Key"]
    semitone_shift: int
    confidence: float
    evidence: str
    families_agree: bool = False
    scale_pcs: Tuple[int, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key.to_dict() if self.key else None,
                "semitone_shift": self.semitone_shift,
                "confidence": round(float(self.confidence), 3),
                "evidence": self.evidence,
                "families_agree": self.families_agree,
                "scale_pcs": list(self.scale_pcs)}


def _family_root(key: "Key") -> int:
    """The relative-major root, so a key and its relative share a value.

    A minor key and its relative major contain the same seven pitch
    classes. A detector choosing between them is choosing a label, not a
    different set of notes, and the engine must not transpose across that
    choice.
    """
    return key.pc if key.mode == "major" else (key.pc + 3) % 12


def _related(a: "Key", b: "Key") -> bool:
    """True when two keys need no transposition between them.

    Three relationships qualify. Relative major/minor share all seven
    notes. A fifth apart -- C# minor over a G# minor beat -- is the
    ordinary dominant pairing. And parallel keys share a tonic: a
    detector calling a beat "C major" when the record is in C minor is
    the commonest key error there is, because the third that separates
    them is carried by the vocal and an 808 bassline does not state one.
    Transposing across any of these would move a vocal that is already
    in the right place.
    """
    if a.pc == b.pc:
        return True
    if _family_root(a) == _family_root(b):
        return True
    gap = (_family_root(a) - _family_root(b)) % 12
    return gap in (5, 7)


def decide_key(vdna: dict, bdna: dict,
               intents: Intents = Intents.AUTO) -> KeyDecision:
    """Reconcile the vocal's and the beat's key estimates.

    Key detectors disagree on roughly 60% of tracks, almost always by a
    relative-major/minor or fifth swap. Taking each side's top label and
    comparing them throws away the agreement that is usually sitting one
    row down: on the reference render the vocal read C# Minor and the
    beat G# Major, but the beat's own second candidate was G# Minor --
    the dominant of the vocal's key, and no transposition at all.

    So both candidate lists are searched for the best-scoring compatible
    pair, and confidence is reported as measured rather than asserted.
    """
    v_key = Key.from_dict(vdna.get("key"))
    b_key = Key.from_dict(bdna.get("key"))
    v_conf = float(vdna.get("key_confidence") or 0.0)
    b_conf = float(bdna.get("key_confidence") or 0.0)

    if intents.key is not None:
        stated = parse_key(intents.key)
        if stated is not None:
            return KeyDecision(
                key=stated, semitone_shift=0, confidence=1.0,
                evidence="you told us the key is %s" % stated.name,
                families_agree=True, scale_pcs=tuple(stated.scale_pcs))
        log.warning("could not parse stated key %r; measuring instead",
                    intents.key)

    if v_key is None and b_key is None:
        return KeyDecision(None, 0, 0.0, "no key could be established")
    if v_key is None and b_key is not None:
        return KeyDecision(b_key, 0, b_conf,
                           "only the beat has a key (%s)" % b_key,
                           True, tuple(b_key.scale_pcs))
    assert v_key is not None
    if b_key is None:
        return KeyDecision(v_key, 0, v_conf,
                           "only the vocal has a key (%s)" % v_key,
                           True, tuple(v_key.scale_pcs))

    pair = _best_compatible_pair(vdna, bdna, v_key, b_key)
    if pair is not None:
        v_cand, b_cand, score = pair
        # The mode comes from the vocal: a sung line states its third, a
        # bassline does not.
        chosen = Key(v_cand.pc, v_cand.mode)
        evidence = ("vocal %s and beat %s are compatible; taking the mode "
                    "from the vocal, no transposition" % (v_cand, b_cand))
        scale = tuple(chosen.scale_pcs)
        if score < KEY_CONFIDENCE_TO_NARROW:
            scale = tuple(sorted(set(v_cand.scale_pcs) | set(b_cand.scale_pcs)))
            evidence += ("; both estimates are uncertain, so the tuner gets "
                         "the full shared scale")
        return KeyDecision(chosen, 0, score, evidence, True, scale)

    if v_conf < KEY_CONFIDENCE_TO_TRANSPOSE or \
            b_conf < KEY_CONFIDENCE_TO_TRANSPOSE:
        scale = tuple(sorted(set(v_key.scale_pcs) | set(b_key.scale_pcs)))
        return KeyDecision(
            v_key, 0, float(min(v_conf, b_conf)),
            "vocal %s and beat %s disagree but at least one estimate is "
            "uncertain (vocal %.2f, beat %.2f); rendering without a "
            "transposition" % (v_key, b_key, v_conf, b_conf),
            False, scale)

    shift, _ = best_shift(v_key, b_key)
    return KeyDecision(
        b_key, int(shift), float(min(v_conf, b_conf)),
        "vocal %s and beat %s are in different keys; shifting the vocal "
        "%+d semitones" % (v_key, b_key, shift),
        False, tuple(b_key.scale_pcs))


def _candidates(dna: dict, top: "Key", conf: float) -> List[Tuple["Key", float]]:
    """A key's candidate list as (Key, score), best first.

    Falls back to the single reported key when no candidate list survived
    analysis, so callers never have to special-case the shape.
    """
    out: List[Tuple["Key", float]] = []
    for entry in (dna.get("key_candidates") or []):
        try:
            name, score = entry[0], float(entry[1])
        except (TypeError, ValueError, IndexError):
            continue
        parsed = parse_key(name)
        if parsed is not None:
            out.append((parsed, score))
    if not out:
        out = [(top, conf)]
    return out


def _best_compatible_pair(vdna: dict, bdna: dict, v_key: "Key", b_key: "Key",
                          v_conf: float = 0.0, b_conf: float = 0.0):
    """Highest-scoring (vocal, beat) candidate pair that needs no transpose.

    Scored as the product of the two candidates' own scores, so a strong
    agreement one row down beats a weak agreement at the top. The
    confidences are the fallback scores for a document that carries no
    candidate list at all -- passing zero there made every such pair
    score zero and lose to the "no pair found" case, which transposed
    keys that were already compatible.
    """
    best = None
    best_score = 0.0
    for v_cand, v_score in _candidates(vdna, v_key, v_conf):
        for b_cand, b_score in _candidates(bdna, b_key, b_conf):
            if not _related(v_cand, b_cand):
                continue
            score = float(v_score) * float(b_score)
            if score > best_score:
                best, best_score = (v_cand, b_cand), score
    if best is None:
        return None
    # Report the geometric mean: a score comparable to the inputs' own
    # confidences rather than the product, which is always smaller.
    return best[0], best[1], float(best_score ** 0.5)
