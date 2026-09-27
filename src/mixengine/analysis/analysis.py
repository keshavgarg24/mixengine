"""
Music information retrieval: rhythm, harmony, structure, pitch.

Every function here follows the same contract: return a result plus a
confidence, never raise, and degrade to a simpler method when the preferred
dependency is missing. The pipeline branches on confidence, so an honest
low-confidence answer is far more useful than a confident wrong one.

Preference order per task
-------------------------
  downbeats  madmom RNN+DBN   ->  librosa beat_track + heuristic downbeats
  f0         torchcrepe       ->  librosa.pyin
  key        note histogram (vocals) / CQT chroma (beats), both K-K scored
  structure  spectral clustering of beat-synchronous features
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.types import FloatSeq

from ..audio import dsp
from ..core.capabilities import CAPS
from ..config import CFG, ANALYSIS_SR, HOP, KK_MAJOR, KK_MINOR
from ..core.keys import Key

warnings.filterwarnings("ignore")
log = logging.getLogger("mixengine.analysis")

_KK_MAJ = np.array(KK_MAJOR)
_KK_MIN = np.array(KK_MINOR)

# Frame RMS below this is not quiet material, it is nothing. Any real
# recording -- however badly gain-staged -- sits well above it, so using it
# as an absolute floor cannot reject usable audio.
SILENCE_FLOOR_DB = -75.0


# ═════════════════════════════════════════════════════════════════════════════
# RHYTHM
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class RhythmResult:
    bpm: float = 0.0
    confidence: float = 0.0
    beats: np.ndarray = field(default_factory=lambda: np.array([]))
    downbeats: np.ndarray = field(default_factory=lambda: np.array([]))
    beats_per_bar: int = 4
    grid_stability: float = 0.0
    alternates: List[float] = field(default_factory=list)
    method: str = "none"
    bar_anchor: dict = field(default_factory=dict)

    @property
    def bar_duration_s(self) -> float:
        return 60.0 / self.bpm * self.beats_per_bar if self.bpm > 0 else 0.0

    @property
    def beat_duration_s(self) -> float:
        return 60.0 / self.bpm if self.bpm > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "bpm": round(float(self.bpm), 2),
            "confidence": round(float(self.confidence), 3),
            "beats": [round(float(b), 4) for b in self.beats],
            "downbeats": [round(float(b), 4) for b in self.downbeats],
            "beats_per_bar": int(self.beats_per_bar),
            "grid_stability": round(float(self.grid_stability), 3),
            "alternates": [round(float(a), 2) for a in self.alternates],
            "method": self.method,
            "bar_anchor": self.bar_anchor,
        }


def analyze_rhythm(y: np.ndarray, sr: int,
                   bpm_hint: Optional[float] = None) -> RhythmResult:
    """Beat and downbeat tracking.

    `bpm_hint` is the producer's tag. It biases but never overrides
    detection -- tags are frequently half/double time or simply wrong, and
    silently trusting them propagates the error through the whole render.
    """
    mono = dsp.to_mono(y).astype(np.float32)
    if len(mono) < sr:
        return RhythmResult()

    res = _rhythm_madmom(mono, sr) if CAPS.madmom else None
    if res is None or res.bpm <= 0:
        res = _rhythm_librosa(mono, sr)

    res.alternates = _tempo_alternates(res.bpm)

    # Reconcile with the producer tag.
    if bpm_hint and bpm_hint > 0 and res.bpm > 0:
        for cand in [res.bpm] + res.alternates:
            if abs(cand - bpm_hint) / bpm_hint < 0.04:
                if abs(cand - res.bpm) > 1e-6:
                    log.info("tempo: adopting %.1f (matches tag %.1f) over %.1f",
                             cand, bpm_hint, res.bpm)
                    scale = cand / res.bpm
                    res.bpm = cand
                    res.beats = res.beats / scale if len(res.beats) else res.beats
                res.confidence = min(1.0, res.confidence + 0.15)
                break

    res.grid_stability = _grid_stability(res.beats)
    if len(res.downbeats) == 0 and len(res.beats) > 0:
        res.downbeats = _infer_downbeats(mono, sr, res.beats, res.beats_per_bar)
    res.downbeats, res.bar_anchor = _anchor_bars_to_drops(
        mono, sr, res.beats, res.downbeats, res.beats_per_bar)
    return res


def _rhythm_madmom(mono: np.ndarray, sr: int) -> Optional[RhythmResult]:
    """madmom RNN + DBN. The accurate path, and the only one giving downbeats."""
    try:
        from madmom.features.downbeats import (
            RNNDownBeatProcessor, DBNDownBeatTrackingProcessor)

        if sr != 44100:
            import librosa
            mono = librosa.resample(mono, orig_sr=sr, target_sr=44100)

        act = RNNDownBeatProcessor()(mono)
        proc = DBNDownBeatTrackingProcessor(
            beats_per_bar=[3, 4], fps=100,
            min_bpm=CFG.analysis.tempo_min, max_bpm=CFG.analysis.tempo_max)
        out = proc(act)               # columns: (time, beat_position_in_bar)
        if out is None or len(out) < 4:
            return None

        beats = out[:, 0].astype(np.float64)
        positions = out[:, 1].astype(int)
        downbeats = beats[positions == 1]
        bpb = int(positions.max()) if positions.size else 4

        intervals = np.diff(beats)
        bpm = 60.0 / float(np.median(intervals)) if intervals.size else 0.0
        # Confidence from how consistent the beat activation is.
        conf = float(np.clip(1.0 - np.std(intervals) / (np.mean(intervals) + 1e-9), 0, 1))

        return RhythmResult(bpm=bpm, confidence=max(conf, 0.55), beats=beats,
                            downbeats=downbeats, beats_per_bar=bpb,
                            method="madmom")
    except Exception as e:
        log.warning("madmom rhythm failed (%s); falling back to librosa", e)
        return None


def _rhythm_librosa(mono: np.ndarray, sr: int) -> RhythmResult:
    """librosa fallback. Good tempo, no true downbeats -- inferred later."""
    import librosa
    try:
        onset_env = librosa.onset.onset_strength(y=mono, sr=sr, hop_length=HOP,
                                                 aggregate=np.median)
        tempo, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_env, sr=sr, hop_length=HOP, trim=False)
        tempo = float(np.atleast_1d(tempo)[0])
        beats = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP)

        # Confidence: how strongly the onset envelope peaks at the beats.
        conf = 0.4
        if len(beat_frames) > 2 and onset_env.size:
            idx = np.clip(beat_frames, 0, len(onset_env) - 1)
            at_beats = float(np.mean(onset_env[idx]))
            overall = float(np.mean(onset_env)) + 1e-9
            conf = float(np.clip((at_beats / overall - 1.0) * 0.6 + 0.35, 0.1, 0.85))

        return RhythmResult(bpm=tempo, confidence=conf, beats=beats,
                            downbeats=np.array([]), beats_per_bar=4,
                            method="librosa")
    except Exception as e:
        log.error("librosa rhythm failed: %s", e)
        return RhythmResult()


def _tempo_alternates(bpm: float) -> List[float]:
    """Metrically plausible alternative readings.

    Octave errors (a 140 BPM beat read as 70) are the single most common
    tempo failure, so the alternatives are carried explicitly rather than
    discarded -- the matcher tries all of them.
    """
    if bpm <= 0:
        return []
    out = []
    for f in (0.5, 2.0, 2.0 / 3.0, 1.5):
        v = bpm * f
        if CFG.analysis.tempo_min * 0.7 <= v <= CFG.analysis.tempo_max * 1.3:
            out.append(round(v, 2))
    return out


def _grid_stability(beats: np.ndarray) -> float:
    """1.0 = perfectly quantised/programmed, <0.8 = live or loose.

    Programmed beats can be stretched aggressively without artifacts;
    loose ones need gentler handling and per-phrase warping.
    """
    if len(beats) < 4:
        return 0.0
    iv = np.diff(beats)
    return float(np.clip(1.0 - (np.std(iv) / (np.mean(iv) + 1e-9)) * 4.0, 0.0, 1.0))


def _infer_downbeats(mono: np.ndarray, sr: int, beats: np.ndarray,
                     bpb: int = 4) -> np.ndarray:
    """Pick the bar-start phase when no downbeat model is available.

    Scores each of the `bpb` possible phases by low-frequency energy at
    those beats -- kick drums land on beat 1 far more often than not. Crude
    next to madmom, but far better than assuming the first beat is bar 1.
    """
    if len(beats) < bpb * 2:
        return beats[:1] if len(beats) else np.array([])

    low = dsp.lowpass(mono, sr, 150.0, order=4)[:, 0]
    env = np.abs(low)

    best_phase, best_score = 0, -np.inf
    for phase in range(bpb):
        idx = np.arange(phase, len(beats), bpb)
        samples = (beats[idx] * sr).astype(int)
        samples = samples[(samples >= 0) & (samples < len(env) - int(0.05 * sr))]
        if samples.size == 0:
            continue
        score = float(np.mean([
            np.max(env[s:s + int(0.05 * sr)]) for s in samples
        ]))
        if score > best_score:
            best_phase, best_score = phase, score
    return beats[np.arange(best_phase, len(beats), bpb)]


BAR_SLIP_TOLERANCE = 0.10   # a beat interval this far off the median is drift
BAR_SLIP_MIN_BEATS = 0.5    # a run of drift that adds up to this has lost count
DROP_JUMP_DB = 10.0         # a drop arrives suddenly...
DROP_SUSTAIN_DB = 6.0       # ...and the bar after it stays that loud


def _anchor_bars_to_drops(mono: np.ndarray, sr: int, beats: np.ndarray,
                          downbeats: np.ndarray, bpb: int
                          ) -> Tuple[np.ndarray, dict]:
    """Re-count bar-ones from the drop wherever the tracker lost count.

    madmom's beat times are reliable wherever there are drums. Through a
    drum-less intro the DBN drifts: on a 150 BPM trap beat it stretched
    eight beats into seven, so every bar-one after the drop sat on beat
    two and a vocal laid on that grid started a beat late. The RNN's
    downbeat activation was near zero throughout, so nothing inside the
    tracker could catch it.

    A drop is the strongest evidence of a bar line a produced beat has:
    the full mix arrives on the one. So wherever a run of stretched or
    compressed intervals adds up to a slipped beat, the first sudden,
    sustained arrival after it becomes bar one and the bars from there
    are counted in tracked beats. A steady tracker is trusted as it is:
    a pickup hit one beat before a correct bar line must move nothing.
    """
    beats = np.asarray(beats, dtype=np.float64)
    out = np.asarray(downbeats, dtype=np.float64)
    info: dict = {"slips": 0, "anchored": []}
    if bpb < 1 or beats.size < 4 * bpb or out.size < 2:
        return out, info
    m = np.ravel(mono)
    iv = np.diff(beats)
    step = float(np.median(iv))
    bar = step * bpb
    loose = np.abs(iv - step) > BAR_SLIP_TOLERANCE * step

    def level(a: float, b: float) -> float:
        seg = m[max(0, int(a * sr)):max(0, int(b * sr))]
        if seg.size == 0:
            return -120.0
        return 20.0 * float(np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-9))

    k = 0
    while k < iv.size:
        if not loose[k]:
            k += 1
            continue
        k0 = k
        while k < iv.size and loose[k]:
            k += 1
        slipped = float(np.sum(iv[k0:k] - step)) / step
        if abs(slipped) < BAR_SLIP_MIN_BEATS:
            continue
        info["slips"] += 1
        best: Optional[Tuple[float, int]] = None
        for i in range(max(0, k - bpb), min(beats.size, k + 2 * bpb + 1)):
            t = beats[i]
            if t < bar or (t + bar) * sr > m.size:
                continue
            jump = level(t, t + 0.15) - level(t - 0.15, t)
            sustain = level(t, t + bar) - level(t - bar, t)
            if (jump >= DROP_JUMP_DB and sustain >= DROP_SUSTAIN_DB
                    and (best is None or jump > best[0])):
                best = (jump, i)
        if best is None:
            continue
        t = float(beats[best[1]])
        if float(np.min(np.abs(out - t))) < 0.5 * step:
            continue
        out = np.concatenate([out[out < t - 0.5 * step], beats[best[1]::bpb]])
        info["anchored"].append(round(t, 3))
        log.info("rhythm: tracker slipped %+.2f beats before %.2fs; counting "
                 "bars from the drop there", slipped, t)
    return out, info


def snap_to_grid(t: float, grid: np.ndarray) -> Tuple[float, float]:
    """Nearest grid point and the distance to it, in seconds."""
    if len(grid) == 0:
        return t, 0.0
    i = int(np.argmin(np.abs(grid - t)))
    return float(grid[i]), float(abs(grid[i] - t))


# ═════════════════════════════════════════════════════════════════════════════
# HARMONY
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class KeyResult:
    key: Optional[Key] = None
    confidence: float = 0.0
    chroma: Optional[np.ndarray] = None
    candidates: List[Tuple[str, float]] = field(default_factory=list)
    method: str = "none"

    def to_dict(self) -> dict:
        return {
            "key": self.key.to_dict() if self.key else None,
            "confidence": round(float(self.confidence), 3),
            "candidates": [(n, round(float(s), 3)) for n, s in self.candidates[:5]],
            "method": self.method,
        }


def _score_profiles(pcp: np.ndarray) -> List[Tuple[Key, float]]:
    """Correlate a 12-bin pitch-class profile against K-K templates."""
    pcp = np.asarray(pcp, dtype=np.float64)
    if pcp.sum() <= 0:
        return []
    pcp = pcp / pcp.sum()

    scored: List[Tuple[Key, float]] = []
    for pc in range(12):
        for mode, template in (("major", _KK_MAJ), ("minor", _KK_MIN)):
            rolled = np.roll(template, pc)
            a = rolled - rolled.mean()
            b = pcp - pcp.mean()
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            corr = float(np.dot(a, b) / denom) if denom > 0 else 0.0
            scored.append((Key(pc, mode), corr))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def detect_key_audio(y: np.ndarray, sr: int,
                     hint: Optional[Key] = None) -> KeyResult:
    """Key detection for instrumentals, from CQT chroma of the harmonic part.

    CQT chroma on the harmonic-separated signal is meaningfully better than
    `chroma_stft` on the raw mix: percussion smears energy across all pitch
    classes and biases the correlation.
    """
    import librosa
    mono = dsp.to_mono(y).astype(np.float32)
    if len(mono) < sr:
        return KeyResult()

    try:
        if sr != ANALYSIS_SR:
            mono = librosa.resample(mono, orig_sr=sr, target_sr=ANALYSIS_SR)
        y_h = librosa.effects.harmonic(mono, margin=3.0)
        chroma = librosa.feature.chroma_cqt(y=y_h, sr=ANALYSIS_SR, hop_length=HOP)
    except Exception as e:
        log.warning("CQT chroma failed (%s); using STFT chroma", e)
        chroma = librosa.feature.chroma_stft(y=mono, sr=sr, hop_length=HOP)

    pcp = chroma.mean(axis=1)
    scored = _score_profiles(pcp)
    if not scored:
        return KeyResult()

    best, best_score = scored[0]
    second_score = scored[1][1] if len(scored) > 1 else 0.0
    gap = best_score - second_score
    conf = float(np.clip(gap / CFG.analysis.key_confidence_gap * 0.5 + best_score * 0.5,
                         0.0, 1.0))

    # A matching producer tag raises confidence; a conflicting one lowers it
    # but does not override a strong detection.
    if hint is not None:
        if hint == best:
            conf = min(1.0, conf + 0.2)
        elif conf < CFG.analysis.key_tag_override_confidence:
            for k, _score in scored[:4]:
                if k == hint:
                    best, conf = hint, max(conf, 0.55)
                    break

    return KeyResult(key=best, confidence=conf, chroma=chroma,
                     candidates=[(k.name, s) for k, s in scored[:5]],
                     method="cqt_chroma_kk")


def detect_key_from_notes(midi_notes: FloatSeq,
                          durations: Optional[FloatSeq] = None
                          ) -> KeyResult:
    """Key detection from a note list -- the right method for vocals.

    Chroma on an isolated monophonic vocal is close to noise: there is no
    harmonic context, only a melody line, and pitch glides smear the bins.
    A duration-weighted histogram of *detected note events* is clean
    evidence by comparison, which is why the vocal path goes f0 -> notes ->
    key rather than straight to chroma.
    """
    if midi_notes is None or len(midi_notes) == 0:
        return KeyResult()
    notes = np.asarray(midi_notes, dtype=np.float64)
    w = (np.ones(len(notes)) if durations is None
         else np.asarray(durations, dtype=np.float64))

    pcp = np.zeros(12)
    for n, weight in zip(notes, w):
        if np.isfinite(n):
            pcp[int(round(n)) % 12] += max(weight, 0.0)
    if pcp.sum() <= 0:
        return KeyResult()

    scored = _score_profiles(pcp)
    best, best_score = scored[0]
    second = scored[1][1] if len(scored) > 1 else 0.0
    conf = float(np.clip((best_score - second) * 4.0 + best_score * 0.4, 0.0, 1.0))
    return KeyResult(key=best, confidence=conf, chroma=pcp[:, None],
                     candidates=[(k.name, s) for k, s in scored[:5]],
                     method="note_histogram_kk")


def beat_sync_chroma(y: np.ndarray, sr: int, beats: np.ndarray) -> np.ndarray:
    """Chroma averaged within each beat -- the feature mashability uses.

    Beat-synchronous framing makes two tracks at different tempos directly
    comparable, which raw frame-wise chroma does not.
    """
    import librosa
    mono = dsp.to_mono(y).astype(np.float32)
    try:
        if sr != ANALYSIS_SR:
            mono = librosa.resample(mono, orig_sr=sr, target_sr=ANALYSIS_SR)
            sr_use = ANALYSIS_SR
        else:
            sr_use = sr
        chroma = librosa.feature.chroma_cqt(y=mono, sr=sr_use, hop_length=HOP)
        if len(beats) < 2:
            return chroma.mean(axis=1, keepdims=True)
        frames = librosa.time_to_frames(beats, sr=sr_use, hop_length=HOP)
        frames = np.clip(frames, 0, chroma.shape[1] - 1)
        return librosa.util.sync(chroma, frames, aggregate=np.median)
    except Exception as e:
        log.warning("beat-sync chroma failed: %s", e)
        return np.zeros((12, 1))


def estimate_chords(y: np.ndarray, sr: int, downbeats: np.ndarray,
                    key: Optional[Key] = None) -> List[dict]:
    """Per-bar chord estimate by template matching over triads.

    Deliberately simple -- root plus quality, one per bar. That resolution
    is what the matcher needs (to check whether sustained vocal notes are
    chord tones); full chord recognition would be more accurate but is a
    much heavier dependency for marginal benefit here.
    """
    import librosa
    if len(downbeats) < 2:
        return []
    mono = dsp.to_mono(y).astype(np.float32)
    try:
        if sr != ANALYSIS_SR:
            mono = librosa.resample(mono, orig_sr=sr, target_sr=ANALYSIS_SR)
        y_h = librosa.effects.harmonic(mono, margin=3.0)
        chroma = librosa.feature.chroma_cqt(y=y_h, sr=ANALYSIS_SR, hop_length=HOP)
    except Exception:
        return []

    templates = {}
    for pc in range(12):
        for quality, intervals in (("maj", (0, 4, 7)), ("min", (0, 3, 7))):
            v = np.zeros(12)
            for i in intervals:
                v[(pc + i) % 12] = 1.0
            templates[(pc, quality)] = v / np.linalg.norm(v)

    out: List[dict] = []
    times = librosa.frames_to_time(np.arange(chroma.shape[1]),
                                   sr=ANALYSIS_SR, hop_length=HOP)
    for bar_i in range(len(downbeats) - 1):
        t0, t1 = downbeats[bar_i], downbeats[bar_i + 1]
        m = (times >= t0) & (times < t1)
        if not np.any(m):
            continue
        seg = chroma[:, m].mean(axis=1)
        norm = np.linalg.norm(seg)
        if norm <= 0:
            continue
        seg = seg / norm
        best, best_s = None, -np.inf
        for kq, tmpl in templates.items():
            s = float(np.dot(seg, tmpl))
            if s > best_s:
                best, best_s = kq, s
        if best is not None:
            out.append({"bar": bar_i, "time": round(float(t0), 3),
                        "root": int(best[0]), "quality": best[1],
                        "confidence": round(float(best_s), 3)})
    return out


def chroma_concentration(chroma: np.ndarray, top_k: int = 3) -> np.ndarray:
    """Per-frame pitch-class concentration: how peaked each chroma frame is.

    Returns one value per frame in [0, 1]: the share of that frame's chroma
    energy held by its `top_k` strongest pitch classes.

    This is the measurement that distinguishes tonal from atonal material,
    and it must be made **per frame**. A triad sounding at one instant puts
    most of its energy in three pitch classes, so a tonal frame scores high.
    A drum hit spreads energy across all twelve, so an atonal frame scores
    near `top_k / 12`.

    Averaging chroma over time first -- as the previous implementation did
    -- destroys exactly this signal. A three-minute song visits most pitch
    classes, so its time-averaged chroma is flat whether or not any single
    moment was tonal, and a flatness test on that average reports almost
    every piece of music as atonal.
    """
    c = np.asarray(chroma, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] < 2 or c.shape[1] == 0:
        return np.zeros(0)
    c = np.maximum(c, 0.0)
    total = c.sum(axis=0)
    ok = total > 1e-9
    if not np.any(ok):
        return np.zeros(0)
    k = int(np.clip(top_k, 1, c.shape[0]))
    top = np.sort(c[:, ok], axis=0)[-k:, :].sum(axis=0)
    return top / total[ok]


def atonality_score(harmonic_ratio: float,
                    concentrations: np.ndarray,
                    n_pitch_classes: int = 12,
                    top_k: int = 3) -> float:
    """How atonal a track is, in [0, 1]. Pure function, no audio deps.

    Combines two independent pieces of evidence:

      * `harmonic_ratio` -- the share of energy HPSS assigns to the
        harmonic component. Drum-only material is overwhelmingly percussive.
      * the median per-frame chroma concentration, which says whether the
        moments of this track have identifiable pitch content at all.

    Both must point the same way for a confident verdict, which is why
    they are averaged rather than OR-ed. The previous implementation OR-ed
    them, so a single over-sensitive test could -- and did -- veto the
    entire harmonic layer on its own.
    """
    uniform = float(top_k) / float(max(n_pitch_classes, 1))
    if concentrations.size == 0:
        conc_atonal = 0.5                       # no evidence either way
    else:
        med = float(np.median(concentrations))
        # Map concentration onto atonality: at the uniform value the track
        # is maximally atonal; well above it, clearly tonal.
        conc_atonal = float(np.clip(
            1.0 - (med - uniform) / max(0.55 - uniform, 1e-6), 0.0, 1.0))

    harm_atonal = float(np.clip(1.0 - harmonic_ratio / 0.45, 0.0, 1.0))
    return float(np.clip(0.55 * conc_atonal + 0.45 * harm_atonal, 0.0, 1.0))


def is_atonal(y: np.ndarray, sr: int, threshold: float = 0.62) -> bool:
    """Detect drum-only / noise beats that have no meaningful key.

    These bypass key matching entirely rather than being assigned a
    spurious key -- they are compatible with any vocal, and forcing a key
    onto them creates false clashes.

    The cost of a false positive here is severe and easy to miss: a beat
    wrongly marked atonal contributes a flat neutral harmonic score to
    every pairing, which silently disables key matching, the Camelot wheel
    and the dissonance penalty for that beat. The threshold is therefore
    set to require real evidence rather than to catch every edge case.
    """
    import librosa
    mono = dsp.to_mono(y).astype(np.float32)
    if len(mono) < sr:
        return False
    try:
        if sr != ANALYSIS_SR:
            mono = librosa.resample(mono, orig_sr=sr, target_sr=ANALYSIS_SR)
        h, _ = librosa.effects.hpss(mono)
        harmonic_ratio = float(np.sum(h ** 2) / (np.sum(mono ** 2) + 1e-12))
        chroma = librosa.feature.chroma_cqt(y=h, sr=ANALYSIS_SR, hop_length=HOP)
        conc = chroma_concentration(chroma, top_k=3)
        score = atonality_score(harmonic_ratio, conc)
        log.debug("atonality: harmonic_ratio=%.3f median_concentration=%.3f score=%.3f",
                  harmonic_ratio,
                  float(np.median(conc)) if conc.size else float("nan"), score)
        return bool(score >= threshold)
    except Exception as e:
        log.warning("atonality detection failed (%s); assuming tonal", e)
        return False


# ═════════════════════════════════════════════════════════════════════════════
# PITCH  (vocals)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class PitchResult:
    f0: np.ndarray = field(default_factory=lambda: np.array([]))
    times: np.ndarray = field(default_factory=lambda: np.array([]))
    voiced: np.ndarray = field(default_factory=lambda: np.array([]))
    confidence: np.ndarray = field(default_factory=lambda: np.array([]))
    notes: List[dict] = field(default_factory=list)
    method: str = "none"

    @property
    def midi(self) -> np.ndarray:
        f = np.where(self.f0 > 0, self.f0, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            return 69.0 + 12.0 * np.log2(f / 440.0)

    def to_dict(self) -> dict:
        voiced_f0 = self.f0[self.voiced] if self.voiced.size else np.array([])
        return {
            "method": self.method,
            "n_notes": len(self.notes),
            "voiced_fraction": (round(float(np.mean(self.voiced)), 3)
                                if self.voiced.size else 0.0),
            "median_f0": (round(float(np.median(voiced_f0)), 2)
                          if voiced_f0.size else None),
            "notes": self.notes[:400],
        }


def track_pitch(y: np.ndarray, sr: int) -> PitchResult:
    """f0 tracking with torchcrepe preferred, librosa.pyin as fallback."""
    mono = dsp.to_mono(y).astype(np.float32)
    if len(mono) < sr * 0.2:
        return PitchResult()

    res = _pitch_torchcrepe(mono, sr) if CAPS.torchcrepe else None
    if res is None or res.f0.size == 0:
        res = _pitch_pyin(mono, sr)
    if res.f0.size:
        res.notes = _segment_notes(res)
    return res


def _decode_viterbi_precise(logits):
    """Viterbi path for the bins, local weighted mean for the cents.

    torchcrepe's own decoders each give something up. `viterbi` finds a
    smooth path but reports each bin's centre plus random dither of up to
    twenty cents, added on purpose to hide the twenty-cent quantisation;
    for a tuning engine that is noise on every frame. `weighted_argmax`
    interpolates between bins for sub-bin precision but follows the raw
    argmax, octave jumps and all. This takes the path from the first and
    the interpolation from the second: the probability-weighted mean of
    the nine bins around the Viterbi bin, exact to a few cents and never
    dithered. Identical on CPU and accelerator to a thousandth of a cent.
    """
    import torch, torchcrepe
    bins, _ = torchcrepe.decode.viterbi(logits)                 # (batch, time)
    with torch.no_grad():
        probs = torch.sigmoid(logits)                           # (batch, bins, time)
    idx = torch.arange(logits.size(1), device=logits.device)
    window = (idx[None, :, None] - bins[:, None, :]).abs() <= 4
    probs = probs * window
    # torchcrepe's own bin-to-cents mapping, minus the dither that
    # `convert.bins_to_cents` adds and cannot be asked not to.
    centres = torchcrepe.CENTS_PER_BIN * idx.float() + 1997.3794084376191
    cents = ((centres[None, :, None] * probs).sum(dim=1)
             / probs.sum(dim=1).clamp_min(1e-9))
    return bins, torchcrepe.convert.cents_to_frequency(cents)


def _pitch_torchcrepe(mono: np.ndarray, sr: int) -> Optional[PitchResult]:
    try:
        import torch, torchcrepe
        target_sr = 16000
        if sr != target_sr:
            import librosa
            mono = librosa.resample(mono, orig_sr=sr, target_sr=target_sr)
        audio = torch.from_numpy(mono[None, :]).float()

        hop = 160                                                # 10 ms
        # The accelerator first; it is where the full model is affordable
        # (`CAPS.crepe_model`). Should it fail, the retry on the CPU drops
        # to the tiny model rather than take an hour with the full one.
        model = CAPS.crepe_model
        devices = [CAPS.device] + (["cpu"] if CAPS.device != "cpu" else [])
        for device in devices:
            try:
                f0, periodicity = torchcrepe.predict(
                    audio, target_sr, hop_length=hop,
                    fmin=CFG.analysis.f0_min, fmax=CFG.analysis.f0_max,
                    model=model, batch_size=512, device=device,
                    decoder=_decode_viterbi_precise,
                    return_periodicity=True)
                break
            except Exception as e:
                if device == devices[-1]:
                    raise
                log.warning("torchcrepe on %s failed (%s); retrying on the cpu",
                            device, e)
                model = "tiny"

        f0 = f0.squeeze(0).cpu().numpy()
        per = periodicity.squeeze(0).cpu().numpy()

        # Median-filter periodicity then smooth f0 -- torchcrepe's own
        # recommended post-processing.
        try:
            per_t = torchcrepe.filter.median(torch.from_numpy(per[None, :]), 3)
            per = per_t.squeeze(0).numpy()
        except Exception:
            pass

        voiced = per > CFG.analysis.voicing_threshold
        f0 = np.where(voiced, f0, 0.0)
        times = np.arange(len(f0)) * hop / target_sr
        return PitchResult(f0=f0, times=times, voiced=voiced,
                           confidence=per,
                           method="torchcrepe" if model == "full" else "torchcrepe_tiny")
    except Exception as e:
        log.warning("torchcrepe failed (%s); falling back to pyin", e)
        return None


def _pitch_pyin(mono: np.ndarray, sr: int) -> PitchResult:
    import librosa
    try:
        f0, voiced_flag, voiced_prob = librosa.pyin(
            mono, fmin=CFG.analysis.f0_min, fmax=CFG.analysis.f0_max,
            sr=sr, hop_length=HOP, fill_na=0.0)
        f0 = np.nan_to_num(f0, nan=0.0)
        voiced = np.asarray(voiced_flag, dtype=bool)
        times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=HOP)
        return PitchResult(f0=f0, times=times, voiced=voiced,
                           confidence=np.nan_to_num(voiced_prob, nan=0.0),
                           method="pyin")
    except Exception as e:
        log.error("pyin failed: %s", e)
        return PitchResult()


def _segment_notes(p: PitchResult) -> List[dict]:
    """Group the f0 contour into discrete note events.

    Splits on unvoiced gaps and on pitch jumps larger than a semitone. The
    resulting note list is what makes reliable vocal key detection, scale-
    aware tuning, and dissonance scoring possible.
    """
    if p.f0.size == 0:
        return []
    midi = p.midi
    dt = float(np.median(np.diff(p.times))) if len(p.times) > 1 else 0.01
    min_frames = max(2, int(CFG.analysis.min_note_dur_s / max(dt, 1e-6)))

    notes: List[dict] = []
    start: Optional[int] = None
    for i in range(len(midi)):
        v = p.voiced[i] and np.isfinite(midi[i])
        if v and start is None:
            start = i
        elif start is not None:
            jump = (v and i > start
                    and abs(midi[i] - np.nanmedian(midi[start:i])) > 1.0)
            if not v or jump:
                if i - start >= min_frames:
                    seg = midi[start:i]
                    seg = seg[np.isfinite(seg)]
                    if seg.size:
                        notes.append({
                            "start": round(float(p.times[start]), 4),
                            "end": round(float(p.times[min(i, len(p.times) - 1)]), 4),
                            "duration": round(float(p.times[min(i, len(p.times) - 1)]
                                                    - p.times[start]), 4),
                            "midi": round(float(np.median(seg)), 3),
                            "pc": int(round(np.median(seg))) % 12,
                            "cents_dev": round(float(
                                (np.median(seg) - round(np.median(seg))) * 100), 1),
                            "vibrato": round(float(np.std(seg) * 100), 1),
                            "confidence": round(float(np.mean(
                                p.confidence[start:i])) if p.confidence.size else 0.5, 3),
                        })
                start = i if jump else None
    return notes


# ═════════════════════════════════════════════════════════════════════════════
# PHRASING  (vocals)
# ═════════════════════════════════════════════════════════════════════════════

def voice_activity_threshold(r_db: np.ndarray) -> Tuple[float, float]:
    """Derive open/close thresholds for voice activity from the level histogram.

    Returns `(open_db, close_db)` with `open_db > close_db` for hysteresis.

    The previous rule -- 95th percentile minus a fixed 38 dB -- assumes the
    gap between voice and silence is always 38 dB. On a clean studio take
    it is far more; on a reverberant phone recording the tail and room tone
    sit well inside 38 dB of the peak, so every frame reads as active and
    the whole take becomes one phrase. That is exactly what happened on the
    engine's only recorded real run: a 170-second vocal produced four
    phrases, one of them 57 seconds long, which silently broke the balance
    stage, level riding, warping and hook detection all at once.

    Frame levels in a vocal recording are strongly bimodal -- speech frames
    and background frames form two clusters -- so the split is found with
    Otsu's method over the level histogram rather than assumed. That adapts
    to whatever the actual gap is, and degrades to a relative rule only
    when the distribution genuinely has no valley to find.
    """
    if r_db.size < 8:
        return -40.0, -50.0

    finite = r_db[np.isfinite(r_db)]
    if finite.size < 8:
        return -40.0, -50.0

    lo, hi = float(np.percentile(finite, 1)), float(np.percentile(finite, 99))
    if hi - lo < 6.0:
        # Essentially constant level: no silence to find. Sit just under the
        # floor so the whole span reads as active rather than none of it.
        return lo - 1.0, lo - 3.0

    hist, edges = np.histogram(np.clip(finite, lo, hi), bins=64, range=(lo, hi))
    centres = (edges[:-1] + edges[1:]) / 2.0
    total = hist.sum()
    if total == 0:
        return hi - 38.0, hi - 44.0

    # Otsu: choose the split maximising between-class variance.
    w0 = np.cumsum(hist) / total
    w1 = 1.0 - w0
    m0 = np.cumsum(hist * centres) / np.maximum(np.cumsum(hist), 1)
    total_mean = float(np.sum(hist * centres) / total)
    with np.errstate(invalid="ignore", divide="ignore"):
        m1 = (total_mean - w0 * m0) / np.maximum(w1, 1e-9)
        between = w0 * w1 * (m0 - m1) ** 2
    between = np.nan_to_num(between, nan=0.0)
    split = float(centres[int(np.argmax(between))])

    # Guard the automatic split against pathological distributions: it must
    # leave real headroom below the loud material and stay above the floor.
    split = float(np.clip(split, lo + 3.0, hi - 8.0))

    # The close threshold must stay *above* the measured noise floor. If it
    # sinks below, the gate can never close: background frames all read as
    # active, the phrase runs on until a random dip in the noise, and the
    # detected end drifts hundreds of milliseconds past the real one.
    floor = float(np.percentile(finite, 5))
    close_db = max(split - 3.0, floor + 2.0)
    open_db = max(split + 1.5, close_db + 1.5)
    return open_db, close_db


def _hysteresis_activity(r_db: np.ndarray, open_db: float,
                         close_db: float) -> np.ndarray:
    """Two-threshold voice activity, so level wobble cannot chop a phrase."""
    active = np.zeros(r_db.size, dtype=bool)
    on = False
    for i, v in enumerate(r_db):
        if not on and v >= open_db:
            on = True
        elif on and v < close_db:
            on = False
        active[i] = on
    return active


def _split_overlong(regions: List[Tuple[int, int]], r_db: np.ndarray,
                    max_frames: int, min_frames: int) -> List[Tuple[int, int]]:
    """Break regions longer than a musical phrase at their quietest interior.

    A phrase is a breath-to-breath unit -- a handful of bars. Anything
    dramatically longer is a detection failure, not a long phrase, and it
    poisons every stage that treats a phrase as the unit of work. Rather
    than accepting it, split recursively at the quietest interior frame,
    which is where a breath almost certainly was.
    """
    out: List[Tuple[int, int]] = []
    stack = list(regions)
    while stack:
        s, e = stack.pop()
        if e - s <= max_frames or e - s < 2 * min_frames:
            out.append((s, e))
            continue
        # Search only the middle so the split cannot produce a sliver.
        margin = max(min_frames, int((e - s) * 0.2))
        lo, hi = s + margin, e - margin
        if hi <= lo:
            out.append((s, e))
            continue
        cut = int(lo + np.argmin(r_db[lo:hi]))
        stack.append((s, cut))
        stack.append((cut, e))
    out.sort()
    return out


def detect_phrases(y: np.ndarray, sr: int) -> List[Tuple[int, int]]:
    """Split a vocal into phrases on silence. Returns sample ranges.

    Phrases are the unit everything downstream operates on: level riding,
    per-phrase warping, downbeat placement, balance measurement over
    vocal-active regions, and arrangement all work phrase-by-phrase rather
    than on the whole take. When this stage is wrong, none of those can be
    right, which makes it one of the highest-leverage functions in the
    engine despite being one of the simplest.
    """
    mono = dsp.to_mono(y)
    if len(mono) < sr * 0.1:
        return []

    frame, hop = int(0.025 * sr), int(0.010 * sr)
    r = dsp.frame_rms(mono, frame, hop)
    r_db = dsp.lin_to_db(r)
    if r_db.size == 0:
        return []

    # Digital silence has no valley to find, and the adaptive threshold
    # would happily place itself under the floor and call the whole file one
    # phrase. There is nothing to detect here, and saying so is correct.
    if float(np.max(r_db)) < SILENCE_FLOOR_DB:
        return []

    open_db, close_db = voice_activity_threshold(r_db)
    active = _hysteresis_activity(r_db, open_db, close_db)

    # Bridge short gaps so a breath doesn't split one phrase into two.
    gap_frames = max(1, int(CFG.analysis.phrase_gap_s * sr / hop))
    active = _close_gaps(active, gap_frames)

    regions: List[Tuple[int, int]] = []
    start = None
    for i, a in enumerate(active):
        if a and start is None:
            start = i
        elif not a and start is not None:
            regions.append((start, i))
            start = None
    if start is not None:
        regions.append((start, len(active)))

    min_frames = max(1, int(CFG.analysis.min_phrase_dur_s * sr / hop))
    max_frames = max(min_frames * 4, int(CFG.analysis.max_phrase_dur_s * sr / hop))
    regions = _split_overlong(regions, r_db, max_frames, min_frames)

    pad = int(0.03 * sr)
    out: List[Tuple[int, int]] = []
    for s, e in regions:
        if e - s < min_frames:
            continue
        out.append((max(0, s * hop - pad), min(len(mono), e * hop + pad)))
    return out


def _close_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill False runs shorter than `max_gap`."""
    m = mask.copy()
    i, n = 0, len(m)
    while i < n:
        if not m[i]:
            j = i
            while j < n and not m[j]:
                j += 1
            if i > 0 and j < n and (j - i) <= max_gap:
                m[i:j] = True
            i = j
        else:
            i += 1
    return m


# A take is not all performance. Phrases closer than this belong to one
# run; the performance is the first run at least LEAD_IN_MIN_RUN_S long,
# and whatever sits before it (a count-in, talk, the noise of a recorder
# being started) is the lead-in. A lead-in carrying more than
# LEAD_IN_MAX_FRACTION of the take's active time is kept: that much sound
# is material, not a lead-in.
LEAD_IN_RUN_GAP_S = 2.0
LEAD_IN_MIN_RUN_S = 6.0
LEAD_IN_MAX_FRACTION = 0.25
LEAD_IN_ISOLATION_S = 3.0
LEAD_IN_HARMONICITY_RATIO = 0.8


def phrase_harmonicity(y: np.ndarray, sr: int,
                       phrases: List[Tuple[int, int]]) -> List[float]:
    """Median periodicity per phrase, 0 (noise) to 1 (a clean voice)."""
    if not phrases:
        return []
    frame, hop = int(0.040 * sr), int(0.010 * sr)
    h = dsp.harmonicity(y, sr, frame, hop)
    out = []
    for s, e in phrases:
        a, b = s // hop, max(s // hop + 1, e // hop)
        seg = h[a:b]
        out.append(float(np.median(seg)) if seg.size else 0.0)
    return out


def performance_span(phrases: List[Tuple[int, int]], sr: int,
                     duration_s: float,
                     harmonicity: Optional[List[float]] = None) -> dict:
    """Where the performance sits inside the take.

    A phone take of a rap started ten seconds before the first line, and
    those ten seconds -- a loud blob of the recorder's own noise -- were as
    loud as the voice. The phrase detector, which knows only level, called
    them a phrase; placement then put that blob on the beat's drop and the
    verse arrived seven bars late. Level cannot settle what is performance
    and what is lead-in; timing can. Lines of a verse come a breath apart,
    a lead-in sits alone before the first of them.

    Returns start/end in seconds, the lead-in and tail lengths, and a
    default -- "trim" or "keep" -- with the reason for it. The person who
    made the recording is asked either way; this is the answer if they
    say nothing.
    """
    ph = [(s / sr, e / sr) for s, e in phrases]
    total_active = sum(e - s for s, e in ph)
    out = {"start_s": 0.0, "end_s": round(float(duration_s), 3),
           "lead_in_s": 0.0, "tail_s": 0.0, "default": "keep",
           "reason": "no lead-in"}
    if len(ph) < 2 or total_active <= 0:
        return out

    # Group phrases into runs.
    runs: List[List[int]] = [[0]]
    for i in range(1, len(ph)):
        if ph[i][0] - ph[i - 1][1] < LEAD_IN_RUN_GAP_S:
            runs[-1].append(i)
        else:
            runs.append([i])
    span_of = lambda run: ph[run[-1]][1] - ph[run[0]][0]        # noqa: E731
    long_runs = [r for r in runs if span_of(r) >= LEAD_IN_MIN_RUN_S]
    if not long_runs:
        long_runs = [max(runs, key=span_of)]
    first, last = long_runs[0][0], long_runs[-1][-1]
    start_s, end_s = ph[first][0], ph[last][1]

    lead = list(range(first))
    tail = list(range(last + 1, len(ph)))
    lead_active = sum(ph[i][1] - ph[i][0] for i in lead)
    tail_active = sum(ph[i][1] - ph[i][0] for i in tail)
    out.update({"start_s": round(start_s, 3), "end_s": round(end_s, 3),
                "lead_in_s": round(lead_active, 3),
                "tail_s": round(tail_active, 3)})
    if not lead and not tail:
        return out

    fraction = (lead_active + tail_active) / total_active
    if fraction >= LEAD_IN_MAX_FRACTION:
        out["reason"] = ("%.0f%% of the take's sound sits outside the main "
                         "run; that is material, not a lead-in"
                         % (fraction * 100))
        return out

    gap = ph[first][0] - ph[first - 1][1] if lead else float("inf")
    voice_like = True
    if harmonicity and len(harmonicity) == len(ph) and lead:
        perf_h = float(np.median([harmonicity[i]
                                  for i in range(first, last + 1)]))
        lead_h = float(np.median([harmonicity[i] for i in lead]))
        voice_like = lead_h >= LEAD_IN_HARMONICITY_RATIO * perf_h
        out["lead_in_harmonicity"] = round(lead_h, 3)
        out["performance_harmonicity"] = round(perf_h, 3)

    if lead and (not voice_like or gap >= LEAD_IN_ISOLATION_S):
        out["default"] = "trim"
        what = "a voice" if voice_like else "noise, not a voice"
        out["reason"] = ("%.1f s of sound before the first line (%s), "
                         "separated from it by %.1f s"
                         % (lead_active, what, gap))
    elif lead:
        out["reason"] = ("%.1f s of voice before the first line, only "
                         "%.1f s ahead of it -- kept unless you say otherwise"
                         % (lead_active, gap))
    else:
        out["default"] = "trim"
        out["reason"] = ("%.1f s of sound after the last line, %.1f s "
                         "behind it" % (tail_active, ph[last + 1][0] - end_s))
    return out


NOISE_VERDICTS = (("clean", 30.0), ("light", 20.0), ("heavy", 14.0),
                  ("severe", -np.inf))


# ── Gain staging ─────────────────────────────────────────────────────────────
# The level every take is brought to before anything measures or mixes it:
# the RMS over its phrases. A phone take at -74 dBFS measured as no loudness
# at all, the balance stage applied no gain because it had nothing to
# compare, and the voice went out 70 dB under the beat with the job marked a
# success. Every threshold downstream -- the expander's, the compressor's --
# assumes a take at a working level, so it is set here, once, and said.
STAGE_TARGET_DB = -20.0
STAGE_CEILING_DB = -1.0       # the peak is never pushed past this
STAGE_DEADBAND_DB = 3.0       # closer than this is left alone
STAGE_MAX_GAIN_DB = 60.0


def stage_level(y: np.ndarray, sr: int,
                phrases: Optional[List[Tuple[int, int]]] = None
                ) -> Tuple[np.ndarray, float]:
    """Bring the take's phrase-level RMS to the working level.

    Returns the take and the gain applied in dB (0.0 when it was left
    alone). Measured over phrases so a take that is mostly silence is
    judged by its words, not its gaps.
    """
    y2 = dsp.as_2d(y)
    if len(y2) == 0:
        return y, 0.0
    if phrases is None:
        phrases = detect_phrases(y2, sr)
    mono = dsp.to_mono(y2)
    parts = [mono[s:e] for s, e in phrases if e > s] if phrases else []
    active = np.concatenate(parts) if parts else mono
    if active.size == 0 or not np.any(active):
        return y, 0.0
    level, peak = dsp.rms_db(active), dsp.peak_db(y2)
    if not (np.isfinite(level) and np.isfinite(peak)):
        return y, 0.0
    gain = min(STAGE_TARGET_DB - level, STAGE_CEILING_DB - peak)
    gain = float(np.clip(gain, -STAGE_MAX_GAIN_DB, STAGE_MAX_GAIN_DB))
    if abs(gain) < STAGE_DEADBAND_DB:
        return y, 0.0
    return (y2 * dsp.db_to_lin(gain)).astype(np.float32), round(gain, 1)


# ── Is there a voice at all ─────────────────────────────────────────────────
# A beat uploaded in the vocal slot, a full song, a test tone: each went
# through as "the vocal", was tuned and placed and mixed over the beat, and
# came back scored in the nineties. Level and pitch cannot tell a voice from
# an instrument; a voice-activity model can. Silero's is two megabytes and
# runs a minute of audio in about a second on a CPU.
# Below this share of phrase frames called speech, there is no voice. Every
# instrumental, tone and noise file measured exactly 0.0; every real take,
# including one at -74 dBFS and one under severe room noise, measured 0.4
# or more. The bar sits near the floor so unusual singing is never blocked
# on the model's uncertainty -- "accept" is the way past if it ever is.
VOICE_MIN_FRACTION = 0.05
_VAD_MODEL = None


def _vad_model():
    global _VAD_MODEL
    if _VAD_MODEL is None:
        from silero_vad import load_silero_vad
        _VAD_MODEL = load_silero_vad()
    return _VAD_MODEL


def voice_presence(y: np.ndarray, sr: int,
                   phrases: Optional[List[Tuple[int, int]]] = None
                   ) -> Optional[dict]:
    """How much of the take's sound the voice model hears as a voice.

    Returns None when the model is not installed or fails -- the check is
    then simply not made, and no question is asked. `speech_in_phrases`
    is the share of the take's own phrases (where it is loud enough to
    be performing) that the model calls speech; below VOICE_MIN_FRACTION
    the verdict is "no_voice".
    """
    if not CAPS.silero_vad:
        return None
    try:
        import librosa
        import torch
        model = _vad_model()
        model.reset_states()
        mono = dsp.to_mono(y).astype(np.float32)
        mono16 = (librosa.resample(mono, orig_sr=sr, target_sr=16000)
                  if sr != 16000 else mono)
        win = 512
        n = (len(mono16) // win) * win
        if n < win:
            return None
        probs = np.empty(n // win, dtype=np.float32)
        with torch.no_grad():
            for i in range(0, n, win):
                probs[i // win] = float(
                    model(torch.from_numpy(mono16[i:i + win]), 16000).item())
        speech = probs > 0.5
        out = {"speech_fraction": round(float(speech.mean()), 3),
               "speech_s": round(float(speech.sum()) * win / 16000.0, 2)}
        if phrases is None:
            phrases = detect_phrases(y, sr)
        mask = np.zeros(probs.size, dtype=bool)
        for s, e in phrases or []:
            a = int(s / sr * 16000 / win)
            b = int(np.ceil(e / sr * 16000 / win))
            mask[max(0, a):max(0, b)] = True
        in_phrases = float(speech[mask].mean()) if mask.any() else 0.0
        out["speech_in_phrases"] = round(in_phrases, 3)
        out["verdict"] = ("voice" if in_phrases >= VOICE_MIN_FRACTION
                          else "no_voice")
        return out
    except Exception as e:
        log.warning("voice presence unavailable (%s)", e)
        return None


def noise_verdict(y: np.ndarray, sr: int,
                  phrases: List[Tuple[int, int]]) -> dict:
    """How far the voice clears what is under it, and a word for it.

    `snr_db` is the median level of the frames inside phrases over the
    median level of the frames between them. The whole-file measure the
    quality probe reports (RMS over the 8th-percentile frame) reads a
    take whose noise never stops as merely quiet, because the quietest
    frames are still noise; this one asks how much louder the words are
    than the gaps, which is what a listener hears.

    Verdicts: clean (>= 30 dB), light (>= 20), heavy (>= 14), severe.
    A severe take cannot be made clean -- the separator and subtraction
    together buy about 12 dB -- and the person is told so. A buffer too
    small to hold both words and a gap is "unknown": nothing is measured,
    so nothing is claimed. Calling that severe told someone whose take
    was a fraction of a second long that their room was too noisy.
    """
    mono = dsp.to_mono(y)
    frame, hop = int(0.025 * sr), int(0.010 * sr)
    r_db = dsp.lin_to_db(dsp.frame_rms(mono, frame, hop))
    out = {"snr_db": 0.0, "verdict": "unknown", "active_db": None,
           "floor_db": None, "gap_fraction": 0.0}
    if r_db.size < 8:
        return out
    mask = np.zeros(r_db.size, dtype=bool)
    for s, e in phrases:
        mask[s // hop: max(s // hop + 1, e // hop)] = True
    if mask.sum() < 4 or (~mask).sum() < 4:
        # No gaps found at all: the noise never stops or the take has no
        # phrases. Fall back to the spread of the level distribution.
        active_db = float(np.percentile(r_db, 85))
        floor_db = float(np.percentile(r_db, 15))
    else:
        active_db = float(np.median(r_db[mask]))
        floor_db = float(np.median(r_db[~mask]))
    snr = active_db - floor_db
    out.update({"snr_db": round(snr, 1), "active_db": round(active_db, 1),
                "floor_db": round(floor_db, 1),
                "gap_fraction": round(float((~mask).mean()), 3)})
    for name, floor in NOISE_VERDICTS:
        if snr >= floor:
            out["verdict"] = name
            break
    return out


def detect_onsets(y: np.ndarray, sr: int) -> np.ndarray:
    """Syllable-level onsets, in seconds."""
    import librosa
    mono = dsp.to_mono(y).astype(np.float32)
    try:
        return librosa.onset.onset_detect(y=mono, sr=sr, hop_length=HOP,
                                          units="time", backtrack=True)
    except Exception:
        return np.array([])


def estimate_vocal_tempo(onsets: np.ndarray) -> Tuple[float, float, List[float]]:
    """Estimate tempo from syllable onset periodicity.

    Returns `(bpm, confidence, alternates)`. See `estimate_vocal_tempo_detailed`
    for the two estimators behind it and how they are reconciled.
    """
    d = estimate_vocal_tempo_detailed(onsets)
    return d["bpm"], d["confidence"], d["alternates"]


def _tempo_autocorrelation(onsets: np.ndarray, frame_rate: float = 200.0
                           ) -> Tuple[float, float]:
    """Tempo from the autocorrelation of the onset train. `(bpm, strength)`.

    An independent second opinion on the histogram estimate. The histogram
    asks "what gap between syllables is most common"; this asks "at what
    period does the whole pattern of syllables repeat", which is a
    different question with a different failure mode. A rap verse with a
    steady flow of sixteenths has a histogram peak at the sixteenth and an
    autocorrelation peak at the bar; a slow ballad with long gaps has a
    diffuse histogram and a clear autocorrelation. Where they agree the
    tempo is real; where they disagree, neither should be trusted alone.
    """
    if onsets.size < 6:
        return 0.0, 0.0
    span = float(onsets[-1] - onsets[0])
    if span < 2.0:
        return 0.0, 0.0
    n = int(np.ceil(span * frame_rate)) + 8
    train = np.zeros(n, dtype=np.float64)
    idx = np.clip(((onsets - onsets[0]) * frame_rate).round().astype(int), 0, n - 1)
    train[idx] = 1.0
    # Widen each onset a little so tempo drift of a few percent still
    # correlates rather than missing by a frame.
    kern = np.exp(-0.5 * (np.arange(-6, 7) / 2.5) ** 2)
    train = np.convolve(train, kern, mode="same")
    train -= train.mean()

    f = np.fft.rfft(train, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    if ac[0] <= 0:
        return 0.0, 0.0
    ac /= ac[0]

    lo = int(60.0 / CFG.analysis.tempo_max * frame_rate)      # fastest beat
    hi = int(min(2.0, span / 2.0) * frame_rate)               # slowest period
    if hi <= lo + 2:
        return 0.0, 0.0
    seg = ac[lo:hi]
    # First clear peak, key-maxima style: the tallest lag is often a bar
    # multiple of the beat, not the beat itself.
    threshold = 0.6 * float(seg.max())
    lag = -1
    for i in range(1, seg.size - 1):
        if seg[i] >= threshold and seg[i] > seg[i - 1] and seg[i] >= seg[i + 1]:
            lag = lo + i
            break
    if lag < 0:
        return 0.0, 0.0
    period = lag / frame_rate
    bpm = 60.0 / period
    while bpm < CFG.analysis.tempo_min:
        bpm *= 2
    while bpm > CFG.analysis.tempo_max:
        bpm /= 2
    return float(bpm), float(np.clip(ac[lag], 0.0, 1.0))


def _octave_related(a: float, b: float, tol: float = 0.05) -> bool:
    """Whether two tempi are the same, or a power-of-two multiple."""
    if a <= 0 or b <= 0:
        return False
    r = np.log2(a / b)
    return abs(r - round(r)) < tol and abs(round(r)) <= 2


def estimate_vocal_tempo_detailed(onsets: np.ndarray) -> Dict:
    """Two independent tempo estimates, reconciled.

    The histogram of inter-onset intervals and the autocorrelation of the
    onset train fail in different ways, so their agreement is the honest
    confidence signal. When they agree -- exactly or by an octave -- the
    combined confidence rises above what either could claim alone. When
    they disagree, confidence is cut and *both* readings are surfaced, so a
    caller that has a beat to check against can try each rather than
    trusting a number that the two methods could not agree on.

    Confidence is genuinely allowed to be near zero: an a cappella with
    rubato phrasing may have no stable tempo at all, and forcing a number
    through the pipeline in that case is what produces the worst results.
    A low value routes the render to phrase-anchored placement instead of
    grid-locked stretching.
    """
    out: Dict = {"bpm": 0.0, "confidence": 0.0, "alternates": [],
                 "histogram_bpm": 0.0, "autocorr_bpm": 0.0,
                 "agreement": "none"}
    if len(onsets) < 6:
        return out
    h_bpm, h_conf, h_alts = _tempo_histogram(np.asarray(onsets, dtype=np.float64))
    a_bpm, a_strength = _tempo_autocorrelation(np.asarray(onsets, dtype=np.float64))
    out["histogram_bpm"] = round(h_bpm, 2)
    out["autocorr_bpm"] = round(a_bpm, 2)

    if h_bpm <= 0 and a_bpm <= 0:
        return out
    if h_bpm <= 0 or a_bpm <= 0:
        bpm = h_bpm or a_bpm
        conf = min(h_conf, 0.45) if h_bpm else min(a_strength * 0.6, 0.45)
        out.update(bpm=round(bpm, 2), confidence=round(conf, 3),
                   alternates=_tempo_alternates(bpm), agreement="single")
        return out

    if _octave_related(h_bpm, a_bpm):
        # Same tempo. The histogram's reading is kept as the figure -- it
        # is refined from the actual intervals -- and confidence rises.
        conf = float(np.clip(max(h_conf, 0.3) + 0.25 * a_strength + 0.15, 0.0, 0.9))
        out.update(bpm=round(h_bpm, 2), confidence=round(conf, 3),
                   alternates=_tempo_alternates(h_bpm),
                   agreement="exact" if abs(h_bpm - a_bpm) / h_bpm < 0.05 else "octave")
        return out

    # Disagreement: the two methods heard different periodicities. Report
    # the histogram's with its confidence halved, and put the other first
    # among the alternates so a beat-grid check can try it.
    conf = float(np.clip(h_conf * 0.5, 0.0, 0.35))
    alts = [round(a_bpm, 2)] + [x for x in _tempo_alternates(h_bpm)
                                if not _octave_related(x, a_bpm)]
    out.update(bpm=round(h_bpm, 2), confidence=round(conf, 3),
               alternates=alts, agreement="disagree")
    log.info("tempo: histogram says %.1f, autocorrelation says %.1f -- "
             "confidence halved", h_bpm, a_bpm)
    return out


def _tempo_histogram(onsets: np.ndarray) -> Tuple[float, float, List[float]]:
    """The inter-onset-interval histogram estimate."""
    if len(onsets) < 6:
        return 0.0, 0.0, []

    iois = np.diff(onsets)
    iois = iois[(iois > 0.08) & (iois < 2.0)]
    if iois.size < 4:
        return 0.0, 0.0, []

    # Histogram of inter-onset intervals; the mode is the likely beat unit.
    hist, edges = np.histogram(iois, bins=60, range=(0.08, 2.0))
    if hist.max() <= 1:
        return 0.0, 0.0, []

    peak = int(np.argmax(hist))

    # Refine using the actual intervals inside the winning bin rather than
    # taking the bin's centre.
    #
    # The histogram is 32 ms wide per bin, so a centre-based reading can be
    # off by 16 ms -- which at 140 BPM is a 6 BPM error. That was not
    # hypothetical: a vocal performed exactly at 140 was measured at 134,
    # the matcher then applied a 4.3% time stretch to "correct" a tempo that
    # was already right, and the stretch destroyed the grid alignment the
    # vocal already had. A tempo error here is uniquely expensive because
    # everything downstream treats it as ground truth.
    lo, hi = edges[peak], edges[min(peak + 2, edges.size - 1)]
    lo = edges[max(peak - 1, 0)]
    in_peak = iois[(iois >= lo) & (iois < hi)]
    peak_ioi = float(np.median(in_peak)) if in_peak.size >= 3 else \
        float((edges[peak] + edges[peak + 1]) / 2)
    if peak_ioi <= 0:
        return 0.0, 0.0, []

    # Syllables usually fall on eighths or sixteenths, so scale into range.
    bpm = 60.0 / peak_ioi
    while bpm < CFG.analysis.tempo_min:
        bpm *= 2
    while bpm > CFG.analysis.tempo_max:
        bpm /= 2

    concentration = float(hist.max() / max(1, hist.sum()))
    conf = float(np.clip(concentration * 2.2, 0.0, 0.75))
    return round(bpm, 2), conf, _tempo_alternates(bpm)


def subdivide(beats: FloatSeq, subdivision_per_beat: int = 4) -> np.ndarray:
    """Expand measured beat times into a finer grid.

    Interpolating between *measured* beats rather than generating positions
    from a tempo is what keeps the grid honest: each beat carries its own
    small timing error, but those errors do not accumulate the way a
    constant-step grid's do.
    """
    b = np.asarray(beats, dtype=np.float64)
    n = max(1, int(subdivision_per_beat))
    if b.size < 2:
        return b
    out = np.empty((b.size - 1) * n + 1, dtype=np.float64)
    for k in range(n):
        out[k::n][:b.size - 1] = b[:-1] + (b[1:] - b[:-1]) * (k / n)
    out[-1] = b[-1]
    return np.sort(out)


def grid_fit_error(onsets: FloatSeq, grid: FloatSeq) -> float:
    """Median distance from these onsets to the nearest grid position.

    A direct measure of "does this performance actually fit this grid?",
    and far more trustworthy than the tempo estimate that proposed it.

    `grid` must be *measured* positions, not a tempo. An earlier version of
    this function took a BPM and generated an evenly-spaced grid, and that
    made it unusable on anything longer than a few bars: a 0.33 BPM error
    -- well inside any tracker's accuracy -- accumulates about 66 ms of
    phase over 28 seconds, which is more than half a sixteenth at 140 BPM.
    The comparison it was supposed to decide flipped entirely depending on
    whether the grid came from the true tempo or the detected one.
    """
    o = np.asarray(onsets, dtype=np.float64)
    g = np.asarray(grid, dtype=np.float64)
    if o.size < 4 or g.size < 2:
        return float("inf")
    g = np.sort(g)
    idx = np.clip(np.searchsorted(g, o), 1, g.size - 1)
    left, right = g[idx - 1], g[idx]
    return float(np.median(np.minimum(np.abs(o - left), np.abs(right - o))))


def verify_tempo(onsets: FloatSeq, detected_bpm: float,
                 reference_grid: FloatSeq,
                 tolerance: float = 0.12) -> Tuple[float, str, float]:
    """Check a detected tempo against the grid the onsets actually fit.

    Returns `(bpm, source, fit_error_s)`. `reference_grid` is the measured
    subdivision grid of the beat the vocal will sit on.

    The estimator's answer is treated as a hypothesis, not a fact, because a
    wrong tempo is uniquely expensive: everything downstream treats it as
    ground truth, and the first thing the renderer does with it is
    time-stretch the vocal. A vocal already performed at the beat's tempo
    but mis-measured a few BPM low gets stretched to "fix" an error it did
    not have -- and the stretch destroys the alignment the performance
    already had.
    """
    o = np.asarray(onsets, dtype=np.float64)
    g = np.asarray(reference_grid, dtype=np.float64)
    if o.size < 6 or g.size < 4 or detected_bpm <= 0:
        return float(detected_bpm), "detected", float("inf")

    beat_bpm = 0.0
    if g.size >= 2:
        step = float(np.median(np.diff(g)))
        if step > 0:
            beat_bpm = 60.0 / (step * 4.0)      # grid is 16ths by convention

    if beat_bpm <= 0:
        return float(detected_bpm), "detected", grid_fit_error(o, g)

    # Each candidate is a hypothesis about the vocal's *true* tempo. Under
    # that hypothesis the renderer will stretch by beat/cand, so the onsets
    # land at o * cand/beat -- and that is what gets scored against the
    # grid. The original scored `o / (cand/detected)`, which is the
    # position the onsets would take if the vocal were at `detected` and
    # were stretched to `cand`: a quantity that answers no question anyone
    # asked. On onsets already sitting on the beat's grid it moved them
    # *off* it for every candidate, so the detected tempo always won and
    # the function could never adopt the beat in the one case it exists
    # for.
    def err_under(cand: float) -> float:
        factor = cand / beat_bpm
        # Fold octave relations: a vocal at twice the beat's tempo is
        # doubletime, which the matcher handles without a stretch.
        while factor >= 1.5:
            factor /= 2.0
        while factor < 0.75:
            factor *= 2.0
        return grid_fit_error(o * factor, g)

    best_bpm, best_src = float(detected_bpm), "detected"
    best_err = err_under(detected_bpm)
    for cand, src in ((beat_bpm, "beat"), (beat_bpm * 2.0, "beat_double"),
                      (beat_bpm * 0.5, "beat_half")):
        if not (CFG.analysis.tempo_min <= cand <= CFG.analysis.tempo_max):
            continue
        err = err_under(cand)
        if err < best_err * (1.0 - tolerance):
            best_bpm, best_src, best_err = cand, "verified_" + src, err

    if best_src != "detected":
        log.info("tempo: onsets fit %.1f BPM (%s, %.0f ms) better than the "
                 "detected %.1f BPM; adopting it",
                 best_bpm, best_src, best_err * 1000, detected_bpm)
    return float(best_bpm), best_src, float(best_err)


def classify_performance_ex(pitch: PitchResult, onsets: np.ndarray,
                            duration_s: float) -> Tuple[str, float, str]:
    """rap | melodic_rap | sung | spoken, with a confidence and the reason.

    Drives materially different treatment: rap gets timing quantisation and
    no tuning; sung gets tuning and gentler timing. Applying a singer's
    chain to a rapper is one of the more audible ways to get this wrong.

    `duration_s` should be the time the voice is actually sounding, not
    the file length: a take that is a third silence read at two syllables
    a second and was called sung. A take with no pitched notes at all is
    not sung either -- it is rap or speech, or the tracker lost the voice
    under noise -- and the low confidence says which questions to ask.
    """
    if duration_s <= 0:
        return "sung", 0.0, "no audio"
    syllable_rate = len(onsets) / duration_s
    notes = pitch.notes or []
    if not notes:
        label = "rap" if syllable_rate >= 2.0 else "spoken"
        return label, 0.3, ("no sustained pitch found; %.1f syllables a "
                            "second" % syllable_rate)

    durations = np.array([n["duration"] for n in notes])
    median_note = float(np.median(durations)) if durations.size else 0.0
    midis = np.array([n["midi"] for n in notes])
    pitch_spread = float(np.std(midis)) if midis.size > 1 else 0.0
    sustained = float(np.mean(durations > 0.30)) if durations.size else 0.0
    facts = ("%.1f syllables a second, notes %.0f ms long, %.0f%% held"
             % (syllable_rate, median_note * 1000, sustained * 100))

    if syllable_rate > 3.6 and median_note < 0.16:
        return ("rap" if pitch_spread < 2.6 else "melodic_rap"), 0.85, facts
    if sustained > 0.34 and pitch_spread > 2.0:
        return "sung", 0.85, facts
    if syllable_rate > 2.6:
        return "melodic_rap", 0.5, facts
    return "sung", 0.45, facts


def classify_performance(pitch: PitchResult, onsets: np.ndarray,
                         duration_s: float) -> str:
    """rap | melodic_rap | sung | spoken. See `classify_performance_ex`."""
    return classify_performance_ex(pitch, onsets, duration_s)[0]


# ═════════════════════════════════════════════════════════════════════════════
# STRUCTURE
# ═════════════════════════════════════════════════════════════════════════════

def analyze_structure(y: np.ndarray, sr: int, beats: np.ndarray,
                      downbeats: np.ndarray) -> List[dict]:
    """Segment into labelled sections using beat-synchronous features.

    Builds a combined chroma+MFCC representation, clusters beat-synchronous
    frames, then labels segments by relative energy: the loudest recurring
    material becomes the chorus, the quietest the intro/outro. This is the
    map the arranger uses to decide where the hook goes.
    """
    import librosa
    if len(beats) < 8:
        return _fallback_sections(y, sr, downbeats)

    mono = dsp.to_mono(y).astype(np.float32)
    try:
        if sr != ANALYSIS_SR:
            mono = librosa.resample(mono, orig_sr=sr, target_sr=ANALYSIS_SR)
        sr_use = ANALYSIS_SR

        chroma = librosa.feature.chroma_cqt(y=mono, sr=sr_use, hop_length=HOP)
        mfcc = librosa.feature.mfcc(y=mono, sr=sr_use, hop_length=HOP, n_mfcc=13)
        rms = librosa.feature.rms(y=mono, hop_length=HOP)

        frames = np.clip(librosa.time_to_frames(beats, sr=sr_use, hop_length=HOP),
                         0, chroma.shape[1] - 1)
        cs = librosa.util.sync(chroma, frames, aggregate=np.median)
        ms = librosa.util.sync(mfcc, frames, aggregate=np.mean)
        rs = librosa.util.sync(rms, frames, aggregate=np.mean)

        feat = np.vstack([
            cs / (np.linalg.norm(cs, axis=0, keepdims=True) + 1e-9),
            ms / (np.linalg.norm(ms, axis=0, keepdims=True) + 1e-9),
        ]).T

        n_seg = int(np.clip(len(beats) // 16, 3, CFG.analysis.target_n_sections))
        bounds = librosa.segment.agglomerative(feat.T, n_seg)
        bounds = np.unique(np.concatenate([[0], bounds, [len(beats) - 1]]))

        energies = rs.flatten()
        energies = energies / (energies.max() + 1e-9)

        sections: List[dict] = []
        for i in range(len(bounds) - 1):
            b0, b1 = int(bounds[i]), int(bounds[i + 1])
            if b1 <= b0:
                continue
            t0 = float(beats[b0])
            t1 = float(beats[min(b1, len(beats) - 1)])
            e = float(np.mean(energies[b0:b1])) if b1 > b0 else 0.0
            sections.append({
                "start": round(t0, 3), "end": round(t1, 3),
                "start_bar": _nearest_bar(t0, downbeats),
                "end_bar": _nearest_bar(t1, downbeats),
                "energy": round(e, 3), "label": "section",
            })

        return _label_sections(sections)
    except Exception as e:
        log.warning("structure analysis failed (%s); using fixed sections", e)
        return _fallback_sections(y, sr, downbeats)


def _label_sections(sections: List[dict]) -> List[dict]:
    """Assign functional labels by energy rank and position."""
    if not sections:
        return sections
    energies = np.array([s["energy"] for s in sections])
    hi = float(np.percentile(energies, 70))
    lo = float(np.percentile(energies, 30))

    for i, s in enumerate(sections):
        first, last = i == 0, i == len(sections) - 1
        if first and s["energy"] <= hi:
            s["label"] = "intro"
        elif last and s["energy"] <= hi:
            s["label"] = "outro"
        elif s["energy"] >= hi:
            s["label"] = "chorus"
        elif s["energy"] <= lo:
            s["label"] = "break"
        else:
            s["label"] = "verse"
    return sections


def _nearest_bar(t: float, downbeats: np.ndarray) -> int:
    if len(downbeats) == 0:
        return 0
    return int(np.argmin(np.abs(downbeats - t)))


def _fallback_sections(y: np.ndarray, sr: int, downbeats: np.ndarray) -> List[dict]:
    """Fixed 8-bar blocks. The terminal rung of the structure ladder."""
    dur = len(dsp.as_2d(y)) / sr
    if len(downbeats) >= 2:
        bar = float(np.median(np.diff(downbeats)))
    else:
        bar = 2.0
    block = bar * 8
    out, t, i = [], 0.0, 0
    while t < dur:
        end = min(t + block, dur)
        # The last block is cut short by the end of the file, and its bar
        # count has to be cut with it: a three-second beat was described as
        # eight bars long because only the times were clipped.
        bars = max(1, min(8, int(round((end - t) / bar)))) if bar > 0 else 8
        out.append({
            "start": round(t, 3), "end": round(end, 3),
            "start_bar": i * 8, "end_bar": i * 8 + bars,
            "energy": 0.5, "label": "section",
        })
        t += block
        i += 1
    return out


# ═════════════════════════════════════════════════════════════════════════════
# SPECTRAL / POCKET
# ═════════════════════════════════════════════════════════════════════════════

def pocket_score(y: np.ndarray, sr: int,
                 low_hz: float = 250.0, high_hz: float = 4000.0) -> float:
    """How much room a beat leaves for a lead vocal. 0 = none, 1 = plenty.

    Measures the beat's spectral density inside the intelligibility band
    relative to its total energy. A sparse trap beat with space in the mids
    scores high; a wall-of-synths beat scores low and will need aggressive
    masking to fit a vocal.
    """
    f, mag = dsp.long_term_spectrum(y, sr)
    if f.size == 0:
        return 0.5
    lin = 10.0 ** (mag / 20.0)
    total = float(np.sum(lin)) + 1e-12
    band = (f >= low_hz) & (f <= high_hz)
    occupancy = float(np.sum(lin[band])) / total
    # Empirically, occupancy ~0.12 is roomy and ~0.45 is crowded.
    return float(np.clip(1.0 - (occupancy - 0.10) / 0.38, 0.0, 1.0))


def pocket_by_section(y: np.ndarray, sr: int, sections: List[dict]) -> Dict[str, float]:
    y2 = dsp.as_2d(y)
    out: Dict[str, List[float]] = {}
    for s in sections:
        a, b = int(s["start"] * sr), int(s["end"] * sr)
        seg = y2[max(0, a):min(len(y2), b)]
        if len(seg) < sr // 2:
            continue
        out.setdefault(s["label"], []).append(pocket_score(seg, sr))
    return {k: round(float(np.mean(v)), 3) for k, v in out.items()}


def spectral_profile(y: np.ndarray, sr: int) -> dict:
    """Broad tonal descriptors used for matching and mastering targets."""
    import librosa
    mono = dsp.to_mono(y).astype(np.float32)
    try:
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=mono, sr=sr)))
        rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=mono, sr=sr)))
        flatness = float(np.mean(librosa.feature.spectral_flatness(y=mono)))
        bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=mono, sr=sr)))
    except Exception:
        centroid = rolloff = flatness = bandwidth = 0.0

    f, mag = dsp.long_term_spectrum(y, sr)
    edges = [0, 120, 500, 2000, 6000, sr / 2]
    lin = 10.0 ** (mag / 20.0)
    total = float(np.sum(lin)) + 1e-12
    bands = []
    for i in range(len(edges) - 1):
        m = (f >= edges[i]) & (f < edges[i + 1])
        bands.append(round(float(np.sum(lin[m]) / total), 4))

    return {
        "centroid_hz": round(centroid, 1),
        "rolloff_hz": round(rolloff, 1),
        "flatness": round(flatness, 5),
        "bandwidth_hz": round(bandwidth, 1),
        "band_energy": {"sub": bands[0], "low": bands[1], "mid": bands[2],
                        "high_mid": bands[3], "air": bands[4]},
    }


def detect_vocal_content(y: np.ndarray, sr: int) -> Tuple[bool, float]:
    """Does this beat already contain vocals (chops, hooks, ad-libs)?

    Beats with their own vocal content collide with the user's vocal in
    both frequency and attention, so the matcher penalises them and the
    mixer ducks those regions harder.
    """
    import librosa
    mono = dsp.to_mono(y).astype(np.float32)
    try:
        if sr != ANALYSIS_SR:
            mono = librosa.resample(mono, orig_sr=sr, target_sr=ANALYSIS_SR)
        # Vocal-range energy that is harmonic and strongly modulated in the
        # 4-8 Hz syllabic range is a reasonable proxy without a full model.
        band = dsp.bandpass(mono, ANALYSIS_SR, 300.0, 3400.0)[:, 0]
        env = dsp.frame_rms(band, int(0.025 * ANALYSIS_SR), int(0.010 * ANALYSIS_SR))
        if env.size < 32:
            return False, 0.0
        env = env - np.mean(env)
        spec = np.abs(np.fft.rfft(env))
        freqs = np.fft.rfftfreq(len(env), d=0.010)
        syll = (freqs >= 3.0) & (freqs <= 8.0)
        ratio = float(np.sum(spec[syll]) / (np.sum(spec) + 1e-9))
        return ratio > 0.22, round(ratio, 3)
    except Exception:
        return False, 0.0
