"""
Putting the vocal in the beat's room.

This is the main reason an automatic mix reads as *pasted on top*. A beat
was produced in one acoustic -- plates, rooms, tails, whatever the producer
chose -- and a vocal recorded in a bedroom arrives with a different one.
Level and EQ do not reconcile that, because the mismatch is in the decay,
not the balance. The ear localises sources by their reverberation, so two
different rooms in one file are heard as two different places however well
the levels are matched.

The approach is to measure rather than assume. A reverberant decay leaves
a signature in the energy envelope after every transient, and that decay
can be read back out:

  **RT60**, per band, from the slope of the envelope's decay following
  strong onsets. Frequency-resolved because real rooms are: high
  frequencies are absorbed by air and soft surfaces and die away first, so
  a single number would put a bright tail on a dark room.

  **DRR**, the direct-to-reverberant ratio, from the energy in the first
  few milliseconds after an onset against the energy in the decay that
  follows. This is what "close" and "far" actually mean, and it is the
  parameter that decides whether the vocal sits in front of the beat or
  inside it.

The vocal is then moved *toward* the beat's space -- never all the way, and
never in the direction of less reverberation. Reverb can be added to a dry
signal; it cannot be removed from a wet one by adding anything, and a take
recorded in a live room needs the dereverb stage rather than this one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.types import FloatSeq

from . import dsp

log = logging.getLogger("mixengine.space")

# Octave-ish bands. Coarse on purpose: estimating a decay needs enough
# energy in the band to see the slope, and narrow bands on sparse material
# give confident-looking numbers from three data points.
BANDS: List[Tuple[float, float]] = [
    (120.0, 500.0), (500.0, 2000.0), (2000.0, 6000.0), (6000.0, 12000.0)]

# Band splitting needs steep filters. A second-order bandpass has skirts
# gentle enough that a loud, slowly-decaying neighbour dominates the band
# being measured: synthesising an impulse response whose 2-6 kHz band was
# given a 0.15 s decay and whose 0.5-2 kHz band was given 1.0 s, then
# measuring it back, returned almost the same decay for both -- the
# measurement was reading leakage from the louder band, not the band.
BAND_ORDER = 6

# Decays outside this range are not rooms. Below 0.08 s the measurement is
# the envelope follower's own release; above 3 s it is a pad or a sustained
# note being read as a tail.
MIN_RT60 = 0.08
MAX_RT60 = 3.0

# How far toward the beat's space the vocal is moved. Matching exactly
# sounds wrong: a lead vocal is conventionally drier and closer than the
# track around it, and taking it all the way puts it behind the beat.
MATCH_AMOUNT = 0.65

# Decay slopes are fitted over this much of the drop, starting below the
# peak to skip the direct sound. The classic T30 window.
FIT_START_DB = -5.0
FIT_END_DB = -35.0

# Below this, the individual decay measurements disagree enough that their
# median is not describing one room. Acting on it would convolve a made-up
# space onto the vocal, which is worse than leaving the mismatch alone.
MIN_CONFIDENCE = 0.25

# A band must agree with itself at least this well before its decay is
# used. Judged per band because the low bands of most produced music carry
# sustained notes that read as decay; those bands fail this on their own
# without dragging the clean high bands down with them.
MIN_BAND_CONFIDENCE = 0.30

# Real rooms ring longer at low frequencies -- absorbers, air and most
# surfaces take the top end first -- by roughly this factor per octave.
# Used to fill in bands that could not be measured directly, so a room
# reconstructed from its high bands is not given a bright, thin tail.
LF_EXTENSION_PER_OCTAVE = 1.25


@dataclass
class SpaceProfile:
    rt60_by_band: Dict[str, float] = field(default_factory=dict)
    rt60_mean: float = 0.0
    drr_db: float = 0.0
    n_decays: int = 0
    confidence: float = 0.0
    note: str = ""
    band_detail: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @property
    def measured(self) -> bool:
        return (self.n_decays >= 3 and self.rt60_mean > 0
                and self.confidence >= MIN_CONFIDENCE)

    def to_dict(self) -> dict:
        return {"rt60_by_band": {k: round(v, 3)
                                 for k, v in self.rt60_by_band.items()},
                "rt60_mean": round(self.rt60_mean, 3),
                "drr_db": round(self.drr_db, 2),
                "n_decays": self.n_decays,
                "confidence": round(self.confidence, 3),
                "measured": self.measured, "note": self.note,
                "band_detail": self.band_detail}


def estimate(y: np.ndarray, sr: int, *,
             onsets: Optional[FloatSeq] = None) -> SpaceProfile:
    """Measure the acoustic space a recording was made or produced in."""
    prof = SpaceProfile()
    mono = dsp.to_mono(dsp.as_2d(y)).astype(np.float32)
    if mono.size < sr:
        prof.note = "too short to measure a decay"
        return prof

    if onsets is None:
        onsets = _onsets(mono, sr)
    o = np.asarray(onsets, dtype=np.float64)
    if o.size < 3:
        prof.note = "not enough transients to read a decay from"
        return prof

    # Each band is judged on its own. A single spread across all bands was
    # the reason this stage declined on nearly every real beat: the low
    # bands, contaminated by sustained bass and pads, disagreed with the
    # high bands that were measuring the room perfectly well, and the
    # combined spread rejected both. Room information is cleanest at high
    # frequencies anyway -- air absorption makes the decay there short and
    # unambiguous -- so the bands that can be measured are exactly the ones
    # worth trusting.
    per_band: Dict[str, Dict[str, float]] = {}
    drr_all: List[float] = []
    total_decays = 0
    for lo, hi in BANDS:
        if hi >= sr * 0.45:
            continue
        band = dsp.to_mono(dsp.bandpass(mono[:, None], sr, lo,
                                        min(hi, sr * 0.44), order=BAND_ORDER))
        env = dsp.envelope_follower(band[:, None], sr, attack_ms=1.0,
                                    release_ms=1.0)
        env = dsp.to_mono(dsp.as_2d(env))
        rts, drrs = _decays(env, sr, o)
        # The track's final decay into silence is the cleanest impulse
        # response most material offers, and it is the one measurement
        # dense material still provides.
        n_onset = len(rts)
        tail = _tail_decay(env, sr)
        if tail is not None:
            rts = rts + [tail]
        if not rts:
            continue
        total_decays += len(rts)
        med = float(np.median(rts))
        # The tail is weighted as three onset decays. It is the one
        # measurement free of the next transient's masking, and on dense
        # material it is the only one there is -- scored as a single
        # sample it could never clear the bar, which defeated the reason
        # it was added.
        if len(rts) > 1:
            spread = float(np.std(rts)) / max(med, 1e-6)
        else:
            spread = 0.25 if tail is not None else 0.6
        n_eff = n_onset + (3 if tail is not None else 0)
        conf = float(np.clip(1.0 - spread, 0.0, 1.0)
                     * np.clip(n_eff / 6.0, 0.0, 1.0))
        key = f"{int(lo)}_{int(hi)}"
        per_band[key] = {"rt60": med, "confidence": conf, "n": len(rts)}
        if conf >= MIN_BAND_CONFIDENCE:
            drr_all.extend(drrs)

    prof.n_decays = total_decays
    prof.band_detail = {k: {kk: round(vv, 3) for kk, vv in d.items()}
                        for k, d in per_band.items()}
    if not per_band:
        prof.note = "no usable decay found; the material may be continuous"
        return prof

    confident = {k: d for k, d in per_band.items()
                 if d["confidence"] >= MIN_BAND_CONFIDENCE}
    if not confident:
        worst = max(per_band.values(), key=lambda d: -d["confidence"])
        prof.note = (f"no band's decay measurements agree with themselves "
                     f"(best confidence {worst['confidence']:.2f})")
        return prof

    # Anchor on the band that agrees with itself best, preferring higher
    # bands on a tie, then check every other confident band against the
    # room prior. A band that rings more than twice as long as the prior
    # predicts from the anchor is not measuring the room: it is measuring a
    # pad or a bass note holding, which is internally consistent -- the
    # same synth sustains the same way every bar -- and so passes the
    # per-band check while being wrong. On the trap fixture the 500-2000 Hz
    # band read 1.36 s against 0.31 s at 6-12 kHz; a real room with that
    # ratio would be a cathedral with a carpeted ceiling.
    keys = [f"{int(lo)}_{int(hi)}" for lo, hi in BANDS]
    centre = {f"{int(lo)}_{int(hi)}": float(np.sqrt(lo * hi)) for lo, hi in BANDS}
    anchor = max(confident, key=lambda k: (round(confident[k]["confidence"], 2),
                                           keys.index(k)))
    kept: Dict[str, float] = {anchor: confident[anchor]["rt60"]}
    rejected: List[str] = []
    for k, d in confident.items():
        if k == anchor:
            continue
        octaves = float(np.log2(centre[anchor] / centre[k]))     # +ve if k lower
        expected = confident[anchor]["rt60"] * (LF_EXTENSION_PER_OCTAVE ** octaves)
        if d["rt60"] > expected * 2.0:
            rejected.append(k)
            continue
        kept[k] = d["rt60"]
    for k in rejected:
        prof.band_detail[k]["contaminated"] = 1.0
    prof.rt60_by_band = kept
    prof.drr_db = float(np.median(drr_all)) if drr_all else 0.0

    # The summary figure comes from *measured* bands only, preferring those
    # above 500 Hz. Extrapolated bands are for synthesis, not for the
    # summary: including them here pulled a clean measurement toward its
    # own prior -- a 0.55 s room measured exactly in its one energetic band
    # was reported as 0.39 s once the empty bands were filled in around it.
    upper = [v for k, v in kept.items() if int(k.split("_")[0]) >= 500]
    prof.rt60_mean = float(np.median(upper)) if upper \
        else float(np.median(list(kept.values())))

    # Overall confidence: how sure the confident bands are, discounted by
    # how much of the spectrum they cover. One clean band is still a real
    # measurement -- it is not discounted below sixty percent of its own
    # confidence -- but four agreeing bands are a better one.
    coverage = len(kept) / max(len(BANDS), 1)
    prof.confidence = float(
        np.mean([confident[k]["confidence"] for k in kept])
        * (0.6 + 0.4 * coverage))
    if prof.confidence < MIN_CONFIDENCE:
        prof.note = (f"decay estimates too uncertain to act on "
                     f"(confidence {prof.confidence:.2f})")
    return prof


def match(vocal: np.ndarray, sr: int, target: SpaceProfile,
          current: Optional[SpaceProfile] = None, *,
          amount: float = MATCH_AMOUNT,
          max_wet: float = 0.35) -> Tuple[np.ndarray, dict]:
    """Move the vocal toward `target`'s space. Returns `(audio, report)`."""
    v = dsp.as_2d(vocal)
    rep: Dict = {"applied": False, "amount": round(float(amount), 3)}
    if not target.measured:
        rep["note"] = target.note or "the beat's space could not be measured"
        return v, rep

    if current is None:
        current = estimate(v, sr)
    rep["beat_space"] = target.to_dict()
    rep["vocal_space"] = current.to_dict()

    # Only ever add. A vocal already wetter than the beat needs less
    # reverberation, and nothing convolved into it can produce that.
    if current.measured and current.rt60_mean >= target.rt60_mean * 0.9:
        rep["note"] = (f"the vocal's room ({current.rt60_mean:.2f}s) is "
                       f"already at or beyond the beat's "
                       f"({target.rt60_mean:.2f}s); dereverb is the stage "
                       f"for this, not reverb")
        return v, rep

    have = current.rt60_mean if current.measured else 0.0
    want = have + (target.rt60_mean - have) * float(np.clip(amount, 0.0, 1.0))
    want = float(np.clip(want, MIN_RT60, MAX_RT60))
    rep["target_rt60"] = round(want, 3)

    ir = synth_rir(sr, want, target.rt60_by_band)
    wet = _convolve(v, ir)

    # Wet level from the *reverberant energy fraction*, not from the DRR in
    # decibels. A DRR of -8 dB and a DRR of -16 dB are 8 dB apart but only
    # 0.86 and 0.98 of the energy reverberant -- almost the same room, very
    # different numbers. Converting first keeps the mapping proportional to
    # what is actually heard.
    #
    # The gap is against the vocal's own reverberant fraction when that
    # could be measured; when it could not, the take is dry and the whole
    # of the target's fraction is the gap. The first version treated an
    # unmeasurable vocal as a zero *decibel* gap, which floored the wet
    # level at 1% -- a dry take going into a 0.9 s room came out dry.
    target_frac = _reverb_fraction(target.drr_db)
    have_frac = _reverb_fraction(current.drr_db) if current.measured else 0.0
    gap = float(np.clip(target_frac - have_frac, 0.0, 1.0))
    wet_frac = float(np.clip(gap * max_wet * amount, 0.0, max_wet))
    rep["wet"] = round(wet_frac, 4)
    rep["target_reverb_fraction"] = round(target_frac, 3)
    rep["vocal_reverb_fraction"] = round(have_frac, 3)

    out = v * (1.0 - wet_frac * 0.35) + wet * wet_frac
    rep["applied"] = True
    log.info("  space: vocal %.2fs -> %.2fs RT60 to sit in the beat's room "
             "(%.0f%% wet)", have, want, wet_frac * 100)
    return out.astype(np.float32), rep


def synth_rir(sr: int, rt60_s: float,
              rt60_by_band: Optional[Dict[str, float]] = None,
              *, predelay_ms: float = 12.0, seed: int = 4) -> np.ndarray:
    """A synthetic impulse response with the requested decay per band.

    Exponentially-decaying noise, filtered into bands and given each band
    its own decay constant, which is what makes it sound like a room rather
    than a spring: a single broadband decay leaves the top end ringing long
    after a real surface would have absorbed it.

    The pre-delay keeps the direct sound separate from the onset of the
    tail. Without it the reverberation starts inside the consonant and
    smears diction, which is the most common way an otherwise correct
    reverb makes a vocal worse.
    """
    rng = np.random.default_rng(seed)
    n = int(sr * min(max(rt60_s, MIN_RT60), MAX_RT60) * 1.2)
    n = max(n, int(sr * 0.05))
    t = np.arange(n) / float(sr)
    noise = rng.normal(0.0, 1.0, n).astype(np.float32)

    out = np.zeros(n, dtype=np.float32)
    bands = fill_bands(rt60_by_band or {}, rt60_s)
    for lo, hi in BANDS:
        if hi >= sr * 0.45:
            continue
        key = f"{int(lo)}_{int(hi)}"
        band_rt = float(bands.get(key, rt60_s))
        band_rt = float(np.clip(band_rt, MIN_RT60, MAX_RT60))
        decay = np.exp(-6.908 * t / band_rt).astype(np.float32)  # ln(1000)
        seg = dsp.to_mono(dsp.bandpass(noise[:, None], sr, lo,
                                       min(hi, sr * 0.44), order=BAND_ORDER))
        out += seg * decay

    pre = int(sr * predelay_ms / 1000.0)
    out = np.concatenate([np.zeros(pre, dtype=np.float32), out])
    peak = float(np.max(np.abs(out))) or 1.0
    return (out / peak).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────

def fill_bands(measured: Dict[str, float], fallback: float) -> Dict[str, float]:
    """A decay for every band, extrapolated from the ones that were measured.

    Bands below the lowest measured one ring longer by
    `LF_EXTENSION_PER_OCTAVE` per octave; bands above the highest decay
    shorter by the same factor. This is what a real room does, and it is
    what keeps a space reconstructed from its clean high bands from
    sounding like a bathroom tiled in glass.
    """
    keys = [f"{int(lo)}_{int(hi)}" for lo, hi in BANDS]
    centres = [np.sqrt(lo * hi) for lo, hi in BANDS]
    known = [(i, measured[k]) for i, k in enumerate(keys) if k in measured]
    out: Dict[str, float] = {}
    for i, k in enumerate(keys):
        if k in measured:
            out[k] = float(measured[k])
            continue
        if not known:
            out[k] = float(fallback)
            continue
        j, rt = min(known, key=lambda p: abs(p[0] - i))
        octaves = np.log2(centres[j] / centres[i])          # +ve when i is lower
        out[k] = float(np.clip(rt * (LF_EXTENSION_PER_OCTAVE ** octaves),
                               MIN_RT60, MAX_RT60))
    return out


def _tail_decay(env: np.ndarray, sr: int) -> Optional[float]:
    """RT60 from the recording's final decay into silence, if it has one.

    The end of a track is usually the one moment nothing else is playing
    over the reverberation, which makes it the cleanest measurement
    available -- and the only one dense material offers at all.
    """
    look = int(sr * 3.0)
    if env.size < look + sr:
        return None
    seg = env[-look:].astype(np.float64)
    db = 20.0 * np.log10(np.maximum(seg, 1e-9))
    # The decay to fit is the one after the *last* strong event, not after
    # the loudest. On dense material the loudest moment is the pile-up of
    # overlapping hits before the ending; a fit from there runs through
    # everything that follows and reads as a decay longer than the room.
    strong = np.flatnonzero(db >= float(db.max()) - 15.0)
    if strong.size == 0:
        return None
    peak_i = int(strong[-1])
    if peak_i > look - int(sr * 0.3):
        return None
    rel = db[peak_i:] - db[peak_i]
    # The tail must actually end quiet, or this is a fade-out or a loop.
    if float(np.min(rel[-int(sr * 0.1):])) > -28.0:
        return None
    lo_i = int(np.argmax(rel <= FIT_START_DB)) if np.any(rel <= FIT_START_DB) else -1
    if lo_i <= 0:
        return None
    end_mask = rel[lo_i:] <= FIT_END_DB
    hi_i = lo_i + int(np.argmax(end_mask)) if np.any(end_mask) else rel.size - 1
    if hi_i - lo_i < int(sr * 0.03):
        return None
    x = np.arange(lo_i, hi_i) / float(sr)
    slope = float(np.polyfit(x, rel[lo_i:hi_i], 1)[0])
    if slope >= -1e-6:
        return None
    rt = -60.0 / slope
    return rt if MIN_RT60 <= rt <= MAX_RT60 else None


def _reverb_fraction(drr_db: float) -> float:
    """Share of the energy that is reverberant, from the direct/reverb ratio."""
    return float(1.0 / (1.0 + 10.0 ** (float(drr_db) / 10.0)))


def _onsets(mono: np.ndarray, sr: int) -> np.ndarray:
    try:
        import librosa
        return librosa.onset.onset_detect(y=mono, sr=sr, units="time",
                                          backtrack=False)
    except Exception:
        return np.zeros(0)


def _decays(env: np.ndarray, sr: int, onsets: np.ndarray
            ) -> Tuple[List[float], List[float]]:
    """Fit a decay slope after each onset that has room to decay into.

    An onset followed closely by another has its tail masked, so only the
    ones with a clear window afterwards are used. On dense material that
    leaves very few, which is the honest outcome -- a wall of sound carries
    no usable information about the room it was made in.
    """
    rts: List[float] = []
    drrs: List[float] = []
    db = 20.0 * np.log10(np.maximum(env, 1e-9))
    direct_n = int(sr * 0.005)
    peak_win = int(sr * 0.03)
    min_window = int(sr * 0.12)

    for k, t in enumerate(onsets):
        start0 = int(t * sr)
        nxt = int(onsets[k + 1] * sr) if k + 1 < onsets.size else len(env)
        if nxt - start0 < min_window or start0 + peak_win >= len(env):
            continue
        # Locate the actual peak within 30 ms of the reported onset, and
        # measure the decay from *there*. Onset detectors disagree about
        # where an onset "is": librosa's raw times sit near the transient,
        # the engine's own are backtracked to the energy minimum before it.
        # Reading the peak from the first 5 ms of a backtracked onset reads
        # the rise instead, and every decay is then measured against the
        # wrong reference -- on one beat that turned 45 usable decays in
        # the top band into 11, and a 0.31 s room into 0.12 s.
        start = start0 + int(np.argmax(db[start0:start0 + peak_win]))
        window = min(nxt - start, int(sr * (MAX_RT60 * 0.6)))
        if window < min_window or start + window > len(env):
            continue
        seg = db[start:start + window]
        peak = float(seg[0])
        rel = seg - peak
        below_start = np.argmax(rel <= FIT_START_DB) if np.any(rel <= FIT_START_DB) else -1
        if below_start <= 0:
            continue
        end_mask = rel[below_start:] <= FIT_END_DB
        below_end = (below_start + int(np.argmax(end_mask))) if np.any(end_mask) \
            else seg.size - 1
        if below_end - below_start < int(sr * 0.02):
            continue

        x = np.arange(below_start, below_end) / float(sr)
        yv = rel[below_start:below_end]
        if x.size < 8:
            continue
        slope = float(np.polyfit(x, yv, 1)[0])          # dB per second
        if slope >= -1e-6:
            continue
        rt = -60.0 / slope
        if not (MIN_RT60 <= rt <= MAX_RT60):
            continue
        rts.append(rt)

        lin = env[start:start + window].astype(np.float64) ** 2
        direct = float(np.sum(lin[:direct_n]))
        tail = float(np.sum(lin[direct_n:]))
        if direct > 0 and tail > 0:
            drrs.append(float(10.0 * np.log10(direct / tail)))
    return rts, drrs


def _convolve(v: np.ndarray, ir: np.ndarray) -> np.ndarray:
    from scipy.signal import fftconvolve
    out = np.empty_like(v)
    for c in range(v.shape[1]):
        wet = fftconvolve(v[:, c], ir)[:len(v)]
        out[:, c] = wet
    peak = float(np.max(np.abs(out))) or 1.0
    ref = float(np.max(np.abs(v))) or 1.0
    return (out / peak * ref).astype(np.float32)
