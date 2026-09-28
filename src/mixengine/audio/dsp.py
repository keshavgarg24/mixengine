"""
DSP primitives built on numpy + scipy only.

Everything here is dependency-light and unit-testable. The heavier
processors (reverb, limiter) come from pedalboard in `mixer.py`, but the
processing that actually determines whether a mix sounds professional --
level riding, ducking, de-essing, spectral unmasking -- lives here, because
it needs to be measurement-driven rather than preset-driven.

Convention: audio arrays are float32/float64 shaped (n_samples, n_channels).
Mono is (n, 1). Helper `as_2d` enforces this.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from ..core.types import FloatSeq
from scipy import signal as sps

EPS = 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# Shape and level helpers
# ─────────────────────────────────────────────────────────────────────────────

def as_2d(x: np.ndarray) -> np.ndarray:
    """Coerce to (n_samples, n_channels)."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return x[:, None]
    if x.ndim == 2:
        # Heuristic: audio is far longer than it is wide.
        return x.T if x.shape[0] < x.shape[1] else x
    raise ValueError(f"unsupported audio shape {x.shape}")


def to_mono(x: np.ndarray) -> np.ndarray:
    """Mono sum as a 1-D array."""
    return as_2d(x).mean(axis=1)


def match_channels(x: np.ndarray, n_ch: int) -> np.ndarray:
    x = as_2d(x)
    if x.shape[1] == n_ch:
        return x
    if x.shape[1] == 1:
        return np.repeat(x, n_ch, axis=1)
    return x.mean(axis=1, keepdims=True) if n_ch == 1 else x[:, :n_ch]


def pad_to(x: np.ndarray, n: int) -> np.ndarray:
    x = as_2d(x)
    if len(x) >= n:
        return x[:n]
    return np.vstack([x, np.zeros((n - len(x), x.shape[1]), dtype=x.dtype)])


def db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def lin_to_db(lin) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(lin), EPS))


def peak_db(x: np.ndarray) -> float:
    x = np.asarray(x)
    return float(lin_to_db(np.max(np.abs(x)))) if x.size else -np.inf


def rms_db(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(lin_to_db(np.sqrt(np.mean(x ** 2)))) if x.size else -np.inf


# ─────────────────────────────────────────────────────────────────────────────
# Envelopes
# ─────────────────────────────────────────────────────────────────────────────

def _coef(time_ms: float, sr: int) -> float:
    """One-pole smoothing coefficient for a given time constant."""
    if time_ms <= 0:
        return 0.0
    return float(np.exp(-1.0 / (max(time_ms, 1e-3) * 0.001 * sr)))


def envelope_follower(x: np.ndarray, sr: int,
                      attack_ms: float = 10.0,
                      release_ms: float = 100.0) -> np.ndarray:
    """Classic attack/release envelope follower on the absolute signal.

    Vectorised where possible but the recursion is inherently sequential;
    `scipy.signal.lfilter` handles the single-coefficient case, and we take
    the elementwise max of the two to emulate asymmetric attack/release
    without a Python loop.
    """
    mono = np.abs(to_mono(x)).astype(np.float64)
    a_att, a_rel = _coef(attack_ms, sr), _coef(release_ms, sr)

    # Fast attack pass and slow release pass, then combine.
    fast = sps.lfilter([1 - a_att], [1, -a_att], mono)
    slow = sps.lfilter([1 - a_rel], [1, -a_rel], mono)
    # Rising edges follow the fast curve, falling edges the slow one.
    env = np.where(fast > slow, fast, slow)
    return env.astype(np.float32)


def smooth(x: np.ndarray, sr: int, time_s: float) -> np.ndarray:
    """Zero-phase smoothing -- no time offset introduced."""
    n = max(3, int(time_s * sr))
    if n % 2 == 0:
        n += 1
    if len(x) <= n * 3:
        return np.asarray(x, dtype=np.float32)
    win = sps.windows.hann(n)
    win /= win.sum()
    return sps.filtfilt(win, [1.0], np.asarray(x, dtype=np.float64)).astype(np.float32)


def frame_rms(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    """RMS per frame, as a 1-D array."""
    mono = to_mono(x).astype(np.float64)
    if len(mono) < frame:
        return np.array([np.sqrt(np.mean(mono ** 2) + EPS)], dtype=np.float32)
    n_frames = 1 + (len(mono) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n_frames)[:, None]
    return np.sqrt(np.mean(mono[idx] ** 2, axis=1) + EPS).astype(np.float32)


def upsample_curve(curve: np.ndarray, n: int) -> np.ndarray:
    """Stretch a per-frame curve to per-sample length by interpolation."""
    curve = np.asarray(curve, dtype=np.float64)
    if len(curve) < 2:
        return np.full(n, curve[0] if len(curve) else 1.0, dtype=np.float32)
    src = np.linspace(0.0, 1.0, len(curve))
    dst = np.linspace(0.0, 1.0, n)
    return np.interp(dst, src, curve).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Filters
# ─────────────────────────────────────────────────────────────────────────────

def _sos_apply(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    x2 = as_2d(x)
    out = np.empty_like(x2)
    for c in range(x2.shape[1]):
        out[:, c] = sps.sosfilt(sos, x2[:, c].astype(np.float64))
    return out


def highpass(x: np.ndarray, sr: int, cutoff_hz: float, order: int = 4) -> np.ndarray:
    cutoff_hz = float(np.clip(cutoff_hz, 20.0, sr * 0.45))
    sos = sps.butter(order, cutoff_hz, btype="highpass", fs=sr, output="sos")
    return _sos_apply(sos, x)


def lowpass(x: np.ndarray, sr: int, cutoff_hz: float, order: int = 4) -> np.ndarray:
    cutoff_hz = float(np.clip(cutoff_hz, 20.0, sr * 0.45))
    sos = sps.butter(order, cutoff_hz, btype="lowpass", fs=sr, output="sos")
    return _sos_apply(sos, x)


def bandpass(x: np.ndarray, sr: int, low_hz: float, high_hz: float,
             order: int = 4) -> np.ndarray:
    low = float(np.clip(low_hz, 20.0, sr * 0.44))
    high = float(np.clip(high_hz, low + 10.0, sr * 0.45))
    sos = sps.butter(order, [low, high], btype="bandpass", fs=sr, output="sos")
    return _sos_apply(sos, x)


def peaking_eq(x: np.ndarray, sr: int, freq_hz: float, gain_db: float,
               q: float = 1.0) -> np.ndarray:
    """RBJ peaking EQ biquad."""
    if abs(gain_db) < 1e-3:
        return as_2d(x)
    freq_hz = float(np.clip(freq_hz, 20.0, sr * 0.45))
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * freq_hz / sr
    alpha = np.sin(w0) / (2.0 * max(q, 0.1))
    cw = np.cos(w0)

    b = np.array([1 + alpha * A, -2 * cw, 1 - alpha * A])
    a = np.array([1 + alpha / A, -2 * cw, 1 - alpha / A])
    sos = np.concatenate([b / a[0], a / a[0]])[None, :]
    return _sos_apply(sos, x)


def shelf_eq(x: np.ndarray, sr: int, freq_hz: float, gain_db: float,
             kind: str = "high", slope: float = 0.9) -> np.ndarray:
    """RBJ low/high shelving filter."""
    if abs(gain_db) < 1e-3:
        return as_2d(x)
    freq_hz = float(np.clip(freq_hz, 20.0, sr * 0.45))
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * freq_hz / sr
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / 2.0 * np.sqrt((A + 1 / A) * (1 / max(slope, 0.1) - 1) + 2)
    two_sqrtA_alpha = 2.0 * np.sqrt(A) * alpha

    if kind == "high":
        b = np.array([A * ((A + 1) + (A - 1) * cw + two_sqrtA_alpha),
                      -2 * A * ((A - 1) + (A + 1) * cw),
                      A * ((A + 1) + (A - 1) * cw - two_sqrtA_alpha)])
        a = np.array([(A + 1) - (A - 1) * cw + two_sqrtA_alpha,
                      2 * ((A - 1) - (A + 1) * cw),
                      (A + 1) - (A - 1) * cw - two_sqrtA_alpha])
    else:
        b = np.array([A * ((A + 1) - (A - 1) * cw + two_sqrtA_alpha),
                      2 * A * ((A - 1) - (A + 1) * cw),
                      A * ((A + 1) - (A - 1) * cw - two_sqrtA_alpha)])
        a = np.array([(A + 1) + (A - 1) * cw + two_sqrtA_alpha,
                      -2 * ((A - 1) + (A + 1) * cw),
                      (A + 1) + (A - 1) * cw - two_sqrtA_alpha])

    sos = np.concatenate([b / a[0], a / a[0]])[None, :]
    return _sos_apply(sos, x)


# ─────────────────────────────────────────────────────────────────────────────
# Dynamics
# ─────────────────────────────────────────────────────────────────────────────

def noise_floor_db(x: np.ndarray, sr: int, percentile: float = 8.0) -> float:
    """Estimate the noise floor from the quietest frames.

    This is what the gate threshold should be derived from. A fixed -35 dB
    threshold either chops quiet phrase tails on a clean recording or lets
    obvious bleed through on a noisy one.
    """
    r = frame_rms(x, frame=int(0.03 * sr), hop=int(0.015 * sr))
    if r.size == 0:
        return -60.0
    return float(lin_to_db(np.percentile(r, percentile)))


def harmonicity(x: np.ndarray, sr: int, frame: int, hop: int,
                f0_min: float = 65.0, f0_max: float = 1000.0) -> np.ndarray:
    """Per-frame periodicity in [0, 1], one value per `hop`.

    The normalised autocorrelation peak inside the voice's pitch-lag
    range. A voiced frame repeats itself once per period and scores high;
    hiss, fan noise and a room tone do not repeat and score low. It is
    the one cheap measurement that tells a loud noise from a loud voice,
    which a level detector by construction cannot.

    Frames are windowed, so the raw autocorrelation decays with lag and
    would under-read low voices; dividing by the window's own
    autocorrelation removes that bias.
    """
    mono = to_mono(x).astype(np.float64)
    if len(mono) < frame or frame < 8:
        return np.zeros(0, dtype=np.float32)
    n_frames = 1 + (len(mono) - frame) // hop
    lag_lo = max(1, int(sr / f0_max))
    lag_hi = min(frame - 1, int(np.ceil(sr / f0_min)))
    if lag_hi <= lag_lo:
        return np.zeros(n_frames, dtype=np.float32)

    win = np.hanning(frame)
    nfft = 1 << int(np.ceil(np.log2(2 * frame)))
    win_ac = np.fft.irfft(np.abs(np.fft.rfft(win, n=nfft)) ** 2, n=nfft)[:frame]
    win_ac = np.maximum(win_ac / max(win_ac[0], EPS), 1e-3)

    out = np.zeros(n_frames, dtype=np.float32)
    chunk = 1024                       # bounds memory on long takes
    for start in range(0, n_frames, chunk):
        stop = min(n_frames, start + chunk)
        idx = np.arange(frame)[None, :] + hop * np.arange(start, stop)[:, None]
        frames = mono[idx]
        frames = (frames - frames.mean(axis=1, keepdims=True)) * win
        spec = np.fft.rfft(frames, n=nfft, axis=1)
        ac = np.fft.irfft(np.abs(spec) ** 2, n=nfft, axis=1)[:, :frame]
        ac = ac / win_ac[None, :]
        e0 = np.maximum(ac[:, 0], EPS)
        peak = ac[:, lag_lo:lag_hi + 1].max(axis=1)
        out[start:stop] = np.clip(peak / e0, 0.0, 1.0)
    return out


def notch(x: np.ndarray, sr: int, freq_hz: float, q: float = 25.0) -> np.ndarray:
    """Remove one spectral line -- mains hum, a motor whine -- at `freq_hz`."""
    freq_hz = float(np.clip(freq_hz, 20.0, sr * 0.45))
    b, a = sps.iirnotch(freq_hz, q, fs=sr)
    return _sos_apply(sps.tf2sos(b, a), x)


def gate(x: np.ndarray, sr: int, threshold_db: float, ratio: float = 4.0,
         attack_ms: float = 2.0, release_ms: float = 120.0,
         floor_db: float = -30.0) -> np.ndarray:
    """Downward expander. Reduces rather than mutes -- a hard gate on a
    vocal chops breath tails and sounds obviously processed."""
    x2 = as_2d(x)
    env = envelope_follower(x2, sr, attack_ms, release_ms)
    env_db = lin_to_db(env)

    below = np.minimum(env_db - threshold_db, 0.0)
    gr_db = np.maximum(below * (ratio - 1.0) / ratio, floor_db)
    g = 10.0 ** (gr_db / 20.0)
    g = smooth(g, sr, 0.005)
    return (x2 * g[:, None]).astype(np.float32)


def compressor(x: np.ndarray, sr: int, threshold_db: float, ratio: float = 3.0,
               attack_ms: float = 8.0, release_ms: float = 90.0,
               knee_db: float = 6.0, makeup: bool = True
               ) -> Tuple[np.ndarray, float]:
    """Soft-knee compressor. Returns `(audio, mean_gain_reduction_db)`.

    The returned GR lets the caller drive the threshold toward a *target*
    gain reduction rather than guessing a fixed threshold -- a -16 dB
    threshold does nothing on a quiet take and crushes a loud one.
    """
    x2 = as_2d(x)
    env = envelope_follower(x2, sr, attack_ms, release_ms)
    env_db = lin_to_db(env)

    over = env_db - threshold_db
    half_knee = knee_db / 2.0
    gr = np.zeros_like(over)

    # Below knee: no reduction. Inside knee: quadratic. Above: linear.
    in_knee = (over > -half_knee) & (over <= half_knee)
    above = over > half_knee
    if knee_db > 0:
        gr[in_knee] = ((1.0 / ratio - 1.0) *
                       (over[in_knee] + half_knee) ** 2 / (2.0 * knee_db))
    gr[above] = (1.0 / ratio - 1.0) * over[above]

    g = 10.0 ** (gr / 20.0)
    out = x2 * g[:, None]

    active = gr[gr < -0.01]
    mean_gr = float(-np.mean(active)) if active.size else 0.0

    if makeup and mean_gr > 0:
        out *= db_to_lin(mean_gr * 0.7)
    return out.astype(np.float32), mean_gr


def auto_compressor(x: np.ndarray, sr: int, target_gr_db: float,
                    ratio: float = 3.0, **kw) -> Tuple[np.ndarray, dict]:
    """Search for the threshold that achieves `target_gr_db` of reduction.

    Bisection over threshold. Five iterations gets within ~0.3 dB, which is
    well below audibility, and removes the single most input-dependent
    magic number in the vocal chain.
    """
    lo, hi = rms_db(x) - 24.0, peak_db(x)
    thr = (lo + hi) / 2.0
    out, gr = x, 0.0
    for _ in range(6):
        thr = (lo + hi) / 2.0
        out, gr = compressor(x, sr, thr, ratio=ratio, **kw)
        if gr < target_gr_db:
            hi = thr           # too little reduction -> lower the threshold
        else:
            lo = thr
    return out, {"threshold_db": round(thr, 2), "gain_reduction_db": round(gr, 2)}


# The shaper runs at this multiple of the sample rate. Waveshaping is the
# one process here that creates frequencies the sample rate cannot hold:
# a 4 kHz sibilant driven into a curve generates harmonics past 20 kHz,
# and at 44.1 kHz those fold back down as inharmonic tones that sound
# like grit rather than warmth. Four times is enough that anything folding
# back lands above hearing at the drives used here.
SATURATE_OVERSAMPLE = 4
# Where the signal is placed on the curve before shaping. The level is
# normalised to this, shaped, then restored, so the character depends on
# `drive_db` alone and not on how loud the take happened to be.
SATURATE_OPERATING_RMS = 0.25


def _shaper(u: np.ndarray, asymmetry: float) -> np.ndarray:
    """A soft curve with a little asymmetry.

    Symmetric curves generate odd harmonics only, which read as hardness.
    A second-order asymmetry adds even harmonics -- the octave, the
    interval the ear hears as warmth rather than distortion -- which is
    the difference between a vocal that sounds driven and one that sounds
    merely louder.
    """
    return np.tanh(u + asymmetry * u * u)


def saturate(x: np.ndarray, sr: int, amount: float = 0.5,
             drive_db: float = 6.0, asymmetry: float = 0.2,
             oversample: int = SATURATE_OVERSAMPLE) -> np.ndarray:
    """Harmonic saturation, level-matched and oversampled.

    What makes a recorded voice sound finished rather than merely clean.
    The harmonics it adds sit above the fundamental, so the voice reads
    louder and closer without its peaks rising -- which is also why it
    earns its place before a limiter: density bought here is density the
    limiter does not have to take out of the transients.

    Level is matched across the process, so `amount` changes the character
    and not the loudness, and an A/B between two settings is a fair one.
    """
    x2 = as_2d(x)
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0.001 or x2.size == 0:
        return x2
    from scipy import signal as sps

    rms = float(np.sqrt(np.mean(np.square(x2.astype(np.float64)))))
    if rms < 1e-9:
        return x2

    scale = (SATURATE_OPERATING_RMS / rms) * db_to_lin(drive_db)
    up = (sps.resample_poly(x2, oversample, 1, axis=0)
          if oversample > 1 else x2.astype(np.float64))
    shaped = _shaper(up * scale, float(asymmetry))
    wet = (sps.resample_poly(shaped, 1, oversample, axis=0)
           if oversample > 1 else shaped)
    wet = pad_to(np.asarray(wet, dtype=np.float32), len(x2))

    # Asymmetry rectifies, which leaves a DC offset behind it.
    wet = wet - np.mean(wet, axis=0, keepdims=True)

    wet_rms = float(np.sqrt(np.mean(np.square(wet.astype(np.float64)))))
    if wet_rms > 1e-9:
        wet = wet * (rms / wet_rms)
    return ((1.0 - amount) * x2 + amount * wet).astype(np.float32)


def soft_clip(x: np.ndarray, sr: int, ceiling_db: float = -1.0,
              knee_db: float = 4.0, oversample: int = SATURATE_OVERSAMPLE
              ) -> Tuple[np.ndarray, dict]:
    """Round off the few samples that stand above everything else.

    A limiter turns its gain down for as long as its release takes, so
    every peak it catches costs the audio around that peak too. A clipper
    shortens the peak itself and touches nothing else, and a peak a few
    samples long is too short for the ear to hear as distortion. Handing
    the isolated peaks to a clipper is what lets the limiter stop working
    so hard, and the limiter's work is what a master loses its snap to.

    Exactly linear below the knee, so quiet material is untouched: only
    what approaches the ceiling is shaped at all.
    """
    x2 = as_2d(x)
    if x2.size == 0:
        return x2, {"applied": False}
    ceiling = db_to_lin(ceiling_db)
    knee = ceiling * db_to_lin(-abs(knee_db))
    span = max(ceiling - knee, 1e-6)
    from scipy import signal as sps

    # Nothing near the ceiling: return the input itself. Resampling up and
    # back is not quite lossless, and a clipper that is not clipping must
    # leave the audio bit-for-bit alone rather than dusting it with
    # conversion error.
    if float(np.abs(x2).max()) <= knee:
        return x2, {"applied": False, "ceiling_db": round(float(ceiling_db), 2),
                    "knee_db": round(float(knee_db), 2),
                    "samples_shaped_pct": 0.0}

    up = (sps.resample_poly(x2, oversample, 1, axis=0)
          if oversample > 1 else x2.astype(np.float64))
    mag = np.abs(up)
    over = mag > knee
    clipped_frac = float(np.mean(over))
    if clipped_frac > 0:
        shaped = knee + span * np.tanh((mag - knee) / span)
        up = np.where(over, np.sign(up) * shaped, up)
    down = (sps.resample_poly(up, 1, oversample, axis=0)
            if oversample > 1 else up)
    out = pad_to(np.asarray(down, dtype=np.float32), len(x2))
    return out, {"applied": bool(clipped_frac > 0),
                 "ceiling_db": round(float(ceiling_db), 2),
                 "knee_db": round(float(knee_db), 2),
                 "samples_shaped_pct": round(100.0 * clipped_frac, 3)}


def parallel_compress(x: np.ndarray, sr: int, amount: float = 0.35,
                      target_gr_db: float = 10.0, ratio: float = 6.0
                      ) -> Tuple[np.ndarray, dict]:
    """Blend a hard-compressed copy under the original.

    The way a vocal is made dense without being made flat. Compressing the
    signal itself to this depth would take the life out of it; running the
    squashed copy underneath instead raises what is quiet -- the tail of a
    word, a breath, the consonant after a loud vowel -- while every peak
    stays where the performance put it.
    """
    x2 = as_2d(x)
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0.001 or x2.size == 0:
        return x2, {"applied": False}
    squashed, info = auto_compressor(
        x2, sr, target_gr_db=target_gr_db, ratio=ratio,
        attack_ms=8.0, release_ms=120.0, knee_db=6.0)
    # Match the copy's level to the source so `amount` is a blend and not
    # a volume control.
    src_rms = float(np.sqrt(np.mean(np.square(x2.astype(np.float64)))))
    cp_rms = float(np.sqrt(np.mean(np.square(squashed.astype(np.float64)))))
    if cp_rms > 1e-9 and src_rms > 1e-9:
        squashed = squashed * (src_rms / cp_rms)
    out = (x2 + amount * squashed) / (1.0 + amount)
    return out.astype(np.float32), {
        "applied": True, "amount": round(amount, 3),
        "copy_gain_reduction_db": info.get("gain_reduction_db", 0.0)}


def deesser(x: np.ndarray, sr: int, low_hz: float = 5000.0,
            high_hz: float = 9500.0, max_gr_db: float = 8.0,
            sensitivity: float = 1.0) -> Tuple[np.ndarray, float]:
    """Split-band de-esser.

    Isolates the sibilance band, measures how far it exceeds its own median,
    and attenuates only that band. The threshold is derived from the signal,
    so it adapts to bright and dark voices without configuration.
    """
    x2 = as_2d(x)
    high = bandpass(x2, sr, low_hz, min(high_hz, sr * 0.45), order=4)
    rest = x2 - high

    env = envelope_follower(high, sr, attack_ms=1.0, release_ms=35.0)
    env_db = lin_to_db(env)

    voiced = env_db[env_db > (np.max(env_db) - 45.0)]
    if voiced.size < 10:
        return x2, 0.0
    threshold = float(np.percentile(voiced, 72.0))

    over = np.maximum(env_db - threshold, 0.0) * float(sensitivity)
    gr_db = -np.minimum(over * 0.65, max_gr_db)
    g = smooth(10.0 ** (gr_db / 20.0), sr, 0.004)

    out = rest + high * g[:, None]
    applied = float(-np.mean(gr_db[gr_db < -0.01])) if np.any(gr_db < -0.01) else 0.0
    return out.astype(np.float32), applied


def level_ride(x: np.ndarray, sr: int, regions: Sequence[Tuple[int, int]],
               max_gain_db: float = 6.0, smoothing_s: float = 0.15,
               weights: Optional[FloatSeq] = None) -> np.ndarray:
    """Per-phrase gain automation toward the median phrase level.

    This is the first thing a human engineer does and the thing most
    auto-mixers skip. Riding levels *before* compression means the
    compressor works on already-consistent material and can stay gentle,
    which is what keeps a vocal sounding natural rather than squashed.
    """
    x2 = as_2d(x)
    if not regions:
        return x2

    levels = []
    for s, e in regions:
        seg = x2[max(0, s):min(len(x2), e)]
        levels.append(rms_db(seg) if len(seg) else -np.inf)

    finite = [l for l in levels if np.isfinite(l)]
    if not finite:
        return x2
    target = float(np.median(finite))

    # `weights` scales how hard each phrase is pulled toward the median:
    # 1.0 is the full correction, 0.5 half of it. The musical salience layer
    # supplies these so a phrase's natural fall-away at its end is not
    # hauled back up, and so the ride is a smoothing rather than a flattening.
    w = list(weights) if weights is not None else [1.0] * len(regions)
    gain_curve = np.ones(len(x2), dtype=np.float32)
    for (s, e), lvl, wt in zip(regions, levels, w):
        if not np.isfinite(lvl):
            continue
        delta = float(np.clip((target - lvl) * float(np.clip(wt, 0.0, 1.5)),
                              -max_gain_db, max_gain_db))
        gain_curve[max(0, s):min(len(x2), e)] = db_to_lin(delta)

    gain_curve = smooth(gain_curve, sr, smoothing_s)
    return (x2 * gain_curve[:, None]).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Ducking and spectral unmasking -- "the pocket"
# ─────────────────────────────────────────────────────────────────────────────

def sidechain_duck(target: np.ndarray, key_signal: np.ndarray, sr: int,
                   depth_db: float = 2.5, attack_ms: float = 12.0,
                   release_ms: float = 220.0,
                   threshold_rel_db: float = -32.0) -> np.ndarray:
    """Duck `target` whenever `key_signal` is present.

    Applied to the beat's *tonal* stems only. Ducking the drums as well
    makes the whole mix pump and destroys the groove -- the reason this
    function exists separately from a full-mix duck is precisely so the
    drums can be excluded.
    """
    tgt = as_2d(target)
    env = envelope_follower(key_signal, sr, attack_ms, release_ms)
    env_db = lin_to_db(env)

    ref = float(np.percentile(env_db[np.isfinite(env_db)], 95.0)) if env_db.size else 0.0
    norm = np.clip((env_db - (ref + threshold_rel_db)) / abs(threshold_rel_db), 0.0, 1.0)

    gr_db = -depth_db * norm
    g = smooth(10.0 ** (gr_db / 20.0), sr, 0.01)
    g = pad_to(g[:, None], len(tgt))[:, 0]
    return (tgt * g[:, None]).astype(np.float32)


def spectral_unmask(beat: np.ndarray, vocal: np.ndarray, sr: int,
                    strength: float = 0.5,
                    low_hz: float = 220.0, high_hz: float = 5200.0,
                    n_fft: int = 2048) -> np.ndarray:
    """Dynamically carve the beat where and when the vocal actually sits.

    Rather than a static EQ dip -- which removes energy even when the vocal
    is silent -- this computes a time-frequency mask from the vocal's own
    spectrogram and attenuates the beat only in the bins the vocal occupies,
    only while it occupies them. Restricted to the intelligibility band so
    the beat keeps its low end and its air.

    This is the processor most responsible for a vocal sounding "in" the
    mix rather than "on top of" it.
    """
    b = as_2d(beat)
    v = to_mono(vocal)
    n = len(b)
    v = np.pad(v, (0, max(0, n - len(v))))[:n]

    hop = n_fft // 4
    win = sps.windows.hann(n_fft, sym=False)
    stft_kw = {"fs": sr, "window": win, "nperseg": n_fft,
               "noverlap": n_fft - hop}

    f, _, Zv = sps.stft(v.astype(np.float64), **stft_kw)
    Vmag = np.abs(Zv)

    band = (f >= low_hz) & (f <= high_hz)
    if not np.any(band) or Vmag.size == 0:
        return b

    # Normalise the vocal's magnitude per frequency bin so the mask reflects
    # relative occupancy rather than absolute level.
    ref = np.percentile(Vmag[band], 96.0)
    if ref <= EPS:
        return b
    occ = np.clip(Vmag / ref, 0.0, 1.0)

    # Soften the band edges so the carve doesn't sound like a filter sweep.
    taper = np.zeros_like(f)
    taper[band] = 1.0
    taper = np.convolve(taper, np.hanning(9) / np.sum(np.hanning(9)), mode="same")

    gain = 1.0 - strength * occ * taper[:, None]
    gain = np.clip(gain, 10 ** (-9.0 / 20.0), 1.0)   # never cut more than 9 dB

    out = np.empty_like(b)
    for c in range(b.shape[1]):
        _, _, Zb = sps.stft(b[:, c].astype(np.float64), **stft_kw)
        m = min(Zb.shape[1], gain.shape[1])
        Zb[:, :m] *= gain[:, :m]
        _, y = sps.istft(Zb, fs=sr, window=win, nperseg=n_fft,
                         noverlap=n_fft - hop)
        out[:, c] = pad_to(np.asarray(y, dtype=np.float64)[:, None], n)[:, 0]
    return out.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Spectrum utilities
# ─────────────────────────────────────────────────────────────────────────────

def long_term_spectrum(x: np.ndarray, sr: int, n_fft: int = 4096
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Average magnitude spectrum in dB. Returns (freqs, magnitude_db)."""
    mono = to_mono(x).astype(np.float64)
    if len(mono) < n_fft:
        mono = np.pad(mono, (0, n_fft - len(mono)))
    f, Pxx = sps.welch(mono, fs=sr, nperseg=n_fft, noverlap=n_fft // 2)
    return f, lin_to_db(np.sqrt(np.maximum(Pxx, EPS)))


def find_resonances(x: np.ndarray, sr: int, n: int = 3,
                    low_hz: float = 150.0, high_hz: float = 6000.0
                    ) -> list:
    """Locate the most prominent narrow peaks in the long-term spectrum.

    Returns `[(freq_hz, excess_db), ...]`. These are the frequencies a human
    engineer would notch -- room modes, mic proximity bumps, nasal
    resonances. Detecting them per-recording is what makes the EQ adaptive
    instead of a preset.
    """
    f, mag = long_term_spectrum(x, sr)
    band = (f >= low_hz) & (f <= high_hz)
    if not np.any(band):
        return []
    fb, mb = f[band], mag[band]

    # Smooth trend line; peaks are what stands above it.
    k = max(5, len(mb) // 24) | 1
    trend = sps.savgol_filter(mb, k, 2)
    excess = mb - trend

    peaks, props = sps.find_peaks(excess, prominence=1.5, distance=max(2, k // 3))
    if peaks.size == 0:
        return []
    order = np.argsort(props["prominences"])[::-1][:n]
    return [(float(fb[peaks[i]]), float(excess[peaks[i]])) for i in order]


def spectral_tilt_match(x: np.ndarray, ref_freqs: np.ndarray,
                        ref_db: np.ndarray, sr: int,
                        strength: float = 0.35, n_bands: int = 6
                        ) -> np.ndarray:
    """Nudge `x`'s broad tonal balance toward a reference curve.

    Deliberately coarse -- a handful of wide shelving/peaking moves, not a
    high-resolution match. Matching a reference too precisely makes every
    output sound identical and strips the beat's character.
    """
    f, mag = long_term_spectrum(x, sr)
    edges = np.geomspace(80.0, min(16000.0, sr * 0.45), n_bands + 1)
    out = as_2d(x)

    for i in range(n_bands):
        lo, hi = edges[i], edges[i + 1]
        m = (f >= lo) & (f < hi)
        rm = (ref_freqs >= lo) & (ref_freqs < hi)
        if not np.any(m) or not np.any(rm):
            continue
        delta = float(np.mean(ref_db[rm]) - np.mean(mag[m]))
        delta = float(np.clip(delta * strength, -3.5, 3.5))
        if abs(delta) < 0.15:
            continue
        centre = float(np.sqrt(lo * hi))
        out = peaking_eq(out, sr, centre, delta, q=0.9)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Stereo
# ─────────────────────────────────────────────────────────────────────────────

def pan(x: np.ndarray, position: float) -> np.ndarray:
    """Constant-power pan. `position` in [-1, 1]."""
    mono = to_mono(x)
    p = float(np.clip(position, -1.0, 1.0))
    angle = (p + 1.0) * np.pi / 4.0
    return np.stack([mono * np.cos(angle), mono * np.sin(angle)],
                    axis=1).astype(np.float32)


def haas_double(x: np.ndarray, sr: int, offset_ms: float, gain_db: float,
                position: float) -> np.ndarray:
    """A delayed, panned copy -- the standard vocal-doubling trick."""
    delay = int(abs(offset_ms) * 0.001 * sr)
    mono = to_mono(x)
    delayed = np.concatenate([np.zeros(delay, dtype=np.float32), mono])[:len(mono)]
    return pan(delayed, position) * db_to_lin(gain_db)


def mono_compatibility_loss_db(x: np.ndarray) -> float:
    """How much level is lost when folded to mono.

    Large losses mean phase cancellation from over-widening -- a real defect
    that shows up on club systems and phone speakers. This is a QC gate,
    not a stylistic preference.
    """
    x2 = as_2d(x)
    if x2.shape[1] < 2:
        return 0.0
    stereo_rms = np.sqrt(np.mean(x2 ** 2) + EPS)
    mono_rms = np.sqrt(np.mean(to_mono(x2) ** 2) + EPS)
    return float(lin_to_db(stereo_rms) - lin_to_db(mono_rms))


def true_peak_db(x: np.ndarray, oversample: int = 4) -> float:
    """Inter-sample peak estimate via oversampling.

    Sample peak alone under-reads by 1-3 dB on limited material; encoders
    and D/A converters see the true peak, which is why -1 dBTP is the
    delivery standard rather than -1 dBFS.
    """
    x2 = as_2d(x)
    n = len(x2)
    if n < 8:
        return peak_db(x2)
    up = sps.resample_poly(x2.astype(np.float64), oversample, 1, axis=0)
    return float(lin_to_db(np.max(np.abs(up))))


def normalize_peak(x: np.ndarray, target_db: float = -1.0) -> np.ndarray:
    tp = true_peak_db(x)
    if not np.isfinite(tp) or tp <= target_db:
        return as_2d(x)
    return (as_2d(x) * db_to_lin(target_db - tp)).astype(np.float32)


def fade(x: np.ndarray, sr: int, fade_in_s: float = 0.005,
         fade_out_s: float = 0.02) -> np.ndarray:
    """Short fades to prevent clicks at boundaries."""
    x2 = as_2d(x).copy()
    ni, no = int(fade_in_s * sr), int(fade_out_s * sr)
    if ni > 0 and len(x2) > ni:
        x2[:ni] *= np.linspace(0.0, 1.0, ni)[:, None]
    if no > 0 and len(x2) > no:
        x2[-no:] *= np.linspace(1.0, 0.0, no)[:, None]
    return x2
