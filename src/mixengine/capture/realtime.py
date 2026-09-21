"""
Real-time frame analysis for vocal capture.

Streaming, allocation-light, and stateful across frames: given consecutive
blocks of microphone audio it reports level, clipping, pitch, voicing,
proximity and noise, at a rate fast enough to drive a live display.

Pitch uses the McLeod Pitch Method rather than YIN. The reason is practical
rather than theoretical: MPM's normalised square difference function is
bounded to [-1, 1], so its clarity value is directly comparable across
frames and a fixed threshold actually means something. YIN's cumulative
mean normalised difference has no such bound, which makes its threshold
material-dependent and its confidence hard to display honestly to a singer.

Two implementation details from the MPM literature are easy to get wrong
and both are handled here:

  * **DC must be removed before the NSDF.** YIN's difference function is
    DC-invariant, so YIN tolerates an offset. MPM does not: a constant
    offset pushes the NSDF toward 1 at every lag and buries the zero
    crossings the key-maximum search depends on.
  * **Overlap is what buys low latency.** The analysis window has to be
    long enough to contain two periods of the lowest expected note -- about
    43 ms for a 100 Hz male low -- but estimates are emitted every hop, so
    a 2048-sample window with a 256-sample hop updates every 5.3 ms while
    still looking back far enough to be accurate.

Nothing here does I/O or touches an audio device. It takes numpy blocks and
returns numbers, which keeps it testable without a microphone and portable
across whatever capture backend the host uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Deque, List, Optional, Tuple
from collections import deque

import numpy as np

# Defaults chosen for 48 kHz capture. The window holds ~43 ms, enough for
# two periods at 100 Hz; the hop gives a ~5.3 ms update rate.
DEFAULT_WINDOW = 2048
DEFAULT_HOP = 256

# MPM key-maximum cutoff. 0.93 is the value used in the Tartini paper.
MPM_CUTOFF = 0.93

# Below this NSDF clarity the frame has no reliable pitch -- a consonant,
# a breath, or silence. Reporting a number anyway is how tuners end up
# flickering wildly on unvoiced material.
MIN_CLARITY = 0.55

# Frames quieter than this are not sung material.
MIN_FRAME_DB = -55.0

F0_MIN_HZ = 65.0        # C2
F0_MAX_HZ = 1200.0      # ~D6


def hz_to_midi(hz: float) -> float:
    return 69.0 + 12.0 * float(np.log2(max(hz, 1e-9) / 440.0))


def midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((float(midi) - 69.0) / 12.0))


def cents_between(hz: float, ref_hz: float) -> float:
    if hz <= 0 or ref_hz <= 0:
        return 0.0
    return 1200.0 * float(np.log2(hz / ref_hz))


# ═════════════════════════════════════════════════════════════════════════════
# Pitch: McLeod Pitch Method
# ═════════════════════════════════════════════════════════════════════════════

def nsdf(x: np.ndarray) -> np.ndarray:
    """Normalised square difference function.

    `n(tau) = 2 * r(tau) / m(tau)`, where `r` is the autocorrelation and
    `m` the summed squared energy of the two overlapping segments. The
    normalisation is what bounds the result to [-1, 1] and makes the peak
    height usable as a confidence value.

    The autocorrelation is computed by FFT because the direct form is
    O(W^2) and would not keep up at a 5 ms update rate; `m` is built by the
    standard incremental recurrence rather than re-summing per lag.
    """
    w = x.size
    if w < 4:
        return np.zeros(0)

    # Zero-pad to at least 2W to avoid circular wrap in the autocorrelation.
    n_fft = 1 << int(np.ceil(np.log2(2 * w)))
    spec = np.fft.rfft(x, n_fft)
    acf = np.fft.irfft(spec * np.conj(spec), n_fft)[:w]

    power = float(np.dot(x, x))
    m = np.empty(w, dtype=np.float64)
    m[0] = 2.0 * power
    # m(tau) = m(tau-1) - x[tau-1]^2 - x[W-tau]^2
    running = 2.0 * power
    for tau in range(1, w):
        running -= x[tau - 1] * x[tau - 1] + x[w - tau] * x[w - tau]
        m[tau] = running

    out = np.zeros(w, dtype=np.float64)
    ok = m > 1e-12
    out[ok] = 2.0 * acf[ok] / m[ok]
    return np.clip(out, -1.0, 1.0)


def _parabolic_peak(y: np.ndarray, i: int) -> Tuple[float, float]:
    """Sub-sample peak position and height by parabolic interpolation.

    Without this the pitch estimate is quantised to integer lags, which at
    a 2048-sample window is worse than 10 cents in the upper register --
    visible as a staircase on any display and useless for cent-accurate
    feedback.
    """
    if i <= 0 or i >= y.size - 1:
        return float(i), float(y[i])
    a, b, c = float(y[i - 1]), float(y[i]), float(y[i + 1])
    denom = a - 2.0 * b + c
    if abs(denom) < 1e-12:
        return float(i), b
    shift = 0.5 * (a - c) / denom
    return float(i) + shift, b - 0.25 * (a - c) * shift


def _key_maxima(n: np.ndarray) -> List[int]:
    """Indices of the maxima that follow each positive zero crossing.

    MPM's rule: only the highest point between a rising and the next
    falling zero crossing is a candidate. Searching all local maxima
    instead is what produces the classic octave jumps, because harmonic
    peaks sit slightly higher than the fundamental's on some material.
    """
    out: List[int] = []
    i = 1
    size = n.size
    # Skip the initial lobe around tau=0, which is always the global max.
    while i < size and n[i] > 0:
        i += 1
    while i < size - 1:
        if n[i] <= 0 < n[i + 1]:          # positive-going zero crossing
            j = i + 1
            best, best_v = j, n[j]
            while j < size - 1 and n[j] >= 0:
                if n[j] > best_v:
                    best, best_v = j, n[j]
                j += 1
            out.append(best)
            i = j
        else:
            i += 1
    return out


@dataclass
class PitchEstimate:
    hz: float = 0.0
    midi: float = 0.0
    clarity: float = 0.0
    voiced: bool = False

    def to_dict(self) -> dict:
        return {"hz": round(self.hz, 2), "midi": round(self.midi, 3),
                "clarity": round(self.clarity, 3), "voiced": self.voiced}


class PitchTracker:
    """Stateful MPM tracker with octave-continuity and light smoothing.

    Per-frame detectors are stateless and therefore jittery. Keeping one
    frame of history lets two cheap corrections happen that matter a great
    deal for a live display: rejecting octave jumps that contradict the
    previous frame, and smoothing the output just enough to stop the
    readout shimmering without adding perceptible lag.
    """

    def __init__(self, sr: int, cutoff: float = MPM_CUTOFF,
                 fmin: float = F0_MIN_HZ, fmax: float = F0_MAX_HZ,
                 smoothing: float = 0.35):
        self.sr = int(sr)
        self.cutoff = float(cutoff)
        self.fmin, self.fmax = float(fmin), float(fmax)
        self.smoothing = float(np.clip(smoothing, 0.0, 0.95))
        self._prev_midi: Optional[float] = None
        self._prev_voiced = False

    def reset(self) -> None:
        self._prev_midi, self._prev_voiced = None, False

    def __call__(self, frame: np.ndarray) -> PitchEstimate:
        x = np.asarray(frame, dtype=np.float64)
        if x.ndim > 1:
            x = x.mean(axis=1)
        if x.size < 64:
            return PitchEstimate()

        # DC removal. Mandatory for MPM -- an offset drives the NSDF toward
        # 1 everywhere and hides the zero crossings entirely.
        x = x - x.mean()
        if float(np.max(np.abs(x))) < 1e-6:
            self._prev_voiced = False
            return PitchEstimate()

        n = nsdf(x)
        if n.size == 0:
            return PitchEstimate()

        lag_min = max(2, int(self.sr / self.fmax))
        lag_max = min(n.size - 2, int(self.sr / self.fmin))
        maxima = [i for i in _key_maxima(n) if lag_min <= i <= lag_max]
        if not maxima:
            self._prev_voiced = False
            return PitchEstimate()

        peak_val = max(float(n[i]) for i in maxima)
        threshold = self.cutoff * peak_val
        # The *first* qualifying maximum, not the highest: the fundamental
        # is the earliest lag that clears the bar, and taking the global max
        # instead is exactly how a detector locks onto a harmonic.
        chosen = next((i for i in maxima if n[i] >= threshold), maxima[0])

        lag, clarity = _parabolic_peak(n, chosen)
        if lag <= 0:
            return PitchEstimate()
        hz = self.sr / lag
        if not (self.fmin <= hz <= self.fmax):
            self._prev_voiced = False
            return PitchEstimate()

        midi = hz_to_midi(hz)
        voiced = clarity >= MIN_CLARITY

        if voiced and self._prev_voiced and self._prev_midi is not None:
            # Octave repair: a jump of almost exactly 12 semitones between
            # adjacent 5 ms frames is a detector artifact, not singing. No
            # voice moves an octave in one hop.
            delta = midi - self._prev_midi
            for octave in (12.0, -12.0, 24.0, -24.0):
                if abs(delta - octave) < 1.0:
                    midi -= octave
                    hz = midi_to_hz(midi)
                    break
            a = self.smoothing
            midi = a * self._prev_midi + (1.0 - a) * midi
            hz = midi_to_hz(midi)

        self._prev_midi = midi if voiced else None
        self._prev_voiced = voiced
        return PitchEstimate(hz=float(hz), midi=float(midi),
                             clarity=float(clarity), voiced=bool(voiced))


# ═════════════════════════════════════════════════════════════════════════════
# Frame analysis
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class FrameStats:
    """Everything measured about one analysis hop."""
    time_s: float = 0.0
    rms_db: float = -120.0
    peak_db: float = -120.0
    clipped: bool = False
    near_clip: bool = False
    pitch: PitchEstimate = field(default_factory=PitchEstimate)
    low_mid_ratio: float = 0.0       # proximity indicator
    sibilance_ratio: float = 0.0
    is_voice: bool = False
    is_plosive: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pitch"] = self.pitch.to_dict()
        for k in ("rms_db", "peak_db", "low_mid_ratio", "sibilance_ratio", "time_s"):
            d[k] = round(float(d[k]), 3)
        return d


class FrameAnalyzer:
    """Streaming analyser. Feed it blocks; it emits one FrameStats per hop.

    Block size is decoupled from hop size deliberately: an audio callback
    delivers whatever the device gives it, which is rarely the analysis hop,
    and a ring buffer here means the caller never has to care.
    """

    def __init__(self, sr: int = 48000, window: int = DEFAULT_WINDOW,
                 hop: int = DEFAULT_HOP):
        self.sr = int(sr)
        self.window = int(window)
        self.hop = int(hop)
        self._buf = np.zeros(0, dtype=np.float32)
        self._n_seen = 0
        self.tracker = PitchTracker(sr)
        self._noise_floor_db = -70.0
        self._quiet_frames: Deque[float] = deque(maxlen=400)
        self._prev_low = 0.0

    @property
    def noise_floor_db(self) -> float:
        return self._noise_floor_db

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)
        self._n_seen = 0
        self._quiet_frames.clear()
        self.tracker.reset()

    def push(self, block: np.ndarray) -> List[FrameStats]:
        """Add audio, return stats for every complete hop it produced."""
        b = np.asarray(block, dtype=np.float32)
        if b.ndim > 1:
            b = b.mean(axis=1)
        self._buf = np.concatenate([self._buf, b]) if self._buf.size else b

        out: List[FrameStats] = []
        while self._buf.size >= self.window:
            frame = self._buf[:self.window]
            out.append(self._analyze(frame))
            self._buf = self._buf[self.hop:]
            self._n_seen += self.hop
        return out

    def _analyze(self, frame: np.ndarray) -> FrameStats:
        t = self._n_seen / float(self.sr)
        peak = float(np.max(np.abs(frame))) if frame.size else 0.0
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2) + 1e-20))
        rms_db = 20.0 * np.log10(max(rms, 1e-10))
        peak_db = 20.0 * np.log10(max(peak, 1e-10))

        st = FrameStats(time_s=t, rms_db=rms_db, peak_db=peak_db,
                        clipped=peak >= 0.999, near_clip=peak >= 0.891)  # -1 dBFS

        # Track the quietest frames to estimate the room's noise floor while
        # recording, so guidance adapts to the actual environment rather
        # than to an assumed one.
        self._quiet_frames.append(rms_db)
        if len(self._quiet_frames) >= 20:
            self._noise_floor_db = float(np.percentile(self._quiet_frames, 10))

        st.is_voice = rms_db > max(MIN_FRAME_DB, self._noise_floor_db + 10.0)

        if st.is_voice:
            st.pitch = self.tracker(frame)
            mag = np.abs(np.fft.rfft(frame * np.hanning(frame.size)))
            freqs = np.fft.rfftfreq(frame.size, 1.0 / self.sr)
            total = float(np.sum(mag)) + 1e-12
            low = float(np.sum(mag[(freqs >= 80) & (freqs <= 250)])) / total
            mid = float(np.sum(mag[(freqs >= 400) & (freqs <= 2000)])) / total
            st.low_mid_ratio = low / max(mid, 1e-6)
            st.sibilance_ratio = float(
                np.sum(mag[(freqs >= 5000) & (freqs <= 9500)])) / total

            # A plosive is a sudden low-frequency burst with almost no
            # corresponding midrange -- the signature of air hitting the
            # capsule rather than a sung note.
            st.is_plosive = bool(low > 0.45 and mid < 0.12 and low > self._prev_low * 2.0)
            self._prev_low = low
        else:
            self.tracker.reset()
            self._prev_low = 0.0
        return st
