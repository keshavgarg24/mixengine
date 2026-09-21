"""
Removing a known beat from a vocal recorded over it.

When a singer records on headphones nothing leaks. When they record on
speakers -- a laptop, a phone, a monitor in the room -- the beat comes back
into the microphone, and every later stage inherits it: the pitch tracker
locks onto the bassline, the onset detector fires on the kick, the mix ends
up with two copies of the instrumental at different levels and different
delays.

The usual answer is to run a source separator. That is the wrong tool here,
and the reason is worth being precise about: **a separator does not know
what the beat is, and we do.** Separation is an underdetermined problem
solved by a model's prior. Removing a *known* signal from a mixture is
echo cancellation, which is determined, has a closed-form solution, and can
remove far more than any reference-free separator because it is not
guessing what to remove.

The method is a least-squares transfer-function estimate in the STFT
domain. The bleed path -- speaker, room, microphone -- is linear and close
enough to time-invariant over a take, so one complex gain per frequency bin
describes it. That gain is estimated only on frames where the singer is
*not* singing, which is the single most important decision here: estimating
across the voice makes the filter learn to cancel the voice, and the
failure is quiet and total. This is the same problem echo cancellers call
double-talk, and the same solution -- freeze adaptation when both are
present.

Two things must be handled before any of this works:

  *Offset.* The recording did not start when the beat did. Found by
  cross-correlation of the onset envelopes, which is robust to the bleed
  being quiet and spectrally distorted.

  *Drift.* Consumer playback clocks are not the recording clock. A phone
  speaker running 40 ppm fast walks a full sample every 25 kHz of samples;
  over three minutes that is a hundred milliseconds, and a fixed offset
  stops being correct halfway through. Corrected by warping the reference
  onto the recording before estimating anything.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from . import dsp

log = logging.getLogger("mixengine.debleed")

# The analysis window must be longer than the bleed path's impulse
# response, or the whole method is invalid: convolution is only
# multiplication in the STFT domain when the response fits inside a frame.
# A small room plus a speaker is 100-300 ms, so 4096 samples (93 ms at
# 44.1 kHz) was too short -- the per-bin phase of the estimate then averaged
# inconsistent frames, and subtracting a wrong-phase estimate *adds* energy.
# Measured on a synthetic 120 ms bleed path, that made the gaps between
# phrases 15.6 dB louder than they started. 16384 samples is 371 ms.
N_FFT = 16384
HOP = 4096

# Which frames are taken to be voice-free, as a percentile of the
# recording-to-reference energy ratio. A quarter is deliberately
# pessimistic: a take with no gaps at all should fail this stage rather
# than have its voice fitted as though it were bleed.
QUIET_PERCENTILE = 25.0

# Coherence below this means the reference does not explain what is in the
# bin, so nothing there is cancelled. This is the guard that keeps the
# filter out of the voice, and it matters more than the estimate itself.
COHERENCE_FLOOR = 0.35

# Never attenuate a bin by more than this. Cancelling to nothing leaves
# holes in the spectrum that read as musical noise; leaving a floor keeps
# the residual continuous and far less noticeable than the artifact.
MAX_ATTENUATION_DB = 18.0

# Below this much measured cancellation the filter has not found a real
# bleed path, and applying it only adds processing to a clean recording.
MIN_USEFUL_DB = 2.0

# How many offset candidates to try. A looping beat's correlation has a
# tall peak at every bar, so the true offset is regularly not the first
# suggestion and sometimes not in the first six: measured on one fixture,
# the same beat at three bleed levels put the correct offset first, fourth,
# and outside the top six.
N_CANDIDATES = 14

# Candidates are scored on this much audio rather than the whole take.
PROBE_SECONDS = 30.0


@dataclass
class BleedReport:
    applied: bool = False
    method: str = "stft_wiener"
    offset_s: float = 0.0
    offset_confidence: float = 0.0
    drift_corrected: bool = False
    estimation_frames: int = 0
    total_frames: int = 0
    cancellation_db: float = 0.0
    bleed_level_db: float = -120.0
    note: str = ""

    def to_dict(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


def offset_candidates(mic: np.ndarray, reference: np.ndarray, sr: int,
                      max_offset_s: float = 30.0, k: int = 6
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """The `k` most likely offsets, best first, with their peak strengths.

    Several candidates rather than one, because a beat is periodic and its
    correlation is too. On a four-bar loop the peak one bar out is nearly as
    tall as the true peak, and the tallest is frequently the wrong one --
    measured on a 140 BPM beat, envelope correlation returned an offset
    exactly one bar (1.72 s) early while reporting full confidence. There is
    no way to break that tie from the correlation alone, so the caller
    resolves it by trying each and keeping whichever actually cancels.

    Envelopes rather than waveforms: the bleed has been through a speaker, a
    room and a microphone, so its waveform correlates poorly with the
    reference while its rhythm survives all three intact.
    """
    empty = (np.zeros(0), np.zeros(0))
    try:
        import librosa
    except ImportError:
        return empty
    a = dsp.to_mono(dsp.as_2d(mic)).astype(np.float32)
    b = dsp.to_mono(dsp.as_2d(reference)).astype(np.float32)
    if a.size < sr or b.size < sr:
        return empty
    hop = 512
    ea = librosa.onset.onset_strength(y=a, sr=sr, hop_length=hop)
    eb = librosa.onset.onset_strength(y=b, sr=sr, hop_length=hop)
    if ea.size < 8 or eb.size < 8:
        return empty
    ea = (ea - ea.mean()) / (ea.std() or 1.0)
    eb = (eb - eb.mean()) / (eb.std() or 1.0)

    n = int(2 ** np.ceil(np.log2(ea.size + eb.size)))
    corr = np.fft.irfft(np.fft.rfft(ea, n) * np.conj(np.fft.rfft(eb, n)), n)
    limit = max(1, min(int(max_offset_s * sr / hop), n // 2))
    lags = np.concatenate([np.arange(limit), np.arange(-limit, 0)])
    search = np.concatenate([corr[:limit], corr[-limit:]])

    # Suppress each winner's neighbourhood before taking the next, so the
    # candidates are genuinely different alignments rather than six samples
    # of one peak.
    guard = max(2, int(0.12 * sr / hop))
    work = search.copy()
    picked, strength = [], []
    for _ in range(k):
        i = int(np.argmax(work))
        if not np.isfinite(work[i]) or work[i] <= 0:
            break
        picked.append(float(lags[i] * hop / sr))
        strength.append(float(work[i]))
        work[max(0, i - guard):i + guard + 1] = -np.inf
    return np.asarray(picked), np.asarray(strength)


def find_offset(mic: np.ndarray, reference: np.ndarray, sr: int,
                max_offset_s: float = 30.0) -> Tuple[float, float]:
    """The single best offset guess. `(seconds, confidence)`.

    Confidence reflects how far the winner stands above the runner-up, not
    how tall it is: on periodic material a very tall peak can still be the
    wrong one, and reporting that as certainty is how the whole stage goes
    silently wrong.
    """
    offs, strengths = offset_candidates(mic, reference, sr, max_offset_s, k=3)
    if offs.size == 0:
        return 0.0, 0.0
    if offs.size == 1:
        return float(offs[0]), 0.5
    ratio = float(strengths[1] / (strengths[0] + 1e-12))
    return float(offs[0]), float(np.clip(1.0 - ratio, 0.0, 1.0))


def cancel(mic: np.ndarray, reference: np.ndarray, sr: int, *,
           offset_s: Optional[float] = None,
           correct_drift: bool = True,
           max_attenuation_db: float = MAX_ATTENUATION_DB
           ) -> Tuple[np.ndarray, dict]:
    """Remove `reference` from `mic`. Returns `(cleaned, report)`.

    Declines, and says so, whenever the measured cancellation would not
    justify the processing -- a clean headphone take runs through this and
    comes out untouched.
    """
    rep = BleedReport()
    m = dsp.as_2d(mic)
    r = dsp.as_2d(reference)
    if m.size == 0 or r.size == 0:
        rep.note = "no audio"
        return m, rep.to_dict()

    mono = dsp.to_mono(m)
    r_mono = dsp.to_mono(r)

    # Each candidate offset is tried and scored by the cancellation it
    # actually achieves, which is the only test that distinguishes the true
    # alignment from the one a bar away on a looping beat.
    if offset_s is not None:
        candidates = np.asarray([float(offset_s)])
    else:
        candidates, _ = offset_candidates(m, r, sr, k=N_CANDIDATES)
        if candidates.size == 0:
            candidates = np.asarray([0.0])

    # Candidates are scored on an excerpt rather than the whole take. Each
    # trial is a full STFT pass over both signals, and twelve of those on a
    # three-minute take is minutes of work to answer a question that a
    # thirty-second window settles just as well.
    probe = min(len(mono), int(sr * PROBE_SECONDS))
    best_offset, best_score = None, -np.inf
    for cand in candidates[:N_CANDIDATES]:
        ref = _place(r_mono, sr, float(cand), len(m))
        if float(np.max(np.abs(ref[:probe]))) < 1e-6:
            continue
        _, stats = _wiener_cancel(mono[:probe], ref[:probe], sr,
                                  max_attenuation_db)
        if stats["cancellation_db"] > best_score:
            best_offset, best_score = float(cand), stats["cancellation_db"]
    if best_offset is None:
        rep.note = "the reference does not overlap the recording"
        return m, rep.to_dict()

    offset_s = best_offset
    ref = _place(r_mono, sr, offset_s, len(m))
    cleaned_mono, stats = _wiener_cancel(mono, ref, sr, max_attenuation_db)
    rep.offset_s = offset_s
    rep.offset_confidence = float(np.clip(
        stats["cancellation_db"] / 12.0, 0.0, 1.0))

    if correct_drift and stats["cancellation_db"] >= MIN_USEFUL_DB:
        ref2, drifted = _correct_drift(mono, ref, sr)
        if drifted:
            cleaned2, stats2 = _wiener_cancel(mono, ref2, sr, max_attenuation_db)
            if stats2["cancellation_db"] > stats["cancellation_db"]:
                cleaned_mono, stats = cleaned2, stats2
                rep.drift_corrected = True

    rep.estimation_frames = int(stats["estimation_frames"])
    rep.total_frames = int(stats["total_frames"])
    rep.cancellation_db = stats["cancellation_db"]
    rep.bleed_level_db = stats["bleed_level_db"]

    if rep.estimation_frames < 8:
        rep.note = ("the singer is audible in almost every frame, so there is "
                    "nowhere to measure the bleed path from")
        return m, rep.to_dict()
    if rep.cancellation_db < MIN_USEFUL_DB:
        rep.note = (f"only {rep.cancellation_db:.1f} dB of the reference is "
                    f"present; the take is effectively clean")
        return m, rep.to_dict()

    rep.applied = True
    log.info("  debleed: removed %.1f dB of beat bleed (offset %.2fs, "
             "%d/%d frames used to estimate)", rep.cancellation_db,
             rep.offset_s, rep.estimation_frames, rep.total_frames)

    if m.shape[1] == 1:
        return cleaned_mono[:, None].astype(np.float32), rep.to_dict()

    # Each channel gets its own filter estimate against the same reference.
    # A two-microphone or stereo recording has two bleed paths, so one
    # filter is wrong for at least one of them -- and the alternative of
    # deriving a per-sample gain from the mono result divides by samples
    # that are near zero in exactly the gaps where the bleed is the whole
    # signal, which is where it must be most accurate.
    out = np.empty_like(m)
    for c in range(m.shape[1]):
        chan, _ = _wiener_cancel(m[:, c].astype(np.float32), ref, sr,
                                 max_attenuation_db)
        out[:, c] = chan
    return out.astype(np.float32), rep.to_dict()


# ─────────────────────────────────────────────────────────────────────────────

def _place(ref: np.ndarray, sr: int, offset_s: float, n: int) -> np.ndarray:
    """Shift the reference by `offset_s` and pad or trim it to `n` samples."""
    shift = int(round(offset_s * sr))
    out = np.zeros(n, dtype=np.float32)
    if shift >= 0:
        take = min(len(ref), n - shift) if shift < n else 0
        if take > 0:
            out[shift:shift + take] = ref[:take]
    else:
        src = -shift
        take = min(len(ref) - src, n)
        if take > 0:
            out[:take] = ref[src:src + take]
    return out


def _correct_drift(mic: np.ndarray, ref: np.ndarray, sr: int
                   ) -> Tuple[np.ndarray, bool]:
    """Warp the reference onto the recording's clock.

    Offsets are measured in windows across the take and fitted; a non-zero
    slope is the playback clock running fast or slow relative to the
    recording clock. Corrected by resampling the reference, which is
    cheaper and safer than warping the take -- the reference is disposable
    and the take is not.
    """
    win = int(sr * 10.0)
    if len(mic) < win * 3:
        return ref, False
    centres, offsets = [], []
    for start in range(0, len(mic) - win, win):
        a = mic[start:start + win]
        b = ref[start:start + win]
        if float(np.max(np.abs(b))) < 1e-5:
            continue
        off, conf = find_offset(a[:, None], b[:, None], sr, max_offset_s=0.4)
        if conf > 0.15:
            centres.append((start + win / 2) / sr)
            offsets.append(off)
    if len(centres) < 3:
        return ref, False

    slope, intercept = np.polyfit(np.asarray(centres), np.asarray(offsets), 1)
    total = abs(slope) * (len(mic) / sr)
    # Below a millisecond over the whole take, resampling costs more in
    # interpolation error than the drift costs in cancellation.
    if total < 0.001:
        return ref, False

    t = np.arange(len(ref), dtype=np.float64) / sr
    src = t - (slope * t + intercept)
    warped = np.interp(src, t, ref).astype(np.float32)
    log.info("  debleed: playback clock drifts %.0f ppm (%.0f ms over the "
             "take); resampling the reference", slope * 1e6, total * 1000)
    return warped, True


def _wiener_cancel(mic: np.ndarray, ref: np.ndarray, sr: int,
                   max_attenuation_db: float
                   ) -> Tuple[np.ndarray, Dict[str, float]]:
    """One complex gain per bin, estimated where the singer is silent.

    Two safeguards, both learned by getting this wrong first:

    **Estimate on the quietest frames only.** The first version accepted any
    frame within 6 dB of the quietest as voice-free, which on a take that is
    half singing admitted more than half the frames. The filter then fit the
    voice as well as the bleed and removed it: cancellation looked like a
    healthy 6 dB while the error against the known clean vocal went from
    -18 dB to -3 dB. A bottom quartile is used instead, and the estimate is
    only as good as the singer's willingness to stop occasionally.

    **Only cancel where the reference explains the recording.** Per-bin
    magnitude-squared coherence between the take and the reference says how
    much of what is at that frequency came from the beat. Where coherence
    is low -- which is exactly where the voice lives -- the gain is left at
    unity. This is what stops the filter reaching into the voice even if a
    voiced frame slipped into the estimate, and it is what makes the whole
    approach safe rather than merely usually-correct.
    """
    stats = {"estimation_frames": 0, "total_frames": 0,
             "cancellation_db": 0.0, "bleed_level_db": -120.0,
             "mean_coherence": 0.0}
    win = np.hanning(N_FFT).astype(np.float32)
    # Pad both sides by a full window so the overlap-add edges, where the
    # reconstruction is not valid, fall outside the real audio.
    pad = N_FFT
    mic_p = np.pad(mic, (pad, pad))
    ref_p = np.pad(ref, (pad, pad))
    D = _stft(mic_p, win, HOP)
    X = _stft(ref_p, win, HOP)
    frames = min(D.shape[0], X.shape[0])
    if frames < 16:
        return mic, stats
    D, X = D[:frames], X[:frames]
    stats["total_frames"] = int(frames)

    d_pow = np.sum(np.abs(D) ** 2, axis=1)
    x_pow = np.sum(np.abs(X) ** 2, axis=1)
    if not np.any(x_pow > 0):
        return mic, stats
    active = x_pow > float(np.median(x_pow[x_pow > 0])) * 0.15
    ok = active & (x_pow > 0)
    if int(ok.sum()) < 16:
        return mic, stats

    ratio = np.full(frames, np.inf)
    ratio[ok] = d_pow[ok] / x_pow[ok]
    finite = ratio[np.isfinite(ratio)]
    cut = float(np.percentile(finite, QUIET_PERCENTILE))
    quiet = ok & (ratio <= cut)
    stats["estimation_frames"] = int(quiet.sum())
    if quiet.sum() < 8:
        return mic, stats

    Dq, Xq = D[quiet], X[quiet]
    Sxy = np.sum(Dq * np.conj(Xq), axis=0)
    Sxx = np.sum(np.abs(Xq) ** 2, axis=0) + 1e-12
    Syy = np.sum(np.abs(Dq) ** 2, axis=0) + 1e-12
    H = Sxy / Sxx
    coh = np.clip((np.abs(Sxy) ** 2) / (Sxx * Syy), 0.0, 1.0)
    stats["mean_coherence"] = float(np.mean(coh))

    # Coherence below the floor means the reference does not explain this
    # bin; above it, the gain ramps in over the remaining range.
    weight = np.clip((coh - COHERENCE_FLOOR) / (1.0 - COHERENCE_FLOOR),
                     0.0, 1.0)
    est = X * (H * weight)[None, :]

    floor_lin = 10.0 ** (-max_attenuation_db / 20.0)
    resid = D - est
    mag = np.abs(resid)
    # A canceller may only ever remove. Clamping the result to the input
    # magnitude makes that structurally true rather than something the
    # estimate has to be good enough to guarantee: whenever the estimate is
    # wrong in phase, subtraction *adds*, and without this clamp the stage
    # can make a recording louder than it found it.
    keep = np.clip(mag, np.abs(D) * floor_lin, np.abs(D))
    phase = np.where(mag > 1e-12, resid / (mag + 1e-12), 1.0)
    Y = keep * phase

    before = float(np.sum(np.abs(Dq) ** 2))
    after = float(np.sum(np.abs(Y[quiet]) ** 2))
    if before > 0 and after > 0:
        stats["cancellation_db"] = float(10.0 * np.log10(before / after))
    bleed = float(np.sum(np.abs(est[quiet]) ** 2))
    if bleed > 0:
        stats["bleed_level_db"] = float(
            10.0 * np.log10(bleed / max(before, 1e-12)))

    full = _istft(Y, win, HOP, len(mic_p) + N_FFT)
    return full[pad:pad + len(mic)].copy(), stats


def _stft(x: np.ndarray, win: np.ndarray, hop: int) -> np.ndarray:
    n_fft = win.size
    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size))
    n_frames = 1 + (x.size - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    return np.fft.rfft(x[idx] * win[None, :], axis=1)


def _istft(X: np.ndarray, win: np.ndarray, hop: int, n: int) -> np.ndarray:
    """Weighted overlap-add. Samples with insufficient window coverage are
    left at zero rather than divided by a near-zero normaliser.

    The first and last half-window of any STFT are covered by only part of
    the window sum. In an unmodified round trip the numerator vanishes with
    the denominator and the division is harmless -- which is exactly why a
    round-trip test that skipped those regions passed while the real path
    produced a sample of 22.0 from an input whose peak was 0.47. Once the
    spectrum has been modified the two no longer cancel. Callers pad their
    input so that the zeroed edges fall outside the audio they care about.
    """
    n_fft = win.size
    frames = np.fft.irfft(X, n=n_fft, axis=1) * win[None, :]
    out = np.zeros(hop * (X.shape[0] - 1) + n_fft, dtype=np.float64)
    norm = np.zeros_like(out)
    w2 = win.astype(np.float64) ** 2
    for i in range(X.shape[0]):
        s = i * hop
        out[s:s + n_fft] += frames[i]
        norm[s:s + n_fft] += w2
    covered = norm > float(np.max(norm)) * 0.5 if norm.size else norm > 0
    out = np.where(covered, out / np.maximum(norm, 1e-12), 0.0)
    return dsp.pad_to(out[:n].astype(np.float32)[:, None], n)[:, 0]
