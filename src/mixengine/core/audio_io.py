"""
Audio loading, saving, and input quality probing.

User uploads are unpredictable: phone recordings, upsampled MP3s, clipped
takes, fake stereo. This module normalises all of that into a known shape
and, critically, *reports* what it found. That report drives every adaptive
parameter downstream -- a mix chain can only adapt to problems it knows
about.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict, field
from typing import Optional, Tuple

import numpy as np

from ..audio import dsp
from ..config import SR
from ..core.capabilities import CAPS

log = logging.getLogger("mixengine.io")

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".aiff", ".aif", ".wma"}


# ─────────────────────────────────────────────────────────────────────────────
# Quality report
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AudioQuality:
    path: str = ""
    duration_s: float = 0.0
    sample_rate: int = 0
    channels: int = 0
    peak_db: float = -np.inf
    true_peak_db: float = -np.inf
    rms_db: float = -np.inf
    noise_floor_db: float = -np.inf
    snr_db: float = 0.0
    clipping_pct: float = 0.0
    bandwidth_hz: float = 0.0
    dc_offset: float = 0.0
    is_fake_stereo: bool = False
    is_inverted_stereo: bool = False
    is_silent: bool = False
    estimated_rt60_s: float = 0.0
    warnings: list = field(default_factory=list)
    # What `load` changed about the file before anything measured it, in
    # words meant for the person who uploaded it.
    repairs: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.is_silent and self.duration_s > 0.5

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, float) and not np.isfinite(v):
                d[k] = None
        return d


# Below this L/R correlation the second channel is the first with its
# polarity flipped -- a mis-wired cable or interface, not a stereo image.
INVERTED_STEREO_CORR = -0.9


def _channel_correlation(y2: np.ndarray) -> Optional[float]:
    """Pearson correlation of the two channels; None unless both carry signal."""
    if y2.shape[1] != 2:
        return None
    a, b = y2[:, 0], y2[:, 1]
    if np.std(a) <= 1e-9 or np.std(b) <= 1e-9:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def probe_quality(y: np.ndarray, sr: int, path: str = "") -> AudioQuality:
    """Measure everything the downstream chain needs to adapt to."""
    y2 = dsp.as_2d(y)
    q = AudioQuality(
        path=path,
        duration_s=len(y2) / sr,
        sample_rate=sr,
        channels=y2.shape[1],
    )

    # Polarity-inverted channels cancel when summed to mono, which every
    # measurement below and the engine itself does: the voice then reads
    # as silence or noise. Measure the take with the second channel flipped
    # back; `load` applies the same flip to the audio it hands out.
    corr = _channel_correlation(y2)
    if corr is not None and corr < INVERTED_STEREO_CORR:
        q.is_inverted_stereo = True
        y2 = y2 * np.array([1.0, -1.0], dtype=y2.dtype)
        corr = -corr
    mono = dsp.to_mono(y2)

    if len(mono) == 0 or np.max(np.abs(mono)) < 1e-6:
        q.is_silent = True
        q.warnings.append("audio is silent or near-silent")
        return q

    q.peak_db = dsp.peak_db(y2)
    q.true_peak_db = dsp.true_peak_db(y2)
    q.rms_db = dsp.rms_db(y2)
    q.noise_floor_db = dsp.noise_floor_db(y2, sr)
    q.snr_db = float(q.rms_db - q.noise_floor_db)
    q.dc_offset = float(np.mean(mono))

    # Clipping: samples at or extremely near full scale.
    clipped = int(np.sum(np.abs(mono) >= 0.9995))
    q.clipping_pct = 100.0 * clipped / max(1, len(mono))

    # Bandwidth: locate the high-frequency cliff. An upsampled 128 kbps MP3
    # shows a hard wall around 15-16 kHz; a real 44.1 kHz recording doesn't.
    q.bandwidth_hz = _estimate_bandwidth(mono, sr)

    # Fake stereo: two channels carrying the same signal.
    q.is_fake_stereo = corr is not None and corr > 0.9995

    q.estimated_rt60_s = _estimate_rt60(mono, sr)

    # -- Warnings ----------------------------------------------------------
    if q.clipping_pct > 0.5:
        q.warnings.append(
            f"{q.clipping_pct:.2f}% of samples are clipped - distortion is "
            f"baked in and cannot be fully removed")
    if q.snr_db < 12:
        q.warnings.append(
            f"low signal-to-noise ratio ({q.snr_db:.1f} dB) - background "
            f"noise will be audible")
    if q.bandwidth_hz < 13000:
        q.warnings.append(
            f"limited bandwidth ({q.bandwidth_hz/1000:.1f} kHz) - source is "
            f"likely a low-bitrate file; top end will sound dull")
    if q.estimated_rt60_s > 0.55:
        q.warnings.append(
            f"heavy room reverb (~{q.estimated_rt60_s:.2f}s) - this limits "
            f"how tight the final mix can sound")
    if abs(q.dc_offset) > 0.01:
        q.warnings.append("DC offset present - will be removed")
    return q


def _estimate_bandwidth(mono: np.ndarray, sr: int) -> float:
    """Find where the spectrum falls off, i.e. the effective bandwidth."""
    f, mag = dsp.long_term_spectrum(mono, sr)
    if len(f) < 8:
        return float(sr / 2)
    band = (f > 200) & (f < sr * 0.48)
    if not np.any(band):
        return float(sr / 2)
    ref = float(np.percentile(mag[band], 90.0))
    # Highest frequency still within 35 dB of the reference level.
    above = f[band][mag[band] > ref - 35.0]
    return float(above[-1]) if above.size else float(sr / 2)


def _estimate_rt60(mono: np.ndarray, sr: int) -> float:
    """Crude reverb-time estimate from the decay after energy peaks.

    Not a calibrated RT60 -- it's a relative indicator good enough to tell a
    dry booth from a bathroom, which is the decision the pipeline needs.
    """
    frame, hop = int(0.02 * sr), int(0.01 * sr)
    r = dsp.frame_rms(mono, frame, hop)
    if len(r) < 30:
        return 0.0
    r_db = dsp.lin_to_db(r)

    decays = []
    peaks = np.where(r_db > np.percentile(r_db, 85))[0]
    for p in peaks[::max(1, len(peaks) // 30)]:
        seg = r_db[p:p + int(1.2 / (hop / sr))]
        if len(seg) < 10:
            continue
        start = seg[0]
        drop = np.where(seg < start - 20.0)[0]
        if drop.size:
            decays.append(drop[0] * hop / sr * 3.0)   # -20 dB -> extrapolate
    if not decays:
        return 0.0
    return float(np.clip(np.median(decays), 0.0, 3.0))


# ─────────────────────────────────────────────────────────────────────────────
# Load / save
# ─────────────────────────────────────────────────────────────────────────────

def _undecodable(path: str, err: Exception) -> str:
    why = str(err).strip() or "no decoder could read it"
    return ("%s could not be decoded as audio (%s). It is not a valid audio "
            "file or it is damaged; export it again as WAV or MP3 and upload "
            "that." % (os.path.basename(path), why))


def decode_check(path: str) -> None:
    """Raise ValueError, in plain words, if `path` cannot be decoded.

    Cheap: it reads a header, or at most one buffer. Run at upload time
    so a damaged file is refused on the spot rather than failing a job a
    minute later with a decoder's empty exception.
    """
    try:
        import soundfile as sf
        sf.info(path)
        return
    except Exception:
        pass
    try:
        import audioread
        with audioread.audio_open(path) as f:
            for _ in f:
                break
    except Exception as e:
        raise ValueError(_undecodable(path, e)) from e


def _native_rate(path: str) -> Optional[int]:
    try:
        import librosa
        return int(librosa.get_samplerate(path))
    except Exception:
        return None


def load(path: str, sr: int = SR, mono: bool = False,
         normalize_format: bool = True) -> Tuple[np.ndarray, int, AudioQuality]:
    """Load audio as (n_samples, n_channels) float32 at `sr`.

    Returns `(audio, sr, quality_report)`. Every change made to the file
    on the way in is written into `quality.repairs`, in words, so the
    person who uploaded it can be told.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"audio file not found: {path}")
    if os.path.getsize(path) == 0:
        raise ValueError(f"audio file is empty: {path}")

    import librosa
    try:
        y, _ = librosa.load(path, sr=sr, mono=mono)
    except Exception as e:
        raise ValueError(_undecodable(path, e)) from e
    y = dsp.as_2d(y)

    q = probe_quality(y, sr, path)

    if normalize_format:
        native = _native_rate(path)
        if native and native < 44100:
            q.repairs.append("resampled from %d Hz to %d Hz; the file has no "
                             "sound above %d Hz" % (native, sr, native // 2))
        if abs(q.dc_offset) > 1e-4:
            y = y - np.mean(y, axis=0, keepdims=True)
            q.repairs.append("a DC offset of %+.3f was removed" % q.dc_offset)
        if q.is_inverted_stereo and y.shape[1] == 2:
            y = y * np.array([1.0, -1.0], dtype=y.dtype)
            q.repairs.append("the two channels were polarity-inverted copies "
                             "of each other; one was flipped back so the "
                             "voice does not cancel in mono")
        if q.is_fake_stereo and y.shape[1] == 2:
            y = y[:, :1]
            q.repairs.append("the two channels were identical and were "
                             "collapsed to one")
        if q.clipping_pct > 0.05:
            y = declip(y)
            q.repairs.append("clipped peaks were reconstructed (%.2f%% of "
                             "samples sat at full scale); some distortion "
                             "is baked into the recording"
                             % q.clipping_pct)
        for line in q.repairs:
            log.info("%s: %s", os.path.basename(path), line)

    return y.astype(np.float32), sr, q


def declip(y: np.ndarray, threshold: float = 0.9995) -> np.ndarray:
    """Soft-repair clipped regions by cubic interpolation across flat tops.

    Cannot restore information that was never recorded, but replacing hard
    flat tops with a smooth arc removes the harsh odd harmonics that make
    clipping so audible.
    """
    y2 = dsp.as_2d(y).copy()
    for c in range(y2.shape[1]):
        ch = y2[:, c]
        mask = np.abs(ch) >= threshold
        if not np.any(mask):
            continue
        # Group consecutive clipped samples.
        idx = np.where(mask)[0]
        splits = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
        for run in splits:
            if len(run) < 2 or len(run) > 512:
                continue
            a, b = run[0] - 1, run[-1] + 1
            if a < 1 or b >= len(ch) - 1:
                continue
            xs = np.array([a - 1, a, b, b + 1])
            ys = ch[xs]
            try:
                poly = np.polyfit(xs, ys, 3)
                ch[run] = np.polyval(poly, run)
            except Exception:
                continue
        y2[:, c] = np.clip(ch, -1.0, 1.0)
    return y2


def save(path: str, y: np.ndarray, sr: int = SR, subtype: str = "PCM_24") -> str:
    """Write audio, creating parent directories as needed."""
    import soundfile as sf
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    y2 = dsp.as_2d(y)
    y2 = np.clip(y2, -1.0, 1.0)
    if y2.shape[1] == 1:
        y2 = y2[:, 0]
    sf.write(path, y2, sr, subtype=subtype)
    return path


def list_audio_files(directory: str) -> list:
    """All audio files in a directory, sorted, non-recursive."""
    if not os.path.isdir(directory):
        return []
    out = [
        os.path.join(directory, f)
        for f in sorted(os.listdir(directory))
        if os.path.splitext(f)[1].lower() in AUDIO_EXTS and not f.startswith(".")
    ]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Loudness
# ─────────────────────────────────────────────────────────────────────────────

def integrated_lufs(y: np.ndarray, sr: int = SR) -> float:
    """ITU-R BS.1770 integrated loudness, with a fallback if pyloudnorm
    is unavailable."""
    y2 = dsp.as_2d(y)
    if len(y2) < sr * 0.4:
        return -np.inf
    if CAPS.pyloudnorm:
        try:
            import pyloudnorm as pyln
            meter = pyln.Meter(sr)
            data = y2[:, 0] if y2.shape[1] == 1 else y2
            val = float(meter.integrated_loudness(data))
            return val if np.isfinite(val) else -np.inf
        except Exception as e:
            log.debug("pyloudnorm failed (%s), using RMS fallback", e)
    # Fallback: K-weighted RMS approximation.
    filtered = dsp.highpass(y2, sr, 60.0, order=2)
    filtered = dsp.shelf_eq(filtered, sr, 1500.0, 4.0, kind="high")
    return float(dsp.rms_db(filtered) - 3.0)


def loudness_region_lufs(y: np.ndarray, sr: int,
                         regions) -> float:
    """Loudness measured over selected regions only.

    This is the correct way to measure a vocal's level: integrated loudness
    across the whole file is dragged down by the silence between phrases, so
    normalising to it makes the vocal too loud while actually singing.
    """
    y2 = dsp.as_2d(y)
    if not regions:
        return integrated_lufs(y2, sr)
    chunks = [y2[max(0, s):min(len(y2), e)] for s, e in regions]
    chunks = [c for c in chunks if len(c) > 0]
    if not chunks:
        return integrated_lufs(y2, sr)
    return integrated_lufs(np.vstack(chunks), sr)


def normalize_lufs(y: np.ndarray, sr: int, target_lufs: float,
                   max_gain_db: float = 24.0) -> np.ndarray:
    current = integrated_lufs(y, sr)
    if not np.isfinite(current):
        return dsp.as_2d(y)
    gain_db = float(np.clip(target_lufs - current, -max_gain_db, max_gain_db))
    return (dsp.as_2d(y) * dsp.db_to_lin(gain_db)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# JSON helpers
# ─────────────────────────────────────────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    """Serialises numpy scalars/arrays that leak into DNA dictionaries."""

    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            return v if np.isfinite(v) else None
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


def write_json(path: str, data: dict) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)
    return path


def read_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        log.warning("could not read %s: %s", path, e)
        return None


def content_key(path: str, version: str = "", chunk: int = 1 << 20) -> str:
    """Content hash of a file, salted with an analyser version.

    Caching analysis by content rather than by filename is what makes
    re-uploading or renaming the same audio free, and what makes editing a
    file in place actually re-analyse it. `version` is mixed in so that
    bumping the analyser invalidates exactly the entries that need it.

    Lives here rather than in either caller because it had been written
    twice -- once in the analysis layer, once in the service -- with
    different version salts, so the CLI and the web importer wrote
    different filenames for the same beat and the catalog listed it twice.
    """
    import hashlib
    h = hashlib.sha256()
    h.update(str(version).encode())
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:24]
