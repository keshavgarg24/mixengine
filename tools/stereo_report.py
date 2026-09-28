"""
What a render's stereo image and transients actually measure.

Written because "the vocals sound too wide" and "it is not crisp" are
both real complaints that no gate in the critic was catching, and neither
can be settled by listening on one pair of speakers. Each number here is
one an engineer would check on a finished master.

    python tools/stereo_report.py FILE [FILE ...]

Correlation is the one to read first. A lead vocal belongs at +1.0: the
same signal in both channels, which is what "centred" means. Anything
below about 0.7 in the vocal band is a lead that has been decorrelated,
and a decorrelated lead is what makes a mix sound washed rather than
close. Mono loss says what a phone speaker throws away.
"""

from __future__ import annotations

import sys
from typing import Dict, List, Tuple

import numpy as np

BANDS: List[Tuple[str, float, float]] = [
    ("low", 20.0, 200.0),
    ("body", 200.0, 800.0),
    ("presence", 800.0, 4000.0),
    ("air", 4000.0, 16000.0),
]


def _load(path: str) -> Tuple[np.ndarray, int]:
    """Read anything the engine can read, at the file's own rate.

    soundfile alone cannot open the m4a a phone records, and comparing a
    render against the take it came from is the main thing this tool is
    for, so it falls back the way the engine does.
    """
    try:
        import soundfile as sf
        y, sr = sf.read(path, always_2d=True, dtype="float32")
        return y, int(sr)
    except Exception:
        import librosa
        y, sr = librosa.load(path, sr=None, mono=False)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 1:
            y = y[:, None]
        elif y.shape[0] < y.shape[1]:
            y = y.T
        return y, int(sr)


def correlation(l: np.ndarray, r: np.ndarray) -> float:
    """Pearson correlation of the two channels, energy-weighted."""
    if l.size == 0:
        return 1.0
    ln, rn = l - l.mean(), r - r.mean()
    d = float(np.sqrt(np.sum(ln * ln) * np.sum(rn * rn)))
    return float(np.sum(ln * rn) / d) if d > 1e-20 else 1.0


def band_filter(y: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    from scipy import signal as sps
    ny = sr / 2.0
    lo_n, hi_n = max(lo / ny, 1e-5), min(hi / ny, 0.999)
    if lo_n >= hi_n:
        return y
    b, a = sps.butter(4, [lo_n, hi_n], btype="band")
    return sps.filtfilt(b, a, y, axis=0)


def crest_db(y: np.ndarray) -> float:
    """Peak over RMS. The headroom a transient still has; limiting eats it."""
    mono = y.mean(axis=1)
    rms = float(np.sqrt(np.mean(np.square(mono.astype(np.float64)))))
    peak = float(np.abs(mono).max())
    if rms <= 1e-12 or peak <= 1e-12:
        return 0.0
    return 20.0 * np.log10(peak / rms)


def short_term_crest(y: np.ndarray, sr: int, win_s: float = 0.4) -> float:
    """Median crest over short windows.

    A whole-file crest factor is dominated by the single loudest moment.
    The median of short windows says whether transients survive *through*
    the track, which is what "crisp" describes.
    """
    mono = y.mean(axis=1).astype(np.float64)
    n = int(win_s * sr)
    if n < 2 or mono.size < n:
        return crest_db(y)
    count = mono.size // n
    out = []
    for i in range(count):
        seg = mono[i * n:(i + 1) * n]
        rms = float(np.sqrt(np.mean(seg * seg)))
        peak = float(np.abs(seg).max())
        if rms > 1e-9 and peak > 1e-9:
            out.append(20.0 * np.log10(peak / rms))
    return float(np.median(out)) if out else 0.0


def report(path: str) -> Dict[str, object]:
    y, sr = _load(path)
    if y.shape[1] == 1:
        y = np.repeat(y, 2, axis=1)
    l, r = y[:, 0].astype(np.float64), y[:, 1].astype(np.float64)
    mid, side = (l + r) / 2.0, (l - r) / 2.0
    m_e = float(np.sum(mid * mid))
    s_e = float(np.sum(side * side))
    total = m_e + s_e

    mono_sum = mid
    full = float(np.sqrt(np.mean(l * l + r * r) / 2.0))
    mono_rms = float(np.sqrt(np.mean(mono_sum * mono_sum)))
    mono_loss = (20.0 * np.log10(mono_rms / full)) if full > 1e-12 and mono_rms > 1e-12 else 0.0

    out: Dict[str, object] = {
        "file": path.rsplit("/", 1)[-1],
        "duration_s": round(y.shape[0] / sr, 1),
        "correlation": round(correlation(l, r), 3),
        "side_energy_pct": round(100.0 * s_e / total, 1) if total > 0 else 0.0,
        "mono_loss_db": round(mono_loss, 2),
        "peak_dbfs": round(20.0 * np.log10(max(float(np.abs(y).max()), 1e-12)), 2),
        "crest_db": round(crest_db(y), 2),
        "short_term_crest_db": round(short_term_crest(y, sr), 2),
    }
    bands: Dict[str, Dict[str, float]] = {}
    for name, lo, hi in BANDS:
        bl = band_filter(y, sr, lo, min(hi, sr / 2.0 - 100))
        be = float(np.sum(bl * bl))
        bands[name] = {
            "correlation": round(correlation(bl[:, 0], bl[:, 1]), 3),
            "energy_pct": round(100.0 * be / max(float(np.sum(y * y)), 1e-20), 1),
        }
    out["bands"] = bands
    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(sr)
        out["lufs"] = round(float(meter.integrated_loudness(y)), 2)
    except Exception:
        pass
    return out


def main(paths: List[str]) -> None:
    rows = [report(p) for p in paths]
    for row in rows:
        print("\n%s  (%.0fs)" % (row["file"], row["duration_s"]))
        print("  correlation      %+.3f   (a centred lead reads near +1.00)"
              % row["correlation"])
        print("  side energy      %5.1f%%" % row["side_energy_pct"])
        print("  mono loss        %+.2f dB" % row["mono_loss_db"])
        print("  peak             %+.2f dBFS" % row["peak_dbfs"])
        print("  crest            %5.2f dB   short-term %5.2f dB"
              % (row["crest_db"], row["short_term_crest_db"]))
        if "lufs" in row:
            print("  loudness         %5.2f LUFS" % row["lufs"])
        print("  per band:")
        for name, b in row["bands"].items():  # type: ignore[union-attr]
            print("    %-9s corr %+.3f   energy %5.1f%%"
                  % (name, b["correlation"], b["energy_pct"]))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    main(sys.argv[1:])
