"""
Source separation and vocal restoration.

Two jobs:
  1. Split a beat into drums / bass / other. This is the highest-leverage
     step in the whole system -- it lets the renderer pitch-shift only the
     tonal stems (leaving drums untouched), duck without pumping the
     groove, and carve masking frequencies surgically.
  2. Isolate vocals from a user upload, but only when the upload actually
     needs it. Running separation on an already-clean acapella adds
     artifacts for no benefit, so we classify first.

Model preference: Mel-Band RoFormer (current SDR leader for vocals) via
`audio-separator`, falling back to Demucs htdemucs, falling back to using
the input unchanged.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np

from ..core import audio_io
from . import dsp
from ..core.capabilities import CAPS
from ..config import SR, ANALYSIS_SR

log = logging.getLogger("mixengine.separation")

STEM_NAMES = ("drums", "bass", "other", "vocals")


# ─────────────────────────────────────────────────────────────────────────────
# Input classification
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VocalInputType:
    kind: str          # "clean_acapella" | "light_bleed" | "full_mix"
    confidence: float
    needs_separation: bool
    reason: str


def classify_vocal_input(y: np.ndarray, sr: int) -> VocalInputType:
    """Decide whether the upload needs separation at all.

    Looks for the signatures of instrumental accompaniment: sustained
    sub-bass energy (a voice has almost none below ~70 Hz), strong
    percussive transients, and a wide spread of simultaneous pitch classes.
    """
    import librosa
    mono = dsp.to_mono(y).astype(np.float32)
    if len(mono) < sr:
        return VocalInputType("clean_acapella", 0.3, False, "too short to classify")

    try:
        if sr != ANALYSIS_SR:
            mono_a = librosa.resample(mono, orig_sr=sr, target_sr=ANALYSIS_SR)
        else:
            mono_a = mono

        f, mag = dsp.long_term_spectrum(mono_a, ANALYSIS_SR)
        lin = 10.0 ** (mag / 20.0)
        total = float(np.sum(lin)) + 1e-12
        sub_energy = float(np.sum(lin[(f > 20) & (f < 80)])) / total

        harm, perc = librosa.effects.hpss(mono_a)
        perc_ratio = float(np.sum(perc ** 2) / (np.sum(mono_a ** 2) + 1e-12))

        chroma = librosa.feature.chroma_cqt(y=harm, sr=ANALYSIS_SR)
        # Polyphony proxy: how many pitch classes are simultaneously strong.
        active = np.mean(np.sum(chroma > 0.55, axis=0))

        score = (min(sub_energy / 0.05, 1.0) * 0.4
                 + min(perc_ratio / 0.35, 1.0) * 0.35
                 + min(active / 4.5, 1.0) * 0.25)

        if score > 0.75:
            return VocalInputType("full_mix", float(score), True,
                                  f"instrumental content detected (score {score:.2f})")
        # Bleed is reported but does not by itself justify separation.
        # Demucs costs minutes and rewrites the take; spending that on a
        # coin-flip reading is how a clean vocal got separated on a score
        # of 0.56. The threshold matches `policy.SEPARATION_CONFIDENCE`,
        # and `separate=always` remains available when the call is wrong.
        if score > 0.32:
            return VocalInputType("light_bleed", float(score), False,
                                  f"background bleed detected (score {score:.2f}) "
                                  f"-- not enough to justify separating")
        return VocalInputType("clean_acapella", float(1.0 - score), False,
                              f"clean isolated vocal (score {score:.2f})")
    except Exception as e:
        log.warning("input classification failed (%s); assuming separation needed", e)
        return VocalInputType("full_mix", 0.4, True, "classification failed")


# ─────────────────────────────────────────────────────────────────────────────
# Separation backends
# ─────────────────────────────────────────────────────────────────────────────

def separate(path: str, out_dir: str, want: str = "all",
             model: Optional[str] = None) -> Dict[str, str]:
    """Separate `path` into stems. Returns {stem_name: file_path}.

    Never raises -- on total failure returns an empty dict and the caller
    proceeds without stems (degraded but functional).
    """
    os.makedirs(out_dir, exist_ok=True)

    if CAPS.audio_separator:
        out = _separate_roformer(path, out_dir, model)
        if out:
            return out
    if CAPS.demucs:
        out = _separate_demucs(path, out_dir)
        if out:
            return out

    log.warning("no separation backend available for %s", os.path.basename(path))
    return {}


def _separate_roformer(path: str, out_dir: str,
                       model: Optional[str] = None) -> Dict[str, str]:
    """Mel-Band RoFormer via the `audio-separator` package.

    Note this yields a 2-stem split (vocals / instrumental). For beats we
    want 4 stems, so the caller runs Demucs for those; RoFormer is used
    where vocal quality specifically matters.
    """
    try:
        from audio_separator.separator import Separator
        model = model or "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
        sep = Separator(output_dir=out_dir, output_format="WAV",
                        log_level=logging.WARNING)
        sep.load_model(model_filename=model)
        files = sep.separate(path)

        out: Dict[str, str] = {}
        for fp in files:
            full = fp if os.path.isabs(fp) else os.path.join(out_dir, fp)
            low = os.path.basename(full).lower()
            if "vocal" in low:
                out["vocals"] = full
            elif "instrument" in low or "no_vocal" in low:
                out["instrumental"] = full
        if out:
            log.info("RoFormer separated %s -> %s",
                     os.path.basename(path), list(out))
        return out
    except Exception as e:
        log.warning("RoFormer separation failed (%s)", e)
        return {}


# Accelerators that could not run the separator in this process. Remembered
# so the second file goes straight to the CPU rather than paying for the
# same failure again.
_UNUSABLE_DEVICES: Set[str] = set()


def _separate_demucs(path: str, out_dir: str,
                     model_name: str = "htdemucs") -> Dict[str, str]:
    """Demucs 4-stem separation.

    Importantly this does NOT hand-chunk the input. Demucs handles long
    files with its own overlap-add; forcing tiny chunks with --no-split
    starves the transformer of temporal context, which is exactly where it
    is strongest, and produces inconsistent separation character between
    chunks.
    """
    try:
        cmd = [sys.executable, "-m", "demucs.separate",
               "-n", model_name, "-o", out_dir, "--filename",
               "{stem}.{ext}", path]
        # The accelerator first, then the CPU. Not every op htdemucs needs
        # exists on every backend -- on Apple's MPS the first convolution
        # fails outright ("output channels > 65536 not supported") -- and
        # a failed accelerator run must cost a retry, not the stems.
        devices = [CAPS.device, "cpu"] if CAPS.device != "cpu" else ["cpu"]
        devices = [d for d in devices if d not in _UNUSABLE_DEVICES] or ["cpu"]
        proc = None
        for device in devices:
            log.info("running demucs on %s (%s) ...", os.path.basename(path), device)
            proc = subprocess.run(cmd + ["-d", device], capture_output=True,
                                  text=True, timeout=3600)
            if proc.returncode == 0:
                break
            if device != "cpu":
                _UNUSABLE_DEVICES.add(device)
            tail = (proc.stderr or "").strip().splitlines()
            log.log(logging.INFO if device != devices[-1] else logging.WARNING,
                    "demucs on %s exited %d: %s", device, proc.returncode,
                    tail[-1] if tail else "")
        if proc is None or proc.returncode != 0:
            return {}

        stem_dir = os.path.join(out_dir, model_name)
        out: Dict[str, str] = {}
        for root, _, files in os.walk(stem_dir):
            for f in files:
                name = os.path.splitext(f)[0].lower()
                if name in STEM_NAMES:
                    out[name] = os.path.join(root, f)
        if out:
            log.info("demucs produced stems: %s", sorted(out))
        return out
    except subprocess.TimeoutExpired:
        log.error("demucs timed out on %s", path)
        return {}
    except Exception as e:
        log.warning("demucs separation failed (%s)", e)
        return {}


def separate_beat_stems(path: str, out_dir: str) -> Dict[str, str]:
    """4-stem split for a catalog beat. Cached by output existence."""
    expected = {n: os.path.join(out_dir, f"{n}.wav") for n in STEM_NAMES}
    if all(os.path.exists(p) for p in list(expected.values())[:3]):
        log.info("stems already present for %s", os.path.basename(out_dir))
        return {k: v for k, v in expected.items() if os.path.exists(v)}

    with tempfile.TemporaryDirectory() as tmp:
        stems = _separate_demucs(path, tmp)
        if not stems:
            return {}
        os.makedirs(out_dir, exist_ok=True)
        out: Dict[str, str] = {}
        for name, src in stems.items():
            y, sr, _ = audio_io.load(src, sr=SR)
            dst = os.path.join(out_dir, f"{name}.wav")
            audio_io.save(dst, y, sr, subtype="PCM_16")
            out[name] = dst
        return out


def load_stems(stem_paths: Dict[str, str], sr: int = SR) -> Dict[str, np.ndarray]:
    """Load stem files into arrays, skipping any that fail."""
    out: Dict[str, np.ndarray] = {}
    for name, path in (stem_paths or {}).items():
        if not path or not os.path.exists(path):
            continue
        try:
            y, _, _ = audio_io.load(path, sr=sr)
            out[name] = y
        except Exception as e:
            log.warning("could not load stem %s: %s", name, e)
    return out


def tonal_sum(stems: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    """Bass + other: everything that carries pitch.

    This is the signal that gets pitch-shifted, ducked, and spectrally
    carved. Keeping it separate from the drums is what makes those
    operations musical rather than destructive.
    """
    parts = [stems[k] for k in ("bass", "other") if k in stems]
    if not parts:
        return None
    n = max(len(p) for p in parts)
    return np.sum([dsp.pad_to(p, n) for p in parts], axis=0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Restoration
# ─────────────────────────────────────────────────────────────────────────────

def denoise(y: np.ndarray, sr: int, strength: float = 0.7) -> np.ndarray:
    """Spectral-subtraction denoiser using a measured noise profile.

    The profile is learned from the quietest frames of this specific
    recording, so it adapts to hiss, hum, or room tone without being told
    which is present.
    """
    y2 = dsp.as_2d(y)
    mono = dsp.to_mono(y2)
    if len(mono) < sr // 2:
        return y2

    from scipy import signal as sps
    n_fft, hop = 2048, 512
    win = sps.windows.hann(n_fft, sym=False)
    kw = {"fs": sr, "window": win, "nperseg": n_fft,
          "noverlap": n_fft - hop}

    _, _, Z = sps.stft(mono.astype(np.float64), **kw)
    mag = np.abs(Z)
    frame_energy = mag.sum(axis=0)
    if frame_energy.size < 8:
        return y2

    quiet = frame_energy <= np.percentile(frame_energy, 12)
    if quiet.sum() < 3:
        return y2
    noise_profile = np.median(mag[:, quiet], axis=1, keepdims=True)

    out = np.empty_like(y2)
    for c in range(y2.shape[1]):
        _, _, Zc = sps.stft(y2[:, c].astype(np.float64), **kw)
        magc, phase = np.abs(Zc), np.angle(Zc)
        cleaned = np.maximum(magc - noise_profile * strength * 1.6,
                             magc * (1.0 - strength * 0.85))
        _, rec = sps.istft(cleaned * np.exp(1j * phase), **kw)
        out[:, c] = dsp.pad_to(np.asarray(rec)[:, None], len(y2))[:, 0]
    return out.astype(np.float32)


def dereverb(y: np.ndarray, sr: int, strength: float = 0.5) -> np.ndarray:
    """Reduce room tail by spectral-envelope suppression.

    A real dereverb model (the UVR de-echo family) is markedly better and
    should be used in production when available. This is the fallback: it
    suppresses bins whose energy substantially exceeds a fast-decaying
    reference envelope, which is where reverb tails live.

    Room reverb is the failure everything downstream inherits -- alignment,
    ducking, and masking all assume a reasonably dry source -- so even a
    partial reduction is worth applying.
    """
    y2 = dsp.as_2d(y)
    if len(y2) < sr // 2 or strength <= 0:
        return y2

    from scipy import signal as sps
    n_fft, hop = 2048, 512
    win = sps.windows.hann(n_fft, sym=False)
    kw = {"fs": sr, "window": win, "nperseg": n_fft,
          "noverlap": n_fft - hop}

    out = np.empty_like(y2)
    for c in range(y2.shape[1]):
        _, _, Z = sps.stft(y2[:, c].astype(np.float64), **kw)
        mag, phase = np.abs(Z), np.angle(Z)

        # Fast-decaying envelope per frequency bin; anything sitting above
        # its own recent decay is direct sound, anything below is tail.
        alpha = 0.62
        env = np.zeros_like(mag)
        env[:, 0] = mag[:, 0]
        for t in range(1, mag.shape[1]):
            env[:, t] = np.maximum(mag[:, t], env[:, t - 1] * alpha)

        ratio = mag / (env + 1e-9)
        gain = np.clip(ratio ** (strength * 1.5), 0.25, 1.0)
        _, rec = sps.istft(mag * gain * np.exp(1j * phase), **kw)
        out[:, c] = dsp.pad_to(np.asarray(rec)[:, None], len(y2))[:, 0]
    return out.astype(np.float32)


def condition_vocal(y: np.ndarray, sr: int, quality) -> Tuple[np.ndarray, dict]:
    """Full restoration chain, applied only where measurement says it's needed.

    Returns `(audio, report)`. Every step is conditional: a clean studio
    take passes through almost untouched, while a noisy phone recording in
    a live room gets the full treatment.
    """
    report: Dict[str, Any] = {"denoise": False, "dereverb": False, "highpass": False}
    out = dsp.as_2d(y)

    # Rumble below 55 Hz is never useful on a vocal.
    out = dsp.highpass(out, sr, 55.0, order=2)
    report["highpass"] = True

    if quality.snr_db < 26.0:
        strength = float(np.clip((26.0 - quality.snr_db) / 22.0, 0.2, 0.85))
        out = denoise(out, sr, strength=strength)
        report["denoise"] = True
        report["denoise_strength"] = round(strength, 2)

    if quality.estimated_rt60_s > 0.28:
        strength = float(np.clip((quality.estimated_rt60_s - 0.25) / 0.6, 0.15, 0.7))
        out = dereverb(out, sr, strength=strength)
        report["dereverb"] = True
        report["dereverb_strength"] = round(strength, 2)

    return out.astype(np.float32), report
