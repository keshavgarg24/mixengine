"""
Runtime capability detection.

The system must produce output on any machine, so every heavy dependency is
optional and every one has a fallback. This module probes what's installed
once at import and exposes booleans the rest of the codebase branches on.

The never-fail contract from the architecture spec is implemented here plus
in the individual fallbacks: madmom missing degrades to librosa beat
tracking, torchcrepe missing degrades to librosa.pyin, Demucs missing means
the vocal is used as-is. Nothing raises.
"""

from __future__ import annotations

import importlib
import logging
import platform
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

log = logging.getLogger("mixengine")


def _has(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


@dataclass
class Capabilities:
    # -- Required ----------------------------------------------------------
    numpy: bool = False
    scipy: bool = False
    librosa: bool = False
    soundfile: bool = False

    # -- Strongly recommended ---------------------------------------------
    pedalboard: bool = False        # reverb, limiter
    pyloudnorm: bool = False        # LUFS metering
    pyrubberband: bool = False      # formant-preserving stretch/shift
    rubberband_cli: bool = False    # the binary pyrubberband shells out to

    # -- Optional, quality-improving --------------------------------------
    madmom: bool = False            # downbeat tracking
    torch: bool = False
    torchcrepe: bool = False        # f0
    demucs: bool = False            # separation
    audio_separator: bool = False   # Mel-Band RoFormer
    whisper: bool = False           # lyrics + intelligibility QC
    sklearn: bool = False           # structure segmentation
    silero_vad: bool = False        # is there a voice in the take at all

    # -- Environment -------------------------------------------------------
    device: str = "cpu"             # cpu | cuda | mps
    platform: str = ""
    python: str = ""

    def summary(self) -> str:
        req = ["librosa", "soundfile"]
        rec = ["pedalboard", "pyloudnorm", "pyrubberband", "rubberband_cli"]
        opt = ["madmom", "torchcrepe", "demucs", "audio_separator", "whisper", "sklearn",
               "silero_vad"]

        def fmt(names):
            return "\n".join(
                f"    {'OK ' if getattr(self, n) else '-- '} {n}" for n in names
            )

        return (
            f"mixengine capabilities\n"
            f"  platform: {self.platform}\n"
            f"  python:   {self.python}\n"
            f"  device:   {self.device}\n"
            f"  required:\n{fmt(req)}\n"
            f"  recommended:\n{fmt(rec)}\n"
            f"  optional:\n{fmt(opt)}\n"
        )

    def to_dict(self) -> dict:
        return asdict(self)

    # -- Derived answers the pipeline actually asks ------------------------

    @property
    def can_render(self) -> bool:
        """Minimum viable: load audio, process it, write it back."""
        return self.librosa and self.soundfile

    @property
    def can_stretch_well(self) -> bool:
        """Formant-preserving stretch and pitch shift are available.

        Only the Rubber Band binary is required. The engine drives it
        directly rather than through pyrubberband, because pyrubberband
        emits every argument as a `key value` pair and so cannot express
        the boolean `--formant` flag at all -- it produces `--formant ""`,
        which Rubber Band rejects with a usage error.
        """
        return self.rubberband_cli

    @property
    def can_separate(self) -> bool:
        return self.demucs or self.audio_separator

    @property
    def tier(self) -> str:
        """Coarse quality tier, useful for setting user expectations."""
        if not self.can_render:
            return "unusable"
        score = sum([
            self.can_stretch_well, self.madmom, self.torchcrepe,
            self.can_separate, self.pedalboard, self.pyloudnorm,
        ])
        return {6: "full", 5: "full", 4: "good", 3: "good"}.get(score, "basic")

    # -- Which implementation each analysis stage would use --------------

    @property
    def crepe_model(self) -> str:
        """The CREPE capacity this machine can afford.

        The full model is the accurate one, and on an accelerator it runs
        faster than real time. On a CPU it runs at a sixteenth of real
        time -- measured on this project's own fixture; decoder and batch
        size made no difference -- which for a four-minute take is an
        hour. That is not a slower analysis, it is no analysis. The tiny
        model on a CPU runs at three times real time and is still a
        better tracker than pyin.
        """
        return "full" if self.device in ("cuda", "mps") else "tiny"

    @property
    def pitch_backend(self) -> str:
        if not self.torchcrepe:
            return "pyin"
        return "torchcrepe" if self.crepe_model == "full" else "torchcrepe_tiny"

    @property
    def separation_backend(self) -> str:
        if self.audio_separator:
            return "audio_separator"
        return "demucs" if self.demucs else "none"

    def analysis_backends(self, separated: bool = True) -> Dict[str, str]:
        """The backend each analysis stage would run with, right now.

        `separated` is whether separation would actually happen: a beat
        whose stems were not requested, or a take clean enough not to need
        them, records "none" however good the installed separator is. An
        analysis document carries this so that a later lookup can tell
        whether the machine it is on could now do better; see
        `improvement_over`.
        """
        return {
            "rhythm": "madmom" if self.madmom else "librosa",
            "pitch": self.pitch_backend,
            "separation": self.separation_backend if separated else "none",
        }


def _detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _detect_rubberband_cli() -> bool:
    import shutil
    return shutil.which("rubberband") is not None


def detect() -> Capabilities:
    c = Capabilities(
        numpy=_has("numpy"),
        scipy=_has("scipy"),
        librosa=_has("librosa"),
        soundfile=_has("soundfile"),
        pedalboard=_has("pedalboard"),
        pyloudnorm=_has("pyloudnorm"),
        pyrubberband=_has("pyrubberband"),
        rubberband_cli=_detect_rubberband_cli(),
        madmom=_has("madmom"),
        torch=_has("torch"),
        torchcrepe=_has("torchcrepe"),
        demucs=_has("demucs"),
        audio_separator=_has("audio_separator"),
        whisper=_has("whisper") or _has("faster_whisper"),
        sklearn=_has("sklearn"),
        silero_vad=_has("silero_vad"),
        platform=f"{platform.system()} {platform.machine()}",
        python=sys.version.split()[0],
    )
    c.device = _detect_device()
    return c


CAPS = detect()


# ─────────────────────────────────────────────────────────────────────────────
# Cache freshness
# ─────────────────────────────────────────────────────────────────────────────
#
# Analyses are cached by content hash and schema version, and installing a
# better backend changes neither. Without the check below, every cached
# beat and vocal kept serving the analysis made before torchcrepe or demucs
# arrived, and the better backend never reached a render. Each stage's
# implementations are listed worst to best.

BACKEND_RANKS: Dict[str, List[str]] = {
    "rhythm": ["librosa", "madmom"],
    "pitch": ["pyin", "torchcrepe_tiny", "torchcrepe"],
    "separation": ["none", "demucs", "audio_separator"],
}


def _rank(stage: str, name: Optional[str]) -> int:
    ranks = BACKEND_RANKS[stage]
    return ranks.index(name) if name in ranks else -1


def improvement_over(recorded: Optional[Dict[str, str]],
                     want_separation: bool = True,
                     now: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Why an analysis made with `recorded` backends is worth redoing here.

    None when this machine could do no better. Otherwise a short reason
    naming the first stage that would improve -- ``"pitch: pyin ->
    torchcrepe"`` -- fit for a log line. Three rules:

    * A better analysis made elsewhere is kept. A stage whose recorded
      backend outranks the one available here is a loss, not an
      improvement, and the document stays.
    * Separation only counts when the caller wants it. A clean take is
      never separated and a beat's stems are only made on request, so
      "none" there is a decision, not a gap.
    * A document that recorded nothing is refreshed once. It predates the
      record, so nothing is known about how it was made.
    """
    now = CAPS.analysis_backends() if now is None else now
    recorded = recorded or {}
    for stage in ("rhythm", "pitch", "separation"):
        if stage == "separation" and not want_separation:
            continue
        have = now.get(stage)
        if have is None:
            continue
        had = recorded.get(stage)
        if _rank(stage, have) > _rank(stage, had):
            return "%s: %s -> %s" % (stage, had or "unrecorded", have)
    return None


def require_render() -> None:
    """Raise only for the genuinely unrecoverable case."""
    if not CAPS.can_render:
        raise RuntimeError(
            "librosa and soundfile are required.\n"
            "    pip install librosa soundfile"
        )
