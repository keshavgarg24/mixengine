"""
mixengine -- automatic vocal + beat song generation.

Pipeline:
    beat_dna     offline catalog analysis (run once per beat)
    vocal_dna    analyse an uploaded vocal
    matching     rank the catalog for that vocal
    pipeline     render, critique, and deliver

Quick start:
    from mixengine import beat_dna, pipeline
    catalog = beat_dna.load_catalog("data/dna/beats")
    result  = pipeline.run("data/vocals/take.wav", catalog, "data/outputs")
"""

__version__ = "1.2.0"

from .config import CFG, VARIANTS, GENRE_PROFILES                # noqa: F401
from .core.capabilities import CAPS                              # noqa: F401
from .core.keys import Key, parse_key                            # noqa: F401
from .core.ir import MusicalIR                                   # noqa: F401
from . import musical                                            # noqa: F401

# The submodules below moved into `core/`, `analysis/` and `audio/` during
# the package restructure, but `__all__` was left advertising them at the
# top level. Since `__all__` only affects `from mixengine import *`, every
# documented import -- `from mixengine import pipeline`, and every notebook
# cell in the README -- raised ImportError instead.
#
# They are bound here as real attributes rather than removed from `__all__`,
# so the documented call sites keep working. Imports are lazy because the
# audio submodules pull in librosa, soundfile and torch: binding them
# eagerly would make `import mixengine` fail on a machine that only needs
# the musical layer, and would cost seconds of import time on one that does.

_SUBMODULES = {
    "analysis": "mixengine.analysis.analysis",
    "beat_dna": "mixengine.analysis.beat_dna",
    "vocal_dna": "mixengine.analysis.vocal_dna",
    "matching": "mixengine.analysis.matching",
    "audio_io": "mixengine.core.audio_io",
    "keys": "mixengine.core.keys",
    "capabilities": "mixengine.core.capabilities",
    "dsp": "mixengine.audio.dsp",
    "separation": "mixengine.audio.separation",
    "transform": "mixengine.audio.transform",
    "mixer": "mixengine.audio.mixer",
    "master": "mixengine.audio.master",
    "critic": "mixengine.audio.critic",
    "pipeline": "mixengine.audio.pipeline",
}


def __getattr__(name):
    """Resolve the historical top-level module names on first access."""
    target = _SUBMODULES.get(name)
    if target is None:
        raise AttributeError("module 'mixengine' has no attribute %r" % name)
    import importlib
    module = importlib.import_module(target)
    globals()[name] = module
    return module


def __dir__():
    return sorted(list(globals().keys()) + list(_SUBMODULES))


__all__ = [
    "CFG", "CAPS", "VARIANTS", "GENRE_PROFILES", "Key", "parse_key",
    "MusicalIR", "musical", "__version__",
] + sorted(_SUBMODULES)
