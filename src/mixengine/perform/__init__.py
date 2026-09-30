"""
Planning a performance, separately from rendering one.

`core/ir.py` describes the engine as UNDERSTAND, then DECIDE, then
EXECUTE. Analysis has always covered the first and the audio stages the
third, but the middle step had nowhere to live: every musical decision was
taken inside the loop that applied it, and existed only as a count in a
report afterwards.

This package is that middle step for the vocal itself. It turns measured
notes into intended ones -- which pitch, at which time, and on what
grounds -- and returns them as data. Nothing here touches audio.

Two things follow from that, and both are the point:

  * the engine can say what it changed and why, note by note, rather than
    reporting an average; and
  * the same plan can be realised by a pitch-shifter, by a voice model, or
    by anything else, because it describes the performance rather than an
    operation on a waveform.
"""

from .plan import (
    INAUDIBLE_CENTS, KEPT_GESTURE, KEPT_IN_TUNE, KEPT_MELISMA,
    KEPT_NO_TARGET, KEPT_TOO_FAR, TUNED,
    NoteTarget, PerformancePlan, build, pitch_targets, snap_to_grid,
)

__all__ = [
    "NoteTarget", "PerformancePlan", "build", "pitch_targets", "snap_to_grid",
    "TUNED", "KEPT_GESTURE", "KEPT_MELISMA", "KEPT_IN_TUNE", "KEPT_TOO_FAR",
    "KEPT_NO_TARGET", "INAUDIBLE_CENTS",
]
