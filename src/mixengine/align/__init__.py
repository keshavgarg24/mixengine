"""
Variable-rate alignment.

A global time-stretch ratio plus per-onset nudges cannot hold a human vocal
to a grid for three minutes; this was measured on a real take rather than
assumed. One ratio fixed to the take's average tempo left a median error of
29 ms against the beat's sixteenth grid, because a performance's tempo is
not constant and the average is wrong almost everywhere.

The fix is a monotonic correspondence between the take's onsets and the
beat's grid, followed by a warp that honours it locally: `dtw` finds the
correspondence, `aligner` decides how much of it to apply, `warp` applies
it.
"""

from . import dtw, warp
from .aligner import AlignReport, align_to_grid, align_to_reference
from .dtw import (Assignment, assign_onsets_to_grid, dtw_path,
                  estimate_tempo_ratio, grid_fit_error, residual_drift)
from .warp import map_times, sanitize_anchors
from .warp import warp as apply_warp

# `warp` deliberately stays bound to the *module*, not to `warp.warp`. Naming
# the function the same as its module means `from mixengine.align import warp`
# silently returns whichever the import order happened to bind last, and the
# failure surfaces as an AttributeError on an unrelated line. The function is
# exported as `apply_warp`.
__all__ = [
    "dtw", "warp",
    "AlignReport", "Assignment",
    "align_to_grid", "align_to_reference",
    "assign_onsets_to_grid", "dtw_path", "residual_drift",
    "estimate_tempo_ratio", "grid_fit_error",
    "apply_warp", "sanitize_anchors", "map_times",
]
