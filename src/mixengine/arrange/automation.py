"""
Applying the plan's curves to audio.

`plan.py` produces a target value per parameter at every moment. This turns
those into actual processing, and it is deliberately honest about which
parameters it can move and which it only records: a report that lists eight
automated parameters when three of them are applied is worse than one that
lists three, because the next person to read it will go looking for the
effect of the other five.

Applied here:

  **Level.** The verse sits back and the hook comes forward. This is the
  single most audible arrangement move there is, and the one that most
  distinguishes a produced record from a vocal at a constant level.

  **Brightness.** High-shelf amount rides with energy. Implemented as a
  crossfade between the dry signal and one fixed shelf rather than a
  time-varying filter, because re-computing biquad coefficients per sample
  either zipper-noises or costs a filter state reset at every block edge,
  and the audible result of the crossfade is the same.

Applied elsewhere: layer gain, which the pipeline rides against the
generated layer bus using `layer_gain_curve` below, because that bus does
not exist yet when this runs.

Recorded but not applied at all: width, saturation, duck depth and send
levels. Those belong to stages that own the relevant signal -- the beat
bus, the send bus -- and moving them from here would mean this module
reaching into three others.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np

from ..audio import dsp

log = logging.getLogger("mixengine.arrange.automation")

# The shelf that brightness rides. 8 kHz: above the consonants, inside the
# air band, and far enough from sibilance that opening it on a hook does
# not also open the esses.
AIR_SHELF_HZ = 8000.0
AIR_MAX_DB = 4.0

# Level ride limits. A wider range than this stops reading as dynamics and
# starts reading as a fader move.
MAX_RIDE_DB = 3.0

# `double_gain_db` is absent deliberately: it is applied, but by the
# pipeline against the generated layer bus rather than here, because this
# function only ever sees the lead. Listing it as recorded-only would send
# the next reader looking for an effect that is real but somewhere else.
RECORDED_ONLY = ("reverb_wet", "delay_wet", "width", "duck_depth_db",
                 "saturation")


def apply_vocal(v: np.ndarray, sr: int, plan) -> Tuple[np.ndarray, dict]:
    """Ride the lead vocal's level and brightness along the plan's contour."""
    out = dsp.as_2d(v)
    n = len(out)
    report: Dict = {"applied": [], "recorded_only": list(RECORDED_ONLY)}
    if plan is None or not plan.automation or n < sr:
        report["note"] = "no contour to follow"
        return out, report

    ride_db = plan.curve("vir_db", n, sr)
    if ride_db.size and float(np.ptp(ride_db)) > 0.05:
        ride_db = np.clip(ride_db, -MAX_RIDE_DB, MAX_RIDE_DB)
        out = out * (10.0 ** (ride_db / 20.0))[:, None]
        report["applied"].append("vir_db")
        report["ride_db"] = {"min": round(float(ride_db.min()), 2),
                             "max": round(float(ride_db.max()), 2)}

    air_db = plan.curve("air_db", n, sr)
    if air_db.size and float(np.ptp(air_db)) > 0.05:
        # One shelf at the maximum, crossfaded by the curve. `amount` is
        # signed: negative energy pulls toward a *darker* version, which is
        # the same shelf applied with the opposite sign.
        bright = dsp.shelf_eq(out, sr, AIR_SHELF_HZ, AIR_MAX_DB, kind="high")
        dark = dsp.shelf_eq(out, sr, AIR_SHELF_HZ, -AIR_MAX_DB, kind="high")
        amount = np.clip(air_db / AIR_MAX_DB, -1.0, 1.0)[:, None]
        up = np.clip(amount, 0.0, 1.0)
        down = np.clip(-amount, 0.0, 1.0)
        out = out * (1.0 - up - down) + bright * up + dark * down
        report["applied"].append("air_db")
        report["air_db"] = {"min": round(float(air_db.min()), 2),
                            "max": round(float(air_db.max()), 2)}

    if not report["applied"]:
        report["note"] = "contour is flat; nothing to ride"
    return out.astype(np.float32), report


LAYER_PARAM = {"double": "double_gain_db", "harmony": "double_gain_db",
               "adlibs": "delay_wet"}


def layer_gain_curve(plan, kind: str, n: int, sr: int) -> Optional[np.ndarray]:
    """The gain, in dB, a generated layer should follow across the song.

    Doubles and harmonies come up into a hook and fall away from it, which
    the contour already describes; a layer held at one level through a
    section change announces itself as an overdub. Returns None when the
    contour is flat, so a caller can skip the multiply entirely rather than
    applying a curve of zeros.
    """
    if plan is None or not plan.automation:
        return None
    param = LAYER_PARAM.get(kind, "double_gain_db")
    curve = plan.curve(param, n, sr)
    return curve if curve.size and float(np.ptp(curve)) > 0.05 else None
