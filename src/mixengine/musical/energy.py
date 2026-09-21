"""
Energy contour: the shape of the song.

Records have an arc. A verse sits back, a pre-hook tightens, a hook opens
up, a bridge pulls away so the last hook lands harder. That arc is not
decoration -- it is most of what separates a produced record from a vocal
playing on top of a loop for three minutes.

The current engine has no concept of it, which is why its output is flat
in a way no parameter tweak can fix: every bar gets the same vocal level,
the same reverb, the same width, the same brightness.

This module makes the arc an explicit object. A target contour is derived
from the section labels and genre convention, and then **both the arranger
and the mixer serve the same curve**: the arranger by adding and removing
elements, the mixer by moving level, space, width and brightness. Because
they share one target, they push in the same direction instead of each
guessing separately.

The contour is also what makes the result *musical* rather than merely
correct. Perceived groove and excitement track event density and beat
salience far more than they track timing minutiae, so shaping density and
emphasis across a song buys more than any amount of extra precision in the
alignment stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.types import Section

# Conventional energy level per section function, 0-1. These are the
# starting points a producer would assume before hearing anything.
#
# Note that the first hook sits at 0.86 rather than 1.0. The ceiling is
# reserved deliberately: later hooks climb above the first one, and a song
# whose opening hook is already at maximum has nowhere left to go for its
# final chorus. Leaving headroom is what makes the arc possible.
BASE_ENERGY: Dict[str, float] = {
    "intro": 0.30,
    "verse": 0.52,
    "prehook": 0.68,
    "chorus": 0.86,
    "hook": 0.86,
    "bridge": 0.42,
    "break": 0.25,
    "inst": 0.55,
    "solo": 0.78,
    "outro": 0.30,
    "section": 0.55,
}

# Genre shaping applied on top of the base curve. Trap and drill live on a
# wide verse-to-hook contrast; R&B and lo-fi are deliberately flatter and
# exaggerating them would read as clumsy.
GENRE_CONTRAST: Dict[str, float] = {
    "trap": 1.15, "drill": 1.20, "hip_hop": 1.05, "melodic_trap": 1.10,
    "boom_bap": 0.90, "pop": 1.10, "rnb": 0.85, "soul": 0.80,
    "afrobeats": 0.95, "dancehall": 1.00, "drum_and_bass": 1.25,
    "house": 1.15, "edm": 1.30, "lofi": 0.65, "jazz": 0.70,
}


@dataclass
class EnergyPoint:
    time: float
    value: float
    label: str = ""


@dataclass
class EnergyContour:
    """A target energy curve over the length of a song."""
    points: List[EnergyPoint] = field(default_factory=list)
    genre: Optional[str] = None

    @property
    def times(self) -> np.ndarray:
        return np.asarray([p.time for p in self.points], dtype=np.float64)

    @property
    def values(self) -> np.ndarray:
        return np.asarray([p.value for p in self.points], dtype=np.float64)

    def at(self, t: float) -> float:
        if not self.points:
            return 0.5
        if len(self.points) == 1:
            return float(self.points[0].value)
        return float(np.interp(float(t), self.times, self.values))

    def peak_time(self) -> float:
        if not self.points:
            return 0.0
        return float(self.points[int(np.argmax(self.values))].time)

    @property
    def contrast(self) -> float:
        """Range between the quietest and loudest moments of the plan."""
        if not self.points:
            return 0.0
        return float(np.max(self.values) - np.min(self.values))

    def to_dict(self) -> dict:
        return {"genre": self.genre,
                "contrast": round(self.contrast, 4),
                "points": [{"time": round(p.time, 3),
                            "value": round(p.value, 4),
                            "label": p.label} for p in self.points]}


def build(sections: Sequence[Section], genre: Optional[str] = None,
          duration_s: Optional[float] = None) -> EnergyContour:
    """Derive a target energy contour from a section map.

    Three musical conventions are applied on top of the per-label base
    values, because a flat lookup would produce the same hook energy in bar
    9 and bar 90 and that is not how records are built:

      * **Later repeats of a section climb.** A second hook is bigger than
        the first, a third bigger still. This is the single most reliable
        arrangement convention in popular music.
      * **The section before a hook lifts toward it**, so the hook is
        arrived at rather than cut to.
      * **The section after a bridge is the peak**, because the bridge
        exists to create the drop that makes it land.
    """
    g = (genre or "").lower().replace(" ", "_")
    contrast = GENRE_CONTRAST.get(g, 1.0)

    if not sections:
        total = float(duration_s or 0.0)
        return EnergyContour(points=[EnergyPoint(0.0, 0.5, "section"),
                                     EnergyPoint(total, 0.5, "section")],
                             genre=genre)

    seen: Dict[str, int] = {}
    raw: List[Tuple[Section, float]] = []
    for i, s in enumerate(sections):
        base = BASE_ENERGY.get(s.label, BASE_ENERGY["section"])
        n = seen.get(s.label, 0)
        seen[s.label] = n + 1

        # Each repeat of a section lifts, with diminishing increments so a
        # long song does not run out of headroom before its final chorus.
        if s.label in ("chorus", "hook") and n > 0:
            base += min(0.06 * n, 0.12)
        elif s.label == "verse" and n > 0:
            base += min(0.07 * n, 0.14)

        # Lift into an upcoming hook.
        if i + 1 < len(sections) and sections[i + 1].label in ("chorus", "hook"):
            if s.label not in ("chorus", "hook"):
                base += 0.08

        # The section after a bridge is the payoff.
        if i > 0 and sections[i - 1].label == "bridge" and s.label in ("chorus", "hook"):
            base = min(1.0, base + 0.08)

        raw.append((s, base))

    # Apply genre contrast around the mean so the average level is
    # preserved and only the spread changes.
    vals = np.asarray([v for _, v in raw], dtype=np.float64)
    mean = float(np.mean(vals))
    vals = np.clip(mean + (vals - mean) * contrast, 0.05, 1.0)

    points: List[EnergyPoint] = []
    for (s, _), v in zip(raw, vals):
        # Two points per section -- one just inside each edge -- so
        # interpolation ramps across the boundary instead of stepping. A
        # hard step in level or brightness at a section line is audible as
        # an edit; a short ramp reads as an arrangement move.
        span = max(s.duration, 1e-3)
        edge = min(0.35, span * 0.12)
        points.append(EnergyPoint(s.start + edge, float(v), s.label))
        points.append(EnergyPoint(max(s.end - edge, s.start + edge), float(v), s.label))

    points.sort(key=lambda p: p.time)
    if points and points[0].time > 0:
        points.insert(0, EnergyPoint(0.0, points[0].value, points[0].label))
    total = float(duration_s or (sections[-1].end if sections else 0.0))
    if points and total > points[-1].time:
        points.append(EnergyPoint(total, points[-1].value, points[-1].label))
    return EnergyContour(points=points, genre=genre)


# ─────────────────────────────────────────────────────────────────────────────
# Mapping energy onto mix parameters
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MixTargets:
    """Mix parameter offsets derived from one point on the energy contour.

    All values are *deltas* applied on top of the genre profile, never
    absolute settings. That keeps the genre profile as the single source
    of truth for what a trap mix sounds like, and lets the contour handle
    only how the song moves within that.
    """
    vir_db: float = 0.0              # vocal-to-instrumental ratio offset
    air_db: float = 0.0              # high shelf
    reverb_wet: float = 0.0
    delay_wet: float = 0.0
    width: float = 0.0               # stereo width offset, -1..+1
    duck_depth_db: float = 0.0
    saturation: float = 0.0          # 0-1 drive offset
    comp_target_gr_db: float = 0.0
    double_gain_db: float = 0.0

    def to_dict(self) -> dict:
        return {k: round(float(v), 3) for k, v in self.__dict__.items()}


def mix_targets(energy: float, *, performance_type: str = "sung",
                genre: Optional[str] = None) -> MixTargets:
    """Translate an energy value in [0, 1] into mix parameter offsets.

    The mapping follows what engineers actually do between a verse and a
    hook: the vocal comes forward, the top end opens, the space gets
    bigger, the beat ducks harder to make room, and doubles come up. Every
    move is small -- the arrangement should be doing most of the work, with
    the mix supporting it rather than substituting for it.
    """
    # Centre on 0.6, which is roughly where a verse sits, so a verse gets
    # offsets near zero and only the extremes move meaningfully.
    e = float(np.clip(energy, 0.0, 1.0))
    d = (e - 0.6) / 0.4                       # -1.5 .. +1.0

    t = MixTargets(
        vir_db=float(np.clip(d * 1.2, -2.0, 1.2)),
        air_db=float(np.clip(d * 1.6, -2.5, 1.8)),
        reverb_wet=float(np.clip(d * 0.035, -0.04, 0.045)),
        delay_wet=float(np.clip(d * 0.03, -0.035, 0.04)),
        width=float(np.clip(d * 0.18, -0.25, 0.20)),
        duck_depth_db=float(np.clip(d * 0.9, -1.0, 1.2)),
        saturation=float(np.clip(d * 0.18, -0.15, 0.22)),
        comp_target_gr_db=float(np.clip(d * 0.8, -1.5, 1.0)),
        double_gain_db=float(np.clip(d * 2.5, -6.0, 2.5)),
    )

    # Rap sits forward and dry by default; opening the reverb on a rap hook
    # the way you would on a sung one washes out the consonants that carry
    # the bars.
    if performance_type in ("rap", "melodic_rap"):
        t.reverb_wet *= 0.55
        t.delay_wet *= 0.8
        t.vir_db += 0.3

    g = (genre or "").lower().replace(" ", "_")
    if g in ("rnb", "soul", "lofi", "jazz"):
        t.vir_db *= 0.7
        t.saturation *= 0.6
    return t


def arrangement_density(energy: float) -> Dict[str, bool]:
    """Which layers should be present at a given energy level.

    Returns a set of on/off decisions the arranger applies to stems and
    generated vocal layers. Muting elements in a verse so the hook has
    somewhere to go is the cheapest and most effective arrangement move
    available once stems exist, and it costs nothing to compute.
    """
    e = float(np.clip(energy, 0.0, 1.0))
    return {
        "drums": e > 0.20,
        "bass": e > 0.30,
        "melody": e > 0.38,
        "pads": e > 0.55,
        "lead_vocal": True,
        "tight_double": e > 0.72,
        "wide_double": e > 0.80,
        "octave_down": e > 0.85,
        "harmony": e > 0.88,
        "adlibs": e > 0.62,
        "whisper_layer": e > 0.90,
    }


def transition_before(section_label: str, energy_jump: float) -> List[str]:
    """Transition effects justified by the size of an energy step.

    A big lift into a hook needs to be set up or it sounds like an edit.
    A small one does not, and decorating it makes the arrangement fussy --
    so the effects are gated on the size of the jump rather than applied at
    every boundary.
    """
    out: List[str] = []
    if energy_jump < 0.12:
        return out
    if energy_jump >= 0.12:
        out.append("reverse_vocal_tail")
    if energy_jump >= 0.20:
        out.append("riser")
        out.append("drum_fill")
    if energy_jump >= 0.30:
        out.append("impact")
        if section_label in ("chorus", "hook"):
            out.append("beat_dropout_1bar")
    return out
