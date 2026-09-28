"""
The arrangement plan: one object both the arranger and the mixer serve.

Until now the engine produced the vocal over the beat, for the vocal's
length, at one level. Every bar got the same treatment, which is why the
output was flat in a way no parameter could fix -- flatness was the design,
not a setting.

This module produces the missing object. It finds the song's structure,
derives an energy arc from it, and turns that arc into two concrete things:

  * **which layers exist where** -- doubles on the hook, harmony only at the
    peak, ad-libs in the gaps, nothing in the first verse;
  * **how the mix moves** -- level, air, space, width and ducking as curves
    over time rather than constants.

Both come from the same contour, so the arrangement and the mix push in the
same direction instead of each guessing separately. That shared target is
the whole reason this is one module and not two.

Nothing here is speculative about what the singer intended. Every decision
traces to something measured -- a phrase that repeats, a section the beat
already has, a genre convention -- and when the measurement is too weak to
support a decision, the plan says so and does less.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.types import FloatSeq

from ..core.types import Section
from ..musical import energy as energy_mod
from . import structure as structure_mod

log = logging.getLogger("mixengine.arrange.plan")

# Layers are requested per phrase from `arrangement_density`, but a layer
# that flickers on and off between adjacent phrases sounds like an
# automation error rather than an arrangement. A layer must be wanted by at
# least this fraction of a section's phrases before it is used there at all.
LAYER_COMMITMENT = 0.5

# Parameters that vary continuously across the song, and the attribute on
# `MixTargets` each one reads.
AUTOMATED = ("vir_db", "air_db", "reverb_wet", "delay_wet", "width",
             "duck_depth_db", "saturation", "double_gain_db")


@dataclass
class SongPlan:
    sections: List[Section] = field(default_factory=list)
    contour: Optional[energy_mod.EnergyContour] = None
    structure: Optional[structure_mod.StructureResult] = None
    layer_regions: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)
    transitions: List[Tuple[float, List[str]]] = field(default_factory=list)
    automation: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    hook_regions: List[Tuple[int, int]] = field(default_factory=list)
    genre: Optional[str] = None
    performance_type: str = "sung"
    duration_s: float = 0.0
    note: str = ""

    def energy_at(self, t: float) -> float:
        return self.contour.at(t) if self.contour else 0.55

    def curve(self, name: str, n: int, sr: int) -> np.ndarray:
        """A parameter's value at every sample, for multiplying into audio.

        Interpolated from the control points rather than stepped, because a
        step in level or brightness at a section line is audible as an edit
        while a ramp across it reads as an arrangement move.
        """
        pair = self.automation.get(name)
        if pair is None:
            return np.zeros(n, dtype=np.float32)
        times, values = pair
        if times.size == 0:
            return np.zeros(n, dtype=np.float32)
        t = np.arange(n, dtype=np.float64) / float(sr)
        return np.interp(t, times, values).astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "genre": self.genre,
            "performance_type": self.performance_type,
            "duration_s": round(self.duration_s, 2),
            "sections": [s.to_dict() for s in self.sections],
            "contour": self.contour.to_dict() if self.contour else None,
            "structure": self.structure.to_dict() if self.structure else None,
            "layers": {k: len(v) for k, v in self.layer_regions.items()},
            "transitions": [{"time": round(t, 3), "effects": e}
                            for t, e in self.transitions],
            "automation": {k: {"min": round(float(v[1].min()), 3),
                               "max": round(float(v[1].max()), 3)}
                           for k, v in self.automation.items() if v[1].size},
            "hook_regions": len(self.hook_regions),
            "note": self.note,
        }


def build(vocal: np.ndarray, sr: int,
          phrases: Sequence[Tuple[int, int]],
          *,
          genre: Optional[str] = None,
          performance_type: str = "sung",
          downbeats: Optional[FloatSeq] = None,
          beats: Optional[FloatSeq] = None,
          beat_sections: Optional[Sequence[Section]] = None,
          allow_layers: bool = True,
          lyrics_doc: Optional[dict] = None) -> SongPlan:
    """Plan the arrangement for one vocal over one beat.

    `lyrics_doc` is the take's transcript, when one was made. It goes to
    the structure pass, where repeated words identify a hook that chroma
    and melodic contour can only guess at.
    """
    n = len(vocal)
    total = n / float(sr)
    plan = SongPlan(genre=genre, performance_type=performance_type,
                    duration_s=total)

    if len(phrases) < 2:
        plan.sections = [replace(s) for s in (beat_sections or [])] or [
            Section(start=0.0, end=total, label="verse")]
        plan.contour = energy_mod.build(plan.sections, genre, total)
        _annotate_sections(plan.sections, plan.contour, None, downbeats)
        plan.note = "not enough phrases to find structure; flat arrangement"
        plan.automation = _automation(plan, performance_type, genre)
        return plan

    # ── 1. What is the hook ───────────────────────────────────────────────
    st = structure_mod.analyze(vocal, sr, phrases, beats=beats,
                               performance_type=performance_type,
                               lyrics_doc=lyrics_doc)
    plan.structure = st
    plan.sections = structure_mod.sections_from_labels(
        st.phrases, st.labels, total, downbeats)

    # The beat has its own structure, and where the two agree the beat wins
    # on boundaries: its section changes are real production events the
    # listener can hear, while the vocal's are inferred from phrase gaps.
    if beat_sections:
        plan.sections = _snap_to_beat_sections(plan.sections, beat_sections)

    # ── 2. The arc ────────────────────────────────────────────────────────
    plan.contour = energy_mod.build(plan.sections, genre, total)
    _annotate_sections(plan.sections, plan.contour, st, downbeats)

    # ── 3. Which layers, where ────────────────────────────────────────────
    plan.hook_regions = [
        (int(f.start * sr), int(f.end * sr))
        for f, label in zip(st.phrases, st.labels) if label == "hook"]
    if allow_layers:
        plan.layer_regions = _layer_regions(st, plan.contour, sr, n)
    if not plan.hook_regions:
        plan.note = st.note or "no hook identified; layers held back"

    # ── 4. Transitions ────────────────────────────────────────────────────
    plan.transitions = _transitions(plan.sections, plan.contour)

    # ── 5. Mix automation ─────────────────────────────────────────────────
    plan.automation = _automation(plan, performance_type, genre)
    return plan


# ─────────────────────────────────────────────────────────────────────────────

def _annotate_sections(sections: Sequence[Section],
                       contour: energy_mod.EnergyContour,
                       st: Optional[structure_mod.StructureResult],
                       downbeats: Optional[FloatSeq]) -> None:
    """Fill each section's reported fields from what was actually decided.

    Energy comes from the contour that drives the mix, bar indices from the
    downbeats the boundaries were snapped to, and confidence from how
    clearly the structure analysis placed the phrases inside the section:
    for a repeated phrase, how strongly it matches the rest of its group;
    for a one-off, how clearly it matches nothing else. Without this the
    report carried the dataclass defaults -- energy 0.5, confidence 0 --
    which read as measurements and were not. Nothing downstream reads
    these fields; they exist for the report and the interface.
    """
    db = np.asarray(downbeats if downbeats is not None else [], dtype=np.float64)
    sim = None
    if (st is not None and st.similarity.size
            and len(st.groups) == len(st.phrases) == st.similarity.shape[0]):
        sim = st.similarity

    def phrase_confidence(i: int) -> float:
        assert sim is not None and st is not None
        others = [j for j in range(sim.shape[0]) if j != i]
        if not others:
            return 0.0
        members = [j for j in others if st.groups[j] == st.groups[i]]
        if members:
            return float(np.mean([sim[i, j] for j in members]))
        return float(1.0 - max(sim[i, j] for j in others))

    for s in sections:
        s.energy = float(contour.at((s.start + s.end) / 2.0))
        if db.size:
            s.start_bar = int(np.argmin(np.abs(db - s.start)))
            s.end_bar = int(np.argmin(np.abs(db - s.end)))
        if sim is not None and st is not None:
            inside = [i for i, f in enumerate(st.phrases)
                      if s.start <= (f.start + f.end) / 2.0 < s.end]
            if inside:
                s.confidence = float(np.mean([phrase_confidence(i)
                                              for i in inside]))


def _snap_to_beat_sections(sections: Sequence[Section],
                           beat_sections: Sequence[Section],
                           tolerance_s: float = 1.6) -> List[Section]:
    """Move vocal section boundaries onto nearby beat section boundaries."""
    edges = sorted({s.start for s in beat_sections} |
                   {s.end for s in beat_sections})
    if not edges:
        return list(sections)
    arr = np.asarray(edges, dtype=np.float64)

    out: List[Section] = []
    prev_end = 0.0
    for s in sections:
        j = int(np.argmin(np.abs(arr - s.start)))
        start = float(arr[j]) if abs(arr[j] - s.start) <= tolerance_s else s.start
        start = max(start, prev_end)
        k = int(np.argmin(np.abs(arr - s.end)))
        end = float(arr[k]) if abs(arr[k] - s.end) <= tolerance_s else s.end
        if end <= start + 0.25:
            end = s.end
        out.append(Section(start=start, end=max(end, start + 0.25),
                           label=s.label, energy=s.energy))
        prev_end = out[-1].end
    return out


def _layer_regions(st: structure_mod.StructureResult,
                   contour: energy_mod.EnergyContour,
                   sr: int, n: int) -> Dict[str, List[Tuple[int, int]]]:
    """Ask the energy contour which layers belong on each phrase.

    A layer is only kept where it is wanted consistently. Asking per phrase
    and acting on every answer makes a double appear for one line and
    vanish for the next, which is heard as a fault rather than a decision --
    so a layer must be wanted by most of a run of phrases before any of
    them gets it.
    """
    wanted: Dict[str, List[Tuple[int, int]]] = {}
    votes: Dict[str, List[Tuple[int, int, bool]]] = {}

    for f, _label in zip(st.phrases, st.labels):
        e = contour.at((f.start + f.end) / 2.0)
        density = energy_mod.arrangement_density(e)
        s_i, e_i = int(f.start * sr), min(int(f.end * sr), n)
        if e_i <= s_i:
            continue
        for name, on in density.items():
            if name == "lead_vocal" or name in ("drums", "bass", "melody", "pads"):
                continue
            votes.setdefault(name, []).append((s_i, e_i, bool(on)))

    for name, entries in votes.items():
        on_count = sum(1 for _, _, on in entries if on)
        if not entries or on_count / len(entries) < LAYER_COMMITMENT:
            # Not wanted consistently across the song -- but a layer wanted
            # on *every* hook and nowhere else is exactly right, so the
            # regions that did vote yes are still kept when there are
            # enough of them to form a run.
            if on_count >= 2:
                wanted[name] = [(s, e) for s, e, on in entries if on]
            continue
        wanted[name] = [(s, e) for s, e, on in entries if on]
    return {k: v for k, v in wanted.items() if v}


def _transitions(sections: Sequence[Section],
                 contour: energy_mod.EnergyContour
                 ) -> List[Tuple[float, List[str]]]:
    out: List[Tuple[float, List[str]]] = []
    for i in range(1, len(sections)):
        prev_e = contour.at(max(sections[i - 1].start,
                                sections[i].start - 0.5))
        next_e = contour.at(min(sections[i].end - 0.01,
                                sections[i].start + 0.5))
        effects = energy_mod.transition_before(sections[i].label,
                                               next_e - prev_e)
        if effects:
            out.append((float(sections[i].start), effects))
    return out


def _automation(plan: SongPlan, performance_type: str,
                genre: Optional[str]
                ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Turn the energy contour into a curve per mix parameter.

    Sampled at the contour's own control points rather than on a fixed
    grid, so a section boundary lands exactly on a control point and the
    ramp across it is the one the contour specified.
    """
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    if plan.contour is None or not plan.contour.points:
        return out
    times = plan.contour.times
    targets = [energy_mod.mix_targets(v, performance_type=performance_type,
                                      genre=genre)
               for v in plan.contour.values]
    for name in AUTOMATED:
        vals = np.asarray([getattr(t, name) for t in targets],
                          dtype=np.float64)
        out[name] = (times, vals)
    return out
