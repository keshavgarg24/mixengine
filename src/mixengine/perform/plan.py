"""
The performance as it should be, before anything renders it.

Every correction the engine makes to a vocal was, until now, a decision
taken inside the code that applied it: the tuner chose a target pitch and
pitch-shifted a slice of audio in the same loop, and the quantiser chose a
target time and moved samples in the same pass. That works, and it has two
costs that matter more as the product grows.

**The decision cannot be inspected.** A user asking "what did you change
about my vocal, and why?" can be told how many notes moved and by how
much on average, because that is all the report carries. The engine knew
far more at the moment it decided -- which chord was underneath, how
exposed the note was, what else it could have chosen -- and threw all of
it away.

**The decision cannot be rendered any other way.** A corrected performance
is a musical object: these syllables, at these pitches, at these times. Once
it exists as data, an audio pitch-shifter is only one of the things that can
realise it. A voice model can sing it. A synthesiser can play it. Nothing
in this module knows or cares which.

So this is the PLAN layer that the architecture note in `core/ir.py`
describes and the pipeline never had:

    UNDERSTAND  ->  PLAN  ->  EXECUTE
    (analysis)     (here)    (render)

It is pure. It takes measurements and returns intentions; it touches no
audio and imports nothing that does. That is what makes it testable
without a render, and what lets the same plan drive a pitch-shifter today
and a voice model later.

The musical judgement is not reimplemented here. `musical.theory` decides
which pitches are legitimate under a chord and `musical.salience` decides
how much a given note's accuracy matters; this module asks them, records
what they said, and writes it down.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..core.types import Note, Phrase, Section
from ..musical import salience, theory

log = logging.getLogger("mixengine.perform")

# Why a note was left where it was, or moved. One of these is recorded for
# every note, so a plan always accounts for the whole performance.
KEPT_GESTURE = "gesture"          # a scoop, fall or slide: expression
KEPT_MELISMA = "melisma"          # several notes on one syllable
KEPT_IN_TUNE = "in_tune"          # already within a few cents of target
KEPT_TOO_FAR = "too_far"          # no legal target close enough to be an error
KEPT_NO_TARGET = "no_target"      # nothing sounding underneath to judge against
TUNED = "tuned"

# Under this the shift is inaudible and applying it only costs quality.
# Just-noticeable pitch difference for a sustained tone is around 5-6
# cents; below that a shift is work done for nobody.
INAUDIBLE_CENTS = 4.0

# Correction is never more than this, whatever the strength asks for.
# Past it a note is not mistuned, it is a different note.
MAX_CORRECTION_SEMITONES = 1.2

# The engine's own ceiling on how much of the error it will take out, so a
# corrected line still reads as performed rather than quantised.
MAX_CORRECTION_FRACTION = 0.95


@dataclass
class NoteTarget:
    """One note, as measured and as intended.

    Both are kept. The difference between them is the correction, and a
    renderer may want either: a pitch-shifter needs the delta, a voice
    model needs the absolute target, and an interface explaining itself to
    the person needs both.
    """
    index: int                     # position in the source note list
    source_midi: float
    source_start: float
    source_end: float
    midi: float                    # target pitch, fractional
    start: float                   # target time
    end: float
    decision: str = KEPT_IN_TUNE
    salience: float = 0.0          # how much accuracy mattered here
    chord: Optional[str] = None    # what was sounding underneath, for the report
    syllable: str = ""
    velocity: float = 0.7
    vibrato_depth_cents: float = 0.0

    @property
    def pitch_shift_cents(self) -> float:
        return (self.midi - self.source_midi) * 100.0

    @property
    def time_shift_s(self) -> float:
        return self.start - self.source_start

    @property
    def moved(self) -> bool:
        return (abs(self.pitch_shift_cents) >= INAUDIBLE_CENTS
                or abs(self.time_shift_s) > 1e-4)

    def to_dict(self) -> dict:
        return {"index": self.index,
                "source_midi": round(self.source_midi, 3),
                "midi": round(self.midi, 3),
                "start": round(self.start, 4), "end": round(self.end, 4),
                "source_start": round(self.source_start, 4),
                "pitch_shift_cents": round(self.pitch_shift_cents, 1),
                "time_shift_ms": round(self.time_shift_s * 1000.0, 1),
                "decision": self.decision,
                "salience": round(self.salience, 3),
                "chord": self.chord, "syllable": self.syllable}


@dataclass
class PerformancePlan:
    """What the engine intends the vocal to be.

    Complete in itself: a renderer needs nothing else to produce this
    performance, and an interface needs nothing else to explain it.
    """
    targets: List[NoteTarget] = field(default_factory=list)
    performance_type: str = "sung"
    tuning_strength: float = 0.0
    timing_strength: float = 0.0
    bar_s: float = 0.0
    beats_per_bar: int = 4
    notes: List[str] = field(default_factory=list)

    # -- What happened, counted ------------------------------------------
    @property
    def tuned(self) -> List[NoteTarget]:
        return [t for t in self.targets if t.decision == TUNED]

    @property
    def retimed(self) -> List[NoteTarget]:
        return [t for t in self.targets if abs(t.time_shift_s) > 1e-4]

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for t in self.targets:
            out[t.decision] = out.get(t.decision, 0) + 1
        return out

    def summary(self) -> str:
        """One line, in words, for a person rather than a log."""
        if not self.targets:
            return "no notes to work with"
        tuned = self.tuned
        bits = []
        if tuned:
            cents = float(np.mean([abs(t.pitch_shift_cents) for t in tuned]))
            bits.append("%d of %d notes tuned by %.0f cents on average"
                        % (len(tuned), len(self.targets), cents))
        else:
            bits.append("every note was already in tune")
        counts = self.counts()
        kept = counts.get(KEPT_GESTURE, 0) + counts.get(KEPT_MELISMA, 0)
        if kept:
            bits.append("%d slides and runs left as performed" % kept)
        if counts.get(KEPT_TOO_FAR):
            bits.append("%d left alone as too far out to be a tuning error"
                        % counts[KEPT_TOO_FAR])
        moved = self.retimed
        if moved:
            ms = float(np.mean([abs(t.time_shift_s) for t in moved])) * 1000.0
            bits.append("%d notes nudged onto the grid by %.0f ms on average"
                        % (len(moved), ms))
        return "; ".join(bits)

    def to_dict(self, include_targets: bool = True) -> dict:
        d: Dict[str, Any] = {
            "performance_type": self.performance_type,
            "tuning_strength": round(self.tuning_strength, 3),
            "timing_strength": round(self.timing_strength, 3),
            "n_notes": len(self.targets),
            "n_tuned": len(self.tuned),
            "n_retimed": len(self.retimed),
            "counts": self.counts(),
            "summary": self.summary(),
            "notes": list(self.notes),
        }
        if self.tuned:
            d["mean_tuning_cents"] = round(
                float(np.mean([abs(t.pitch_shift_cents) for t in self.tuned])), 1)
            d["max_tuning_cents"] = round(
                float(np.max([abs(t.pitch_shift_cents) for t in self.tuned])), 1)
        if include_targets:
            # Capped: a three-minute take has hundreds of notes and the
            # whole list does not belong in a job result.
            d["targets"] = [t.to_dict() for t in self.targets[:400]]
        return d


def pitch_targets(notes: Sequence[Note], context: Any, *,
                  strength: float = 0.5,
                  max_correction_semitones: float = MAX_CORRECTION_SEMITONES
                  ) -> List[NoteTarget]:
    """Where each note should sit in pitch, and why.

    `context` is an `audio.tuning.HarmonicContext` -- taken structurally
    rather than by import, because it belongs to the module that applies
    this plan and importing it here would make the plan layer depend on
    the renderer it exists to be independent of.

    The judgement is `musical.theory`'s: which pitches are legitimate
    under the chord sounding at that instant, narrowed when the note is
    exposed. The amount taken is `musical.salience`'s: a held note over a
    cadence is pulled firmly and a passing sixteenth is barely touched.
    """
    out: List[NoteTarget] = []
    for i, n in enumerate(notes):
        t = NoteTarget(index=i, source_midi=float(n.midi),
                       source_start=float(n.start), source_end=float(n.end),
                       midi=float(n.midi), start=float(n.start),
                       end=float(n.end), velocity=float(n.velocity),
                       vibrato_depth_cents=float(n.vibrato_depth_cents))

        # Gestures and runs are how a singer sounds like a person.
        if n.is_transition:
            t.decision = KEPT_GESTURE
            out.append(t)
            continue
        if n.is_melisma:
            t.decision = KEPT_MELISMA
            out.append(t)
            continue

        chord = context.chord_at(n.start)
        key = context.key_at(n.start)
        t.chord = _chord_name(chord)
        if chord is None and key is None:
            t.decision = KEPT_NO_TARGET
            out.append(t)
            continue

        sal = salience.pitch_salience(
            n, section=context.section_at(n.start),
            is_cadence=context.is_cadence(n.start),
            tessitura_high_midi=context.tessitura_high_midi,
            is_exposed_texture=context.sparse_backing)
        t.salience = float(sal)

        candidates = theory.tuning_candidates(chord, key, context.genre,
                                              exposed=sal >= 1.0)
        target = theory.nearest_target(n.midi, candidates,
                                       max_semitones=max_correction_semitones)
        if target is None:
            t.decision = KEPT_TOO_FAR
            out.append(t)
            continue

        take = float(np.clip(strength * sal, 0.0, MAX_CORRECTION_FRACTION))
        delta = (target - n.midi) * take
        if abs(delta) * 100.0 < INAUDIBLE_CENTS:
            t.decision = KEPT_IN_TUNE
            out.append(t)
            continue

        t.midi = float(n.midi + delta)
        t.decision = TUNED
        out.append(t)
    return out


def snap_to_grid(targets: Sequence[NoteTarget], grid: Sequence[float], *,
                 strength: float = 0.0,
                 downbeats: Sequence[float] = (),
                 bar_s: float = 0.0, beats_per_bar: int = 4,
                 phrases: Sequence[Phrase] = (),
                 sections: Sequence[Section] = (),
                 max_move_s: float = 0.12) -> None:
    """Move each target toward its nearest grid slot, in place.

    Weighted by `musical.salience`, so a downbeat is pulled firmly and an
    off-beat sixteenth is barely touched. That weighting is the difference
    between a vocal that sits in the pocket and one that has been
    quantised flat.

    Notes keep their length: a note is moved, never stretched, because
    stretching one note inside a phrase changes the singer's diction and
    moving it does not.
    """
    if strength <= 0.0 or not targets:
        return
    g = np.asarray([float(x) for x in grid], dtype=np.float64)
    g = g[np.isfinite(g)]
    if g.size < 2:
        return
    g.sort()
    for t in targets:
        if t.decision in (KEPT_GESTURE, KEPT_MELISMA):
            continue
        k = int(np.searchsorted(g, t.start))
        near = [g[j] for j in (k - 1, k) if 0 <= j < g.size]
        if not near:
            continue
        slot = min(near, key=lambda s: abs(s - t.start))
        err = float(slot - t.start)
        if abs(err) > max_move_s:
            continue
        sal = salience.timing_salience(
            t.start, downbeats=downbeats, bar_duration_s=bar_s,
            beats_per_bar=beats_per_bar,
            phrase=_containing_phrase(phrases, t.start),
            section=_containing_section(sections, t.start))
        move = err * float(np.clip(strength * sal, 0.0, 1.0))
        if abs(move) < 1e-4:
            continue
        length = t.end - t.start
        t.start += move
        t.end = t.start + length


def build(notes: Sequence[Note], context: Any, *,
          performance_type: str = "sung",
          tuning_strength: float = 0.0,
          timing_strength: float = 0.0,
          grid: Sequence[float] = (),
          downbeats: Sequence[float] = (),
          bar_s: float = 0.0, beats_per_bar: int = 4,
          phrases: Sequence[Phrase] = (),
          syllables: Sequence[str] = ()) -> PerformancePlan:
    """The whole plan: what to sing, where, and at what pitch."""
    plan = PerformancePlan(performance_type=performance_type,
                           tuning_strength=float(tuning_strength),
                           timing_strength=float(timing_strength),
                           bar_s=float(bar_s),
                           beats_per_bar=int(beats_per_bar))
    if not notes:
        plan.notes.append("no notes were tracked in this take")
        return plan

    plan.targets = pitch_targets(notes, context, strength=tuning_strength)
    for i, t in enumerate(plan.targets):
        if i < len(syllables):
            t.syllable = str(syllables[i])

    if timing_strength > 0 and len(grid) >= 2:
        snap_to_grid(plan.targets, grid, strength=timing_strength,
                     downbeats=downbeats, bar_s=bar_s,
                     beats_per_bar=beats_per_bar, phrases=phrases,
                     sections=getattr(context, "sections", ()) or ())
    return plan


# ─────────────────────────────────────────────────────────────────────────────

def _chord_name(chord: Any) -> Optional[str]:
    if chord is None:
        return None
    names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    try:
        return "%s%s" % (names[int(chord.root) % 12],
                         "" if chord.quality == "maj" else chord.quality)
    except Exception:                                    # noqa: BLE001
        return None


def _containing_phrase(phrases: Sequence[Phrase], t: float) -> Optional[Phrase]:
    for p in phrases:
        if p.start <= t < p.end:
            return p
    return None


def _containing_section(sections: Sequence[Section], t: float) -> Optional[Section]:
    for s in sections:
        if s.start <= t < s.end:
            return s
    return None
