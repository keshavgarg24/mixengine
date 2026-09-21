"""
Musical pitch correction.

This replaces scale-snapping with something closer to what a person does
when they tune a vocal by hand.

The difference that matters is *what the note is judged against*. Scale
snapping asks "is this pitch in the key?", which is the wrong question: a
major seventh held over a minor iv chord is in the key and sounds like a
mistake, while a flat third over a major chord is outside the key and is
the defining sound of most rap, soul and blues melody. The question a
producer actually asks is "what chord is underneath this note, right now,
and how exposed is it?"

So every note is evaluated against three things:

  * **The chord sounding at that instant**, with the key as a fallback when
    no chord is known. Chord tones and deliberate tensions are legitimate
    targets; only the genuinely harsh intervals -- a pitch a minor ninth
    above a chord tone -- are treated as wrong.
  * **How exposed the note is.** A two-second note over a cadence is under
    scrutiny; a passing sixteenth in a fast bar is not. Correction strength
    scales with that exposure, which is what keeps a corrected line sounding
    performed rather than quantised.
  * **Whether it is a note at all.** Scoops, falls, slides and melisma runs
    are gestures. Correcting them is the single most recognisable way an
    automatic tuner announces itself, so they are skipped outright.

Genre matters and is not cosmetic. In the blues-derived genres the flat
third, flat fifth and flat seventh are protected, because a tuner that
"fixes" them has removed the reason the melody works.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from ..core.types import FloatSeq

from . import dsp
from ..core.types import (
    ChordEvent, KeyRegion, Note, Section,
    ATTACK_SLIDE, RELEASE_FALL,
)
from ..musical import salience, theory

log = logging.getLogger("mixengine.tuning")


@dataclass
class HarmonicContext:
    """What is sounding underneath the vocal, as a function of time.

    Deliberately small: the tuner needs to ask a handful of questions at a
    point in time, and taking the whole IR would couple it to a structure
    it does not otherwise need.
    """
    chords: Sequence[ChordEvent] = ()
    key_regions: Sequence[KeyRegion] = ()
    sections: Sequence[Section] = ()
    cadence_times: FloatSeq = ()
    genre: Optional[str] = None
    tessitura_high_midi: float = 0.0
    sparse_backing: bool = False        # exposed texture: errors show more

    def chord_at(self, t: float) -> Optional[ChordEvent]:
        for c in self.chords:
            if c.start <= t < c.end:
                return c
        return None

    def key_at(self, t: float) -> Optional[KeyRegion]:
        for k in self.key_regions:
            if k.start <= t < k.end:
                return k
        return self.key_regions[0] if self.key_regions else None

    def section_at(self, t: float) -> Optional[Section]:
        for s in self.sections:
            if s.start <= t < s.end:
                return s
        return None

    def is_cadence(self, t: float, tol: float = 0.35) -> bool:
        return any(abs(float(c) - t) <= tol for c in self.cadence_times)

    @staticmethod
    def from_beat_dna(bdna: dict, semitone_shift: int = 0,
                      genre: Optional[str] = None,
                      tessitura_high_midi: float = 0.0) -> "HarmonicContext":
        """Build from a beat DNA document, following any transposition.

        The shift is applied here rather than at the call site because the
        beat may have been pitch-shifted during rendering. Tuning a vocal
        to the chords of the *untransposed* beat would be confidently and
        consistently wrong -- by exactly the shift amount.
        """
        chords: List[ChordEvent] = []
        for c in (bdna.get("chords") or []):
            try:
                ev = ChordEvent.from_dict(c)
            except Exception:
                continue
            ev.root = (ev.root + int(semitone_shift)) % 12
            chords.append(ev)
        # Per-bar chord estimates carry no end time; give each one a span
        # running to the next, so `chord_at` can answer at all.
        for i, ev in enumerate(chords):
            if ev.end <= ev.start:
                ev.end = chords[i + 1].start if i + 1 < len(chords) else ev.start + 4.0

        keys: List[KeyRegion] = []
        k = bdna.get("key")
        if k and k.get("pc") is not None:
            keys.append(KeyRegion(
                start=0.0, end=float(bdna.get("duration_s") or 1e6),
                pc=(int(k["pc"]) + int(semitone_shift)) % 12,
                mode=str(k.get("mode", "minor")),
                confidence=float(bdna.get("key_confidence") or 0.0)))

        sections = [Section.from_dict(s) for s in (bdna.get("sections") or [])]
        pocket = float(bdna.get("pocket_score") or 0.5)
        return HarmonicContext(
            chords=chords, key_regions=keys, sections=sections,
            cadence_times=theory.detect_cadences(chords, keys) and
            [c.time for c in theory.detect_cadences(chords, keys)] or [],
            genre=genre or bdna.get("genre"),
            tessitura_high_midi=tessitura_high_midi,
            sparse_backing=pocket > 0.7)


@dataclass
class TuningReport:
    enabled: bool = True
    method: str = "musical"
    notes_considered: int = 0
    notes_corrected: int = 0
    notes_skipped_transition: int = 0
    notes_skipped_melisma: int = 0
    notes_refused_far: int = 0
    mean_correction_cents: float = 0.0
    max_correction_cents: float = 0.0
    chord_aware_fraction: float = 0.0
    blue_notes_preserved: int = 0

    def to_dict(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


def tune_musical(y: np.ndarray, sr: int, notes: Sequence[Note],
                 context: HarmonicContext, *,
                 base_strength: float = 0.5,
                 max_correction_semitones: float = 1.2,
                 pitch_shift_fn: Optional[Callable] = None
                 ) -> Tuple[np.ndarray, dict]:
    """Correct intonation with musical judgement. Returns `(audio, report)`.

    `pitch_shift_fn(segment, sr, semitones) -> segment` is injected so this
    module stays independent of whichever stretch backend is installed, and
    so the decision logic can be tested without one.
    """
    v = dsp.as_2d(y)
    rep = TuningReport(enabled=base_strength > 0.0,
                       notes_considered=len(notes))
    if base_strength <= 0 or not notes:
        rep.enabled = False
        return v, rep.to_dict()

    if pitch_shift_fn is None:
        from .transform import pitch_shift as _ps
        pitch_shift_fn = _ps

    out = v.copy()
    corrections: List[float] = []
    chord_hits = 0

    for n in notes:
        start = int(n.start * sr)
        end = min(int(n.end * sr), len(v))
        if end <= start or start >= len(v):
            continue

        # Gestures are performance, not error.
        if n.is_transition:
            rep.notes_skipped_transition += 1
            continue
        if n.is_melisma:
            rep.notes_skipped_melisma += 1
            continue

        chord = context.chord_at(n.start)
        key = context.key_at(n.start)
        if chord is not None:
            chord_hits += 1

        # How much scrutiny is this note under?
        sal = salience.pitch_salience(
            n, section=context.section_at(n.start),
            is_cadence=context.is_cadence(n.start),
            tessitura_high_midi=context.tessitura_high_midi,
            is_exposed_texture=context.sparse_backing)

        exposed = sal >= 1.0
        candidates = theory.tuning_candidates(chord, key, context.genre,
                                              exposed=exposed)
        target = theory.nearest_target(n.midi, candidates,
                                       max_semitones=max_correction_semitones)
        if target is None:
            # Too far from any legal target to be a tuning error. Forcing it
            # would produce a confident wrong answer; leaving it alone is
            # the honest outcome.
            rep.notes_refused_far += 1
            continue

        if key is not None and context.genre:
            from ..musical.theory import BLUES_FAMILY, blue_notes
            g = str(context.genre).lower().replace(" ", "_")
            if g in BLUES_FAMILY and (int(round(n.midi)) % 12) in blue_notes(key.pc):
                rep.blue_notes_preserved += 1

        delta = (target - n.midi) * float(np.clip(base_strength * sal, 0.0, 0.95))
        if abs(delta) < 0.04:                      # under ~4 cents: inaudible
            continue

        seg = v[start:end]
        if len(seg) < int(0.03 * sr):
            continue
        try:
            shifted = dsp.pad_to(pitch_shift_fn(seg, sr, float(delta)), len(seg))
        except Exception:
            continue

        # Crossfade the edges or the note boundaries click.
        fade = min(int(0.006 * sr), len(seg) // 4)
        if fade > 2:
            ramp = np.linspace(0.0, 1.0, fade)[:, None]
            shifted[:fade] = shifted[:fade] * ramp + seg[:fade] * (1 - ramp)
            shifted[-fade:] = (shifted[-fade:] * ramp[::-1]
                               + seg[-fade:] * (1 - ramp[::-1]))
        out[start:end] = shifted
        corrections.append(delta * 100.0)

    rep.notes_corrected = len(corrections)
    if corrections:
        rep.mean_correction_cents = float(np.mean(np.abs(corrections)))
        rep.max_correction_cents = float(np.max(np.abs(corrections)))
    rep.chord_aware_fraction = (chord_hits / len(notes)) if notes else 0.0

    if corrections:
        log.info("  tuned %d/%d notes (mean %.0f cents, %.0f%% chord-aware, "
                 "%d gestures preserved)", rep.notes_corrected,
                 rep.notes_considered, rep.mean_correction_cents,
                 rep.chord_aware_fraction * 100.0,
                 rep.notes_skipped_transition + rep.notes_skipped_melisma)
    return out.astype(np.float32), rep.to_dict()


def notes_from_dna(note_dicts: Sequence[dict],
                   detect_gestures: bool = True) -> List[Note]:
    """Convert DNA note dicts into `Note` objects, inferring gestures.

    The analyser stores median pitch per note but not how the note was
    approached or released. Those are recoverable from the relationship
    between consecutive notes, and they are what tells the tuner which
    notes to leave alone -- so they are inferred here rather than being
    lost.
    """
    notes: List[Note] = []
    for d in note_dicts:
        try:
            notes.append(Note.from_dict(d))
        except Exception:
            continue
    if not detect_gestures:
        return notes

    for i, n in enumerate(notes):
        # Wide pitch spread within a note is a slide or scoop, not a
        # steady note that happens to be out of tune.
        if n.vibrato_depth_cents > 70.0:
            n.attack = ATTACK_SLIDE
        if i > 0:
            prev = notes[i - 1]
            gap = n.start - prev.end
            step = n.midi - prev.midi
            # Legato motion of more than a tone is an approach gesture.
            if gap < 0.06 and abs(step) > 2.0:
                n.attack = ATTACK_SLIDE
            # A short note dropping steeply straight after a long one is a fall.
            if gap < 0.08 and step < -2.5 and n.duration < 0.2:
                n.release = RELEASE_FALL
        # Rapid same-syllable runs read as melisma.
        if 0 < i < len(notes) - 1:
            if (n.duration < 0.16
                    and n.start - notes[i - 1].end < 0.05
                    and notes[i + 1].start - n.end < 0.05):
                n.is_melisma = True
    return notes
