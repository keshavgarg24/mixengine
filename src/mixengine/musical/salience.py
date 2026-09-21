"""
Musical salience: deciding where accuracy matters.

The problem with uniform correction
-----------------------------------
An engine that corrects every moment equally sounds processed, and it
sounds processed for a reason that has nothing to do with the quality of
its algorithms: it spends its correction budget in the wrong places.

A syllable landing 15 ms late on a downbeat is plainly audible. The same
15 ms on an off-beat sixteenth in the middle of a fast rap bar is not --
and "fixing" it costs the flow its character while buying nothing. A note
held for two seconds over a cadence exposes every cent of pitch error; a
passing sixteenth does not, and tuning it hard is how a line loses its
shape.

So the engine asks a different question at every moment: *how much does
accuracy matter right here?* Salience answers it, and every correction
stage -- timing, pitch, level, de-essing, layering -- scales its strength
by the answer.

Timing and pitch have genuinely different salience profiles, so they get
separate functions rather than one shared weight:

  * **Timing** salience follows the metrical hierarchy. Strong beats and
    phrase boundaries matter; subdivisions between them do not.
  * **Pitch** salience follows exposure. Duration, register and harmonic
    context matter; metrical position barely does.

All functions return a multiplier centred near 1.0, so a stage's base
strength stays meaningful and the weighting only redistributes it.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.types import FloatSeq

from ..core.types import Note, Phrase, Section

# Hook and chorus sections carry the track. An error there is heard by
# every listener; the same error in a second verse often is not.
_SECTION_WEIGHT = {
    "hook": 1.50, "chorus": 1.50, "prehook": 1.20,
    "verse": 1.00, "bridge": 1.10, "intro": 0.85,
    "outro": 0.80, "break": 0.80, "inst": 0.60, "solo": 1.10,
    "section": 1.00,
}


# ═════════════════════════════════════════════════════════════════════════════
# Metrical position
# ═════════════════════════════════════════════════════════════════════════════

def metrical_weight(t: float, downbeats: FloatSeq,
                    bar_duration_s: float, subdivision: int = 16,
                    beats_per_bar: int = 4) -> float:
    """Weight from where `t` falls in the bar.

    Implements the standard metrical hierarchy: the downbeat is the
    strongest position, then the other beats, then eighths, then the rest.
    Listeners track the pulse at these levels, which is why errors are
    audible in proportion to the strength of the position they land on.
    """
    dbs = np.asarray(downbeats, dtype=np.float64)
    if dbs.size == 0 or bar_duration_s <= 0:
        return 1.0

    i = int(np.searchsorted(dbs, float(t)) - 1)
    if i < 0:
        bar_start = float(dbs[0])
    else:
        bar_start = float(dbs[min(i, dbs.size - 1)])

    rel = (float(t) - bar_start) / bar_duration_s
    if not (0.0 <= rel < 1.0):
        rel = rel % 1.0

    slot = rel * subdivision
    nearest = int(round(slot)) % subdivision
    # Distance to the nearest slot, in slots: a hit that is not near any
    # grid position is off-grid material and should not be pulled hard.
    dist = abs(slot - round(slot))

    per_beat = max(1, subdivision // max(beats_per_bar, 1))
    if nearest == 0:
        w = 2.0                                   # downbeat
    elif nearest % per_beat == 0:
        w = 1.4                                   # other beats
    elif nearest % max(per_beat // 2, 1) == 0:
        w = 0.9                                   # eighths
    else:
        w = 0.6                                   # sixteenths and finer

    # Taper off for material that is genuinely between grid positions.
    return float(w * (1.0 - 0.4 * min(dist * 2.0, 1.0)))


# ═════════════════════════════════════════════════════════════════════════════
# Timing salience
# ═════════════════════════════════════════════════════════════════════════════

def timing_salience(t: float, *,
                    downbeats: FloatSeq = (),
                    bar_duration_s: float = 0.0,
                    subdivision: int = 16,
                    beats_per_bar: int = 4,
                    phrase: Optional[Phrase] = None,
                    section: Optional[Section] = None,
                    is_cadence: bool = False,
                    word_stress: float = 0.5) -> float:
    """How much timing accuracy matters at `t`.

    Returned values run roughly 0.3 to 3.5. Multiply a stage's base
    correction strength by this to concentrate the work where it is heard.
    """
    w = metrical_weight(t, downbeats, bar_duration_s, subdivision, beats_per_bar)

    if phrase is not None:
        # A phrase entry sets the listener's expectation for everything
        # that follows it, so getting the entry right matters more than
        # any single syllable inside the phrase.
        if abs(t - phrase.anchor) < 0.12:
            w *= 1.6
        elif abs(t - phrase.end) < 0.12:
            w *= 1.15

    if is_cadence:
        w *= 1.35

    # Stressed syllables carry the rhythm of the line.
    w *= 0.85 + 0.45 * float(np.clip(word_stress, 0.0, 1.0))

    if section is not None:
        w *= _SECTION_WEIGHT.get(section.label, 1.0)

    return float(np.clip(w, 0.25, 3.5))


# ═════════════════════════════════════════════════════════════════════════════
# Pitch salience
# ═════════════════════════════════════════════════════════════════════════════

def pitch_salience(note: Note, *,
                   section: Optional[Section] = None,
                   is_cadence: bool = False,
                   tessitura_high_midi: float = 0.0,
                   is_exposed_texture: bool = False) -> float:
    """How much pitch accuracy matters for a given note.

    Driven by exposure rather than metre. The dominant term is duration:
    the ear needs roughly a couple of hundred milliseconds of steady tone
    to judge intonation at all, and beyond that every extra moment makes
    an error more obvious.
    """
    d = note.duration

    # Below ~0.12 s pitch is barely perceptible; the curve saturates once
    # the note is long enough to be clearly judged.
    if d < 0.12:
        w = 0.35
    elif d < 0.25:
        w = 0.7
    else:
        w = 1.0 + 0.6 * float(np.clip((d - 0.25) / 1.0, 0.0, 1.0))

    # Notes at the top of a singer's range are both harder to pitch and
    # more exposed, so errors there are the ones listeners notice.
    if tessitura_high_midi > 0 and note.midi >= tessitura_high_midi - 2.0:
        w *= 1.25

    # Sustained notes with vibrato are being performed deliberately; heavy
    # correction there fights the singer rather than helping them.
    if note.vibrato_depth_cents > 25.0:
        w *= 0.8

    if note.is_melisma:
        w *= 0.6          # runs are gesture, not target pitches

    if is_cadence:
        w *= 1.4

    if is_exposed_texture:
        w *= 1.3          # sparse backing hides nothing

    w *= 0.9 + 0.3 * float(np.clip(note.velocity, 0.0, 1.0))

    if section is not None:
        w *= _SECTION_WEIGHT.get(section.label, 1.0)

    return float(np.clip(w, 0.2, 3.5))


# ═════════════════════════════════════════════════════════════════════════════
# Spectral salience
# ═════════════════════════════════════════════════════════════════════════════

def deess_salience(t: float, *,
                   section: Optional[Section] = None,
                   in_sibilant_span: bool = True,
                   local_brightness: float = 0.5) -> float:
    """How hard the de-esser should work at `t`.

    The production rule this encodes: set the de-esser so it moves on the
    words that have a problem, not constantly. Over-applied de-essing
    turns sibilants into lisps, which is a worse and more obvious defect
    than the sibilance it was removing.

    With measured phoneme positions available, `in_sibilant_span` is a real
    measurement rather than a guess, which is what makes surgical de-essing
    possible at all.
    """
    if not in_sibilant_span:
        return 0.15                       # essentially idle between sibilants
    w = 1.0 + 1.0 * float(np.clip((local_brightness - 0.5) * 2.0, 0.0, 1.0))
    if section is not None:
        w *= _SECTION_WEIGHT.get(section.label, 1.0) * 0.5 + 0.5
    return float(np.clip(w, 0.2, 2.5))


def level_salience(t: float, *,
                   phrase: Optional[Phrase] = None,
                   section: Optional[Section] = None,
                   word_stress: float = 0.5) -> float:
    """How much level consistency matters at `t`.

    Level riding should smooth phrase-to-phrase inconsistency without
    flattening intra-phrase dynamics, which are the singer's expression.
    Weighting by phrase rather than by instant is what preserves that.
    """
    w = 1.0
    if phrase is not None and phrase.duration > 0:
        pos = (t - phrase.start) / phrase.duration
        # Phrase ends fall away naturally; holding them up sounds unnatural.
        if pos > 0.85:
            w *= 0.7
    w *= 0.9 + 0.2 * float(np.clip(word_stress, 0.0, 1.0))
    if section is not None:
        w *= _SECTION_WEIGHT.get(section.label, 1.0) * 0.4 + 0.6
    return float(np.clip(w, 0.3, 2.0))


# ═════════════════════════════════════════════════════════════════════════════
# Error weighting for the critic
# ═════════════════════════════════════════════════════════════════════════════

def weighted_error(errors: FloatSeq, weights: FloatSeq) -> float:
    """Salience-weighted aggregate error.

    The critic must not report a plain mean. A take that is immaculate
    everywhere except its downbeats is worse than one that is slightly
    loose throughout, and an unweighted average says the opposite.
    """
    e = np.asarray(errors, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if e.size == 0:
        return 0.0
    if w.size != e.size:
        w = np.ones_like(e)
    w = np.maximum(w, 1e-6)
    return float(np.sum(np.abs(e) * w) / np.sum(w))


def perceptible_timing_threshold_ms(subdivision_ms: float) -> float:
    """Below this, a timing deviation is not worth correcting.

    Two bounds apply. Listeners struggle to identify timing discrepancies
    of about 30 ms or less even in controlled listening, and a deviation
    small relative to the current subdivision is inaudible in context.
    Correcting below this threshold spends stretch artifacts on something
    nobody can hear, which is a straight loss.
    """
    return float(max(12.0, min(30.0, subdivision_ms * 0.12)))
