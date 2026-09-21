"""
Harmonic reasoning.

Pure music theory over pitch classes. No audio, no numpy arrays of samples,
no I/O -- which means every rule in here is unit-testable and every
decision the engine makes about consonance can be inspected and argued
with.

Why the engine needs this
-------------------------
Deciding whether a sung note is "right" by checking it against the song's
global key is the approximation that makes automatic tuning sound bland.
A note can be perfectly in key and still be the wrong note: a major 7th
held over a minor iv chord is in the scale and sounds like a mistake. The
question a producer actually asks is *what chord is underneath this note,
right now, and for how long is the note exposed?*

So consonance here is always evaluated against the chord sounding at that
instant, with the scale as a fallback when no chord is known, and with
three refinements that stop the result from being mechanically "correct"
but musically dead:

  * **Available tensions.** A 9th over a major chord is consonant and
    beautiful. Treating it as an error and dragging it to the root is a
    downgrade.
  * **Avoid notes.** A pitch sitting a minor 9th above a chord tone is the
    genuinely harsh interval, and it is what should be corrected -- not
    every non-triad tone.
  * **Blue notes.** A flat third over a major chord is the defining sound
    of blues, rock, soul and most rap melody. It is not out of tune. Any
    system that "fixes" it has destroyed the performance, so the blues
    degrees are explicitly protected in the genres where they belong.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

from ..core.types import FloatSeq
from ..core.types import (
    Cadence, ChordEvent, KeyRegion,
    CADENCE_AUTHENTIC, CADENCE_DECEPTIVE, CADENCE_HALF, CADENCE_PLAGAL,
    FUNC_APPLIED, FUNC_CHROMATIC, FUNC_DOMINANT, FUNC_SUBDOMINANT, FUNC_TONIC,
)

# ─────────────────────────────────────────────────────────────────────────────
# Scales
# ─────────────────────────────────────────────────────────────────────────────

SCALES: Dict[str, Tuple[int, ...]] = {
    "major":          (0, 2, 4, 5, 7, 9, 11),
    "minor":          (0, 2, 3, 5, 7, 8, 10),     # natural minor / aeolian
    "harmonic_minor": (0, 2, 3, 5, 7, 8, 11),
    "melodic_minor":  (0, 2, 3, 5, 7, 9, 11),
    "dorian":         (0, 2, 3, 5, 7, 9, 10),
    "phrygian":       (0, 1, 3, 5, 7, 8, 10),
    "lydian":         (0, 2, 4, 6, 7, 9, 11),
    "mixolydian":     (0, 2, 4, 5, 7, 9, 10),
    "locrian":        (0, 1, 3, 5, 6, 8, 10),
    "minor_penta":    (0, 3, 5, 7, 10),
    "major_penta":    (0, 2, 4, 7, 9),
    "blues":          (0, 3, 5, 6, 7, 10),
}

# The three degrees that read as expressive inflection rather than error
# when they appear over a major tonality.
BLUE_DEGREES: Tuple[int, ...] = (3, 6, 10)        # b3, b5, b7

# Genres where the blues degrees are part of the language and must never be
# "corrected" toward the diatonic scale.
BLUES_FAMILY: FrozenSet[str] = frozenset({
    "trap", "drill", "hip_hop", "boom_bap", "rnb", "soul", "blues", "rock",
    "funk", "gospel", "jazz", "melodic_trap", "afrobeats", "dancehall",
})

# Diatonic triad qualities, scale degree -> quality.
_MAJOR_TRIADS = ("maj", "min", "min", "maj", "maj", "min", "dim")
_MINOR_TRIADS = ("min", "dim", "maj", "min", "min", "maj", "maj")

# Harmonic function by scale degree (0-indexed).
_MAJOR_FUNCTIONS = (FUNC_TONIC, FUNC_SUBDOMINANT, FUNC_TONIC, FUNC_SUBDOMINANT,
                    FUNC_DOMINANT, FUNC_TONIC, FUNC_DOMINANT)
_MINOR_FUNCTIONS = (FUNC_TONIC, FUNC_SUBDOMINANT, FUNC_TONIC, FUNC_SUBDOMINANT,
                    FUNC_DOMINANT, FUNC_SUBDOMINANT, FUNC_DOMINANT)

_ROMAN = ("I", "II", "III", "IV", "V", "VI", "VII")


def scale_pcs(tonic_pc: int, mode: str) -> Tuple[int, ...]:
    """Pitch classes of a scale."""
    steps = SCALES.get(mode, SCALES["major"])
    return tuple((int(tonic_pc) + s) % 12 for s in steps)


def relative_major_pc(pc: int, mode: str) -> int:
    """The major key sharing this key's notes.

    A minor key and its relative major contain identical pitch classes, so
    normalising both sides to this root before comparing is what makes a
    vocal in E minor correctly score as a perfect match against a beat in
    G major, rather than as a three-semitone clash.
    """
    return (int(pc) + 3) % 12 if mode == "minor" else int(pc) % 12


def degree_of(pc: int, tonic_pc: int, mode: str) -> Optional[int]:
    """Scale degree (0-6) of a pitch class, or None if chromatic."""
    pcs = scale_pcs(tonic_pc, mode)
    try:
        return pcs.index(int(pc) % 12)
    except ValueError:
        return None


def is_diatonic(pc: int, tonic_pc: int, mode: str) -> bool:
    return degree_of(pc, tonic_pc, mode) is not None


def blue_notes(tonic_pc: int) -> FrozenSet[int]:
    """The blues inflections relative to a tonic."""
    return frozenset((int(tonic_pc) + d) % 12 for d in BLUE_DEGREES)


# ─────────────────────────────────────────────────────────────────────────────
# Chord construction and labelling
# ─────────────────────────────────────────────────────────────────────────────

def diatonic_triad(tonic_pc: int, mode: str, degree: int) -> Tuple[int, str]:
    """`(root_pc, quality)` of the triad on a scale degree (0-indexed)."""
    pcs = scale_pcs(tonic_pc, mode)
    d = int(degree) % 7
    qualities = _MAJOR_TRIADS if mode == "major" else _MINOR_TRIADS
    return pcs[d], qualities[d]


def roman_numeral(chord: ChordEvent, key: KeyRegion) -> str:
    """Roman-numeral label, e.g. 'IV', 'vi', 'bVII', 'V/V'.

    Purely for logs and user-facing explanations -- nothing branches on the
    string. But being able to read a progression back as 'i bVII bVI V' is
    what makes the harmonic layer debuggable by someone who knows music
    rather than only by someone who knows the code.
    """
    d = degree_of(chord.root, key.pc, key.mode)
    if d is not None:
        num = _ROMAN[d]
    else:
        # Chromatic root: name it by distance above the tonic with a flat.
        semis = (chord.root - key.pc) % 12
        nearest = min(range(7),
                      key=lambda i: abs(((scale_pcs(key.pc, key.mode)[i] - key.pc) % 12) - semis))
        num = "b" + _ROMAN[nearest]
    if chord.quality in ("min", "dim"):
        num = num.lower()
    if chord.quality == "dim":
        num += "o"
    elif chord.quality == "aug":
        num += "+"
    elif chord.quality in ("sus2", "sus4"):
        num += chord.quality
    return num


def harmonic_function(chord: ChordEvent, key: KeyRegion) -> str:
    """Classify a chord as tonic, subdominant, dominant, applied or chromatic.

    Function is what tells the arranger where a phrase is going. A dominant
    chord creates the expectation that resolves on the next downbeat, and
    that expectation is where listeners feel a phrase land.
    """
    d = degree_of(chord.root, key.pc, key.mode)
    if d is None:
        # A major or dominant-quality chord a fifth above a diatonic root is
        # an applied dominant -- V/V and friends -- not merely chromatic.
        target = (chord.root + 5) % 12
        if chord.quality == "maj" and is_diatonic(target, key.pc, key.mode):
            return FUNC_APPLIED
        return FUNC_CHROMATIC
    table = _MAJOR_FUNCTIONS if key.mode == "major" else _MINOR_FUNCTIONS
    func = table[d]
    # In minor, a major chord on degree 5 is a raised-leading-tone V: strongly
    # dominant, more so than the diatonic minor v.
    if key.mode == "minor" and d == 4 and chord.quality == "maj":
        return FUNC_DOMINANT
    return func


def annotate_functions(chords: Sequence[ChordEvent],
                       key_regions: Sequence[KeyRegion]) -> List[ChordEvent]:
    """Fill in `function` on each chord using the key in force at its onset."""
    out: List[ChordEvent] = []
    for c in chords:
        k = _key_at(key_regions, c.start)
        if k is not None:
            c.function = harmonic_function(c, k)
        out.append(c)
    return out


def _key_at(key_regions: Sequence[KeyRegion], t: float) -> Optional[KeyRegion]:
    for k in key_regions:
        if k.start <= t < k.end:
            return k
    return key_regions[0] if key_regions else None


def _refine_boundary(chords, old_key, new_key, detected_at: int,
                     window: int, score_key) -> int:
    """Find where a modulation actually began, not where it was confirmed.

    A windowed detector notices a key change late: the window has to fill
    with enough of the new key before it outscores the old one, and the
    hysteresis that suppresses false positives delays it further. Reporting
    the confirmation point as the boundary would place the key change
    several chords after the first chord of the new key, which then makes
    every per-section transposition decision in that gap wrong.

    So once a switch is confirmed, walk backwards over single chords and
    find the earliest one that already favoured the new key.
    """
    earliest = detected_at
    for i in range(detected_at, max(0, detected_at - window) - 1, -1):
        seg = chords[i:i + 1]
        if score_key(seg, new_key[0], new_key[1]) >= score_key(seg, old_key[0], old_key[1]):
            earliest = i
        else:
            break
    return earliest


# ─────────────────────────────────────────────────────────────────────────────
# Consonance: tensions and avoid notes
# ─────────────────────────────────────────────────────────────────────────────

def available_tensions(chord: ChordEvent, key: Optional[KeyRegion] = None) -> FrozenSet[int]:
    """Non-triad pitch classes that sound intentional over this chord.

    Restricted to scale members when a key is known, because a tension that
    is outside the key reads as a wrong note rather than as colour.
    """
    root = int(chord.root) % 12
    if chord.quality == "maj":
        candidates = {2, 9, 11}              # 9th, 13th, maj7
    elif chord.quality == "min":
        candidates = {2, 5, 9, 10}           # 9th, 11th, 13th, b7
    elif chord.quality == "dim":
        candidates = {9}                     # 13th / dim7
    elif chord.quality == "aug":
        candidates = {2, 10}
    elif chord.quality in ("sus2", "sus4"):
        candidates = {10, 2, 5}
    elif chord.quality == "pow":
        # A power chord has no third, so both are available and neither is
        # a clash. This is why they are so common under sung melodies.
        candidates = {2, 3, 4, 5, 9, 10}
    else:
        candidates = {2, 9}
    pcs = {(root + c) % 12 for c in candidates}
    if key is not None:
        pcs &= set(scale_pcs(key.pc, key.mode))
    return frozenset(pcs - chord.tones)


def avoid_notes(chord: ChordEvent) -> FrozenSet[int]:
    """Pitch classes that genuinely clash: a minor 9th above a chord tone.

    This is the interval the ear rejects, and it is a far better definition
    of "wrong note" than "not in the triad". Applying correction only to
    these leaves tensions and passing tones alone, which is the difference
    between a tuner that preserves a performance and one that flattens it.
    """
    tones = chord.tones
    bad = set()
    for t in tones:
        cand = (t + 1) % 12
        if cand not in tones:
            bad.add(cand)
    return frozenset(bad)


def consonance(pc: int, chord: Optional[ChordEvent],
               key: Optional[KeyRegion] = None,
               genre: Optional[str] = None) -> float:
    """How consonant a pitch class is right now, in [0, 1].

    The ordering encodes the musical priorities directly: chord tones are
    unambiguous, tensions are colour, blue notes are expression in the
    genres that use them, other scale tones are fine in passing, and only
    minor-ninth clashes score as genuinely wrong.
    """
    p = int(pc) % 12
    if chord is not None:
        if p in chord.tones:
            return 1.0
        if p in avoid_notes(chord):
            # A blue note that happens to land on an avoid pitch is still
            # expression in blues-derived music, so it is rescued here
            # rather than being corrected away.
            if _is_blue(p, key, genre):
                return 0.6
            return 0.05
        if p in available_tensions(chord, key):
            return 0.85
    if _is_blue(p, key, genre):
        return 0.7
    if key is not None and is_diatonic(p, key.pc, key.mode):
        return 0.6
    return 0.2


def _is_blue(pc: int, key: Optional[KeyRegion], genre: Optional[str]) -> bool:
    if key is None or genre is None:
        return False
    if genre.lower().replace(" ", "_") not in BLUES_FAMILY:
        return False
    return (int(pc) % 12) in blue_notes(key.pc)


def tuning_candidates(chord: Optional[ChordEvent], key: Optional[KeyRegion],
                      genre: Optional[str] = None,
                      exposed: bool = True) -> FrozenSet[int]:
    """Pitch classes a note may legitimately be corrected *to*.

    `exposed` distinguishes the two cases that matter. A long or accented
    note is under scrutiny, so the target set is narrow -- chord tones and
    deliberate tensions. A short passing note is not, so the whole scale is
    fair game and pulling it to a chord tone would flatten the line.
    """
    out: Set[int] = set()
    if chord is not None:
        out |= set(chord.tones)
        if exposed:
            out |= set(available_tensions(chord, key))
    if key is not None and (not exposed or chord is None):
        out |= set(scale_pcs(key.pc, key.mode))
    if key is not None and genre and genre.lower().replace(" ", "_") in BLUES_FAMILY:
        out |= set(blue_notes(key.pc))
    if chord is not None:
        out -= set(avoid_notes(chord)) - _blue_set(key, genre)
    return frozenset(out) if out else frozenset(range(12))


def _blue_set(key: Optional[KeyRegion], genre: Optional[str]) -> Set[int]:
    if key is None or not genre:
        return set()
    if genre.lower().replace(" ", "_") not in BLUES_FAMILY:
        return set()
    return set(blue_notes(key.pc))


def nearest_target(midi: float, candidates: Iterable[int],
                   max_semitones: float = 1.5) -> Optional[float]:
    """Closest MIDI pitch in `candidates` (as pitch classes) within a limit.

    The cap matters. A note more than roughly a tone away from any legal
    target is not a tuning error -- it is a different note, or a detection
    failure. Dragging it into place produces a confident wrong answer,
    which is worse than leaving it alone, so this returns None instead.
    """
    cands = {int(c) % 12 for c in candidates}
    if not cands:
        return None
    base = int(round(midi))
    best, best_d = None, float("inf")
    for cand in range(base - 3, base + 4):
        if cand % 12 not in cands:
            continue
        d = abs(midi - cand)
        if d < best_d:
            best, best_d = float(cand), d
    if best is None or best_d > max_semitones:
        return None
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Progression-level reasoning
# ─────────────────────────────────────────────────────────────────────────────

def detect_cadences(chords: Sequence[ChordEvent],
                    key_regions: Sequence[KeyRegion],
                    min_strength: float = 0.35) -> List[Cadence]:
    """Find resolution points in a progression.

    Cadences are the moments a listener expects a phrase to land on, which
    makes them the highest-salience positions in the track: an error there
    is far more audible than the same error mid-phrase.
    """
    out: List[Cadence] = []
    for i in range(len(chords) - 1):
        a, b = chords[i], chords[i + 1]
        k = _key_at(key_regions, b.start)
        if k is None:
            continue
        fa = harmonic_function(a, k)
        da = degree_of(a.root, k.pc, k.mode)
        db = degree_of(b.root, k.pc, k.mode)
        conf = min(a.confidence or 0.5, b.confidence or 0.5)

        kind, strength = None, 0.0
        if da == 4 and db == 0:
            kind, strength = CADENCE_AUTHENTIC, 1.0
        elif da == 3 and db == 0:
            kind, strength = CADENCE_PLAGAL, 0.7
        elif da == 4 and db == 5:
            kind, strength = CADENCE_DECEPTIVE, 0.6
        elif db == 4 and fa != FUNC_DOMINANT:
            kind, strength = CADENCE_HALF, 0.45
        if kind is None:
            continue
        s = strength * max(conf, 0.3)
        if s >= min_strength:
            out.append(Cadence(time=b.start, kind=kind, strength=round(s, 4)))
    return out


def progression_fingerprint(chords: Sequence[ChordEvent],
                            key: KeyRegion) -> Tuple[str, ...]:
    """Roman-numeral sequence, deduplicated -- a comparable progression shape.

    Two beats with the same fingerprint support the same melody, regardless
    of key or tempo. That makes this a cheap and strong matching signal
    that global key comparison misses entirely.
    """
    out: List[str] = []
    for c in chords:
        r = roman_numeral(c, key)
        if not out or out[-1] != r:
            out.append(r)
    return tuple(out)


def detect_modulations(chords: Sequence[ChordEvent],
                       window: int = 8,
                       min_gain: float = 0.18) -> List[KeyRegion]:
    """Segment a progression into key regions.

    Slides a window over the chords and scores every candidate key by how
    much of the windowed chord content is diatonic to it, weighted by
    duration. A new region is opened only when a different key wins by a
    clear margin, because a single borrowed chord is colour, not a
    modulation, and splitting on it would fragment the analysis.
    """
    if not chords:
        return []
    n = len(chords)
    w = max(3, min(int(window), n))

    def score_key(seg: Sequence[ChordEvent], pc: int, mode: str) -> float:
        pcs = set(scale_pcs(pc, mode))
        total = sum(max(c.duration, 1e-6) for c in seg)
        if total <= 0:
            return 0.0
        hit = 0.0
        for c in seg:
            inside = len(c.tones & pcs) / max(len(c.tones), 1)
            # A chord whose root is the tonic is much stronger evidence of
            # the key than one that merely shares notes with it.
            if (c.root % 12) == pc:
                inside = min(1.0, inside + 0.25)
            hit += inside * max(c.duration, 1e-6)
        return hit / total

    candidates = [(pc, mode) for pc in range(12) for mode in ("major", "minor")]

    # Score every candidate key in every window up front. Comparing the
    # challenger against the incumbent *within the same window* is the only
    # comparison that means anything: comparing it against the incumbent's
    # best score from an earlier window measures how good that earlier
    # window was, not whether the key has changed.
    window_scores: List[Dict[Tuple[int, str], float]] = []
    for i in range(n):
        seg = chords[max(0, i - w // 2): min(n, i + w // 2 + 1)]
        window_scores.append({c: score_key(seg, c[0], c[1]) for c in candidates})

    regions: List[KeyRegion] = []
    cur = max(window_scores[0].items(), key=lambda kv: kv[1])[0]
    start = chords[0].start
    pending: Optional[Tuple[int, str]] = None
    pending_since = 0
    # A real modulation persists. Requiring the challenger to win several
    # consecutive windows stops a single borrowed chord -- which is colour,
    # not a key change -- from fragmenting the analysis.
    hysteresis = max(2, w // 3)

    for i in range(1, n):
        scores = window_scores[i]
        best = max(scores.items(), key=lambda kv: kv[1])[0]
        gain = scores[best] - scores[cur]
        same_family = (relative_major_pc(*best) == relative_major_pc(*cur))

        if best != cur and not same_family and gain > min_gain:
            if pending != best:
                pending, pending_since = best, i
            elif i - pending_since + 1 >= hysteresis:
                boundary = chords[_refine_boundary(
                    chords, cur, pending, pending_since, w, score_key)].start
                if boundary > start:
                    regions.append(KeyRegion(
                        start=start, end=boundary, pc=cur[0], mode=cur[1],
                        confidence=round(window_scores[pending_since - 1][cur], 4)))
                    start = boundary
                cur, pending = pending, None
        else:
            pending = None

    regions.append(KeyRegion(start=start, end=chords[-1].end,
                             pc=cur[0], mode=cur[1],
                             confidence=round(window_scores[-1][cur], 4)))
    return regions


# ─────────────────────────────────────────────────────────────────────────────
# Harmony generation (for vocal stacks)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HarmonyVoice:
    """A generated harmony line as per-note MIDI offsets from the lead."""
    name: str
    offsets: Tuple[float, ...]
    mean_interval: float = 0.0


def generate_harmony(lead_midi: FloatSeq,
                     times: FloatSeq,
                     chord_at,
                     key_at,
                     interval: str = "third",
                     direction: int = 1,
                     tessitura: Optional[Tuple[float, float]] = None
                     ) -> HarmonyVoice:
    """Build a harmony line that stays inside the harmony at every moment.

    A fixed transposition is not a harmony -- shift a melody up four
    semitones and half the notes land outside the chord. A real harmony
    part moves in thirds *of the scale*, so the interval alternates between
    three and four semitones depending on where in the key the note sits.
    That alternation is exactly what this computes, snapping each harmony
    note to the nearest chord tone or scale tone above or below the lead.

    `tessitura` clamps the result into the singer's comfortable range, and
    when a note would fall outside it the line is folded by an octave
    rather than being allowed to sit somewhere the voice cannot go.
    """
    steps = {"third": 2, "fourth": 3, "fifth": 4, "sixth": 5, "octave": 7}
    step = steps.get(interval, 2) * (1 if direction >= 0 else -1)

    offsets: List[float] = []
    for midi, t in zip(lead_midi, times):
        chord = chord_at(t)
        key = key_at(t)
        # Accept a KeyRegion (which carries `.key`) as well as a bare Key.
        key_obj = getattr(key, "key", key)
        scale = sorted(scale_pcs(key_obj.pc, key_obj.mode)) if key_obj is not None else []
        chord_tones = sorted(chord.tones) if chord is not None else []

        # The ladder is always the *scale*. Walking the chord's tones was
        # the original implementation, and it is wrong in a way that only
        # shows once a real chord is supplied: a triad has three notes per
        # octave, so two steps up a chord ladder is a fifth or a sixth, not
        # a third. The first caller to pass chords got harmonies at +6 to
        # +10 semitones. The one test that existed used no chord and so
        # walked the scale, which happened to be right.
        pool = scale or chord_tones
        if not pool:
            offsets.append(float(step))
            continue

        base = int(round(midi))
        ladder = [p for oct_ in range(-2, 3)
                  for p in (base - (base % 12) + oct_ * 12 + pc for pc in pool)]
        ladder = sorted(set(ladder))
        try:
            idx = min(range(len(ladder)), key=lambda i: abs(ladder[i] - midi))
        except ValueError:
            offsets.append(float(step))
            continue
        target_idx = int(min(max(idx + step, 0), len(ladder) - 1))
        target = float(ladder[target_idx])

        # Reconcile with the chord and with the singer. Two things can be
        # wrong with the raw ladder tone. Against the chord: a scale tone a
        # semitone above a chord tone is a minor ninth against it, the
        # avoid-note case. Against the lead: a chromatic lead note -- a
        # blue note, a slide caught mid-way -- sits off the ladder, so the
        # tone a third up the ladder can be under a minor third from what
        # was actually sung, and a harmony is never a second; nor is it a
        # tritone, which is where the avoid-note step lands when the lead
        # is itself a tension. The fix is the same in every case: continue
        # along the ladder, away from the lead, to the next tone that is
        # clean on both counts. Dropping onto the chord tone below was the
        # first implementation, and in a major key that always lands a
        # major second above the lead: consonant with the chord, a clash
        # with the singer.
        sign = 1 if direction >= 0 else -1
        for _ in range(4):
            pc = int(round(target)) % 12
            gap = abs(target - float(midi))
            avoid = (bool(chord_tones) and pc not in chord_tones
                     and (pc - 1) % 12 in chord_tones)
            if not avoid and gap >= 2.5 and not 5.5 <= gap < 6.5:
                break
            nxt = target_idx + sign
            if not 0 <= nxt < len(ladder):
                break
            target_idx = nxt
            target = float(ladder[target_idx])

        if tessitura is not None:
            lo, hi = tessitura
            while target > hi and target - 12 > lo:
                target -= 12
            while target < lo and target + 12 < hi:
                target += 12
        offsets.append(target - float(midi))

    mean = sum(offsets) / len(offsets) if offsets else 0.0
    return HarmonyVoice(name="%s_%s" % (interval, "up" if direction >= 0 else "down"),
                        offsets=tuple(round(o, 3) for o in offsets),
                        mean_interval=round(mean, 3))


def singable(midi: float, voice_low: float, voice_high: float,
             margin: float = 2.0) -> bool:
    """Whether a pitch sits within a voice's usable range.

    Generated stacks that sit outside the singer's range are the fastest
    way to make a track sound synthetic, because a real person could not
    have produced them.
    """
    return (voice_low - margin) <= midi <= (voice_high + margin)
