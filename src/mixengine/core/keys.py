"""
Musical key representation, parsing, and harmonic compatibility.

Handles the `"eb_minor"` format used in the beats collection, converts to and
from Camelot notation, and answers the question the matcher actually needs:
"how well do these two keys work together, and what pitch shift gets them
there?"

Design note -- the "family root" concept
----------------------------------------
Every key belongs to a relative major/minor pair sharing the same notes
(A minor and C major are the same seven pitches). Normalising both keys to
the major of their pair ("family root") before comparing means a vocal in
E minor over a beat in G major correctly scores as a perfect match rather
than as a 3-semitone clash.

    minor -> family root = (pc + 3) % 12
    major -> family root = pc
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..config import NOTE_NAMES

# ─────────────────────────────────────────────────────────────────────────────
# Pitch-class parsing
# ─────────────────────────────────────────────────────────────────────────────

_PC: Dict[str, int] = {
    "c": 0, "b#": 0,
    "c#": 1, "db": 1,
    "d": 2,
    "d#": 3, "eb": 3,
    "e": 4, "fb": 4,
    "f": 5, "e#": 5,
    "f#": 6, "gb": 6,
    "g": 7,
    "g#": 8, "ab": 8,
    "a": 9,
    "a#": 10, "bb": 10,
    "b": 11, "cb": 11,
}

_MINOR_WORDS = {"min", "minor", "m", "moll", "aeolian"}
_MAJOR_WORDS = {"maj", "major", "m", "dur", "ionian"}


@dataclass(frozen=True)
class Key:
    """A musical key: pitch class 0-11 (0=C) plus mode."""
    pc: int
    mode: str            # "major" | "minor"

    def __post_init__(self) -> None:
        if not 0 <= self.pc <= 11:
            raise ValueError(f"pitch class out of range: {self.pc}")
        if self.mode not in ("major", "minor"):
            raise ValueError(f"unknown mode: {self.mode}")

    # -- Representations ---------------------------------------------------

    @property
    def name(self) -> str:
        return f"{NOTE_NAMES[self.pc]} {self.mode.capitalize()}"

    @property
    def slug(self) -> str:
        """Round-trips with the collection's `key` field, e.g. 'd#_minor'."""
        return f"{NOTE_NAMES[self.pc].lower().replace('#', '#')}_{self.mode}"

    @property
    def family_root(self) -> int:
        """The major key of this key's relative major/minor pair."""
        return (self.pc + 3) % 12 if self.mode == "minor" else self.pc

    @property
    def camelot(self) -> str:
        """Camelot wheel code, e.g. '2A' for Eb minor.

        A lookup table rather than modular arithmetic: the anchor offsets
        are easy to get subtly wrong and the table is trivially auditable.
        """
        return _CAMELOT_TABLE[(self.pc, self.mode)]

    @property
    def scale_pcs(self) -> Tuple[int, ...]:
        """The seven pitch classes of this key's diatonic scale."""
        steps = (0, 2, 4, 5, 7, 9, 11) if self.mode == "major" else (0, 2, 3, 5, 7, 8, 10)
        return tuple((self.pc + s) % 12 for s in steps)

    @property
    def tonic_triad(self) -> Tuple[int, int, int]:
        third = 4 if self.mode == "major" else 3
        return (self.pc, (self.pc + third) % 12, (self.pc + 7) % 12)

    def transposed(self, semitones: int) -> "Key":
        return Key((self.pc + semitones) % 12, self.mode)

    def to_dict(self) -> dict:
        return {"pc": self.pc, "mode": self.mode,
                "name": self.name, "camelot": self.camelot}

    @staticmethod
    def from_dict(d: Optional[dict]) -> Optional["Key"]:
        if not d or d.get("pc") is None:
            return None
        return Key(int(d["pc"]), str(d.get("mode", "major")))

    def __str__(self) -> str:
        return self.name


# Explicit Camelot table -- unambiguous and trivially auditable.
_CAMELOT_TABLE: Dict[Tuple[int, str], str] = {
    (8, "minor"): "1A",  (11, "major"): "1B",   # Ab minor  / B major
    (3, "minor"): "2A",  (6, "major"): "2B",    # Eb minor  / F# major
    (10, "minor"): "3A", (1, "major"): "3B",    # Bb minor  / Db major
    (5, "minor"): "4A",  (8, "major"): "4B",    # F minor   / Ab major
    (0, "minor"): "5A",  (3, "major"): "5B",    # C minor   / Eb major
    (7, "minor"): "6A",  (10, "major"): "6B",   # G minor   / Bb major
    (2, "minor"): "7A",  (5, "major"): "7B",    # D minor   / F major
    (9, "minor"): "8A",  (0, "major"): "8B",    # A minor   / C major
    (4, "minor"): "9A",  (7, "major"): "9B",    # E minor   / G major
    (11, "minor"): "10A", (2, "major"): "10B",  # B minor   / D major
    (6, "minor"): "11A", (9, "major"): "11B",   # F# minor  / A major
    (1, "minor"): "12A", (4, "major"): "12B",   # Db minor  / E major
}

_CAMELOT_REVERSE: Dict[str, Key] = {
    v: Key(k[0], k[1]) for k, v in _CAMELOT_TABLE.items()
}


def parse_key(value) -> Optional[Key]:
    """Parse a key from the many shapes it arrives in.

    Accepts the collection's `"eb_minor"` slug, display names like
    `"F# Minor"`, Camelot codes like `"2A"`, and already-parsed dicts.
    Returns None for anything unrecognised rather than raising -- bad tags
    are common and should degrade to "unknown", not crash the pipeline.

    >>> parse_key("eb_minor").name
    'D# Minor'
    >>> parse_key("eb_minor").camelot
    '2A'
    >>> parse_key("F# Major").pc
    6
    """
    if value is None:
        return None
    if isinstance(value, Key):
        return value
    if isinstance(value, dict):
        return Key.from_dict(value)

    s = str(value).strip().lower()
    if not s:
        return None

    # Camelot code?
    up = s.upper().replace(" ", "")
    if up in _CAMELOT_REVERSE:
        return _CAMELOT_REVERSE[up]

    # Normalise separators: "eb_minor", "eb minor", "eb-minor", "ebm"
    s = s.replace("-", " ").replace("_", " ").replace("/", " ")
    s = s.replace("♯", "#").replace("♭", "b")
    parts = [p for p in s.split() if p]

    if not parts:
        return None

    if len(parts) == 1:
        token = parts[0]
        # "ebm" / "f#min" / "amaj" -- split note from mode
        for note_len in (2, 1):
            if len(token) > note_len and token[:note_len] in _PC:
                cand_note, cand_mode = token[:note_len], token[note_len:]
                if cand_mode in _MINOR_WORDS or cand_mode.startswith("min"):
                    return Key(_PC[cand_note], "minor")
                if cand_mode in _MAJOR_WORDS or cand_mode.startswith("maj"):
                    return Key(_PC[cand_note], "major")
        # Bare note name -> assume major
        if token in _PC:
            return Key(_PC[token], "major")
        return None

    note, mode_word = parts[0], parts[1]
    if note not in _PC:
        return None
    mode = "minor" if (mode_word in _MINOR_WORDS or mode_word.startswith("min")) else "major"
    return Key(_PC[note], mode)


# ─────────────────────────────────────────────────────────────────────────────
# Compatibility
# ─────────────────────────────────────────────────────────────────────────────

def circle_distance(pc_a: int, pc_b: int) -> int:
    """Steps between two pitch classes on the circle of fifths (0-6).

    Adjacent on the circle (a perfect fifth apart) = 1 step. This is the
    right distance metric for harmonic compatibility; chromatic distance
    is not (C and G are 7 semitones apart but maximally compatible).
    """
    # Multiplying by 7 mod 12 maps chromatic position to circle-of-fifths
    # position, because 7 semitones is one fifth.
    a = (pc_a * 7) % 12
    b = (pc_b * 7) % 12
    d = abs(a - b) % 12
    return min(d, 12 - d)


def camelot_neighbours(key: Key, include_self: bool = True) -> List[str]:
    """The classic harmonic-mixing compatible set.

    Same code, the relative major/minor (same number, other letter), and
    +/-1 on the wheel. This is what DJs use and it holds up well as a
    first-pass filter.
    """
    code = key.camelot
    number = int(code[:-1])
    letter = code[-1]
    other = "B" if letter == "A" else "A"

    out: List[str] = []
    if include_self:
        out.append(code)
    out.append(f"{number}{other}")
    out.append(f"{(number % 12) + 1}{letter}")
    out.append(f"{((number - 2) % 12) + 1}{letter}")
    return out


def compatible_camelot_set(key: Key, max_shift: int = 0) -> List[str]:
    """Camelot codes reachable from `key` given up to `max_shift` semitones.

    With max_shift=0 this returns 4 codes (~17% of a uniformly distributed
    catalog). With max_shift=1 it opens up to roughly half the catalog,
    which is the main lever the relaxation ladder pulls.
    """
    codes: List[str] = []
    for s in range(-abs(max_shift), abs(max_shift) + 1):
        codes.extend(camelot_neighbours(key.transposed(s)))
    return list(dict.fromkeys(codes))        # ordered de-duplication


def best_shift(vocal: Key, beat: Key, max_shift: int = 2) -> Tuple[int, float]:
    """Find the pitch shift (applied to the *beat*) that best aligns the keys.

    Returns `(semitones, quality)` where quality is 0-1. Searching and
    scoring is the correct approach; clamping a large required shift to the
    maximum is strictly worse than not shifting at all, because it lands on
    a key matching neither source.

    >>> best_shift(parse_key("e_minor"), parse_key("g_major"))[0]
    0
    """
    best_s, best_q = 0, -1.0
    for s in range(-abs(max_shift), abs(max_shift) + 1):
        shifted = beat.transposed(s)
        # Circle-of-fifths distance between family roots: 0 = same key family.
        d = circle_distance(vocal.family_root, shifted.family_root)
        harmonic = 1.0 - (d / 6.0)
        # Mode agreement is a secondary bonus -- a minor vocal over a major
        # beat in the same family still works, but matching modes is safer.
        mode_bonus = 0.08 if vocal.mode == shifted.mode else 0.0
        # Penalise the shift itself: every semitone costs audible quality.
        cost = 0.11 * abs(s)
        q = harmonic + mode_bonus - cost
        if q > best_q:
            best_s, best_q = s, q
    return best_s, float(max(0.0, min(1.0, best_q)))


def key_compatibility(vocal: Optional[Key], beat: Optional[Key],
                      max_shift: int = 2) -> Tuple[int, float]:
    """Public entry point used by the matcher. Handles missing keys.

    An unknown key on either side returns a neutral score with no shift --
    the pairing is neither rewarded nor rejected on harmonic grounds, and
    the other scoring terms decide.
    """
    if vocal is None or beat is None:
        return 0, 0.5
    return best_shift(vocal, beat, max_shift=max_shift)


def notes_in_key(pcs, key: Key) -> float:
    """Fraction of the given pitch classes that fall inside `key`'s scale.

    Used both for key detection cross-checks and for the dissonance penalty.
    """
    if pcs is None or len(pcs) == 0:
        return 0.0
    scale = set(key.scale_pcs)
    hits = sum(1 for p in pcs if int(p) % 12 in scale)
    return hits / len(pcs)
