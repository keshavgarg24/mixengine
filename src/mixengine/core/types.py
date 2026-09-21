"""
Value types for the Musical Intermediate Representation.

Every type here is a plain dataclass with no audio dependencies beyond
numpy. They are the shared vocabulary between analysis, planning, and
rendering: analysis produces them, the planner reasons over them, and the
renderer consumes them.

Two conventions hold throughout:

  * Times are in **seconds**, floats, measured from the start of the
    source audio. Sample positions appear only where a stage genuinely
    needs them, and are always named `*_samples`.
  * Every measured quantity carries a `confidence` in [0, 1] wherever the
    measurement can fail. Downstream stages branch on confidence rather
    than assuming a value is trustworthy, which is what lets the engine
    degrade honestly instead of confidently producing nonsense.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# Anything the numeric code accepts as "a list of floats": a real list, a
# tuple, or a numpy array. Annotating these parameters as `FloatSeq`
# is what mypy wanted but not what the callers pass -- every grid, onset
# list and downbeat list in the engine is an ndarray by the time it reaches
# a function, and `ndarray` is not a `Sequence`. The alias states the
# actual contract rather than a narrower one the code has never honoured.
FloatSeq = Union[Sequence[float], np.ndarray]

# ─────────────────────────────────────────────────────────────────────────────
# Enumerated vocabularies
#
# Plain string constants rather than Enum: these round-trip through JSON
# without custom encoders, and the DNA documents are meant to be readable
# by hand and by other services.
# ─────────────────────────────────────────────────────────────────────────────

# Harmonic function of a chord within its key.
FUNC_TONIC = "T"
FUNC_SUBDOMINANT = "S"
FUNC_DOMINANT = "D"
FUNC_APPLIED = "A"           # secondary dominant / applied chord
FUNC_CHROMATIC = "X"         # borrowed or non-functional

CADENCE_AUTHENTIC = "authentic"     # V -> I
CADENCE_PLAGAL = "plagal"           # IV -> I
CADENCE_DECEPTIVE = "deceptive"     # V -> vi
CADENCE_HALF = "half"               # -> V

# How a note begins and ends. Drives whether the tuner is allowed to touch it.
ATTACK_CLEAN = "clean"
ATTACK_SCOOP = "scoop"              # approached from below
ATTACK_FALL_IN = "fall_in"          # approached from above
ATTACK_SLIDE = "slide"              # legato from the previous note

RELEASE_CLEAN = "clean"
RELEASE_FALL = "fall"
RELEASE_RISE = "rise"
RELEASE_SLIDE = "slide"

# Phoneme classes we care about for mixing decisions.
PHON_VOWEL = "vowel"
PHON_SIBILANT = "sibilant"          # s, sh, z, zh -- de-esser targets
PHON_PLOSIVE = "plosive"            # p, b, t, d, k, g -- de-plosive targets
PHON_FRICATIVE = "fricative"
PHON_NASAL = "nasal"
PHON_OTHER = "other"

SECTION_LABELS = ("intro", "verse", "prehook", "chorus", "hook", "bridge",
                  "break", "inst", "solo", "outro", "section")


def _round_list(xs: FloatSeq, nd: int = 4) -> List[float]:
    return [round(float(x), nd) for x in xs]


# ═════════════════════════════════════════════════════════════════════════════
# TIME
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class TempoCurve:
    """Tempo as a function of time, not a scalar.

    A human take has no single tempo and neither does a live-played beat.
    Carrying the curve is what makes variable-rate alignment possible: a
    global stretch ratio can only ever be right on average, and "right on
    average" is exactly what a listener hears as drift.

    `times` and `bpm` are parallel arrays. A constant-tempo source is
    represented by a two-point curve, so callers never need a special case.
    """
    times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    bpm: np.ndarray = field(default_factory=lambda: np.zeros(0))
    confidence: float = 0.0
    source: str = "none"             # detected | tag | user | click | none

    @staticmethod
    def constant(bpm: float, duration_s: float = 600.0,
                 confidence: float = 1.0, source: str = "user") -> "TempoCurve":
        return TempoCurve(times=np.array([0.0, float(duration_s)]),
                          bpm=np.array([float(bpm), float(bpm)]),
                          confidence=confidence, source=source)

    @staticmethod
    def from_beats(beat_times: FloatSeq,
                   confidence: float = 0.5,
                   source: str = "detected") -> "TempoCurve":
        """Derive an instantaneous tempo curve from beat positions."""
        b = np.asarray(beat_times, dtype=np.float64)
        if b.size < 2:
            return TempoCurve(confidence=0.0, source=source)
        iois = np.diff(b)
        # Guard against zero/negative intervals from bad trackers.
        iois = np.where(iois > 1e-6, iois, np.nan)
        inst = 60.0 / iois
        good = np.isfinite(inst)
        if not np.any(good):
            return TempoCurve(confidence=0.0, source=source)
        # Sample the curve at interval midpoints -- that is where each
        # instantaneous tempo estimate actually applies.
        mids = (b[:-1] + b[1:]) / 2.0
        return TempoCurve(times=mids[good], bpm=inst[good],
                          confidence=confidence, source=source)

    @property
    def is_known(self) -> bool:
        return self.bpm.size > 0 and self.confidence > 0.0

    @property
    def mean_bpm(self) -> float:
        if self.bpm.size == 0:
            return 0.0
        return float(np.median(self.bpm))

    @property
    def stability(self) -> float:
        """1.0 = metronomic, < 0.8 = live or drifting.

        Expressed as 1 - coefficient of variation, scaled so that the
        useful discrimination happens in the top of the range: programmed
        material clusters above 0.9 and human playing below it.
        """
        if self.bpm.size < 3:
            return 0.0
        m = float(np.mean(self.bpm))
        if m <= 0:
            return 0.0
        cv = float(np.std(self.bpm)) / m
        return float(np.clip(1.0 - cv * 4.0, 0.0, 1.0))

    def at(self, t: float) -> float:
        """Tempo at time `t`, linearly interpolated and edge-held."""
        if self.bpm.size == 0:
            return 0.0
        if self.bpm.size == 1:
            return float(self.bpm[0])
        return float(np.interp(float(t), self.times, self.bpm))

    def to_dict(self) -> dict:
        return {"times": _round_list(self.times, 4),
                "bpm": _round_list(self.bpm, 3),
                "mean_bpm": round(self.mean_bpm, 2),
                "stability": round(self.stability, 4),
                "confidence": round(float(self.confidence), 4),
                "source": self.source}

    @staticmethod
    def from_dict(d: Optional[dict]) -> "TempoCurve":
        if not d:
            return TempoCurve()
        return TempoCurve(times=np.asarray(d.get("times") or [], dtype=np.float64),
                          bpm=np.asarray(d.get("bpm") or [], dtype=np.float64),
                          confidence=float(d.get("confidence") or 0.0),
                          source=str(d.get("source") or "none"))


@dataclass
class Meter:
    """Time signature, valid from `start` until the next Meter."""
    numerator: int = 4
    denominator: int = 4
    start: float = 0.0

    @property
    def beats_per_bar(self) -> int:
        return int(self.numerator)

    @property
    def is_compound(self) -> bool:
        """6/8, 9/8, 12/8 -- felt in dotted beats, which changes subdivision."""
        return self.denominator == 8 and self.numerator in (6, 9, 12)

    @property
    def natural_subdivision(self) -> int:
        """Grid resolution this meter is naturally quantised against."""
        return 12 if self.is_compound else 16

    def to_dict(self) -> dict:
        return {"numerator": self.numerator, "denominator": self.denominator,
                "start": round(self.start, 4)}


@dataclass
class Beat:
    """One beat position on the grid."""
    time: float
    index: int = 0
    bar: int = 0
    position_in_bar: int = 1         # 1-based; 1 == downbeat
    strength: float = 1.0            # tracker activation / salience
    is_downbeat: bool = False

    def to_dict(self) -> dict:
        return {"time": round(self.time, 4), "index": self.index,
                "bar": self.bar, "position_in_bar": self.position_in_bar,
                "strength": round(float(self.strength), 4),
                "is_downbeat": self.is_downbeat}


@dataclass
class GrooveTemplate:
    """The systematic microtiming and accent signature of a performance.

    This is the single most important object in the IR, and it exists
    because of a result that is easy to get backwards.

    The intuition most automated systems encode is "human timing is
    irregular, so add irregularity to sound human." The empirical
    literature does not support it. When expert performances had their
    microtiming deviations scaled to different magnitudes, groove ratings
    were high at or below the originally performed magnitude and fell when
    deviations were exaggerated -- and fully quantised versions rated as
    highly as the original performance. Across commercial tracks, perceived
    groove correlated with beat salience and event density but not with
    either microtiming measure. Injecting random jitter therefore makes a
    render worse, not more human.

    What does carry feel is *systematic, directional* offset: a snare
    consistently behind the beat, a hat consistently ahead of it, a
    consistent swing ratio. That is a property of the track, it repeats
    bar after bar, and it can be measured and transferred.

    So the engine never quantises a vocal to a mathematical grid and never
    adds jitter. It measures the beat's own groove and aligns the vocal to
    *that*. `offsets_ms` is the measured deviation of each subdivision slot
    from its mathematically exact position, averaged over the track;
    `consistency` says how reliably that pattern repeats, and therefore how
    much it should be trusted.

    Slot indexing: slot `i` of `subdivision` slots per bar corresponds to
    the exact position `bar_start + i * bar_duration / subdivision`.
    """
    subdivision: int = 16                 # slots per bar (16 = 16ths, 12 = compound)
    offsets_ms: np.ndarray = field(default_factory=lambda: np.zeros(0))
    velocities: np.ndarray = field(default_factory=lambda: np.zeros(0))
    swing_ratio: float = 0.5              # 0.5 straight, 0.667 triplet swing
    consistency: float = 0.0              # 0-1, how reliably the pattern repeats
    per_element: Dict[str, np.ndarray] = field(default_factory=dict)
    n_bars_observed: int = 0
    source: str = "none"                  # measured | genre_prior | straight | none

    @staticmethod
    def straight(subdivision: int = 16) -> "GrooveTemplate":
        """A perfectly quantised grid. The correct default, not a fallback.

        Programmed trap and drill really are quantised; imposing a swung or
        jittered template on them would be the error.
        """
        n = int(subdivision)
        return GrooveTemplate(subdivision=n, offsets_ms=np.zeros(n),
                              velocities=np.ones(n), swing_ratio=0.5,
                              consistency=1.0, source="straight")

    @property
    def is_meaningful(self) -> bool:
        """Whether this template should influence alignment at all.

        A template measured from too few bars, or one whose pattern does not
        repeat, is noise. Applying it would be exactly the random-jitter
        mistake this class exists to avoid.
        """
        return bool(self.offsets_ms.size > 0
                    and self.n_bars_observed >= 4
                    and self.consistency >= 0.5)

    @property
    def max_offset_ms(self) -> float:
        return float(np.max(np.abs(self.offsets_ms))) if self.offsets_ms.size else 0.0

    def offset_at_slot(self, slot: int) -> float:
        """Microtiming offset in milliseconds for a subdivision slot."""
        if self.offsets_ms.size == 0:
            return 0.0
        return float(self.offsets_ms[int(slot) % self.offsets_ms.size])

    def accent_at_slot(self, slot: int) -> float:
        if self.velocities.size == 0:
            return 1.0
        return float(self.velocities[int(slot) % self.velocities.size])

    def scaled(self, amount: float) -> "GrooveTemplate":
        """Return this template with its deviations scaled.

        Values above 1.0 are deliberately permitted but should be used
        sparingly: exaggerating measured microtiming reduces perceived
        groove rather than increasing it.
        """
        a = float(amount)
        return GrooveTemplate(
            subdivision=self.subdivision,
            offsets_ms=self.offsets_ms * a,
            velocities=self.velocities.copy(),
            swing_ratio=0.5 + (self.swing_ratio - 0.5) * a,
            consistency=self.consistency,
            per_element={k: v * a for k, v in self.per_element.items()},
            n_bars_observed=self.n_bars_observed, source=self.source)

    def to_dict(self) -> dict:
        return {"subdivision": int(self.subdivision),
                "offsets_ms": _round_list(self.offsets_ms, 2),
                "velocities": _round_list(self.velocities, 3),
                "swing_ratio": round(float(self.swing_ratio), 4),
                "consistency": round(float(self.consistency), 4),
                "per_element": {k: _round_list(v, 2)
                                for k, v in self.per_element.items()},
                "n_bars_observed": int(self.n_bars_observed),
                "source": self.source,
                "is_meaningful": self.is_meaningful}

    @staticmethod
    def from_dict(d: Optional[dict]) -> "GrooveTemplate":
        if not d:
            return GrooveTemplate()
        return GrooveTemplate(
            subdivision=int(d.get("subdivision", 16)),
            offsets_ms=np.asarray(d.get("offsets_ms") or [], dtype=np.float64),
            velocities=np.asarray(d.get("velocities") or [], dtype=np.float64),
            swing_ratio=float(d.get("swing_ratio", 0.5)),
            consistency=float(d.get("consistency", 0.0)),
            per_element={k: np.asarray(v, dtype=np.float64)
                         for k, v in (d.get("per_element") or {}).items()},
            n_bars_observed=int(d.get("n_bars_observed", 0)),
            source=str(d.get("source", "none")))


# ═════════════════════════════════════════════════════════════════════════════
# HARMONY
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class KeyRegion:
    """A key that holds over a span of time.

    Songs modulate. A single global key is an approximation that breaks
    exactly where it matters most -- a bridge in the relative major, a
    final-chorus lift -- so the IR carries regions and every harmonic
    question is asked at a specific time.
    """
    start: float
    end: float
    pc: int                          # tonic pitch class, 0 = C
    mode: str = "minor"              # major | minor
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {"start": round(self.start, 4), "end": round(self.end, 4),
                "pc": int(self.pc), "mode": self.mode,
                "confidence": round(float(self.confidence), 4)}

    @staticmethod
    def from_dict(d: dict) -> "KeyRegion":
        return KeyRegion(start=float(d.get("start", 0.0)),
                         end=float(d.get("end", 0.0)),
                         pc=int(d.get("pc", 0)),
                         mode=str(d.get("mode", "minor")),
                         confidence=float(d.get("confidence", 0.0)))


@dataclass
class ChordEvent:
    """One chord, with enough detail to reason about consonance.

    `quality` is the triad type; `extensions` holds added scale degrees as
    semitone offsets from the root (10 = b7, 14 -> 2 for a 9th, and so on,
    stored reduced into 0-11). Keeping extensions separate from quality
    means a tuner can treat a 9th as available without the chord having to
    be spelled as a distinct symbol.
    """
    start: float
    end: float
    root: int                        # pitch class 0-11
    quality: str = "maj"             # maj | min | dim | aug | sus2 | sus4 | pow
    extensions: Tuple[int, ...] = ()     # extra pitch classes, absolute 0-11
    bass: Optional[int] = None       # pitch class if inverted / slash chord
    function: str = FUNC_TONIC
    confidence: float = 0.0

    _TRIADS = {
        "maj": (0, 4, 7),
        "min": (0, 3, 7),
        "dim": (0, 3, 6),
        "aug": (0, 4, 8),
        "sus2": (0, 2, 7),
        "sus4": (0, 5, 7),
        "pow": (0, 7),               # no third: compatible with either mode
    }

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def tones(self) -> frozenset:
        """Pitch classes that are unambiguously consonant over this chord."""
        intervals = self._TRIADS.get(self.quality, self._TRIADS["maj"])
        pcs = {(self.root + i) % 12 for i in intervals}
        pcs |= {int(e) % 12 for e in self.extensions}
        if self.bass is not None:
            pcs.add(int(self.bass) % 12)
        return frozenset(pcs)

    @property
    def has_third(self) -> bool:
        return self.quality not in ("pow", "sus2", "sus4")

    @property
    def third_pc(self) -> Optional[int]:
        if not self.has_third:
            return None
        return (self.root + (4 if self.quality in ("maj", "aug") else 3)) % 12

    def to_dict(self) -> dict:
        return {"start": round(self.start, 4), "end": round(self.end, 4),
                "root": int(self.root), "quality": self.quality,
                "extensions": [int(e) for e in self.extensions],
                "bass": (int(self.bass) if self.bass is not None else None),
                "function": self.function,
                "confidence": round(float(self.confidence), 4)}

    @staticmethod
    def from_dict(d: dict) -> "ChordEvent":
        # The per-bar chord estimator emits `time` (the bar's onset) rather
        # than `start`, and no end at all. Reading only `start` silently
        # collapsed every chord to the span [0, 0], so `chord_at()` answered
        # None for the whole track and the chord-aware tuner quietly
        # degraded to scale-aware -- with nothing in the output saying so.
        start = d.get("start", d.get("time", 0.0))
        return ChordEvent(
            start=float(start), end=float(d.get("end", 0.0)),
            root=int(d.get("root", 0)), quality=str(d.get("quality", "maj")),
            extensions=tuple(int(e) for e in (d.get("extensions") or ())),
            bass=(int(d["bass"]) if d.get("bass") is not None else None),
            function=str(d.get("function", FUNC_TONIC)),
            confidence=float(d.get("confidence", 0.0)))


@dataclass
class Cadence:
    """A harmonic resolution point.

    Cadences are where listeners expect phrases to land, which makes them
    high-salience: a timing or tuning error on a cadence is far more
    audible than the same error mid-phrase.
    """
    time: float
    kind: str = CADENCE_AUTHENTIC
    strength: float = 0.5

    def to_dict(self) -> dict:
        return {"time": round(self.time, 4), "kind": self.kind,
                "strength": round(float(self.strength), 4)}


# ═════════════════════════════════════════════════════════════════════════════
# MELODY / PERFORMANCE
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Note:
    """One sung or rapped note event.

    `midi` is fractional -- the measured pitch, not a quantised one. The
    distance from `round(midi)` is the singer's intonation, and destroying
    it is how automatic tuning starts sounding synthetic, so it is carried
    explicitly rather than rounded away at analysis time.
    """
    start: float
    end: float
    midi: float
    velocity: float = 0.7            # 0-1, from RMS over the note
    vibrato_rate_hz: float = 0.0
    vibrato_depth_cents: float = 0.0
    attack: str = ATTACK_CLEAN
    release: str = RELEASE_CLEAN
    is_melisma: bool = False         # multiple notes on one syllable
    word_index: Optional[int] = None
    confidence: float = 0.5

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def pc(self) -> int:
        return int(round(self.midi)) % 12

    @property
    def cents_dev(self) -> float:
        """Deviation from equal-tempered pitch, in cents (-50..+50)."""
        return float((self.midi - round(self.midi)) * 100.0)

    @property
    def is_transition(self) -> bool:
        """True when the note is an approach or departure gesture.

        Scoops, falls and slides are expression, not error. The tuner must
        leave them alone; correcting them is the single most recognisable
        way an automatic pitch correction gives itself away.
        """
        return (self.attack in (ATTACK_SCOOP, ATTACK_FALL_IN, ATTACK_SLIDE)
                or self.release in (RELEASE_FALL, RELEASE_RISE, RELEASE_SLIDE))

    def to_dict(self) -> dict:
        return {"start": round(self.start, 4), "end": round(self.end, 4),
                "midi": round(float(self.midi), 3), "pc": self.pc,
                "duration": round(self.duration, 4),
                "cents_dev": round(self.cents_dev, 1),
                "velocity": round(float(self.velocity), 3),
                "vibrato_rate_hz": round(float(self.vibrato_rate_hz), 2),
                "vibrato_depth_cents": round(float(self.vibrato_depth_cents), 1),
                "attack": self.attack, "release": self.release,
                "is_melisma": self.is_melisma, "word_index": self.word_index,
                "confidence": round(float(self.confidence), 3)}

    @staticmethod
    def from_dict(d: dict) -> "Note":
        return Note(
            start=float(d.get("start", 0.0)), end=float(d.get("end", 0.0)),
            midi=float(d.get("midi", 0.0)),
            velocity=float(d.get("velocity", 0.7)),
            vibrato_rate_hz=float(d.get("vibrato_rate_hz", 0.0)),
            vibrato_depth_cents=float(d.get("vibrato_depth_cents", 0.0)),
            attack=str(d.get("attack", ATTACK_CLEAN)),
            release=str(d.get("release", RELEASE_CLEAN)),
            is_melisma=bool(d.get("is_melisma", False)),
            word_index=d.get("word_index"),
            confidence=float(d.get("confidence", 0.5)))


@dataclass
class ExpressionProfile:
    """Aggregate description of how the singer performs, not what they sing."""
    mean_vibrato_rate_hz: float = 0.0
    mean_vibrato_depth_cents: float = 0.0
    scoop_fraction: float = 0.0          # fraction of notes approached from below
    melisma_fraction: float = 0.0
    mean_intonation_cents: float = 0.0   # absolute deviation from ET
    intonation_spread_cents: float = 0.0
    dynamic_range_db: float = 0.0
    timing_bias_ms: float = 0.0          # negative = ahead of the beat
    timing_spread_ms: float = 0.0

    def to_dict(self) -> dict:
        return {k: round(float(v), 3) for k, v in asdict(self).items()}


# ═════════════════════════════════════════════════════════════════════════════
# LANGUAGE
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Word:
    text: str
    start: float
    end: float
    stress: float = 0.5              # 0-1, lexical/metrical emphasis
    confidence: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        return {"text": self.text, "start": round(self.start, 4),
                "end": round(self.end, 4), "stress": round(float(self.stress), 3),
                "confidence": round(float(self.confidence), 3)}

    @staticmethod
    def from_dict(d: dict) -> "Word":
        return Word(text=str(d.get("text", "")),
                    start=float(d.get("start", 0.0)),
                    end=float(d.get("end", 0.0)),
                    stress=float(d.get("stress", 0.5)),
                    confidence=float(d.get("confidence", 0.0)))


@dataclass
class Phoneme:
    """A phone with its class.

    The class is what the mixer needs: sibilants tell the de-esser exactly
    which milliseconds to act on, plosives tell the de-plosive stage the
    same. Acting on measured phone positions rather than a broadband
    threshold is the difference between a de-esser that moves on problem
    words and one that lisps the whole vocal.
    """
    symbol: str
    start: float
    end: float
    kind: str = PHON_OTHER
    word_index: Optional[int] = None

    def to_dict(self) -> dict:
        return {"symbol": self.symbol, "start": round(self.start, 4),
                "end": round(self.end, 4), "kind": self.kind,
                "word_index": self.word_index}


@dataclass
class LineGroup:
    """A set of lyric lines that repeat -- the raw material for hook finding."""
    line_indices: Tuple[int, ...] = ()
    text: str = ""
    similarity: float = 0.0
    mean_energy: float = 0.0

    def to_dict(self) -> dict:
        return {"line_indices": list(self.line_indices), "text": self.text,
                "similarity": round(float(self.similarity), 4),
                "mean_energy": round(float(self.mean_energy), 4)}


@dataclass
class RepetitionMap:
    """Which lyric lines repeat, and how strongly.

    This is how the hook is found. The alternative currently in the engine
    -- assume the back half of the take is the hook -- is a guess that is
    wrong for any song with a hook at the top.
    """
    lines: List[Tuple[float, float]] = field(default_factory=list)   # line spans
    line_texts: List[str] = field(default_factory=list)
    groups: List[LineGroup] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"lines": [[round(a, 4), round(b, 4)] for a, b in self.lines],
                "line_texts": list(self.line_texts),
                "groups": [g.to_dict() for g in self.groups]}


# ═════════════════════════════════════════════════════════════════════════════
# STRUCTURE
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Section:
    start: float
    end: float
    label: str = "section"
    energy: float = 0.5              # 0-1, normalised within the track
    start_bar: int = 0
    end_bar: int = 0
    confidence: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def n_bars(self) -> int:
        return max(0, self.end_bar - self.start_bar)

    def to_dict(self) -> dict:
        return {"start": round(self.start, 4), "end": round(self.end, 4),
                "label": self.label, "energy": round(float(self.energy), 4),
                "start_bar": self.start_bar, "end_bar": self.end_bar,
                "confidence": round(float(self.confidence), 4)}

    @staticmethod
    def from_dict(d: dict) -> "Section":
        return Section(start=float(d.get("start", 0.0)),
                       end=float(d.get("end", 0.0)),
                       label=str(d.get("label", "section")),
                       energy=float(d.get("energy", 0.5)),
                       start_bar=int(d.get("start_bar", 0)),
                       end_bar=int(d.get("end_bar", 0)),
                       confidence=float(d.get("confidence", 0.0)))


@dataclass
class Phrase:
    """A sung phrase: the unit every downstream stage operates on.

    `has_pickup` matters for placement. A phrase that begins with a pickup
    should be aligned so the *stressed* syllable lands on the downbeat,
    not its first sound -- aligning the pickup itself to the bar line
    pushes the whole phrase a beat late, which is one of the more common
    ways automatic placement sounds wrong.
    """
    start: float
    end: float
    start_sample: int = 0
    end_sample: int = 0
    energy: float = 0.5
    has_pickup: bool = False
    stressed_onset: Optional[float] = None    # where the downbeat should land
    word_indices: Tuple[int, ...] = ()
    note_indices: Tuple[int, ...] = ()

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def anchor(self) -> float:
        """The moment that should coincide with a strong grid position."""
        return self.stressed_onset if self.stressed_onset is not None else self.start

    def to_dict(self) -> dict:
        return {"start": round(self.start, 4), "end": round(self.end, 4),
                "start_sample": int(self.start_sample),
                "end_sample": int(self.end_sample),
                "energy": round(float(self.energy), 4),
                "has_pickup": self.has_pickup,
                "stressed_onset": (round(self.stressed_onset, 4)
                                   if self.stressed_onset is not None else None),
                "word_indices": list(self.word_indices),
                "note_indices": list(self.note_indices)}

    @staticmethod
    def from_dict(d: dict) -> "Phrase":
        return Phrase(start=float(d.get("start", 0.0)),
                      end=float(d.get("end", 0.0)),
                      start_sample=int(d.get("start_sample", 0)),
                      end_sample=int(d.get("end_sample", 0)),
                      energy=float(d.get("energy", 0.5)),
                      has_pickup=bool(d.get("has_pickup", False)),
                      stressed_onset=d.get("stressed_onset"),
                      word_indices=tuple(d.get("word_indices") or ()),
                      note_indices=tuple(d.get("note_indices") or ()))


# ═════════════════════════════════════════════════════════════════════════════
# ACOUSTIC
# ═════════════════════════════════════════════════════════════════════════════

# Octave band centres used for reverberation measurement.
RT60_BANDS_HZ = (125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0)


@dataclass
class SpaceFingerprint:
    """The acoustic space a recording sits in.

    Matching the vocal's space to the beat's is what stops a vocal sounding
    pasted on top of an instrumental. The literature is encouraging here:
    blind estimates of just two quantities -- reverberation time and the
    direct-to-reverberant ratio -- are enough to synthesise an impulse
    response that matches plausibly. Everything else in this struct refines
    that baseline rather than being required for it.
    """
    rt60_bands: Tuple[float, ...] = ()       # parallel to RT60_BANDS_HZ
    rt60_broadband: float = 0.0
    drr_db: float = 0.0                      # direct-to-reverberant ratio
    predelay_ms: float = 0.0
    early_density: float = 0.0               # reflections per second, 0-80 ms
    confidence: float = 0.0

    @property
    def is_dry(self) -> bool:
        return self.rt60_broadband < 0.25

    @property
    def is_usable(self) -> bool:
        """Can a dereverb stage plausibly rescue this into a dry source?"""
        return self.rt60_broadband < 0.6

    def to_dict(self) -> dict:
        return {"rt60_bands": _round_list(self.rt60_bands, 3),
                "rt60_band_freqs": list(RT60_BANDS_HZ),
                "rt60_broadband": round(float(self.rt60_broadband), 3),
                "drr_db": round(float(self.drr_db), 2),
                "predelay_ms": round(float(self.predelay_ms), 2),
                "early_density": round(float(self.early_density), 2),
                "confidence": round(float(self.confidence), 4)}

    @staticmethod
    def from_dict(d: Optional[dict]) -> "SpaceFingerprint":
        if not d:
            return SpaceFingerprint()
        return SpaceFingerprint(
            rt60_bands=tuple(float(x) for x in (d.get("rt60_bands") or ())),
            rt60_broadband=float(d.get("rt60_broadband", 0.0)),
            drr_db=float(d.get("drr_db", 0.0)),
            predelay_ms=float(d.get("predelay_ms", 0.0)),
            early_density=float(d.get("early_density", 0.0)),
            confidence=float(d.get("confidence", 0.0)))


# Band edges used for the spectral budget. Chosen on musical function
# rather than equal spacing: these are the ranges engineers actually
# negotiate between a vocal and a beat.
BUDGET_BANDS: Tuple[Tuple[str, float, float], ...] = (
    ("sub",       20.0,    60.0),    # 808 fundamental; a vocal has nothing here
    ("low",       60.0,   120.0),    # kick body, 808 harmonics
    ("lowmid",   120.0,   350.0),    # vocal weight vs bass -- the contested zone
    ("mid",      350.0,  1500.0),    # vocal body; the vocal should win
    ("presence",1500.0,  5000.0),    # intelligibility; the vocal must win
    ("air",     8000.0, 16000.0),    # hats and vocal air share this
)


@dataclass
class SpectralProfile:
    """Long-term tonal description plus the per-band occupancy figures the
    mixer's frequency budget reasons over."""
    freqs: np.ndarray = field(default_factory=lambda: np.zeros(0))
    mag_db: np.ndarray = field(default_factory=lambda: np.zeros(0))
    band_energy: Dict[str, float] = field(default_factory=dict)
    centroid_hz: float = 0.0
    rolloff_hz: float = 0.0
    flatness: float = 0.0
    bandwidth_hz: float = 0.0

    def to_dict(self, decimate: int = 1) -> dict:
        step = max(1, int(decimate))
        return {"freqs": _round_list(self.freqs[::step], 1),
                "mag_db": _round_list(self.mag_db[::step], 2),
                "band_energy": {k: round(float(v), 5)
                                for k, v in self.band_energy.items()},
                "centroid_hz": round(float(self.centroid_hz), 1),
                "rolloff_hz": round(float(self.rolloff_hz), 1),
                "flatness": round(float(self.flatness), 6),
                "bandwidth_hz": round(float(self.bandwidth_hz), 1)}


@dataclass
class LoudnessProfile:
    integrated_lufs: float = float("-inf")
    short_term_lufs: np.ndarray = field(default_factory=lambda: np.zeros(0))
    loudness_range_lu: float = 0.0
    true_peak_db: float = float("-inf")
    peak_db: float = float("-inf")
    rms_db: float = float("-inf")

    @property
    def crest_factor_db(self) -> float:
        if not (np.isfinite(self.peak_db) and np.isfinite(self.rms_db)):
            return 0.0
        return float(self.peak_db - self.rms_db)

    @property
    def is_mastered(self) -> bool:
        """Heuristic: already limited and loud, so do not re-compress it.

        `bool(...)` is not redundant -- numpy comparisons return `np.bool_`,
        which is not JSON serialisable and would break IR persistence.
        """
        return bool(np.isfinite(self.integrated_lufs)
                    and self.integrated_lufs > -11.0
                    and self.loudness_range_lu < 6.0)

    def to_dict(self) -> dict:
        def f(v):
            v = float(v)
            return round(v, 2) if np.isfinite(v) else None
        return {"integrated_lufs": f(self.integrated_lufs),
                "loudness_range_lu": round(float(self.loudness_range_lu), 2),
                "true_peak_db": f(self.true_peak_db),
                "peak_db": f(self.peak_db), "rms_db": f(self.rms_db),
                "crest_factor_db": round(self.crest_factor_db, 2),
                "is_mastered": self.is_mastered}


@dataclass
class VoiceProfile:
    """What kind of instrument this voice is.

    Drives the high-pass corner, the de-esser sensitivity, the frequency
    budget in the low-mids, and which harmony intervals are singable for
    generated stacks.
    """
    f0_low_hz: float = 0.0           # 5th percentile of voiced f0
    f0_high_hz: float = 0.0          # 95th percentile
    f0_median_hz: float = 0.0
    voice_type: str = ""             # male_bass .. female_soprano
    tessitura_low_midi: float = 0.0  # comfortable range, 25th-75th percentile
    tessitura_high_midi: float = 0.0
    sibilance_ratio: float = 0.0
    resonances: Tuple[Tuple[float, float], ...] = ()   # (hz, excess_db)
    noise_floor_db: float = -120.0

    @property
    def range_semitones(self) -> float:
        if self.f0_low_hz <= 0 or self.f0_high_hz <= 0:
            return 0.0
        return float(12.0 * np.log2(self.f0_high_hz / self.f0_low_hz))

    def to_dict(self) -> dict:
        return {"f0_low_hz": round(float(self.f0_low_hz), 1),
                "f0_high_hz": round(float(self.f0_high_hz), 1),
                "f0_median_hz": round(float(self.f0_median_hz), 1),
                "voice_type": self.voice_type,
                "tessitura_low_midi": round(float(self.tessitura_low_midi), 2),
                "tessitura_high_midi": round(float(self.tessitura_high_midi), 2),
                "range_semitones": round(self.range_semitones, 2),
                "sibilance_ratio": round(float(self.sibilance_ratio), 5),
                "resonances": [[round(float(f), 1), round(float(e), 2)]
                               for f, e in self.resonances],
                "noise_floor_db": round(float(self.noise_floor_db), 2)}


# ═════════════════════════════════════════════════════════════════════════════
# PROVENANCE
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ProcessingStep:
    """One transform applied to the audio, recorded so a render is reproducible
    and so the critic can attribute a defect to the stage that caused it."""
    stage: str
    method: str
    params: Dict = field(default_factory=dict)
    metrics_before: Dict = field(default_factory=dict)
    metrics_after: Dict = field(default_factory=dict)
    seconds: float = 0.0
    note: str = ""

    def to_dict(self) -> dict:
        return {"stage": self.stage, "method": self.method, "params": self.params,
                "metrics_before": self.metrics_before,
                "metrics_after": self.metrics_after,
                "seconds": round(float(self.seconds), 3), "note": self.note}
