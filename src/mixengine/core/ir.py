"""
The Musical Intermediate Representation.

One structure describes a vocal take and an instrumental alike. Analysis
fills it, the planner reasons over it, the renderer executes against it.

Why this exists
---------------
The previous design fused analysis and rendering into a single function
and passed flat dictionaries of numbers between stages. That makes three
things impossible: there is nowhere for a musical decision to live, nothing
downstream can ask a musical question ("what chord is under this note?"),
and a failure cannot be attributed to the stage that caused it.

The IR separates the three things a production system actually does:

    UNDERSTAND  ->  DECIDE  ->  EXECUTE
    (analysis)      (plan)      (render)

Everything here belongs to UNDERSTAND. It is inert data plus lookups, with
no audio processing and no decisions. Every field is optional, because
sources differ -- a drum loop has no notes, an a cappella has no chords --
and every stage is expected to check `confidence` rather than assume.

Design rules
------------
  * Times in seconds from the start of the source.
  * Lists are kept sorted by start time; the lookups below rely on it.
  * Nothing raises for missing data. Absent information returns None or a
    documented neutral value, because an honest "unknown" routes the
    pipeline correctly while a fabricated value corrupts everything
    downstream of it.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .types import (
    Beat, Cadence, ChordEvent, ExpressionProfile, GrooveTemplate, KeyRegion,
    LoudnessProfile, Meter, Note, Phoneme, Phrase, ProcessingStep,
    RepetitionMap, Section, SpaceFingerprint, SpectralProfile, TempoCurve,
    VoiceProfile, Word,
    PHON_PLOSIVE, PHON_SIBILANT,
)

IR_VERSION = "2.0.0"

# What kind of source an IR describes. Routing decisions branch on this.
ROLE_VOCAL = "vocal"
ROLE_BEAT = "beat"
ROLE_MIX = "mix"


def _starts(items: Sequence) -> List[float]:
    return [float(getattr(x, "start", getattr(x, "time", 0.0))) for x in items]


@dataclass
class MusicalIR:
    """Everything the engine knows about one piece of audio."""

    # ── Identity ──────────────────────────────────────────────────────────
    source_id: str = ""
    source_path: str = ""
    role: str = ROLE_VOCAL
    version: str = IR_VERSION
    duration_s: float = 0.0
    sample_rate: int = 48000

    # ── Time ──────────────────────────────────────────────────────────────
    tempo: TempoCurve = field(default_factory=TempoCurve)
    beats: List[Beat] = field(default_factory=list)
    meters: List[Meter] = field(default_factory=lambda: [Meter()])
    groove: GrooveTemplate = field(default_factory=GrooveTemplate)

    # ── Harmony ───────────────────────────────────────────────────────────
    key_regions: List[KeyRegion] = field(default_factory=list)
    chords: List[ChordEvent] = field(default_factory=list)
    cadences: List[Cadence] = field(default_factory=list)
    is_atonal: bool = False

    # ── Melody and performance ────────────────────────────────────────────
    notes: List[Note] = field(default_factory=list)
    f0_times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    f0_hz: np.ndarray = field(default_factory=lambda: np.zeros(0))
    f0_voiced: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    onsets: np.ndarray = field(default_factory=lambda: np.zeros(0))
    expression: ExpressionProfile = field(default_factory=ExpressionProfile)
    performance_type: str = ""            # rap | melodic_rap | sung | spoken

    # ── Language ──────────────────────────────────────────────────────────
    words: List[Word] = field(default_factory=list)
    phonemes: List[Phoneme] = field(default_factory=list)
    repetition: RepetitionMap = field(default_factory=RepetitionMap)

    # ── Structure ─────────────────────────────────────────────────────────
    sections: List[Section] = field(default_factory=list)
    phrases: List[Phrase] = field(default_factory=list)
    energy_times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    energy: np.ndarray = field(default_factory=lambda: np.zeros(0))

    # ── Acoustic ──────────────────────────────────────────────────────────
    stem_paths: Dict[str, str] = field(default_factory=dict)
    space: SpaceFingerprint = field(default_factory=SpaceFingerprint)
    spectrum: SpectralProfile = field(default_factory=SpectralProfile)
    loudness: LoudnessProfile = field(default_factory=LoudnessProfile)
    voice: VoiceProfile = field(default_factory=VoiceProfile)
    timbre_embedding: np.ndarray = field(default_factory=lambda: np.zeros(0))

    # ── Provenance ────────────────────────────────────────────────────────
    genre: Optional[str] = None
    mood: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    processing_log: List[ProcessingStep] = field(default_factory=list)
    confidence: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    status: str = "ok"
    error: str = ""

    # ══════════════════════════════════════════════════════════════════════
    # Derived summaries
    # ══════════════════════════════════════════════════════════════════════

    @property
    def bpm(self) -> float:
        """Representative tempo. Zero means genuinely unknown.

        Callers must treat zero as "no grid available" and route to
        phrase-anchored placement rather than substituting a guess.
        """
        return self.tempo.mean_bpm

    @property
    def has_grid(self) -> bool:
        return len(self.beats) >= 4 and self.tempo.is_known

    @property
    def downbeats(self) -> List[float]:
        return [b.time for b in self.beats if b.is_downbeat]

    @property
    def beat_times(self) -> np.ndarray:
        return np.asarray([b.time for b in self.beats], dtype=np.float64)

    @property
    def beats_per_bar(self) -> int:
        return self.meters[0].beats_per_bar if self.meters else 4

    @property
    def bar_duration_s(self) -> float:
        bpm = self.bpm
        return (60.0 / bpm) * self.beats_per_bar if bpm > 0 else 0.0

    @property
    def n_bars(self) -> int:
        bd = self.bar_duration_s
        return int(self.duration_s / bd) if bd > 0 else 0

    @property
    def key(self) -> Optional[KeyRegion]:
        """The dominant key region by total duration, or None."""
        if not self.key_regions:
            return None
        return max(self.key_regions, key=lambda k: (k.end - k.start) * max(k.confidence, 0.01))

    @property
    def syllable_rate(self) -> float:
        """Onsets per second over the whole take."""
        if self.duration_s <= 0:
            return 0.0
        return float(len(self.onsets)) / self.duration_s

    @property
    def voiced_fraction(self) -> float:
        if self.f0_voiced.size == 0:
            return 0.0
        return float(np.mean(self.f0_voiced))

    def conf(self, field_name: str, default: float = 0.0) -> float:
        return float(self.confidence.get(field_name, default))

    # ══════════════════════════════════════════════════════════════════════
    # Temporal lookups
    # ══════════════════════════════════════════════════════════════════════

    def meter_at(self, t: float) -> Meter:
        out = self.meters[0] if self.meters else Meter()
        for m in self.meters:
            if m.start <= t:
                out = m
            else:
                break
        return out

    def beat_at(self, t: float) -> Optional[Beat]:
        """The beat at or immediately before `t`."""
        if not self.beats:
            return None
        i = bisect.bisect_right(_starts(self.beats), float(t)) - 1
        return self.beats[i] if i >= 0 else None

    def nearest_beat(self, t: float) -> Optional[Beat]:
        if not self.beats:
            return None
        times = self.beat_times
        return self.beats[int(np.argmin(np.abs(times - float(t))))]

    def nearest_downbeat(self, t: float) -> Optional[float]:
        db = self.downbeats
        if not db:
            return None
        arr = np.asarray(db, dtype=np.float64)
        return float(arr[int(np.argmin(np.abs(arr - float(t))))])

    def bar_of(self, t: float) -> int:
        b = self.beat_at(t)
        if b is not None:
            return b.bar
        bd = self.bar_duration_s
        return int(t / bd) if bd > 0 else 0

    def bar_start(self, bar: int) -> Optional[float]:
        for b in self.beats:
            if b.bar == bar and b.is_downbeat:
                return b.time
        bd = self.bar_duration_s
        if bd <= 0:
            return None
        origin = self.downbeats[0] if self.downbeats else 0.0
        return origin + bar * bd

    def subdivision_grid(self, subdivision: int = 16,
                         apply_groove: bool = True) -> np.ndarray:
        """Grid positions in seconds at `subdivision` slots per bar.

        With `apply_groove` the returned positions carry the track's own
        measured microtiming, so aligning to this grid places material in
        the pocket rather than on a mathematically exact but rhythmically
        dead grid. That distinction is the whole point of carrying a groove
        template, so it defaults to on.
        """
        db = self.downbeats
        bd = self.bar_duration_s
        if len(db) < 2 or bd <= 0:
            # No bar information: fall back to an even division of the beats.
            times = self.beat_times
            if times.size < 2:
                return np.zeros(0)
            step = float(np.median(np.diff(times)))
            per_beat = max(1, int(round(subdivision / max(self.beats_per_bar, 1))))
            fine = []
            for i in range(times.size - 1):
                for k in range(per_beat):
                    fine.append(times[i] + step * k / per_beat)
            fine.append(float(times[-1]))
            return np.asarray(fine, dtype=np.float64)

        use_groove = apply_groove and self.groove.is_meaningful
        out: List[float] = []
        for bi in range(len(db) - 1):
            start, nxt = db[bi], db[bi + 1]
            span = nxt - start
            # A downbeat gap wildly off the nominal bar length means a
            # dropped or spurious downbeat; fall back to nominal rather
            # than smearing the whole bar's grid.
            if not (0.5 * bd < span < 2.0 * bd):
                span = bd
            for slot in range(subdivision):
                t = start + span * slot / subdivision
                if use_groove:
                    t += self.groove.offset_at_slot(slot) / 1000.0
                out.append(t)
        out.append(float(db[-1]))
        return np.asarray(out, dtype=np.float64)

    # ══════════════════════════════════════════════════════════════════════
    # Harmonic lookups
    # ══════════════════════════════════════════════════════════════════════

    def key_at(self, t: float) -> Optional[KeyRegion]:
        for k in self.key_regions:
            if k.start <= t < k.end:
                return k
        return self.key

    def chord_at(self, t: float) -> Optional[ChordEvent]:
        if not self.chords:
            return None
        i = bisect.bisect_right([c.start for c in self.chords], float(t)) - 1
        if i < 0:
            return None
        c = self.chords[i]
        return c if t < c.end else None

    def chords_between(self, a: float, b: float) -> List[ChordEvent]:
        return [c for c in self.chords if c.end > a and c.start < b]

    def is_cadence_near(self, t: float, tol: float = 0.25) -> bool:
        return any(abs(c.time - t) <= tol for c in self.cadences)

    # ══════════════════════════════════════════════════════════════════════
    # Structural lookups
    # ══════════════════════════════════════════════════════════════════════

    def section_at(self, t: float) -> Optional[Section]:
        for s in self.sections:
            if s.start <= t < s.end:
                return s
        return None

    def sections_labelled(self, *labels: str) -> List[Section]:
        want = set(labels)
        return [s for s in self.sections if s.label in want]

    def phrase_at(self, t: float) -> Optional[Phrase]:
        for p in self.phrases:
            if p.start <= t < p.end:
                return p
        return None

    def phrase_index_at(self, t: float) -> Optional[int]:
        for i, p in enumerate(self.phrases):
            if p.start <= t < p.end:
                return i
        return None

    def energy_at(self, t: float) -> float:
        if self.energy.size == 0:
            s = self.section_at(t)
            return s.energy if s else 0.5
        return float(np.interp(float(t), self.energy_times, self.energy))

    # ══════════════════════════════════════════════════════════════════════
    # Melodic and linguistic lookups
    # ══════════════════════════════════════════════════════════════════════

    def note_at(self, t: float) -> Optional[Note]:
        for n in self.notes:
            if n.start <= t < n.end:
                return n
        return None

    def notes_between(self, a: float, b: float) -> List[Note]:
        return [n for n in self.notes if n.end > a and n.start < b]

    def word_at(self, t: float) -> Optional[Word]:
        if not self.words:
            return None
        i = bisect.bisect_right([w.start for w in self.words], float(t)) - 1
        if i < 0:
            return None
        w = self.words[i]
        return w if t < w.end else None

    def sibilant_spans(self) -> List[Tuple[float, float]]:
        """Where the de-esser should act, from measured phone positions.

        Targeting measured sibilants instead of a broadband threshold is
        what lets de-essing be surgical: it moves on the words that have a
        problem and leaves the rest of the vocal untouched.
        """
        return [(p.start, p.end) for p in self.phonemes if p.kind == PHON_SIBILANT]

    def plosive_spans(self) -> List[Tuple[float, float]]:
        return [(p.start, p.end) for p in self.phonemes if p.kind == PHON_PLOSIVE]

    # ══════════════════════════════════════════════════════════════════════
    # Bookkeeping
    # ══════════════════════════════════════════════════════════════════════

    def log(self, stage: str, method: str, **kw) -> None:
        self.processing_log.append(ProcessingStep(stage=stage, method=method, **kw))

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def failed(self, reason: str) -> "MusicalIR":
        self.status = "failed"
        self.error = reason
        return self

    # ══════════════════════════════════════════════════════════════════════
    # Serialisation
    # ══════════════════════════════════════════════════════════════════════

    def to_dict(self, *, include_dense: bool = False) -> dict:
        """Serialise to JSON-compatible primitives.

        Dense per-frame arrays (f0 contours, full spectra) are excluded by
        default: they dominate the document size, are cheap to recompute,
        and nothing that reads a stored IR from a database needs them.
        """
        d = {
            "source_id": self.source_id, "source_path": self.source_path,
            "role": self.role, "version": self.version,
            "duration_s": round(self.duration_s, 3),
            "sample_rate": self.sample_rate,
            "status": self.status, "error": self.error,

            "tempo": self.tempo.to_dict(),
            "bpm": round(self.bpm, 2),
            "beats": [b.to_dict() for b in self.beats],
            "meters": [m.to_dict() for m in self.meters],
            "groove": self.groove.to_dict(),
            "has_grid": self.has_grid,

            "key_regions": [k.to_dict() for k in self.key_regions],
            "chords": [c.to_dict() for c in self.chords],
            "cadences": [c.to_dict() for c in self.cadences],
            "is_atonal": self.is_atonal,

            "n_notes": len(self.notes),
            "notes": [n.to_dict() for n in self.notes],
            "onsets": [round(float(o), 4) for o in self.onsets],
            "expression": self.expression.to_dict(),
            "performance_type": self.performance_type,
            "syllable_rate": round(self.syllable_rate, 3),

            "words": [w.to_dict() for w in self.words],
            "phonemes": [p.to_dict() for p in self.phonemes],
            "repetition": self.repetition.to_dict(),

            "sections": [s.to_dict() for s in self.sections],
            "phrases": [p.to_dict() for p in self.phrases],

            "stem_paths": dict(self.stem_paths),
            "space": self.space.to_dict(),
            "spectrum": self.spectrum.to_dict(decimate=8),
            "loudness": self.loudness.to_dict(),
            "voice": self.voice.to_dict(),

            "genre": self.genre, "mood": list(self.mood),
            "metadata": dict(self.metadata),
            "processing_log": [p.to_dict() for p in self.processing_log],
            "confidence": {k: round(float(v), 4) for k, v in self.confidence.items()},
            "warnings": list(self.warnings),
        }
        if include_dense:
            d["f0_times"] = [round(float(x), 4) for x in self.f0_times]
            d["f0_hz"] = [round(float(x), 2) for x in self.f0_hz]
            d["f0_voiced"] = [bool(x) for x in self.f0_voiced]
            d["energy_times"] = [round(float(x), 4) for x in self.energy_times]
            d["energy"] = [round(float(x), 4) for x in self.energy]
            d["timbre_embedding"] = [round(float(x), 6) for x in self.timbre_embedding]
        return d

    @staticmethod
    def from_dict(d: dict) -> "MusicalIR":
        ir = MusicalIR(
            source_id=str(d.get("source_id", "")),
            source_path=str(d.get("source_path", "")),
            role=str(d.get("role", ROLE_VOCAL)),
            version=str(d.get("version", IR_VERSION)),
            duration_s=float(d.get("duration_s", 0.0)),
            sample_rate=int(d.get("sample_rate", 48000)),
            status=str(d.get("status", "ok")), error=str(d.get("error", "")),
            tempo=TempoCurve.from_dict(d.get("tempo")),
            groove=GrooveTemplate.from_dict(d.get("groove")),
            is_atonal=bool(d.get("is_atonal", False)),
            performance_type=str(d.get("performance_type", "")),
            genre=d.get("genre"), mood=list(d.get("mood") or []),
            metadata=dict(d.get("metadata") or {}),
            confidence=dict(d.get("confidence") or {}),
            warnings=list(d.get("warnings") or []),
            space=SpaceFingerprint.from_dict(d.get("space")),
        )
        ir.beats = [Beat(time=float(b["time"]), index=int(b.get("index", 0)),
                         bar=int(b.get("bar", 0)),
                         position_in_bar=int(b.get("position_in_bar", 1)),
                         strength=float(b.get("strength", 1.0)),
                         is_downbeat=bool(b.get("is_downbeat", False)))
                    for b in (d.get("beats") or [])]
        ir.meters = [Meter(numerator=int(m.get("numerator", 4)),
                           denominator=int(m.get("denominator", 4)),
                           start=float(m.get("start", 0.0)))
                     for m in (d.get("meters") or [])] or [Meter()]
        ir.key_regions = [KeyRegion.from_dict(k) for k in (d.get("key_regions") or [])]
        ir.chords = [ChordEvent.from_dict(c) for c in (d.get("chords") or [])]
        ir.cadences = [Cadence(time=float(c["time"]), kind=str(c.get("kind", "")),
                               strength=float(c.get("strength", 0.5)))
                       for c in (d.get("cadences") or [])]
        ir.notes = [Note.from_dict(n) for n in (d.get("notes") or [])]
        ir.onsets = np.asarray(d.get("onsets") or [], dtype=np.float64)
        ir.words = [Word.from_dict(w) for w in (d.get("words") or [])]
        ir.phonemes = [Phoneme(symbol=str(p.get("symbol", "")),
                               start=float(p.get("start", 0.0)),
                               end=float(p.get("end", 0.0)),
                               kind=str(p.get("kind", "other")),
                               word_index=p.get("word_index"))
                       for p in (d.get("phonemes") or [])]
        ir.sections = [Section.from_dict(s) for s in (d.get("sections") or [])]
        ir.phrases = [Phrase.from_dict(p) for p in (d.get("phrases") or [])]
        ir.stem_paths = dict(d.get("stem_paths") or {})
        if d.get("f0_hz"):
            ir.f0_times = np.asarray(d.get("f0_times") or [], dtype=np.float64)
            ir.f0_hz = np.asarray(d["f0_hz"], dtype=np.float64)
            ir.f0_voiced = np.asarray(d.get("f0_voiced") or [], dtype=bool)
        if d.get("energy"):
            ir.energy_times = np.asarray(d.get("energy_times") or [], dtype=np.float64)
            ir.energy = np.asarray(d["energy"], dtype=np.float64)
        return ir

    def summary(self) -> str:
        """One-line human description, for logs and user-facing reports."""
        bits = [self.source_id or self.role]
        k = self.key
        if self.is_atonal:
            bits.append("atonal")
        elif k is not None:
            from .keys import Key
            bits.append(Key(k.pc, k.mode).name)
        if self.bpm > 0:
            bits.append("%.0f BPM" % self.bpm)
        else:
            bits.append("no stable tempo")
        if self.performance_type:
            bits.append(self.performance_type.replace("_", " "))
        if self.groove.is_meaningful:
            bits.append("swing %.2f" % self.groove.swing_ratio)
        if self.notes:
            bits.append("%d notes" % len(self.notes))
        if self.sections:
            bits.append("%d sections" % len(self.sections))
        return " | ".join(bits)
