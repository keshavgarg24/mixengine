"""
The score: what a voice is asked to sing, in a form nothing else is needed to read.

A `PerformancePlan` is the engine's own working object. A score is what
leaves it: notes in seconds and fractional MIDI, the words on them, the
expression measured on each, the tempo, and the language. Any renderer --
the engine's pitch-shifter, a singing synthesiser, a speech model asked to
rap -- can be given one and needs nothing else.

Two exports, because they serve different readers. The JSON carries
everything, including pitch to the cent. The MIDI file carries what MIDI
can -- integer pitch, velocity, tempo, and the words as lyric events -- so a
person can open it in a synthesiser by hand and hear what a model would be
given. MIDI lyric text is Latin-1 unless told otherwise; Hindi and Punjabi
are written to it as UTF-8, which is why the file says so on both save and
load.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .lyric_align import assign_words
from .plan import PerformancePlan

PPQ = 480
MIDI_CHARSET = "utf-8"


@dataclass
class Score:
    language: Optional[str]
    bpm: float
    beats_per_bar: int = 4
    performance_type: str = "sung"
    notes: List[Dict[str, Any]] = field(default_factory=list)
    lyrics_placed: bool = False
    lyrics_reason: Optional[str] = None
    lyric_coverage: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"language": self.language, "bpm": round(self.bpm, 3),
                "beats_per_bar": self.beats_per_bar,
                "performance_type": self.performance_type,
                "lyrics_placed": self.lyrics_placed,
                "lyrics_reason": self.lyrics_reason,
                "lyric_coverage": self.lyric_coverage,
                "notes": self.notes}


def build_score(plan: PerformancePlan, *, bpm: float,
                language: Optional[str] = None,
                lyric_doc: Optional[dict] = None) -> Score:
    """The plan, with words on its notes, as a score."""
    if not np.isfinite(bpm) or bpm <= 0:
        raise ValueError("a score needs a tempo; got %r" % (bpm,))
    placed = assign_words(plan.targets, lyric_doc)
    score = Score(language=language, bpm=float(bpm),
                  beats_per_bar=plan.beats_per_bar,
                  performance_type=plan.performance_type,
                  lyrics_placed=placed["notes_with_text"] > 0,
                  lyrics_reason=placed["reason"],
                  lyric_coverage=placed["coverage"])
    for t in plan.targets:
        score.notes.append({
            "start": round(t.start, 4), "end": round(t.end, 4),
            "midi": round(t.midi, 3),
            "velocity": round(t.velocity, 3),
            "vibrato_depth_cents": round(t.vibrato_depth_cents, 1),
            "syllable": t.syllable,
            "decision": t.decision,
        })
    return score


def _ticks(seconds: float, bpm: float) -> int:
    return int(round(seconds * bpm / 60.0 * PPQ))


def to_midi(score: Score, path: str) -> None:
    """Write the score as a standard MIDI file with its words as lyric events."""
    from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo

    mf = MidiFile(ticks_per_beat=PPQ, charset=MIDI_CHARSET)
    track = MidiTrack()
    mf.tracks.append(track)
    track.append(MetaMessage("set_tempo", tempo=bpm2tempo(score.bpm), time=0))
    track.append(MetaMessage("time_signature", numerator=score.beats_per_bar,
                             denominator=4, time=0))
    if score.language:
        track.append(MetaMessage("text", text="language=%s" % score.language, time=0))

    events = []
    for n in score.notes:
        on, off = _ticks(n["start"], score.bpm), _ticks(n["end"], score.bpm)
        off = max(off, on + 1)
        pitch = int(np.clip(round(n["midi"]), 0, 127))
        vel = int(np.clip(round(n["velocity"] * 127.0), 1, 127))
        events.append((on, 1, "on", pitch, vel, n["syllable"]))
        events.append((off, 0, "off", pitch, 0, ""))
    events.sort(key=lambda e: (e[0], e[1]))

    last = 0
    for tick, _, kind, pitch, vel, text in events:
        if kind == "on" and text:
            track.append(MetaMessage("lyrics", text=text, time=tick - last))
            last = tick
        track.append(Message("note_on" if kind == "on" else "note_off",
                             note=pitch, velocity=vel, time=tick - last))
        last = tick
    mf.save(path)
