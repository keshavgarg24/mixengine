"""
Which word sits on which note.

Until now nothing in the engine tied a note to a word: `Note.word_index`
was never set and `NoteTarget.syllable` was filled only by a caller that
passed syllables, which none did. A plan could say "this pitch, at this
time" and nothing about what was being sung there -- and a voice that
sings a score needs exactly that.

The assignment uses the transcript's word times, and those are only
trusted when `lyrics.timing_usable` says so. On rap and sung material they
agree with the audio's onsets only some of the time, so when they do not,
this assigns nothing and says why, rather than writing words onto notes
they were never sung on.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from ..analysis import lyrics

# Word times are good to a few tens of milliseconds at best.
WORD_NOTE_TOLERANCE_S = 0.06
# A word shorter than this is given this much, so a zero-length word still
# lands on the note it starts in.
MIN_WORD_SPAN_S = 0.05
# Overlap with a word that makes a note part of it.
MIN_CONTINUATION_OVERLAP_S = 0.03

CONTINUATION = "-"           # the MIDI and score convention for "hold the syllable"


def assign_words(targets: Sequence[Any], doc: Optional[dict], *,
                 min_probability: float = lyrics.WORD_MIN_PROB) -> Dict[str, Any]:
    """Write each word onto the note it was sung on. Returns what happened.

    A word goes to the note it overlaps most. Several words on one note
    (fast rap packs a note with syllables) are joined; a word spanning
    several notes (a melisma) is carried by the first and the others hold
    it. Notes with no word stay empty.
    """
    out: Dict[str, Any] = {"words": 0, "assigned": 0, "coverage": 0.0,
                           "notes": len(targets), "notes_with_text": 0,
                           "reason": None}
    for t in targets:
        t.syllable = ""
    if not targets:
        out["reason"] = "there are no notes to put words on"
        return out
    if not doc or not doc.get("lines"):
        out["reason"] = "there is no transcript"
        return out
    if not lyrics.timing_usable(doc):
        out["reason"] = ("the transcript's timing does not agree with the "
                         "audio, so words were not placed on notes")
        return out

    words = lyrics.words_of(doc, min_probability)
    out["words"] = len(words)
    held = []
    for w in words:
        span_end = max(w.end, w.start + MIN_WORD_SPAN_S)
        lo, hi = w.start - WORD_NOTE_TOLERANCE_S, span_end + WORD_NOTE_TOLERANCE_S
        best, best_overlap = None, 0.0
        for t in targets:
            overlap = min(hi, t.source_end) - max(lo, t.source_start)
            if overlap > best_overlap:
                best, best_overlap = t, overlap
        if best is None:
            continue
        best.syllable = (best.syllable + " " + w.text).strip()
        out["assigned"] += 1
        held.append((w, span_end, best))

    for w, span_end, carrier in held:
        for t in targets:
            if t is carrier or t.syllable:
                continue
            if min(span_end, t.source_end) - max(w.start, t.source_start) \
                    >= MIN_CONTINUATION_OVERLAP_S and t.source_start > carrier.source_start:
                t.syllable = CONTINUATION

    out["notes_with_text"] = sum(1 for t in targets if t.syllable)
    out["coverage"] = round(out["assigned"] / len(words), 3) if words else 0.0
    return out
