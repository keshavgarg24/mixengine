"""
A voice can only sing what the score says, so the score has to say it.

Nothing in the engine used to tie a note to a word, so a plan could name a
pitch and a time and nothing about what was sung there. Words are now
placed on notes by the transcript's own times -- and only when those times
are trusted. The score then leaves the engine as JSON (everything) or MIDI
(what MIDI can carry), and the MIDI must survive Hindi and Punjabi, which
its default text encoding cannot hold.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mido import MidiFile                                              # noqa: E402

from mixengine import perform                                          # noqa: E402
from mixengine.perform import PerformancePlan, NoteTarget              # noqa: E402


def _target(i, start, end, midi=60.0, vel=0.7):
    return NoteTarget(index=i, source_midi=midi, source_start=start,
                      source_end=end, midi=midi, start=start, end=end,
                      velocity=vel)


def _plan(spans):
    return PerformancePlan(targets=[_target(i, a, b) for i, (a, b) in enumerate(spans)],
                           performance_type="sung", bar_s=2.0, beats_per_bar=4)


def _doc(words, reliable=True):
    return {"lines": [{"text": " ".join(w[0] for w in words), "start": words[0][1],
                       "end": words[-1][2],
                       "words": [{"text": t, "start": a, "end": b, "probability": 0.9}
                                 for t, a, b in words]}] if words else [],
            "timing": {"reliable": reliable}}


class TestWordsOnNotes(unittest.TestCase):

    def test_each_word_lands_on_the_note_it_was_sung_on(self):
        plan = _plan([(1.0, 1.4), (1.5, 1.9), (2.0, 2.4)])
        out = perform.assign_words(plan.targets, _doc(
            [("one", 1.0, 1.35), ("two", 1.5, 1.85), ("three", 2.0, 2.35)]))
        self.assertEqual([t.syllable for t in plan.targets], ["one", "two", "three"])
        self.assertEqual(out["coverage"], 1.0)

    def test_a_word_across_two_notes_is_carried_by_the_first_and_held(self):
        plan = _plan([(1.0, 1.5), (1.5, 2.0)])
        perform.assign_words(plan.targets, _doc([("love", 1.0, 2.0)]))
        self.assertEqual([t.syllable for t in plan.targets],
                         ["love", perform.lyric_align.CONTINUATION])

    def test_two_words_on_one_note_are_joined_in_order(self):
        plan = _plan([(1.0, 1.6)])
        perform.assign_words(plan.targets, _doc([("a", 1.0, 1.2), ("b", 1.3, 1.5)]))
        self.assertEqual(plan.targets[0].syllable, "a b")

    def test_a_word_where_no_note_was_is_left_out_and_counted(self):
        plan = _plan([(1.0, 1.4)])
        out = perform.assign_words(plan.targets, _doc(
            [("here", 1.0, 1.3), ("nowhere", 5.0, 5.3)]))
        self.assertEqual(plan.targets[0].syllable, "here")
        self.assertEqual(out["assigned"], 1)
        self.assertEqual(out["coverage"], 0.5)

    def test_a_zero_length_word_still_lands_where_it_starts(self):
        plan = _plan([(1.0, 1.4)])
        perform.assign_words(plan.targets, _doc([("hey", 1.1, 1.1)]))
        self.assertEqual(plan.targets[0].syllable, "hey")

    def test_notes_with_no_word_stay_empty(self):
        plan = _plan([(1.0, 1.4), (3.0, 3.4)])
        perform.assign_words(plan.targets, _doc([("only", 1.0, 1.3)]))
        self.assertEqual(plan.targets[1].syllable, "")

    def test_it_does_not_depend_on_the_note_order_of_the_words(self):
        plan = _plan([(1.0, 1.4), (1.5, 1.9)])
        perform.assign_words(plan.targets, _doc(
            [("second", 1.5, 1.85), ("first", 1.0, 1.35)]))
        self.assertEqual([t.syllable for t in plan.targets], ["first", "second"])


class TestItRefusesWhenTheTimesCannotBeTrusted(unittest.TestCase):

    def test_unreliable_timing_places_nothing_and_says_why(self):
        plan = _plan([(1.0, 1.4)])
        out = perform.assign_words(plan.targets, _doc([("w", 1.0, 1.3)], reliable=False))
        self.assertEqual(plan.targets[0].syllable, "")
        self.assertEqual(out["assigned"], 0)
        self.assertIn("timing", out["reason"])

    def test_stale_words_are_cleared_not_kept(self):
        plan = _plan([(1.0, 1.4)])
        plan.targets[0].syllable = "stale"
        perform.assign_words(plan.targets, _doc([("w", 1.0, 1.3)], reliable=False))
        self.assertEqual(plan.targets[0].syllable, "")

    def test_no_transcript_is_reported_as_such(self):
        plan = _plan([(1.0, 1.4)])
        self.assertIn("transcript", perform.assign_words(plan.targets, None)["reason"])
        self.assertIn("transcript", perform.assign_words(plan.targets, {"lines": []})["reason"])

    def test_no_notes_is_reported_as_such(self):
        self.assertIn("notes", perform.assign_words([], _doc([("w", 1, 2)]))["reason"])

    def test_running_it_twice_gives_the_same_answer(self):
        plan = _plan([(1.0, 1.5), (1.5, 2.0)])
        doc = _doc([("love", 1.0, 2.0)])
        perform.assign_words(plan.targets, doc)
        first = [t.syllable for t in plan.targets]
        perform.assign_words(plan.targets, doc)
        self.assertEqual([t.syllable for t in plan.targets], first)


class TestTheScore(unittest.TestCase):

    def test_it_carries_what_a_voice_needs(self):
        plan = _plan([(1.0, 1.5)])
        plan.targets[0].vibrato_depth_cents = 30.0
        score = perform.build_score(plan, bpm=120.0, language="hi",
                                    lyric_doc=_doc([("नमस्ते", 1.0, 1.4)]))
        d = score.to_dict()
        self.assertEqual((d["language"], d["bpm"]), ("hi", 120.0))
        self.assertEqual(d["notes"][0]["syllable"], "नमस्ते")
        self.assertEqual(d["notes"][0]["vibrato_depth_cents"], 30.0)
        self.assertTrue(d["lyrics_placed"])

    def test_a_score_with_untrusted_words_says_so(self):
        plan = _plan([(1.0, 1.5)])
        score = perform.build_score(plan, bpm=120.0, language="en",
                                    lyric_doc=_doc([("w", 1.0, 1.4)], reliable=False))
        self.assertFalse(score.lyrics_placed)
        self.assertIsNotNone(score.lyrics_reason)

    def test_a_score_needs_a_tempo(self):
        for bad in (0.0, -90.0, float("nan")):
            with self.assertRaises(ValueError):
                perform.build_score(_plan([(1.0, 1.5)]), bpm=bad)


class TestMidi(unittest.TestCase):

    def _roundtrip(self, plan, doc, bpm=120.0, language="hi"):
        score = perform.build_score(plan, bpm=bpm, language=language, lyric_doc=doc)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.mid")
            perform.to_midi(score, path)
            return MidiFile(path, charset="utf-8")

    def test_hindi_and_punjabi_survive_the_file(self):
        plan = _plan([(1.0, 1.4), (1.5, 1.9)])
        mf = self._roundtrip(plan, _doc([("नमस्ते", 1.0, 1.35), ("ਸਤ ਸ੍ਰੀ", 1.5, 1.85)]))
        lyr = [m.text for m in mf.tracks[0] if m.type == "lyrics"]
        self.assertEqual(lyr, ["नमस्ते", "ਸਤ ਸ੍ਰੀ"])

    def test_the_tempo_and_language_are_in_the_file(self):
        from mido import tempo2bpm
        mf = self._roundtrip(_plan([(1.0, 1.4)]), None, bpm=93.0, language="pa")
        tempo = next(m for m in mf.tracks[0] if m.type == "set_tempo")
        self.assertAlmostEqual(tempo2bpm(tempo.tempo), 93.0, delta=0.01)
        self.assertIn("language=pa", [m.text for m in mf.tracks[0] if m.type == "text"])

    def test_notes_sit_at_their_times_in_ticks(self):
        # one second at 120 bpm is two beats: 960 ticks
        mf = self._roundtrip(_plan([(1.0, 1.5)]), None)
        t, on, off = 0, None, None
        for m in mf.tracks[0]:
            t += m.time
            if m.type == "note_on" and on is None:
                on = t
            if m.type == "note_off":
                off = t
        self.assertEqual((on, off), (960, 1440))

    def test_pitch_and_velocity_map_to_midi_ranges(self):
        plan = PerformancePlan(targets=[_target(0, 1.0, 1.4, midi=60.4, vel=1.0),
                                        _target(1, 1.5, 1.9, midi=200.0, vel=0.0)])
        mf = self._roundtrip(plan, None)
        ons = [m for m in mf.tracks[0] if m.type == "note_on"]
        self.assertEqual((ons[0].note, ons[0].velocity), (60, 127))
        self.assertEqual(ons[1].note, 127)                  # clamped
        self.assertGreaterEqual(ons[1].velocity, 1)         # never a silent note-on

    def test_every_note_on_is_closed(self):
        plan = _plan([(1.0, 1.4), (1.4, 1.8), (1.8, 2.2)])
        mf = self._roundtrip(plan, None)
        kinds = [m.type for m in mf.tracks[0] if m.type in ("note_on", "note_off")]
        self.assertEqual(kinds.count("note_on"), kinds.count("note_off"))

    def test_a_zero_length_note_still_has_a_length(self):
        mf = self._roundtrip(_plan([(1.0, 1.0)]), None)
        t = on = off = 0
        for m in mf.tracks[0]:
            t += m.time
            if m.type == "note_on":
                on = t
            if m.type == "note_off":
                off = t
        self.assertGreater(off, on)


class TestAsPerformed(unittest.TestCase):

    def _notes(self):
        from mixengine.core.types import ATTACK_SLIDE, Note
        a = Note(start=1.0, end=1.4, midi=60.37, velocity=0.9, vibrato_depth_cents=25.0)
        b = Note(start=1.5, end=1.9, midi=62.0, attack=ATTACK_SLIDE)
        c = Note(start=2.0, end=2.1, midi=64.0, is_melisma=True)
        return [a, b, c]

    def test_nothing_is_moved_and_the_pitch_keeps_its_cents(self):
        plan = perform.as_performed(self._notes())
        t = plan.targets[0]
        self.assertEqual((t.start, t.end, t.midi), (1.0, 1.4, 60.37))
        self.assertFalse(any(x.moved for x in plan.targets))

    def test_expression_is_carried_across(self):
        t = perform.as_performed(self._notes()).targets[0]
        self.assertEqual((t.velocity, t.vibrato_depth_cents), (0.9, 25.0))

    def test_gestures_and_runs_are_named_for_a_voice_that_must_reproduce_them(self):
        d = [x.decision for x in perform.as_performed(self._notes()).targets]
        self.assertEqual(d, [perform.KEPT_NO_TARGET, perform.KEPT_GESTURE,
                             perform.KEPT_MELISMA])


if __name__ == "__main__":
    unittest.main()
