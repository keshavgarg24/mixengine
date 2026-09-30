"""
The performance as intended, decided without touching audio.

These pin the layer that turns measured notes into intended ones. The
point of its existing is that the decision can be checked on its own: no
pitch-shifter, no render, no sample rate. Every test here runs in
milliseconds for that reason.
"""

import unittest

from mixengine import perform
from mixengine.audio.tuning import HarmonicContext
from mixengine.core.types import (ATTACK_SCOOP, ChordEvent, KeyRegion, Note,
                                  RELEASE_CLEAN)


def _note(midi, start=0.0, dur=0.5, **kw):
    return Note(start=start, end=start + dur, midi=midi, **kw)


def _context(root=0, quality="maj", genre=None):
    """One chord held for a minute, in its own key."""
    return HarmonicContext(
        chords=[ChordEvent(start=0.0, end=60.0, root=root, quality=quality,
                           confidence=0.9)],
        key_regions=[KeyRegion(start=0.0, end=60.0, pc=root, mode="major",
                               confidence=0.9)],
        genre=genre)


class TestPitchTargets(unittest.TestCase):

    def test_a_flat_note_is_pulled_toward_the_chord_tone(self):
        """C major underneath, a note 40 cents under E."""
        targets = perform.pitch_targets([_note(63.6)], _context(), strength=0.8)
        t = targets[0]
        self.assertEqual(t.decision, perform.TUNED)
        self.assertGreater(t.pitch_shift_cents, 0.0)
        self.assertLessEqual(t.midi, 64.0)

    def test_a_note_already_in_tune_is_left_alone(self):
        targets = perform.pitch_targets([_note(64.0)], _context(), strength=0.8)
        self.assertEqual(targets[0].decision, perform.KEPT_IN_TUNE)
        self.assertEqual(targets[0].pitch_shift_cents, 0.0)

    def test_a_scoop_is_performance_and_is_kept(self):
        n = _note(63.6, attack=ATTACK_SCOOP, release=RELEASE_CLEAN)
        targets = perform.pitch_targets([n], _context(), strength=1.0)
        self.assertEqual(targets[0].decision, perform.KEPT_GESTURE)
        self.assertEqual(targets[0].midi, n.midi)

    def test_a_melisma_is_kept(self):
        n = _note(63.6, is_melisma=True)
        targets = perform.pitch_targets([n], _context(), strength=1.0)
        self.assertEqual(targets[0].decision, perform.KEPT_MELISMA)

    def test_a_note_far_from_any_target_is_refused_not_dragged(self):
        """Three semitones from anything legal is a different note or a
        detection failure, and forcing it invents a confident wrong answer."""
        targets = perform.pitch_targets([_note(62.5)], _context(root=0),
                                        strength=1.0,
                                        max_correction_semitones=0.3)
        self.assertEqual(targets[0].decision, perform.KEPT_TOO_FAR)
        self.assertEqual(targets[0].midi, 62.5)

    def test_nothing_underneath_means_no_judgement(self):
        empty = HarmonicContext()
        targets = perform.pitch_targets([_note(63.6)], empty, strength=1.0)
        self.assertEqual(targets[0].decision, perform.KEPT_NO_TARGET)

    def test_strength_scales_how_much_is_taken(self):
        gentle = perform.pitch_targets([_note(63.6)], _context(), strength=0.2)
        firm = perform.pitch_targets([_note(63.6)], _context(), strength=0.9)
        self.assertLess(gentle[0].pitch_shift_cents,
                        firm[0].pitch_shift_cents)

    def test_no_strength_changes_nothing(self):
        targets = perform.pitch_targets([_note(63.6)], _context(), strength=0.0)
        self.assertEqual(targets[0].midi, 63.6)

    def test_every_note_is_accounted_for(self):
        """A plan that silently drops notes cannot be trusted to describe
        the performance."""
        notes = [_note(63.6), _note(64.0, 1.0), _note(62.5, 2.0),
                 _note(67.0, 3.0, attack=ATTACK_SCOOP)]
        targets = perform.pitch_targets(notes, _context(), strength=0.8)
        self.assertEqual(len(targets), len(notes))
        self.assertEqual([t.index for t in targets], [0, 1, 2, 3])
        for t in targets:
            self.assertTrue(t.decision)


class TestSnapToGrid(unittest.TestCase):

    GRID = [i * 0.25 for i in range(64)]      # sixteenths at 60 BPM

    def test_a_late_note_is_pulled_toward_its_slot(self):
        t = perform.pitch_targets([_note(64.0, start=1.03)], _context(),
                                  strength=0.0)
        perform.snap_to_grid(t, self.GRID, strength=1.0, bar_s=1.0)
        self.assertLess(abs(t[0].start - 1.0), 0.03)

    def test_a_note_keeps_its_length(self):
        t = perform.pitch_targets([_note(64.0, start=1.03, dur=0.4)],
                                  _context(), strength=0.0)
        perform.snap_to_grid(t, self.GRID, strength=1.0, bar_s=1.0)
        self.assertAlmostEqual(t[0].end - t[0].start, 0.4, places=5)

    def test_a_note_far_from_the_grid_is_not_dragged(self):
        """Beyond the move limit it is not a timing error, and hauling it
        in would break the phrase around it."""
        t = perform.pitch_targets([_note(64.0, start=1.4)], _context(),
                                  strength=0.0)
        perform.snap_to_grid(t, self.GRID, strength=1.0, bar_s=1.0,
                             max_move_s=0.05)
        self.assertEqual(t[0].start, 1.4)

    def test_gestures_are_not_retimed(self):
        n = _note(64.0, start=1.03, attack=ATTACK_SCOOP)
        t = perform.pitch_targets([n], _context(), strength=0.0)
        perform.snap_to_grid(t, self.GRID, strength=1.0, bar_s=1.0)
        self.assertEqual(t[0].start, 1.03)

    def test_no_strength_moves_nothing(self):
        t = perform.pitch_targets([_note(64.0, start=1.03)], _context(),
                                  strength=0.0)
        perform.snap_to_grid(t, self.GRID, strength=0.0, bar_s=1.0)
        self.assertEqual(t[0].start, 1.03)


class TestThePlanExplainsItself(unittest.TestCase):

    def setUp(self):
        notes = [_note(63.6), _note(64.0, 1.0), _note(67.0, 2.0),
                 _note(71.6, 3.0, attack=ATTACK_SCOOP)]
        self.plan = perform.build(notes, _context(), tuning_strength=0.8,
                                  timing_strength=0.0)

    def test_it_counts_every_note(self):
        self.assertEqual(sum(self.plan.counts().values()),
                         len(self.plan.targets))

    def test_the_summary_is_a_sentence_about_this_take(self):
        s = self.plan.summary()
        self.assertIn("notes tuned", s)
        self.assertIn("left as performed", s)

    def test_it_serialises_for_the_interface(self):
        d = self.plan.to_dict()
        self.assertEqual(d["n_notes"], 4)
        self.assertIn("summary", d)
        self.assertEqual(len(d["targets"]), 4)
        self.assertIn("pitch_shift_cents", d["targets"][0])

    def test_an_empty_take_says_so_instead_of_failing(self):
        plan = perform.build([], _context(), tuning_strength=0.8)
        self.assertEqual(plan.targets, [])
        self.assertIn("no notes", plan.summary() + " ".join(plan.notes))


class TestTheTunerCarriesOutThePlan(unittest.TestCase):
    """The audio stage must do what the plan says and nothing else."""

    def test_the_tuner_reports_the_plan_it_followed(self):
        import numpy as np
        from mixengine.audio import tuning
        sr = 22050
        t = np.arange(sr) / sr
        y = (0.2 * np.sin(2 * np.pi * 329.6 * t)).astype(np.float32)[:, None]
        notes = [_note(63.6, start=0.1, dur=0.5)]
        _, report = tuning.tune_musical(
            y, sr, notes, _context(), base_strength=0.8,
            pitch_shift_fn=lambda seg, _sr, _semi: seg)
        self.assertIn("plan", report)
        self.assertEqual(report["plan"]["n_notes"], 1)
        self.assertEqual(report["plan"]["targets"][0]["decision"],
                         perform.TUNED)


if __name__ == "__main__":
    unittest.main()
