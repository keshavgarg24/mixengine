"""
The score has to carry how a note was sung, not only which note it was.

The analyser wrote the spread of each note under the key "vibrato";
`Note.from_dict` reads "vibrato_depth_cents". The two never met, so every
note reached the performance plan with vibrato 0 and velocity 0.7 -- a
plan that describes a row of identical notes, which is all a voice model
would then have sung. These tests pin the measurement itself and the
handoff that was broken.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mixengine.analysis import analysis                                # noqa: E402
from mixengine.audio import tuning                                     # noqa: E402
from mixengine.core.types import ATTACK_SLIDE, Note                   # noqa: E402

DT = 0.01
SR = 22050


def _contour(seconds, vib_hz=0.0, vib_cents=0.0, glide_cents=0.0, noise=0.0, seed=0):
    t = np.arange(int(seconds / DT)) * DT
    cents = vib_cents * np.sin(2 * np.pi * vib_hz * t) + glide_cents * t / seconds
    rng = np.random.default_rng(seed)
    cents = cents + rng.normal(0.0, noise, t.size)
    return 60.0 + cents / 100.0


class TestVibrato(unittest.TestCase):

    def test_a_vibrato_is_read_at_its_rate_and_depth(self):
        rate, depth = analysis.note_vibrato(_contour(1.0, 6.0, 40.0), DT)
        self.assertAlmostEqual(rate, 6.0, delta=0.4)
        self.assertAlmostEqual(depth, 40.0, delta=8.0)

    def test_a_slow_and_a_fast_vibrato_are_told_apart(self):
        slow, _ = analysis.note_vibrato(_contour(1.2, 4.5, 35.0), DT)
        fast, _ = analysis.note_vibrato(_contour(1.2, 7.0, 35.0), DT)
        self.assertLess(slow, fast)

    def test_a_steady_note_has_none(self):
        self.assertEqual(analysis.note_vibrato(_contour(1.0, noise=6.0), DT), (0.0, 0.0))

    def test_a_glide_is_not_vibrato(self):
        self.assertEqual(analysis.note_vibrato(_contour(1.0, glide_cents=90.0), DT),
                         (0.0, 0.0))

    def test_broadband_jitter_is_not_vibrato_however_large(self):
        for seed in range(5):
            self.assertEqual(
                analysis.note_vibrato(_contour(1.0, noise=40.0, seed=seed), DT),
                (0.0, 0.0), f"seed {seed}")

    def test_a_note_too_short_to_hold_a_cycle_has_none(self):
        self.assertEqual(analysis.note_vibrato(_contour(0.15, 6.0, 40.0), DT), (0.0, 0.0))

    def test_a_vibrato_riding_a_glide_is_still_found(self):
        rate, depth = analysis.note_vibrato(
            _contour(1.2, 6.0, 35.0, glide_cents=80.0), DT)
        self.assertAlmostEqual(rate, 6.0, delta=0.5)
        self.assertGreater(depth, 20.0)


class TestVelocity(unittest.TestCase):

    def test_louder_notes_score_higher(self):
        v = analysis.note_velocities([-30, -30, -30, -24, -18, -30, -36, -30])
        self.assertGreater(v[4], v[3])
        self.assertGreater(v[3], v[6])

    def test_gaining_the_whole_take_changes_nothing(self):
        lv = [-31.0, -27.5, -22.0, -29.0, -35.0, -26.0, -24.0]
        a = analysis.note_velocities(lv)
        b = analysis.note_velocities([x + 20.0 for x in lv])
        np.testing.assert_allclose(a, b, atol=1e-9)

    def test_an_even_take_stays_at_the_default(self):
        v = analysis.note_velocities([-28.0] * 8)
        self.assertTrue(all(abs(x - 0.7) < 1e-9 for x in v))

    def test_too_few_notes_to_be_relative_stay_at_the_default(self):
        self.assertEqual(analysis.note_velocities([-20.0, -40.0]), [0.7, 0.7])

    def test_it_stays_inside_the_unit_range(self):
        v = analysis.note_velocities([-80, -20, -20, -20, -20, -20, 0, -20])
        self.assertTrue(all(0.0 <= x <= 1.0 for x in v))


def _take():
    """Eight 0.6 s notes. 0 is steady and quiet, 1 vibrato and loud, 2 steady
    and loud, 3 steady and quiet; the rest repeat the quiet ones so the
    median sits low enough for the loud notes to stand out."""
    seconds = 8 * 0.7
    t = np.arange(int(seconds / DT)) * DT
    f0 = np.zeros_like(t)
    amp = np.zeros(int(seconds * SR))
    spec = [(60.0, 0.0, 0.05), (62.0, 40.0, 0.3), (64.0, 0.0, 0.3), (65.0, 0.0, 0.05),
            (60.0, 0.0, 0.05), (62.0, 0.0, 0.05), (64.0, 0.0, 0.05), (65.0, 0.0, 0.05)]
    for k, (midi, vib, a) in enumerate(spec):
        lo, hi = int((k * 0.7 + 0.05) / DT), int((k * 0.7 + 0.65) / DT)
        tt = t[lo:hi] - t[lo]
        cents = vib * np.sin(2 * np.pi * 6.0 * tt)
        f0[lo:hi] = 440.0 * 2 ** ((midi + cents / 100.0 - 69.0) / 12.0)
        amp[int((k * 0.7 + 0.05) * SR):int((k * 0.7 + 0.65) * SR)] = a
    voiced = f0 > 0
    p = analysis.PitchResult(f0=f0, times=t, voiced=voiced,
                             confidence=np.where(voiced, 0.9, 0.0))
    p.notes = analysis._segment_notes(p)
    return p, amp.astype(np.float32)


class TestTheHandoffThatWasBroken(unittest.TestCase):

    def test_expression_survives_into_the_note_the_plan_reads(self):
        p, mono = _take()
        analysis.annotate_expression(p.notes, p, mono, SR)
        self.assertEqual(len(p.notes), 8)
        notes = [Note.from_dict(n) for n in p.notes]
        self.assertGreater(notes[1].vibrato_depth_cents, 20.0)
        self.assertAlmostEqual(notes[1].vibrato_rate_hz, 6.0, delta=0.6)
        self.assertEqual(notes[0].vibrato_depth_cents, 0.0)
        self.assertGreater(notes[2].velocity, notes[0].velocity)
        self.assertGreater(notes[1].velocity, notes[3].velocity)

    def test_the_old_constants_are_gone(self):
        p, mono = _take()
        analysis.annotate_expression(p.notes, p, mono, SR)
        self.assertGreater(len({round(n["velocity"], 2) for n in p.notes}), 1)


class TestSlideRuleStillMeansSpread(unittest.TestCase):

    def _note(self, **extra):
        d = {"start": 0.0, "end": 0.6, "midi": 60.0, "confidence": 0.9}
        d.update(extra)
        return tuning.notes_from_dna([d])[0]

    def test_a_wide_spread_is_a_slide(self):
        self.assertEqual(self._note(pitch_spread_cents=90.0).attack, ATTACK_SLIDE)

    def test_a_deep_vibrato_is_not_mistaken_for_one(self):
        n = self._note(pitch_spread_cents=30.0, vibrato_depth_cents=85.0)
        self.assertNotEqual(n.attack, ATTACK_SLIDE)

    def test_a_document_from_before_the_split_still_reads(self):
        self.assertEqual(self._note(vibrato=90.0).attack, ATTACK_SLIDE)


if __name__ == "__main__":
    unittest.main()
