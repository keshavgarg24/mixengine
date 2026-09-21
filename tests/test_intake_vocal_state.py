"""
Vocal state: raw, tuned, or finished.

The engine's worst failure was treating a finished vocal as a raw take
-- retuning 317 of 506 notes that were already within 14 cents of the
grid, compressing a vocal whose phrases varied by under 1 dB, and
stripping a reverb that was a mix decision. Each of those is visible in
the signal if the question is asked.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mixengine.analysis.intake import _crest_db, detect_vocal_state  # noqa: E402
from mixengine.core.intents import Intents                        # noqa: E402

SR = 44100


def dna_with_notes(cents_offsets, spread_db=4.0, rt60=0.2,
                   deviation_cents=None):
    """A vocal DNA document whose notes sit at given cent offsets."""
    notes = []
    t = 0.0
    for i, off in enumerate(cents_offsets):
        midi = 57 + (i % 7) + off / 100.0
        notes.append({"midi": midi, "start": t, "duration": 0.4})
        t += 0.5
    if deviation_cents is None:
        deviation_cents = float(np.mean(np.abs(cents_offsets)))
    return {"notes": notes,
            "phrase_level_spread_db": spread_db,
            "tuning_deviation_cents": deviation_cents,
            "estimated_rt60_s": rt60}


def signal(crest_db=14.0, seconds=6.0, seed=1):
    """Noise bursts whose peak-to-RMS ratio is set to `crest_db`."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    y = rng.standard_normal(n)
    duty = 10.0 ** (-crest_db / 20.0)
    gate = (np.arange(n) % 1000) < max(1, int(1000 * duty * 4))
    y = y * gate
    y /= (np.abs(y).max() + 1e-12)
    return y.astype(np.float32)


class TestVocalState(unittest.TestCase):

    def test_notes_on_the_grid_read_as_tuned(self):
        dna = dna_with_notes([2, -3, 1, 4, -2, 0, 3, -1] * 4)
        st = detect_vocal_state(signal(), SR, dna)
        self.assertGreater(st.tuned_fraction, 0.8)
        self.assertIn(st.state, ("tuned", "finished"))

    def test_notes_spread_across_the_semitone_read_as_raw(self):
        dna = dna_with_notes([40, -35, 22, -48, 31, -27, 44, -19] * 4,
                             spread_db=5.0)
        st = detect_vocal_state(signal(), SR, dna)
        self.assertLess(st.tuned_fraction, 0.5)
        self.assertEqual(st.state, "raw")

    def test_narrow_phrase_spread_indicates_compression(self):
        tuned = [2, -3, 1, 4, -2, 0, 3, -1] * 4
        finished = detect_vocal_state(signal(), SR,
                                      dna_with_notes(tuned, spread_db=0.97))
        self.assertEqual(finished.state, "finished")

    def test_tuned_but_dynamic_is_not_finished(self):
        tuned = [2, -3, 1, 4, -2, 0, 3, -1] * 4
        st = detect_vocal_state(signal(), SR,
                                dna_with_notes(tuned, spread_db=5.5))
        self.assertEqual(st.state, "tuned")

    def test_reverb_on_a_finished_vocal_is_intentional(self):
        tuned = [2, -3, 1, 4, -2, 0, 3, -1] * 4
        st = detect_vocal_state(signal(), SR,
                                dna_with_notes(tuned, spread_db=0.97,
                                               rt60=0.77))
        self.assertTrue(st.reverb_is_intentional)

    def test_reverb_on_a_raw_vocal_is_a_room(self):
        raw = [40, -35, 22, -48, 31, -27, 44, -19] * 4
        st = detect_vocal_state(signal(), SR,
                                dna_with_notes(raw, spread_db=5.0, rt60=0.77))
        self.assertFalse(st.reverb_is_intentional)

    def test_intent_overrides_measurement(self):
        raw = [40, -35, 22, -48, 31, -27, 44, -19] * 4
        st = detect_vocal_state(signal(), SR, dna_with_notes(raw),
                                Intents.from_dict({"vocal_state": "finished"}))
        self.assertEqual(st.state, "finished")
        self.assertEqual(st.confidence, 1.0)
        self.assertIn("you told us", st.evidence)

    def test_too_few_notes_is_low_confidence_not_a_guess(self):
        st = detect_vocal_state(signal(), SR, dna_with_notes([1, 2, 3]))
        self.assertLess(st.confidence, 0.5)

    def test_report_is_serialisable(self):
        st = detect_vocal_state(signal(), SR, dna_with_notes([2, -3, 1, 4]))
        d = st.to_dict()
        self.assertIn("state", d)
        self.assertIsInstance(d["confidence"], float)

    def test_rt60_read_from_nested_quality_location(self):
        """Production DNA (vocal_dna.py) nests the AudioQuality payload
        under "quality" with no top-level `estimated_rt60_s` key. RT60
        must still be found there."""
        tuned = [2, -3, 1, 4, -2, 0, 3, -1] * 4
        dna = dna_with_notes(tuned, spread_db=0.97)
        del dna["estimated_rt60_s"]
        dna["quality"] = {"estimated_rt60_s": 0.77}
        st = detect_vocal_state(signal(), SR, dna)
        self.assertTrue(st.reverb_is_intentional)
        self.assertEqual(st.rt60_s, 0.77)

    def test_crest_db_handles_stereo_input(self):
        """A stereo (n, 2) signal must downmix to a sensible crest factor,
        not silently collapse to the `active.size < 128` guard's 0.0."""
        mono_sig = signal()
        stereo = np.stack([mono_sig, mono_sig], axis=1)
        crest = _crest_db(stereo)
        self.assertGreater(crest, 1.0)
        self.assertAlmostEqual(crest, _crest_db(mono_sig), places=5)


if __name__ == "__main__":
    unittest.main()
