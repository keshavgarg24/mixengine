"""
Relationship: was this vocal recorded to this beat?

If it was, the alignment problem is one number -- the lag between them
-- and every stage that would "fix" the timing is doing damage. The
same cross-correlation that detects the relationship measures the lag,
so detection and alignment are one operation.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mixengine.analysis.intake import detect_relationship          # noqa: E402
from mixengine.core.intents import Intents                         # noqa: E402

SR = 22050


def pulse_train(seconds, period_s, sr=SR, seed=0, jitter=0.0):
    """Impulses at a fixed period -- a stand-in for an onset envelope."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    y = np.zeros(n, dtype=np.float32)
    t = 0.0
    while t < seconds:
        idx = int((t + rng.normal(0, jitter)) * sr)
        if 0 <= idx < n:
            y[idx:idx + 64] = 1.0
        t += period_s
    return y + rng.standard_normal(n).astype(np.float32) * 0.01


def dna(duration, bpm):
    return {"duration_s": duration, "bpm": bpm}


class TestRelationship(unittest.TestCase):

    def test_same_performance_reads_as_locked(self):
        beat = pulse_train(20.0, 0.5)
        vocal = np.concatenate([np.zeros(int(1.5 * SR), dtype=np.float32),
                                beat[:int(18.5 * SR)]])
        r = detect_relationship(vocal, beat, SR,
                                dna(20.0, 120.0), dna(20.0, 120.0))
        self.assertEqual(r.state, "locked")
        self.assertAlmostEqual(r.offset_s, 1.5, delta=0.05)

    def test_unrelated_audio_reads_as_free(self):
        beat = pulse_train(20.0, 0.5, seed=1)
        vocal = pulse_train(20.0, 0.37, seed=99, jitter=0.02)
        r = detect_relationship(vocal, beat, SR,
                                dna(20.0, 162.0), dna(20.0, 120.0))
        self.assertEqual(r.state, "free")

    def test_different_durations_are_not_locked(self):
        beat = pulse_train(20.0, 0.5)
        vocal = beat[:int(8.0 * SR)]
        r = detect_relationship(vocal, beat, SR,
                                dna(8.0, 120.0), dna(20.0, 120.0))
        self.assertEqual(r.state, "free")

    def test_half_time_tempo_is_not_a_disagreement(self):
        """Trap is written at 146 and felt at 73. That is one tempo."""
        beat = pulse_train(20.0, 0.5)
        vocal = np.concatenate([np.zeros(int(0.25 * SR), dtype=np.float32),
                                beat[:int(19.75 * SR)]])
        r = detect_relationship(vocal, beat, SR,
                                dna(20.0, 73.0), dna(20.0, 146.0))
        self.assertEqual(r.state, "locked")

    def test_intent_overrides_measurement(self):
        beat = pulse_train(20.0, 0.5, seed=1)
        vocal = pulse_train(20.0, 0.37, seed=99)
        r = detect_relationship(vocal, beat, SR,
                                dna(20.0, 162.0), dna(20.0, 120.0),
                                Intents.from_dict({"relationship": "locked"}))
        self.assertEqual(r.state, "locked")
        self.assertIn("you told us", r.evidence)

    def test_offset_is_reported_even_when_free(self):
        beat = pulse_train(20.0, 0.5, seed=1)
        vocal = pulse_train(20.0, 0.37, seed=99)
        r = detect_relationship(vocal, beat, SR,
                                dna(20.0, 162.0), dna(20.0, 120.0))
        self.assertIsInstance(r.offset_s, float)

    def test_report_is_serialisable(self):
        beat = pulse_train(10.0, 0.5)
        r = detect_relationship(beat, beat, SR, dna(10.0, 120.0),
                                dna(10.0, 120.0))
        self.assertIn("state", r.to_dict())


if __name__ == "__main__":
    unittest.main()
