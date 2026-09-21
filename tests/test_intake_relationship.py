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


def pulse_train_with_accents(seconds, period_s, accent_times, accent_amp,
                             bed_amp=1.0, sr=SR, seed=0):
    """A regular pulse train with a few pulses boosted to accent_amp --
    a quiet, regular bed plus a handful of genuinely loud transients."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    y = np.zeros(n, dtype=np.float32)
    t = 0.0
    while t < seconds:
        idx = int(t * sr)
        if 0 <= idx < n:
            y[idx:idx + 64] = bed_amp
        t += period_s
    for at in accent_times:
        idx = int(at * sr)
        if 0 <= idx < n - 64:
            y[idx:idx + 64] = accent_amp
    return y + rng.standard_normal(n).astype(np.float32) * 0.01


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

    # -- Fix round 1 regression tests -----------------------------------

    def test_loud_transients_over_a_quiet_bed_recover_the_lag(self):
        """A hard clip maps every frame above its cap to the identical
        value -- it cannot tell a 2x-median accent from a 20x-median
        one. This bed's own period is a divisor of the lag, so its bulk
        correlation is close to tied against period-multiple decoys;
        only the three accented pulses (20x the bed's own amplitude)
        resolve which lag is true. The two 20s clips are sliced from a
        shared 30s source starting 5s in, so neither carries a
        digital-silence edge the way the other fixtures in this file
        do -- the accents are the only landmark available.
        """
        source = pulse_train_with_accents(30.0, 0.5, [8.0, 11.7, 15.4],
                                          accent_amp=20.0, seed=100)
        beat = source[int(5.0 * SR):int(25.0 * SR)]
        vocal = source[int(6.5 * SR):int(26.5 * SR)]
        r = detect_relationship(vocal, beat, SR,
                                dna(20.0, 120.0), dna(20.0, 120.0))
        self.assertEqual(r.state, "locked")
        self.assertAlmostEqual(r.offset_s, -1.5, delta=0.1)

    def test_short_pair_recovers_correct_lag(self):
        """`_best_lag` built `lags` at a fixed length (from MAX_LAG_S)
        but sliced `corr` -- whose length shrinks with input -- to the
        same fixed bounds; numpy clamps a too-large slice instead of
        raising, so short input silently indexed `lags` under the wrong
        length assumption. A 4s pair is well inside the regime where
        that mismatch used to bite (roughly audio under ~10s combined).
        """
        content = pulse_train(4.0, 0.4, seed=2)
        beat = np.concatenate([np.zeros(int(1.0 * SR), dtype=np.float32),
                               content[:int(3.0 * SR)]]).astype(np.float32)
        vocal = content
        r = detect_relationship(vocal, beat, SR,
                                dna(4.0, 120.0), dna(4.0, 120.0))
        self.assertAlmostEqual(r.offset_s, -1.0, delta=0.1)

    def test_long_lag_offset_matches_actual_frame_rate(self):
        """hop = round(sr / ENVELOPE_SR) is 220 at sr=22050 -- an actual
        frame rate of ~100.23 Hz, not the nominal 100. Converting frames
        to seconds with the nominal rate drifts by about 0.23% of the
        lag itself: negligible at a fraction of a second, but ~18ms by
        8s here (and ~45ms at the 20s MAX_LAG_S limit), which this
        tolerance is tight enough to catch.
        """
        beat = pulse_train(20.0, 0.5)
        vocal = np.concatenate([np.zeros(int(8.0 * SR), dtype=np.float32),
                                beat[:int(12.0 * SR)]]).astype(np.float32)
        r = detect_relationship(vocal, beat, SR,
                                dna(20.0, 120.0), dna(20.0, 120.0))
        self.assertEqual(r.state, "locked")
        self.assertAlmostEqual(r.offset_s, 8.0, delta=0.01)


if __name__ == "__main__":
    unittest.main()
