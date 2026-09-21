"""
Intake against real musical audio rather than pulse trains.

The synthetic fixtures in the other intake tests are exactly periodic,
which makes them a harder correlation problem than real music and an
easier one for a detector to accidentally pass. These tests build their
material from an actual beat so the detector faces real onset density,
real reverb tails and real spectral content.

The locked case is constructed rather than assumed: a "vocal" is made by
high-passing the beat hard -- roughly what a separated vocal looks like
spectrally -- and delaying it by a known lag. It is therefore genuinely
locked to the beat, and the measured offset has a ground truth to be
wrong about.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mixengine.analysis import intake                             # noqa: E402
from mixengine.audio import dsp                                   # noqa: E402

SR = 22050
FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "beat.wav")


def load_beat():
    import soundfile as sf
    y, sr = sf.read(FIXTURE, dtype="float32", always_2d=True)
    if sr != SR:
        import librosa
        y = np.column_stack([
            librosa.resample(np.ascontiguousarray(y[:, c]),
                             orig_sr=sr, target_sr=SR)
            for c in range(y.shape[1])])
    return y, SR


@unittest.skipUnless(os.path.exists(FIXTURE), "beat fixture missing")
class TestRelationshipOnRealAudio(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.beat, cls.sr = load_beat()
        cls.dna = {"duration_s": len(cls.beat) / cls.sr, "bpm": 120.0}

    def locked_vocal(self, lag_s):
        """Real audio, genuinely locked to the beat, spectrally unlike it."""
        voc = dsp.highpass(self.beat, self.sr, 300.0, order=4)
        pad = np.zeros((int(lag_s * self.sr), voc.shape[1]), dtype=voc.dtype)
        return np.vstack([pad, voc])[:len(self.beat)]

    def test_a_locked_take_is_recognised(self):
        r = intake.detect_relationship(self.locked_vocal(1.75), self.beat,
                                       self.sr, self.dna, self.dna)
        self.assertEqual(r.state, "locked")

    def test_the_measured_offset_is_accurate(self):
        """Task 7 applies this number directly, so it has to be right."""
        for lag in (0.5, 1.75, 3.0):
            r = intake.detect_relationship(self.locked_vocal(lag), self.beat,
                                           self.sr, self.dna, self.dna)
            self.assertAlmostEqual(r.offset_s, lag, delta=0.05,
                                   msg="lag %.2fs recovered as %.3fs"
                                       % (lag, r.offset_s))

    def test_unrelated_material_is_not_claimed_as_locked(self):
        """The failure that matters most: a false 'locked' skips every
        correction a genuinely mismatched pair needs."""
        rng = np.random.default_rng(7)
        noise = rng.standard_normal(self.beat.shape).astype(np.float32) * 0.1
        r = intake.detect_relationship(noise, self.beat, self.sr,
                                       self.dna, self.dna)
        self.assertEqual(r.state, "free")

    def test_a_length_mismatch_is_not_locked(self):
        half = self.beat[:len(self.beat) // 2]
        r = intake.detect_relationship(
            half, self.beat, self.sr,
            {"duration_s": len(half) / self.sr, "bpm": 120.0}, self.dna)
        self.assertEqual(r.state, "free")


if __name__ == "__main__":
    unittest.main()
