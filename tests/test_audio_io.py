"""
What a take looks like after `audio_io.load`: the format repairs the
loader makes before anything measures or mixes it.
"""

import os
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mixengine.core import audio_io                               # noqa: E402

SR = audio_io.SR


def _voice(seconds=2.0):
    t = np.arange(int(SR * seconds)) / SR
    return (0.5 * np.sin(2 * np.pi * 220 * t)
            * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)


class TestInvertedStereo(unittest.TestCase):
    """A mis-wired cable hands over L and -L. Every stage sums to mono, so
    left alone the voice cancels and the take is judged as noise."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.voice = _voice()

    def tearDown(self):
        self.tmp.cleanup()

    def _load(self, left, right):
        path = os.path.join(self.tmp.name, "take.wav")
        sf.write(path, np.stack([left, right], axis=1), SR, subtype="FLOAT")
        return audio_io.load(path, sr=SR)

    def test_an_inverted_pair_is_flipped_back(self):
        y, _, q = self._load(self.voice, -self.voice)
        self.assertTrue(q.is_inverted_stereo)
        self.assertFalse(q.is_silent)
        self.assertTrue(any("polarity" in w for w in q.warnings))
        mono = y.mean(axis=1)
        self.assertGreater(np.std(mono), 0.9 * np.std(self.voice))
        # Flipped back, the two channels are one signal: collapsed to mono.
        self.assertEqual(y.shape[1], 1)

    def test_the_measurements_see_the_voice_not_the_cancellation(self):
        q = audio_io.probe_quality(np.stack([self.voice, -self.voice], 1), SR)
        self.assertTrue(q.is_inverted_stereo)
        self.assertGreater(q.rms_db, -20.0)

    def test_a_real_stereo_image_is_left_alone(self):
        lag = int(0.02 * SR)
        right = np.roll(self.voice, lag)
        y, _, q = self._load(self.voice, right)
        self.assertFalse(q.is_inverted_stereo)
        self.assertFalse(q.is_fake_stereo)
        self.assertEqual(y.shape[1], 2)
        np.testing.assert_allclose(y[:, 1], right, atol=1e-4)

    def test_one_silent_channel_is_not_a_polarity_fault(self):
        y, _, q = self._load(self.voice, np.zeros_like(self.voice))
        self.assertFalse(q.is_inverted_stereo)
        self.assertFalse(q.is_silent)


if __name__ == "__main__":
    unittest.main()
