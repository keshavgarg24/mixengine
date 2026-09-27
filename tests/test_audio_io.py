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
        self.assertTrue(any("polarity" in r for r in q.repairs))
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


class TestWhatLoadTellsThePerson(unittest.TestCase):
    """Every change made on the way in is said in words, and a file that
    cannot be read is refused in words."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def path(self, name):
        return os.path.join(self.tmp.name, name)

    def test_a_damaged_file_is_refused_with_its_name_and_a_way_forward(self):
        bad = self.path("take.wav")
        with open(bad, "wb") as f:
            f.write(b"RIFF" + bytes(range(256)) * 200)
        for call in (audio_io.decode_check, lambda p: audio_io.load(p, sr=SR)):
            with self.assertRaises(ValueError) as cm:
                call(bad)
            msg = str(cm.exception)
            self.assertIn("take.wav", msg)
            self.assertIn("could not be decoded", msg)
            self.assertIn("WAV or MP3", msg)

    def test_a_good_file_passes_the_decode_check(self):
        good = self.path("take.wav")
        sf.write(good, _voice(), SR, subtype="FLOAT")
        audio_io.decode_check(good)

    def test_repairs_are_listed_in_words(self):
        voice = _voice()
        clipped = np.clip(voice * 8, -1, 1) + 0.2
        p = self.path("take.wav")
        sf.write(p, np.stack([clipped, clipped], 1), SR, subtype="FLOAT")
        y, _, q = audio_io.load(p, sr=SR)
        text = " / ".join(q.repairs)
        self.assertIn("DC offset", text)
        self.assertIn("clipped peaks", text)
        self.assertIn("identical", text)
        self.assertEqual(y.shape[1], 1)
        self.assertLess(abs(float(y.mean())), 1e-3)

    def test_a_low_rate_file_says_what_it_lost(self):
        p = self.path("phone.wav")
        sf.write(p, _voice()[::6], SR // 6, subtype="FLOAT")
        _, _, q = audio_io.load(p, sr=SR)
        self.assertTrue(any("resampled from %d Hz" % (SR // 6) in r
                            for r in q.repairs), q.repairs)

    def test_a_clean_file_needs_no_repair(self):
        p = self.path("take.wav")
        sf.write(p, _voice(), SR, subtype="FLOAT")
        _, _, q = audio_io.load(p, sr=SR)
        self.assertEqual(q.repairs, [])


class TestDeclip(unittest.TestCase):
    """The repaired arc used to be clipped straight back to full scale, so
    hard clipping -- the case the repair exists for -- was left as it was."""

    def test_hard_clipping_is_actually_repaired(self):
        t = np.arange(SR * 4) / SR
        clean = np.sin(2 * np.pi * 220 * t).astype(np.float32)
        clipped = np.clip(clean * 3, -1, 1).astype(np.float32)
        out = audio_io.declip(clipped)[:, 0]
        at_ceiling = lambda a: int((np.abs(a) >= 0.9995 * np.abs(a).max()).sum())  # noqa: E731
        self.assertLess(at_ceiling(out), at_ceiling(clipped) * 0.05)
        self.assertGreater(float(np.corrcoef(out, clean)[0, 1]), 0.999)
        self.assertLessEqual(float(np.abs(out).max()), 1.0)

    def test_a_clean_take_passes_through_untouched(self):
        y = _voice()
        np.testing.assert_array_equal(audio_io.declip(y)[:, 0], y)


if __name__ == "__main__":
    unittest.main()
