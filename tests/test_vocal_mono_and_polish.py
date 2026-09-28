"""
A lead vocal arrives centred, and the chain that makes it sound produced.

Three things are pinned here. That a take recorded on two microphones
reaches the mix as one channel rather than as a smeared stereo pair, and
that it gets there without the naive sum that cancels the voice. That
saturation adds harmonics without adding aliasing or level, so the setting
controls character and nothing else. And that the clipper in the master
touches only what approaches the ceiling.
"""

import unittest

import numpy as np

from mixengine.audio import dsp
from mixengine.core import audio_io

SR = 44100


def _voice(n: int, sr: int = SR, f0: float = 150.0) -> np.ndarray:
    """A harmonic-rich tone that behaves like a voice under these tests."""
    t = np.arange(n) / sr
    y = sum((1.0 / h) * np.sin(2 * np.pi * f0 * h * t) for h in (1, 2, 3, 4, 5))
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 2.0 * t)
    return (0.25 * y * env).astype(np.float32)


class TestTwoMicrophoneTake(unittest.TestCase):
    """The case that motivated all of this: a phone with two microphones.

    The take reaching the engine is one voice arriving twice, a fraction
    of a millisecond apart. Summed as it stands the two copies comb-filter
    and the voice hollows out; on the real take that settled this, the
    presence band lost 6 dB and the transcriber went from 80 words to
    none at all.
    """

    def setUp(self):
        n = SR * 3
        rng = np.random.default_rng(0)
        v = _voice(n)
        lag = 11                       # what the real take measured
        # The far microphone is not a quieter copy of the near one. It
        # hears the same voice late, plus the room -- reflections that
        # correlate with nothing. That is what puts the real take's
        # channels at 0.88 rather than 0.999, and a fixture without it
        # tests a case that does not occur.
        room = np.convolve(v, rng.standard_normal(2205).astype(np.float32)
                           * np.exp(-np.arange(2205) / 400.0), mode="same")
        room *= 0.35 * float(np.abs(v).max()) / max(float(np.abs(room).max()), 1e-9)
        far = np.roll(v, lag) * 0.6 + room
        far += 0.004 * rng.standard_normal(n).astype(np.float32)
        self.take = np.stack([far.astype(np.float32), v], axis=1)

    def test_the_take_comes_out_as_one_channel(self):
        out, rep = audio_io.fold_to_mono(self.take, SR)
        self.assertEqual(out.shape[1], 1)
        self.assertTrue(rep["applied"])

    def test_the_clearer_microphone_is_the_one_kept(self):
        out, rep = audio_io.fold_to_mono(self.take, SR)
        self.assertEqual(rep["method"], "better_channel")
        self.assertEqual(rep["kept_channel"], 1)

    def test_the_voice_is_not_hollowed_out(self):
        """The point of the exercise. Against a naive sum, which is the
        obvious thing to do and the thing that ruins the take."""
        out, _ = audio_io.fold_to_mono(self.take, SR)
        naive = dsp.to_mono(self.take)[:, None]

        def presence(x):
            b = dsp.bandpass(dsp.as_2d(x), SR, 800.0, 4000.0)
            return float(np.sqrt(np.mean(np.square(b.astype(np.float64)))))

        self.assertGreater(presence(out), presence(naive) * 1.3)

    def test_the_person_is_told_what_happened(self):
        _, rep = audio_io.fold_to_mono(self.take, SR)
        self.assertIn("two microphones", rep["note"])

    def test_the_take_is_not_modified_in_place(self):
        """The caller keeps using the array it passed in."""
        before = self.take.copy()
        audio_io.fold_to_mono(self.take, SR)
        self.assertTrue(np.array_equal(before, self.take))


class TestOtherShapesOfTake(unittest.TestCase):

    def test_a_mono_take_is_left_alone(self):
        v = _voice(SR)[:, None]
        out, rep = audio_io.fold_to_mono(v, SR)
        self.assertEqual(rep["method"], "already_mono")
        self.assertTrue(np.array_equal(dsp.as_2d(v), out))

    def test_a_duplicated_channel_is_summed(self):
        v = _voice(SR)
        out, rep = audio_io.fold_to_mono(np.stack([v, v], axis=1), SR)
        self.assertEqual(rep["method"], "sum")
        self.assertEqual(out.shape[1], 1)
        np.testing.assert_allclose(out[:, 0], v, atol=1e-5)

    def test_an_inverted_channel_survives_instead_of_cancelling(self):
        """Summed as it arrives this take is silence."""
        v = _voice(SR)
        out, rep = audio_io.fold_to_mono(np.stack([v, -v], axis=1), SR)
        self.assertTrue(rep["polarity_flipped"])
        self.assertGreater(float(np.abs(out).max()), 0.1)

    def test_a_silent_channel_does_not_halve_the_take(self):
        v = _voice(SR)
        out, rep = audio_io.fold_to_mono(
            np.stack([np.zeros_like(v), v], axis=1), SR)
        self.assertEqual(rep["method"], "better_channel")
        self.assertEqual(rep["kept_channel"], 1)


class TestSaturation(unittest.TestCase):

    def setUp(self):
        self.n = SR * 2
        t = np.arange(self.n) / SR
        self.tone = (0.3 * np.sin(2 * np.pi * 4000.0 * t)).astype(np.float32)[:, None]

    def _spectrum(self, y):
        w = np.hanning(len(y))
        return (np.abs(np.fft.rfft(y[:, 0] * w)),
                np.fft.rfftfreq(len(y), 1.0 / SR))

    def test_it_adds_harmonics(self):
        out = dsp.saturate(self.tone, SR, amount=1.0, drive_db=12.0)
        mag, f = self._spectrum(out)
        fund = mag[(f > 3900) & (f < 4100)].max()
        second = mag[(f > 7900) & (f < 8100)].max()
        self.assertGreater(second / fund, 1e-2)

    def test_it_does_not_alias(self):
        """A 4 kHz tone's harmonics land above it. Anything appearing
        *below* it folded back, and folded-back tones are inharmonic --
        the difference between warmth and grit."""
        out = dsp.saturate(self.tone, SR, amount=1.0, drive_db=12.0)
        mag, f = self._spectrum(out)
        fund = mag[(f > 3900) & (f < 4100)].max()
        below = mag[(f > 100) & (f < 3500)].max()
        self.assertLess(20 * np.log10(below / fund), -100.0)

    def test_the_amount_changes_character_and_not_loudness(self):
        quiet = dsp.saturate(self.tone, SR, amount=0.2, drive_db=9.0)
        heavy = dsp.saturate(self.tone, SR, amount=0.9, drive_db=9.0)

        def rms(x):
            return float(np.sqrt(np.mean(np.square(x.astype(np.float64)))))

        self.assertAlmostEqual(rms(quiet), rms(heavy), delta=rms(quiet) * 0.05)

    def test_no_amount_is_a_no_op(self):
        out = dsp.saturate(self.tone, SR, amount=0.0)
        np.testing.assert_allclose(out, self.tone)


class TestSoftClipper(unittest.TestCase):

    def test_quiet_material_is_untouched(self):
        """Only what approaches the ceiling may be shaped; a clipper that
        touches everything is a distortion unit."""
        t = np.arange(SR) / SR
        quiet = (0.05 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)[:, None]
        out, rep = dsp.soft_clip(quiet, SR, ceiling_db=-1.0)
        self.assertFalse(rep["applied"])
        np.testing.assert_allclose(out, quiet, atol=1e-6)

    def test_peaks_are_brought_down(self):
        t = np.arange(SR) / SR
        loud = (0.99 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)[:, None]
        out, rep = dsp.soft_clip(loud, SR, ceiling_db=-6.0)
        self.assertTrue(rep["applied"])
        self.assertLess(float(np.abs(out).max()), float(np.abs(loud).max()))


class TestParallelCompression(unittest.TestCase):

    def test_it_raises_what_is_quiet_without_moving_the_peaks(self):
        n = SR * 2
        v = _voice(n)
        v[:SR] *= 0.15                       # a quiet half and a loud half
        x = v[:, None]
        out, rep = dsp.parallel_compress(x, SR, amount=0.5)
        self.assertTrue(rep["applied"])

        def rms(seg):
            return float(np.sqrt(np.mean(np.square(seg.astype(np.float64)))))

        before = rms(x[SR:]) / max(rms(x[:SR]), 1e-12)
        after = rms(out[SR:]) / max(rms(out[:SR]), 1e-12)
        self.assertLess(after, before)
        self.assertLessEqual(float(np.abs(out).max()),
                             float(np.abs(x).max()) * 1.05)


if __name__ == "__main__":
    unittest.main()
