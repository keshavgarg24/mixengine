"""
Laying the vocal's own bar grid over the beat's.

Every attempt to align a vocal to a beat by correlating one against the
other failed for the same reason: the two share no content, so the
correlation peaks at chance. This is the approach that replaced them --
track each signal's bar-ones independently, then find the shift that
lays one grid on the other. Measured on real stems it recovered known
shifts of 0.9, 0.7 and 1.2 seconds to within 6 ms.

The pure-logic tests below need no tracker. The end-to-end tests run
madmom on real audio and are skipped where it is not installed.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mixengine.audio import dsp, transform                        # noqa: E402
from mixengine.core.capabilities import CAPS                      # noqa: E402

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "beat.wav")
BAR = 2.4


class TestGridAlignmentLogic(unittest.TestCase):

    def test_matching_grids_need_no_shift(self):
        beat = np.arange(0.0, 60.0, BAR)
        shift, conf = transform._grid_alignment(beat.copy(), beat, BAR)
        self.assertAlmostEqual(shift, 0.0, places=3)
        self.assertGreater(conf, 0.99)

    def test_a_late_vocal_is_moved_earlier(self):
        beat = np.arange(0.0, 60.0, BAR)
        vocal = beat + 0.9
        shift, conf = transform._grid_alignment(vocal, beat, BAR)
        self.assertAlmostEqual(shift, -0.9, places=3)
        self.assertGreater(conf, 0.99)

    def test_an_early_vocal_is_moved_later(self):
        beat = np.arange(0.0, 60.0, BAR)
        vocal = beat - 0.7
        shift, _ = transform._grid_alignment(vocal, beat, BAR)
        self.assertAlmostEqual(shift, 0.7, places=3)

    def test_whole_bar_offsets_are_invisible(self):
        """A vocal one bar late is on the grid. Nothing to correct."""
        beat = np.arange(0.0, 60.0, BAR)
        shift, conf = transform._grid_alignment(beat[:-1] + BAR, beat, BAR)
        self.assertAlmostEqual(shift, 0.0, places=3)
        self.assertGreater(conf, 0.99)

    def test_scattered_ones_report_low_confidence(self):
        """A vocal whose ones fall anywhere in the bar has no grid, and
        must say so rather than hand back a confident random shift."""
        rng = np.random.default_rng(3)
        beat = np.arange(0.0, 120.0, BAR)
        vocal = beat + rng.uniform(-BAR / 2, BAR / 2, size=beat.size)
        _, conf = transform._grid_alignment(vocal, beat, BAR)
        self.assertLess(conf, transform.VOCAL_GRID_CONFIDENCE)

    def test_empty_input_is_safe(self):
        shift, conf = transform._grid_alignment(np.zeros(0), np.arange(5.0),
                                                BAR)
        self.assertEqual((shift, conf), (0.0, 0.0))


def _reference_bar(y, sr):
    """The fixture's bar length from a plain 4/4 downbeat pass."""
    import warnings
    from madmom.features.downbeats import (DBNDownBeatTrackingProcessor,
                                           RNNDownBeatProcessor)
    mono = np.ascontiguousarray(y.mean(axis=1)).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        act = RNNDownBeatProcessor()(mono)
        out = DBNDownBeatTrackingProcessor(beats_per_bar=[4], fps=100)(act)
    ones = out[out[:, 1] == 1][:, 0]
    return float(np.median(np.diff(ones))) if ones.size > 2 else 0.0


@unittest.skipUnless(CAPS.madmom and os.path.exists(FIXTURE),
                     "needs madmom and the beat fixture")
class TestGridAlignmentOnAudio(unittest.TestCase):
    """The tracker on real audio, with a ground truth to be wrong about.

    A vocal is made by high-passing the beat -- roughly what a separated
    vocal looks like spectrally -- and delaying it a known amount. Its
    bar-ones are the beat's, shifted, so the recovered shift has a truth.
    """

    @classmethod
    def setUpClass(cls):
        import soundfile as sf
        y, sr = sf.read(FIXTURE, dtype="float32", always_2d=True)
        cls.beat, cls.sr = y, sr
        # Production gets the bar from the beat's DNA. Here the fixture's
        # own 4/4 reading stands in for it, rather than a hardcoded guess.
        cls.bar = _reference_bar(y, sr)
        cls.beat_ones = (transform.vocal_downbeats(y, sr, cls.bar)
                         if cls.bar > 0 else None)

    def shifted_vocal(self, lag_s):
        v = dsp.highpass(self.beat, self.sr, 300.0, order=4)
        n = int(abs(lag_s) * self.sr)
        if lag_s > 0:
            return np.vstack([np.zeros((n, v.shape[1]), v.dtype), v])
        return v[n:]

    def test_the_beat_has_a_grid(self):
        self.assertIsNotNone(self.beat_ones)
        self.assertGreater(self.bar, 0.0)

    def test_known_shifts_are_recovered(self):
        if self.beat_ones is None:
            self.skipTest("tracker found no grid on the fixture")
        for lag in (0.5, -0.4):
            ones = transform.vocal_downbeats(self.shifted_vocal(lag),
                                             self.sr, self.bar)
            self.assertIsNotNone(ones, "no grid on the shifted vocal")
            shift, conf = transform._grid_alignment(ones, self.beat_ones,
                                                    self.bar)
            self.assertGreaterEqual(conf, transform.VOCAL_GRID_CONFIDENCE)
            self.assertAlmostEqual(shift, -lag, delta=0.06,
                                   msg="lag %+.2f recovered as %+.3f"
                                       % (lag, shift))


if __name__ == "__main__":
    unittest.main()


class TestGridTempoRatio(unittest.TestCase):
    """The stretch ratio from the vocal's own bars, not a tempo histogram."""

    def test_same_tempo_is_ratio_one(self):
        ones = np.arange(0.0, 100.0, BAR)
        r = transform.grid_tempo_ratio(ones, BAR)
        self.assertIsNotNone(r)
        self.assertAlmostEqual(r, 1.0, places=5)

    def test_a_slow_vocal_reads_above_one(self):
        """Bars 1% longer than the beat's -> ratio 1.01, i.e. the vocal
        must be shortened by that factor to match."""
        ones = np.arange(0.0, 100.0, BAR * 1.01)
        r = transform.grid_tempo_ratio(ones, BAR)
        self.assertAlmostEqual(r, 1.01, places=4)

    def test_jitter_does_not_bias_the_fit(self):
        """Per-bar jitter of +/-30 ms averages out over forty bars; a
        median-of-diffs would not, and a histogram certainly did not."""
        rng = np.random.default_rng(5)
        ones = np.arange(0.0, 100.0, BAR) + rng.uniform(-0.03, 0.03, 42)
        r = transform.grid_tempo_ratio(ones, BAR)
        self.assertAlmostEqual(r, 1.0, places=3)

    def test_too_few_bars_is_none(self):
        self.assertIsNone(transform.grid_tempo_ratio(np.arange(0.0, 10.0, BAR),
                                                     BAR))
        self.assertIsNone(transform.grid_tempo_ratio(None, BAR))
