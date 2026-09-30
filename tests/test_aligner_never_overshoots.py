"""
Correcting timing must move an onset toward the grid, never past it.

The aligner scales each correction by how much the position matters: a
downbeat is worth more than an off-beat sixteenth, so it is pulled harder.
That weight reaches 2.0, and it was multiplying the correction rather than
selecting it -- so at full strength a syllable 30 ms late was moved 60 ms
and arrived 30 ms early. The error was mirrored, not corrected, and it
happened on exactly the positions a listener locks onto.

The weight now decides *which* onsets are worth moving; how far is capped
at the error itself.
"""

import unittest

import numpy as np

from mixengine.align import aligner
from mixengine.audio import timing
from mixengine.core.types import Phrase

SR = 22050
BPM = 120.0
BEAT = 60.0 / BPM
BAR = BEAT * 4


def _context():
    return timing.TimingContext(
        beats=np.arange(0.0, 32.0, BEAT),
        downbeats=np.arange(0.0, 32.0, BAR),
        bar_duration_s=BAR, beats_per_bar=4)


def _take(error_s: float, n: int = 48):
    """Onsets sitting a fixed distance off every sixteenth of the grid."""
    ctx = _context()
    grid = ctx.target_grid(16, apply_groove=True)[:n]
    onsets = [float(g) + error_s for g in grid]
    y = np.zeros((int(SR * 32), 1), dtype=np.float32)
    for o in onsets:
        i = int(o * SR)
        y[i:i + 200, 0] = 1.0
    return ctx, grid, onsets, y


class TestCorrectionNeverPassesTheGrid(unittest.TestCase):
    """Drift correction is off throughout: it is a uniform time scaling
    applied to every onset, so it shows up in the reported displacement
    and would mask the per-onset move this is about."""

    def test_a_late_onset_is_never_moved_past_the_beat(self):
        err = 0.030
        ctx, _, onsets, y = _take(err)
        for strength in (0.45, 0.8, 1.0):
            _, rep = aligner.align_to_grid(
                y, SR, onsets, ctx, strength=strength, subdivision=16,
                correct_drift=False)
            self.assertLessEqual(
                rep["max_move_ms"], err * 1000.0 + 0.5,
                f"at strength {strength} an onset {err * 1000:.0f} ms out "
                f"moved {rep['max_move_ms']:.1f} ms, past the grid")

    def test_an_early_onset_is_never_moved_past_the_beat(self):
        err = -0.030
        ctx, _, onsets, y = _take(err)
        _, rep = aligner.align_to_grid(y, SR, onsets, ctx, strength=1.0,
                                       subdivision=16, correct_drift=False)
        self.assertLessEqual(rep["max_move_ms"], abs(err) * 1000.0 + 0.5)

    def test_full_strength_still_lands_on_the_grid(self):
        """Capping the move must not stop it doing its job."""
        ctx, _, onsets, y = _take(0.030)
        _, rep = aligner.align_to_grid(y, SR, onsets, ctx, strength=1.0,
                                       subdivision=16, correct_drift=False)
        self.assertLess(rep["error_after_ms"], 3.0)
        self.assertLess(rep["error_after_ms"], rep["error_before_ms"])

    def test_a_gentle_strength_corrects_only_part_of_the_error(self):
        """The knob must still mean something after the cap."""
        ctx, _, onsets, y = _take(0.030)
        _, gentle = aligner.align_to_grid(y, SR, onsets, ctx, strength=0.3,
                                          subdivision=16, correct_drift=False)
        _, firm = aligner.align_to_grid(y, SR, onsets, ctx, strength=1.0,
                                        subdivision=16, correct_drift=False)
        self.assertGreater(gentle["error_after_ms"], firm["error_after_ms"])


class TestPhraseEntriesAreWeighted(unittest.TestCase):
    """`phrases` was accepted by this function and never read, so the
    entry weighting it exists for was never applied."""

    def test_entries_are_counted_when_phrases_are_given(self):
        ctx, grid, onsets, y = _take(0.030)
        phrases = [Phrase(start=float(grid[k]), end=float(grid[k]) + 1.0,
                          start_sample=int(grid[k] * SR),
                          end_sample=int((grid[k] + 1.0) * SR))
                   for k in (0, 16, 32)]
        _, rep = aligner.align_to_grid(y, SR, onsets, ctx, strength=0.5,
                                       subdivision=16, phrases=phrases,
                                       correct_drift=False)
        self.assertGreater(rep["phrase_entries"], 0)

    def test_without_phrases_nothing_is_weighted_as_an_entry(self):
        ctx, _, onsets, y = _take(0.030)
        _, rep = aligner.align_to_grid(y, SR, onsets, ctx, strength=0.5,
                                       subdivision=16, correct_drift=False)
        self.assertEqual(rep["phrase_entries"], 0)

    def test_an_entry_is_pulled_at_least_as_hard_as_the_rest(self):
        ctx, grid, onsets, y = _take(0.030)
        phrases = [Phrase(start=float(grid[0]), end=float(grid[0]) + 1.0,
                          start_sample=0, end_sample=int(SR))]
        _, without = aligner.align_to_grid(y, SR, onsets, ctx, strength=0.3,
                                           subdivision=16, correct_drift=False)
        _, with_ = aligner.align_to_grid(y, SR, onsets, ctx, strength=0.3,
                                         subdivision=16, phrases=phrases,
                                         correct_drift=False)
        self.assertLessEqual(with_["error_after_ms"],
                             without["error_after_ms"] + 1e-6)


if __name__ == "__main__":
    unittest.main()
