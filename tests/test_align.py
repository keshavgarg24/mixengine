"""
Tests for variable-rate alignment.

Every case here except the pure-property ones corresponds to a defect that
was found by measuring the engine rather than by reading it, and each names
the failure it prevents. Three of them are guarding against mistakes made
while building this module, which is the reason they are worth keeping: the
code looked correct in all three cases and produced plausible numbers.

No audio backend is required. Where audio is involved the signal is a click
train, so an onset's position can be measured exactly rather than estimated.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mixengine.align import aligner, dtw                               # noqa: E402
from mixengine.align import warp as warp_mod                           # noqa: E402
from mixengine.audio.timing import TimingContext                       # noqa: E402
from mixengine.core.types import GrooveTemplate                        # noqa: E402

SR = 22050


def context(bpm=120.0, bars=20, groove=None):
    beat = 60.0 / bpm
    bar = beat * 4
    return TimingContext(
        beats=np.arange(0, bar * bars, beat),
        downbeats=np.arange(0, bar * bars, bar),
        bar_duration_s=bar, beats_per_bar=4,
        groove=groove or GrooveTemplate(), grid_stability=1.0)


def click_train(times, sr=SR, duration=None, freq=700.0):
    dur = duration if duration is not None else float(max(times)) + 1.0
    y = np.zeros(int(dur * sr), dtype=np.float32)
    n = int(sr * 0.012)
    w = (np.exp(-np.arange(n) / (sr * 0.004))
         * np.sin(2 * np.pi * freq * np.arange(n) / sr)).astype(np.float32)
    for t in times:
        i = int(t * sr)
        if 0 <= i < len(y) - n:
            y[i:i + n] += 0.6 * w
    return y


def find_clicks(y, sr=SR, rel=0.25):
    m = np.abs(y[:, 0] if y.ndim > 1 else y)
    if m.max() <= 0:
        return np.zeros(0)
    th = m.max() * rel
    out, i = [], 0
    while i < len(m):
        if m[i] > th:
            j = i
            while j < len(m) and m[j] > th * 0.3:
                j += 1
            out.append((i + int(np.argmax(m[i:j]))) / sr)
            i = j + int(sr * 0.02)
        else:
            i += 1
    return np.asarray(out)


# ═════════════════════════════════════════════════════════════════════════════

class TestAssignOnsetsToGrid(unittest.TestCase):

    def setUp(self):
        self.grid = np.arange(0.0, 60.0, 0.125)

    def test_recovers_exact_slots_when_clean(self):
        rng = np.random.default_rng(7)
        true = np.sort(rng.choice(np.arange(4, 460), 120, replace=False))
        onsets = self.grid[true] + rng.normal(0, 0.008, 120)
        a = dtw.assign_onsets_to_grid(onsets, self.grid, max_move_s=0.05)
        self.assertTrue(np.array_equal(a.slot_index, true))

    def test_assignment_is_strictly_monotonic(self):
        """Nearest-neighbour is not, and a non-monotonic assignment asks the
        warp for a negative time interval."""
        rng = np.random.default_rng(3)
        true = np.sort(rng.choice(np.arange(4, 460), 180, replace=False))
        onsets = self.grid[true] * 1.004 + rng.normal(0, 0.015, 180)

        nearest = np.abs(onsets[:, None] - self.grid[None, :]).argmin(axis=1)
        self.assertGreater(int((np.diff(nearest) <= 0).sum()), 0,
                           "fixture no longer exercises the failure")

        a = dtw.assign_onsets_to_grid(onsets, self.grid, max_move_s=0.09)
        used = a.slot_index[a.assigned]
        self.assertTrue(np.all(np.diff(used) > 0))

    def test_no_two_onsets_share_a_slot(self):
        rng = np.random.default_rng(3)
        true = np.sort(rng.choice(np.arange(4, 460), 180, replace=False))
        onsets = self.grid[true] * 1.004 + rng.normal(0, 0.015, 180)
        a = dtw.assign_onsets_to_grid(onsets, self.grid, max_move_s=0.09)
        used = a.slot_index[a.assigned].tolist()
        self.assertEqual(len(used), len(set(used)))

    def test_unreachable_onsets_are_skipped_not_forced(self):
        onsets = np.array([0.0, 0.125, 7.3117, 0.375, 0.5])
        onsets = np.sort(onsets)
        a = dtw.assign_onsets_to_grid(onsets, self.grid, max_move_s=0.004)
        far = int(np.argmin(np.abs(onsets - 7.3117)))
        self.assertFalse(bool(a.assigned[far]))
        self.assertAlmostEqual(float(a.targets[far]), 7.3117, places=6)

    def test_empty_inputs_are_safe(self):
        a = dtw.assign_onsets_to_grid([], self.grid)
        self.assertEqual(a.n_assigned, 0)
        b = dtw.assign_onsets_to_grid([1.0, 2.0], [])
        self.assertEqual(b.n_assigned, 0)


class TestTempoRatioEstimation(unittest.TestCase):
    """The drift measurement, and why the obvious one does not work."""

    def _take(self, drift, seed):
        grid = np.arange(0.0, 60.0, 0.125)
        rng = np.random.default_rng(seed)
        true = np.sort(rng.choice(np.arange(4, 460), 180, replace=False))
        return grid, np.sort(grid[true] * drift + rng.normal(0, 0.015, 180))

    def test_recovers_known_drift(self):
        for drift, seed in ((1.004, 3), (0.997, 11), (1.012, 2), (0.988, 9)):
            with self.subTest(drift=drift):
                grid, onsets = self._take(drift, seed)
                ratio, _, gain = dtw.estimate_tempo_ratio(onsets, grid)
                self.assertAlmostEqual(ratio, drift, places=3)
                self.assertGreater(gain, 0.1)

    def test_on_tempo_take_is_left_alone(self):
        grid, onsets = self._take(1.0, 5)
        ratio, _, _ = dtw.estimate_tempo_ratio(onsets, grid)
        self.assertLess(abs(ratio - 1.0), 0.001)

    def test_residual_drift_aliases_and_must_not_be_used_to_measure_drift(self):
        """Guards the reason `estimate_tempo_ratio` exists.

        Once a take has drifted past half a slot the assignment moves it to
        the next slot and the error resets, so a fit through the errors sees
        almost nothing. The first implementation used that fit and stretched
        takes by a factor-of-eight-wrong amount.
        """
        grid, onsets = self._take(0.997, 11)
        a = dtw.assign_onsets_to_grid(onsets, grid, max_move_s=0.09)
        slope, _ = dtw.residual_drift(a)
        truth = -0.003
        self.assertLess(abs(slope), abs(truth) * 0.5,
                        "residual_drift unexpectedly saw the drift; if this "
                        "now works, the comment in estimate_tempo_ratio needs "
                        "revisiting")
        ratio, _, _ = dtw.estimate_tempo_ratio(onsets, grid)
        self.assertAlmostEqual(ratio, 0.997, places=3)


class TestSanitizeAnchors(unittest.TestCase):

    def test_unreachable_anchor_is_dropped_not_clamped(self):
        """Clamping and carrying the remainder forward displaced every later
        anchor; onsets landed a median 24.5 ms from the slot they were given.
        An anchor that cannot be honoured must be given up instead."""
        src = np.array([0.0, 1.0, 1.05, 2.0, 3.0])
        dst = np.array([0.0, 1.0, 1.40, 2.0, 3.0])       # 0.05 -> 0.40 s
        s, d, info = warp_mod.sanitize_anchors(src, dst, 3.0)
        self.assertGreaterEqual(info["ratio_clamped"], 1)
        for keep in (2.0, 3.0):
            k = int(np.argmin(np.abs(s - keep)))
            self.assertAlmostEqual(float(d[k]), keep, places=6)

    def test_output_is_monotonic_in_both_axes(self):
        rng = np.random.default_rng(1)
        src = np.sort(rng.uniform(0.2, 19.8, 200))
        dst = src + rng.normal(0, 0.05, 200)
        s, d, _ = warp_mod.sanitize_anchors(src, dst, 20.0)
        self.assertTrue(np.all(np.diff(s) > 0))
        self.assertTrue(np.all(np.diff(d) > 0))

    def test_ratios_stay_within_bounds(self):
        rng = np.random.default_rng(2)
        src = np.sort(rng.uniform(0.2, 19.8, 300))
        dst = src + rng.normal(0, 0.12, 300)
        s, d, _ = warp_mod.sanitize_anchors(src, dst, 20.0)
        r = warp_mod.anchor_ratios(s, d)
        self.assertGreaterEqual(float(r.min()), warp_mod.MIN_RATIO - 1e-9)
        self.assertLessEqual(float(r.max()), warp_mod.MAX_RATIO + 1e-9)

    def test_endpoints_are_pinned(self):
        s, d, _ = warp_mod.sanitize_anchors([1.0, 2.0], [1.1, 2.1], 4.0)
        self.assertAlmostEqual(float(s[0]), 0.0)
        self.assertAlmostEqual(float(d[0]), 0.0)
        self.assertAlmostEqual(float(s[-1]), 4.0)


class TestNeighbourLimiting(unittest.TestCase):

    def test_result_satisfies_the_warps_ratio_bounds(self):
        """A margin expressed as a fraction of the gap allowed local ratios
        up to 1.7 against a limit of 1.25, so the warp discarded those
        anchors and the onsets did not move at all."""
        rng = np.random.default_rng(4)
        onsets = np.sort(rng.uniform(0, 30, 200))
        move = rng.normal(0, 0.04, 200)
        limited = aligner._limit_by_neighbours(onsets, move)
        ratios = 1.0 + np.diff(limited) / np.diff(onsets)
        self.assertGreaterEqual(float(ratios.min()), warp_mod.MIN_RATIO - 1e-6)
        self.assertLessEqual(float(ratios.max()), warp_mod.MAX_RATIO + 1e-6)

    def test_only_ever_shrinks_moves(self):
        rng = np.random.default_rng(5)
        onsets = np.sort(rng.uniform(0, 30, 120))
        move = rng.normal(0, 0.05, 120)
        limited = aligner._limit_by_neighbours(onsets, move)
        self.assertLessEqual(float(np.abs(limited).sum()),
                             float(np.abs(move).sum()) + 1e-6)

    def test_legal_moves_pass_through_untouched(self):
        onsets = np.arange(0.0, 10.0, 0.5)
        move = np.full(onsets.size, 0.01)          # uniform shift, ratio 1.0
        limited = aligner._limit_by_neighbours(onsets, move)
        np.testing.assert_allclose(limited, move, atol=1e-9)


class TestSegmentWarpLength(unittest.TestCase):

    def test_identity_map_preserves_positions(self):
        """Laying segments end to end made every crossfade consume length.
        Ninety-three segments at 6 ms lost half a second, which read as 70 ms
        of onset displacement from a warp that changes nothing."""
        times = np.arange(0.4, 12.0, 0.31)
        y = click_train(times, duration=13.0)
        s = np.concatenate(([0.0], times, [13.0]))
        out = warp_mod._warp_segments(y[:, None], SR, s, s.copy())
        found = find_clicks(out)
        self.assertGreaterEqual(len(found), len(times) - 1)
        err = [float(np.min(np.abs(found - t))) for t in times]
        self.assertLess(float(np.median(err)) * 1000, 3.0)

    def test_output_length_follows_the_anchors(self):
        times = np.arange(0.4, 8.0, 0.4)
        y = click_train(times, duration=9.0)
        s = np.concatenate(([0.0], times, [9.0]))
        d = s * 1.1
        out = warp_mod._warp_segments(y[:, None], SR, s, d)
        self.assertAlmostEqual(len(out) / SR, 9.9, delta=0.1)


class TestAlignToGrid(unittest.TestCase):

    def test_declines_on_too_few_onsets(self):
        y = click_train([0.5, 1.0, 1.5])
        out, rep = aligner.align_to_grid(y[:, None], SR, [0.5, 1.0, 1.5],
                                         context(), strength=0.8)
        self.assertFalse(rep["enabled"])
        self.assertIn("onsets", rep["note"])

    def test_declines_on_zero_strength(self):
        ctx = context()
        times = ctx.target_grid(16)[4:60:2]
        y = click_train(times)
        out, rep = aligner.align_to_grid(y[:, None], SR, times, ctx, strength=0.0)
        self.assertFalse(rep["enabled"])

    def test_leaves_an_already_tight_take_alone(self):
        ctx = context()
        times = ctx.target_grid(16)[4:120:3]
        y = click_train(times)
        _, rep = aligner.align_to_grid(y[:, None], SR, times, ctx, strength=0.8)
        self.assertEqual(rep["moved"], 0)
        self.assertEqual(rep["below_threshold"], rep["assigned"])

    def test_removes_tempo_drift_and_lands_on_the_grid(self):
        ctx = context()
        grid = ctx.target_grid(16)
        rng = np.random.default_rng(11)
        sl = np.sort(rng.choice(np.arange(4, grid.size - 20), 120, replace=False))
        times = np.sort(grid[sl] * 1.005 + rng.normal(0, 0.014, sl.size))
        y = click_train(times, duration=float(times[-1]) + 2.0)

        before = dtw.grid_fit_error(times, grid)
        out, rep = aligner.align_to_grid(y[:, None], SR, times, ctx, strength=0.85)

        self.assertTrue(rep["enabled"])
        self.assertTrue(rep["drift_corrected"])
        self.assertAlmostEqual(rep["drift_slope"], 0.005, places=2)

        found = find_clicks(out)
        after = dtw.grid_fit_error(found, grid)
        self.assertLess(after, before * 0.5)
        self.assertGreaterEqual(len(found), int(len(times) * 0.95))

    def test_report_prediction_matches_the_rendered_audio(self):
        """A report that disagrees with the output is worse than no report:
        the repair loop acts on it."""
        ctx = context()
        grid = ctx.target_grid(16)
        rng = np.random.default_rng(21)
        sl = np.sort(rng.choice(np.arange(4, grid.size - 20), 110, replace=False))
        times = np.sort(grid[sl] * 1.004 + rng.normal(0, 0.016, sl.size))
        y = click_train(times, duration=float(times[-1]) + 2.0)
        out, rep = aligner.align_to_grid(y[:, None], SR, times, ctx, strength=0.9)
        measured = dtw.grid_fit_error(find_clicks(out), grid) * 1000
        self.assertLess(abs(measured - rep["error_after_ms"]), 8.0)


class TestDtwPath(unittest.TestCase):

    def test_diagonal_cost_matrix_gives_a_diagonal_path(self):
        n = 40
        cost = np.ones((n, n))
        np.fill_diagonal(cost, 0.0)
        rows, cols, _ = dtw_path_safe(cost)
        self.assertTrue(np.all(rows == cols))

    def test_path_is_monotonic(self):
        rng = np.random.default_rng(6)
        cost = rng.random((60, 70))
        rows, cols, _ = dtw_path_safe(cost)
        self.assertTrue(np.all(np.diff(rows) >= 0))
        self.assertTrue(np.all(np.diff(cols) >= 0))

    def test_no_more_than_one_consecutive_hold(self):
        """A path that parks on one source frame holds a phoneme, which on
        playback is a stutter rather than a timing error."""
        rng = np.random.default_rng(8)
        cost = rng.random((50, 90))
        rows, cols, _ = dtw_path_safe(cost, max_consecutive=1)
        for seq in (rows, cols):
            held = np.diff(seq) == 0
            run = 0
            for h in held:
                run = run + 1 if h else 0
                self.assertLessEqual(run, 1)


def dtw_path_safe(cost, **kw):
    return dtw.dtw_path(cost, gully=0.0, **kw)


if __name__ == "__main__":
    unittest.main(verbosity=2)
