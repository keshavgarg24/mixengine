"""
Where the vocal starts in the beat, and what the mixer receives.

Two renders of real takes showed the same three faults: the vocal was
laid on the right bar line but at the top of the beat, so the drop hit
mid-verse; the downbeat tracker, free to pick any tempo, counted a rap
take at half time and the take was never stretched; and the mixer was
handed the noisy original while the analysis had been made on the
restored one. The tests below pin each of those down without audio.
"""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mixengine.analysis import analysis                             # noqa: E402
from mixengine.audio import separation, transform                   # noqa: E402
from mixengine.core import audio_io                                 # noqa: E402

BAR = 2.4
SR = 1000


class TestTempoWindow(unittest.TestCase):

    def test_window_sits_around_the_beats_tempo(self):
        lo, hi = transform._tempo_window(1.6, 4)          # 150 BPM
        self.assertAlmostEqual(lo, 138.0, places=3)
        self.assertAlmostEqual(hi, 162.0, places=3)

    def test_a_count_the_tracker_cannot_resolve_is_skipped(self):
        self.assertIsNone(transform._tempo_window(1.6, 8))   # 300 BPM
        self.assertIsNone(transform._tempo_window(0.0, 4))

    def test_the_double_count_of_a_slow_beat_is_kept(self):
        self.assertIsNotNone(transform._tempo_window(2.5, 8))  # 192 BPM


def _sections(*spans):
    return [{"start": a, "end": z, "label": label} for a, z, label in spans]


class TestPlaceAtSection(unittest.TestCase):

    grid = np.arange(0.0, 60.0, BAR)

    def vocal(self, seconds):
        return np.ones((int(seconds * SR), 1), dtype=np.float32)

    def test_the_vocal_moves_whole_bars_to_the_drop(self):
        secs = _sections((0.0, 9.6, "intro"), (9.6, 40.0, "chorus"),
                         (40.0, 60.0, "outro"))
        v, info = transform.place_at_section(self.vocal(20), SR, [(100, 5000)],
                                             self.grid, secs)
        self.assertEqual(info["method"], "section_entry")
        self.assertEqual(info["moved_bars"], 4)
        self.assertAlmostEqual(info["moved_s"], 9.6, places=3)
        self.assertEqual(len(v), 20 * SR + int(9.6 * SR))
        self.assertEqual(float(np.abs(v[:int(9.6 * SR)]).max()), 0.0)

    def test_a_break_before_the_drop_is_skipped_too(self):
        secs = _sections((0.0, 7.2, "intro"), (7.2, 9.6, "break"),
                         (9.6, 40.0, "chorus"))
        _, info = transform.place_at_section(self.vocal(20), SR, [(100, 5000)],
                                             self.grid, secs)
        self.assertEqual(info["section"], "chorus")
        self.assertEqual(info["moved_bars"], 4)

    def test_no_sections_means_no_move(self):
        v, info = transform.place_at_section(self.vocal(20), SR, [(100, 5000)],
                                             self.grid, None)
        self.assertEqual(info["method"], "none")
        self.assertEqual(len(v), 20 * SR)

    def test_a_beat_with_no_intro_leaves_the_vocal_alone(self):
        secs = _sections((0.0, 40.0, "chorus"), (40.0, 60.0, "outro"))
        v, info = transform.place_at_section(self.vocal(20), SR, [(100, 5000)],
                                             self.grid, secs)
        self.assertEqual(info["moved_bars"], 0)
        self.assertEqual(len(v), 20 * SR)

    def test_a_long_intro_is_not_imposed_on_the_listener(self):
        secs = _sections((0.0, 48.0, "intro"), (48.0, 60.0, "chorus"))
        v, info = transform.place_at_section(self.vocal(20), SR, [(100, 5000)],
                                             self.grid, secs)
        self.assertEqual(info["method"], "kept")
        self.assertEqual(len(v), 20 * SR)

    def test_leading_silence_is_trimmed_to_reach_the_drop(self):
        secs = _sections((0.0, 9.6, "intro"), (9.6, 60.0, "chorus"))
        first = int(30.1 * SR)
        v, info = transform.place_at_section(self.vocal(40), SR, [(first, first + 900)],
                                             self.grid, secs)
        self.assertEqual(info["moved_bars"], -9)
        self.assertEqual(len(v), 40 * SR - int(21.6 * SR))

    def test_never_cuts_into_the_first_phrase(self):
        secs = _sections((0.0, 40.0, "chorus"), (40.0, 60.0, "outro"))
        first = int(1.3 * SR)                       # nearest bar line is 2.4
        v, info = transform.place_at_section(self.vocal(20), SR, [(first, first + 900)],
                                             self.grid, secs)
        self.assertEqual(info["method"], "kept")
        self.assertEqual(len(v), 20 * SR)


class TestSeparatorDenoise(unittest.TestCase):
    """The separator runs as a denoiser only where subtraction cannot cope,
    and never on a take that already came out of it."""

    sr = 8000

    def setUp(self):
        rng = np.random.default_rng(1)
        self.noisy = (rng.normal(0, 0.03, (self.sr * 2, 1))).astype(np.float32)
        t = np.arange(self.sr * 2) / self.sr
        self.tone = (0.1 * np.sin(2 * np.pi * 220 * t))[:, None].astype(np.float32)

    def fake_separate(self, calls):
        def _separate(path, out_dir, want="all", model=None):
            calls.append(want)
            stem = os.path.join(out_dir, "vocals.wav")
            audio_io.save(stem, self.tone, self.sr)
            return {"vocals": stem}
        return _separate

    def condition(self, snr_db, separated=False, can_separate=True, fake=None):
        calls = []
        quality = SimpleNamespace(snr_db=snr_db, estimated_rt60_s=0.1,
                                  noise_floor_db=-30.0)
        with mock.patch.object(separation, "CAPS",
                               SimpleNamespace(can_separate=can_separate)), \
                mock.patch.object(separation, "separate",
                                  fake or self.fake_separate(calls)):
            out, report = separation.condition_vocal(self.noisy, self.sr, quality,
                                                     separated=separated)
        return out, report, calls

    def test_a_take_the_noise_nearly_covers_goes_through_the_separator(self):
        out, report, calls = self.condition(6.0)
        self.assertEqual(calls, ["vocals"])
        self.assertTrue(report["separator_denoise"])
        self.assertEqual(len(out), len(self.noisy))
        # What comes back is the stem, not the input.
        c = np.corrcoef(out[:, 0], self.tone[:, 0])[0, 1]
        self.assertGreater(c, 0.9)

    def test_a_take_that_was_already_separated_is_not_separated_again(self):
        _, report, calls = self.condition(6.0, separated=True)
        self.assertEqual(calls, [])
        self.assertNotIn("separator_denoise", report)

    def test_ordinary_noise_stays_with_subtraction(self):
        _, report, calls = self.condition(20.0)
        self.assertEqual(calls, [])
        self.assertTrue(report["denoise"])

    def test_no_stem_back_falls_through_to_subtraction(self):
        out, report, _ = self.condition(6.0, fake=lambda *a, **k: {})
        self.assertNotIn("separator_denoise", report)
        self.assertTrue(report["denoise"])
        self.assertTrue(np.all(np.isfinite(out)))

    def test_without_a_separator_nothing_changes(self):
        _, report, calls = self.condition(6.0, can_separate=False)
        self.assertEqual(calls, [])
        self.assertNotIn("separator_denoise", report)


if __name__ == "__main__":
    unittest.main()


class TestBarAnchor(unittest.TestCase):
    """Bar-ones re-counted from the drop where the tracker lost a beat.

    A 150 BPM grid; the tracker stretches the six drum-less beats before
    12.8 s into five, so its bar-ones after that point sit one beat after
    the real bar lines -- exactly what happened on a real trap beat.
    """

    SR = 2000
    STEP = 0.4

    def _audio(self, drop_s):
        rng = np.random.default_rng(0)
        y = rng.standard_normal(int(40 * self.SR)).astype(np.float32) * 0.01
        if drop_s is not None:
            y[int(drop_s * self.SR):] *= 30.0
        return y

    def _steady(self):
        beats = np.arange(0.0, 40.0, self.STEP)
        return beats, beats[::4]

    def _slipped(self):
        real = np.arange(0.0, 40.0, self.STEP)
        beats = np.concatenate([real[real < 10.4],
                                np.linspace(10.4, 12.8, 6)[:-1],
                                real[real >= 12.8]])
        return beats, beats[::4]

    def test_a_slipped_tracker_counts_bars_from_the_drop(self):
        beats, downs = self._slipped()
        self.assertAlmostEqual(downs[downs > 12.8][0], 13.2, places=6)
        new, info = analysis._anchor_bars_to_drops(self._audio(12.8), self.SR,
                                                   beats, downs, 4)
        self.assertEqual(info["anchored"], [12.8])
        after = new[new >= 12.8]
        np.testing.assert_allclose(after, 12.8 + 1.6 * np.arange(after.size),
                                   atol=1e-6)
        np.testing.assert_allclose(new[new < 12.6], downs[downs < 12.6])

    def test_a_steady_tracker_is_trusted(self):
        beats, downs = self._steady()
        new, info = analysis._anchor_bars_to_drops(self._audio(12.8), self.SR,
                                                   beats, downs, 4)
        np.testing.assert_allclose(new, downs)
        self.assertEqual(info, {"slips": 0, "anchored": []})

    def test_a_pickup_before_a_correct_bar_line_moves_nothing(self):
        """The drop arrives a beat early but the tracker never slipped."""
        beats, downs = self._steady()
        new, _ = analysis._anchor_bars_to_drops(self._audio(12.4), self.SR,
                                                beats, downs, 4)
        np.testing.assert_allclose(new, downs)

    def test_a_slip_with_no_drop_after_it_is_left_alone(self):
        beats, downs = self._slipped()
        new, info = analysis._anchor_bars_to_drops(self._audio(None), self.SR,
                                                   beats, downs, 4)
        np.testing.assert_allclose(new, downs)
        self.assertEqual(info, {"slips": 1, "anchored": []})

    def test_a_drop_already_on_a_bar_line_is_kept(self):
        beats, downs = self._slipped()
        new, info = analysis._anchor_bars_to_drops(self._audio(13.2), self.SR,
                                                   beats, downs, 4)
        np.testing.assert_allclose(new, downs)
        self.assertEqual(info["anchored"], [])
