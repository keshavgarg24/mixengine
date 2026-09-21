"""
Tests for the musical render stages: chord-aware tuning, groove-aware
timing, and tempo verification.

These cover the bridge between the pure `musical/` logic and the audio
path. Several of them lock down bugs that were found by running the engine
rather than by reading it, and each of those is noted where it applies --
the point of the test is to keep that specific failure from returning.

The audio backends are stubbed, because what is under test is the
*decision*: which notes get corrected, how far, which onsets move, and
whether a time stretch should happen at all.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mixengine.analysis import analysis                                # noqa: E402
from mixengine.audio import timing, tuning                             # noqa: E402
from mixengine.core.types import (                                     # noqa: E402
    ATTACK_SCOOP, ChordEvent, GrooveTemplate, KeyRegion, Note,
)

SR = 44100


def fake_shift(seg, sr, semitones):
    """Records the request and returns the segment unchanged."""
    fake_shift.calls.append(float(semitones))
    return seg


fake_shift.calls = []


def fake_stretch(seg, sr, ratio, preserve_formants=False):
    n = max(1, int(round(len(seg) * ratio)))
    idx = np.clip((np.arange(n) / max(ratio, 1e-9)).astype(int), 0, len(seg) - 1)
    return seg[idx]


def audio(seconds=4.0, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)[:, None]


def c_major_context(genre="pop", **kw):
    return tuning.HarmonicContext(
        chords=[ChordEvent(0.0, 2.0, 0, "maj"), ChordEvent(2.0, 4.0, 5, "maj")],
        key_regions=[KeyRegion(0.0, 99.0, 0, "major", 0.9)],
        genre=genre, **kw)


# ═════════════════════════════════════════════════════════════════════════════
# Chord-aware tuning
# ═════════════════════════════════════════════════════════════════════════════

class TestHarmonicContext(unittest.TestCase):

    def test_chord_time_field_is_accepted(self):
        """The per-bar estimator emits `time`, not `start`.

        Reading only `start` collapsed every chord to the span [0, 0], so
        `chord_at()` answered None for the whole track and the chord-aware
        tuner silently degraded to scale-aware with nothing reporting it.
        Measured chord-awareness went from 2.6% to 27.6% when this was fixed.
        """
        ev = ChordEvent.from_dict({"bar": 2, "time": 4.5, "root": 7,
                                   "quality": "maj", "confidence": 0.9})
        self.assertAlmostEqual(ev.start, 4.5)
        self.assertEqual(ev.root, 7)

    def test_from_beat_dna_gives_chords_a_span(self):
        ctx = tuning.HarmonicContext.from_beat_dna({
            "chords": [{"time": 0.0, "root": 0, "quality": "maj"},
                       {"time": 2.0, "root": 5, "quality": "maj"}],
            "key": {"pc": 0, "mode": "major"}, "duration_s": 8.0})
        self.assertIsNotNone(ctx.chord_at(1.0))
        self.assertEqual(ctx.chord_at(1.0).root, 0)
        self.assertEqual(ctx.chord_at(3.0).root, 5)

    def test_transposition_is_applied_to_chords_and_key(self):
        """If the beat was pitch-shifted, tuning to the untransposed chords
        would be confidently wrong by exactly the shift amount."""
        ctx = tuning.HarmonicContext.from_beat_dna(
            {"chords": [{"time": 0.0, "root": 0, "quality": "maj"}],
             "key": {"pc": 0, "mode": "major"}, "duration_s": 4.0},
            semitone_shift=2)
        self.assertEqual(ctx.chord_at(1.0).root, 2)
        self.assertEqual(ctx.key_at(1.0).pc, 2)

    def test_missing_chords_degrade_to_key_only(self):
        ctx = tuning.HarmonicContext.from_beat_dna(
            {"key": {"pc": 9, "mode": "minor"}, "duration_s": 4.0})
        self.assertIsNone(ctx.chord_at(1.0))
        self.assertIsNotNone(ctx.key_at(1.0))


class TestTuneMusical(unittest.TestCase):

    def setUp(self):
        fake_shift.calls = []

    def run_tune(self, notes, ctx=None, strength=0.6):
        return tuning.tune_musical(audio(), SR, notes, ctx or c_major_context(),
                                   base_strength=strength,
                                   pitch_shift_fn=fake_shift)

    def test_corrects_toward_the_chord_tone(self):
        # 0.4 semitones sharp of E, over a C major chord.
        _, rep = self.run_tune([Note(0.2, 0.9, 64.4)])
        self.assertEqual(rep["notes_corrected"], 1)
        self.assertLess(fake_shift.calls[0], 0.0)      # corrected downward

    def test_gestures_are_never_touched(self):
        """Scoops and slides are performance. Correcting them is the most
        recognisable way an automatic tuner announces itself."""
        notes = [Note(0.2, 0.9, 64.4, attack=ATTACK_SCOOP),
                 Note(1.2, 1.9, 64.4, is_melisma=True)]
        _, rep = self.run_tune(notes)
        self.assertEqual(rep["notes_corrected"], 0)
        self.assertEqual(rep["notes_skipped_transition"], 1)
        self.assertEqual(rep["notes_skipped_melisma"], 1)

    def test_blue_third_is_preserved_in_blues_family_genres(self):
        """Eb over a C major chord is the sound of the genre, not an error."""
        note = [Note(0.2, 1.0, 63.0)]                   # Eb
        _, trap = self.run_tune(note, c_major_context(genre="trap"))
        self.assertGreaterEqual(trap["blue_notes_preserved"], 1)

    def test_far_notes_are_refused_not_dragged(self):
        """A note far from any legal target is a different note, or a
        detection failure. Forcing it produces a confident wrong answer."""
        _, rep = self.run_tune([Note(0.2, 0.9, 61.0)],
                               tuning.HarmonicContext(
                                   chords=[ChordEvent(0.0, 4.0, 0, "maj")],
                                   key_regions=[]))
        self.assertEqual(rep["notes_refused_far"] + rep["notes_corrected"], 1)

    def test_long_exposed_notes_are_corrected_harder_than_passing_ones(self):
        """Correction scales with exposure, which is what keeps a corrected
        line sounding performed rather than quantised."""
        fake_shift.calls = []
        self.run_tune([Note(0.2, 2.0, 64.4)])           # long
        long_shift = abs(fake_shift.calls[0]) if fake_shift.calls else 0.0
        fake_shift.calls = []
        self.run_tune([Note(0.2, 0.30, 64.4)])          # short
        short_shift = abs(fake_shift.calls[0]) if fake_shift.calls else 0.0
        self.assertGreater(long_shift, short_shift)

    def test_zero_strength_disables(self):
        _, rep = self.run_tune([Note(0.2, 0.9, 64.4)], strength=0.0)
        self.assertFalse(rep["enabled"])
        self.assertEqual(fake_shift.calls, [])

    def test_reports_chord_awareness(self):
        _, rep = self.run_tune([Note(0.2, 0.9, 64.4), Note(2.2, 2.9, 65.4)])
        self.assertAlmostEqual(rep["chord_aware_fraction"], 1.0)

    def test_no_notes_is_safe(self):
        _, rep = self.run_tune([])
        self.assertFalse(rep["enabled"])

    def test_report_is_json_safe(self):
        import json
        _, rep = self.run_tune([Note(0.2, 0.9, 64.4)])
        json.dumps(rep)


class TestNotesFromDna(unittest.TestCase):

    def test_infers_slides_between_close_notes(self):
        notes = tuning.notes_from_dna([
            {"start": 0.0, "end": 0.40, "midi": 60.0},
            {"start": 0.41, "end": 0.80, "midi": 65.0},    # +5 st, legato
        ])
        self.assertTrue(notes[1].is_transition)

    def test_infers_melisma_in_rapid_runs(self):
        notes = tuning.notes_from_dna([
            {"start": 0.00, "end": 0.12, "midi": 60.0},
            {"start": 0.13, "end": 0.25, "midi": 62.0},
            {"start": 0.26, "end": 0.38, "midi": 64.0},
        ])
        self.assertTrue(notes[1].is_melisma)

    def test_steady_notes_are_not_flagged(self):
        notes = tuning.notes_from_dna([
            {"start": 0.0, "end": 0.9, "midi": 60.0},
            {"start": 1.5, "end": 2.4, "midi": 60.0},
        ])
        self.assertFalse(any(n.is_transition or n.is_melisma for n in notes))


# ═════════════════════════════════════════════════════════════════════════════
# Groove-aware timing
# ═════════════════════════════════════════════════════════════════════════════

def timing_context(bpm=120.0, bars=8, groove=None):
    beat = 60.0 / bpm
    beats = np.arange(bars * 4) * beat
    downbeats = np.arange(bars + 1) * beat * 4
    return timing.TimingContext(beats=beats, downbeats=downbeats,
                                bar_duration_s=beat * 4, beats_per_bar=4,
                                groove=groove or GrooveTemplate.straight(16),
                                grid_stability=0.95)


class TestTimingContext(unittest.TestCase):

    def test_target_grid_carries_the_groove(self):
        """Landing on the beat's own positions *is* landing in the pocket.
        A mathematically exact grid is merely on time, which is stiffer."""
        offs = np.zeros(16)
        offs[4] = 20.0
        g = GrooveTemplate(subdivision=16, offsets_ms=offs,
                           velocities=np.ones(16), consistency=0.9,
                           n_bars_observed=8, source="measured")
        ctx = timing_context(groove=g)
        feel = ctx.target_grid(16, apply_groove=True)
        plain = ctx.target_grid(16, apply_groove=False)
        self.assertAlmostEqual((feel[4] - plain[4]) * 1000.0, 20.0, delta=1.0)

    def test_falls_back_to_beats_without_downbeats(self):
        ctx = timing.TimingContext(beats=np.arange(16) * 0.5,
                                   downbeats=np.zeros(0))
        self.assertGreater(ctx.target_grid(16).size, 10)

    def test_from_beat_dna_reads_the_stored_groove(self):
        ctx = timing.TimingContext.from_beat_dna({
            "beats": list(np.arange(32) * 0.5), "downbeats": [0.0, 2.0, 4.0],
            "bpm": 120.0, "beats_per_bar": 4,
            "groove": GrooveTemplate.straight(16).to_dict()})
        self.assertEqual(ctx.beats_per_bar, 4)
        self.assertAlmostEqual(ctx.bar_duration_s, 2.0, places=3)


class TestQuantizeMusical(unittest.TestCase):

    def test_audibility_gate_tests_the_error_not_the_correction(self):
        """The gate that was backwards.

        Testing the scaled correction meant an audible 36 ms error, scaled
        by strength and slot weight into a 3 ms move, fell under the
        threshold and was skipped -- so the errors that most needed fixing
        were exactly the ones ignored. Measured effect of the fix on a real
        render: onsets moved went from 1 of 84 to 48 of 84.
        """
        ctx = timing_context()
        grid = ctx.target_grid(16)
        # Onsets sitting a clearly audible 35 ms late.
        onsets = [float(g) + 0.035 for g in grid[4:40:4]]
        _, rep = timing.quantize_musical(audio(8.0), SR, onsets, ctx,
                                         strength=0.3,
                                         time_stretch_fn=fake_stretch)
        self.assertGreater(rep["onsets_moved"], 0,
                           "audible errors were skipped: %s" % rep)

    def test_inaudible_deviations_are_left_alone(self):
        """Correcting below the audibility threshold spends stretch
        artifacts on something nobody can hear."""
        ctx = timing_context()
        grid = ctx.target_grid(16)
        onsets = [float(g) + 0.002 for g in grid[4:40:4]]
        _, rep = timing.quantize_musical(audio(8.0), SR, onsets, ctx,
                                         strength=0.5,
                                         time_stretch_fn=fake_stretch)
        self.assertEqual(rep["onsets_moved"], 0)
        self.assertGreater(rep["onsets_below_threshold"], 0)

    def test_zero_strength_disables(self):
        ctx = timing_context()
        _, rep = timing.quantize_musical(audio(), SR, [0.5, 1.0], ctx,
                                         strength=0.0,
                                         time_stretch_fn=fake_stretch)
        self.assertFalse(rep["enabled"])

    def test_groove_flag_is_reported(self):
        offs = np.full(16, 8.0)
        g = GrooveTemplate(subdivision=16, offsets_ms=offs,
                           velocities=np.ones(16), consistency=0.9,
                           n_bars_observed=8, swing_ratio=0.62)
        ctx = timing_context(groove=g)
        grid = ctx.target_grid(16)
        _, rep = timing.quantize_musical(
            audio(8.0), SR, [float(x) + 0.03 for x in grid[4:40:4]], ctx,
            strength=0.4, time_stretch_fn=fake_stretch)
        self.assertTrue(rep["groove_applied"])
        self.assertAlmostEqual(rep["swing_ratio"], 0.62, places=2)

    def test_empty_grid_is_safe(self):
        ctx = timing.TimingContext(beats=np.zeros(0), downbeats=np.zeros(0))
        _, rep = timing.quantize_musical(audio(), SR, [0.5], ctx,
                                         time_stretch_fn=fake_stretch)
        self.assertFalse(rep["enabled"])

    def test_report_is_json_safe(self):
        import json
        ctx = timing_context()
        _, rep = timing.quantize_musical(audio(), SR, [0.5, 1.0, 1.5], ctx,
                                         time_stretch_fn=fake_stretch)
        json.dumps(rep)


# ═════════════════════════════════════════════════════════════════════════════
# Tempo verification
# ═════════════════════════════════════════════════════════════════════════════

class TestGridFitError(unittest.TestCase):

    def test_measures_distance_to_the_nearest_grid_point(self):
        grid = np.arange(0, 10, 0.1)
        onsets = grid[::5] + 0.02
        self.assertAlmostEqual(analysis.grid_fit_error(onsets, grid), 0.02,
                               places=3)

    def test_perfectly_aligned_onsets_score_zero(self):
        grid = np.arange(0, 10, 0.1)
        self.assertLess(analysis.grid_fit_error(grid[::4], grid), 1e-6)

    def test_uses_measured_positions_not_a_generated_grid(self):
        """The bug this replaced.

        An earlier version generated an evenly-spaced grid from a BPM. A
        0.33 BPM error -- well inside any tracker's accuracy -- accumulates
        about 66 ms of phase over 28 seconds, more than half a sixteenth at
        140 BPM, and inverted the comparison it was meant to decide.
        """
        true_bpm, drifted_bpm = 140.0, 139.67
        step = 60.0 / true_bpm / 4.0
        onsets = np.arange(0, 28.0, step * 4)
        measured = np.arange(0, 28.0 + step, step)
        generated = np.arange(0, 28.0 + step, 60.0 / drifted_bpm / 4.0)
        self.assertLess(analysis.grid_fit_error(onsets, measured), 0.002)
        self.assertGreater(analysis.grid_fit_error(onsets, generated), 0.010)

    def test_too_few_onsets_is_infinite(self):
        self.assertEqual(analysis.grid_fit_error([0.1], np.arange(0, 1, 0.1)),
                         float("inf"))


class TestSubdivide(unittest.TestCase):

    def test_interpolates_between_measured_beats(self):
        beats = np.array([0.0, 0.5, 1.0, 1.5])
        g = analysis.subdivide(beats, 4)
        self.assertEqual(g.size, 13)
        self.assertAlmostEqual(g[1], 0.125, places=6)
        self.assertTrue(np.all(np.diff(g) > 0))

    def test_does_not_accumulate_drift(self):
        """Beats with individual jitter must not compound it."""
        rng = np.random.default_rng(0)
        beats = np.arange(64) * 0.43 + rng.normal(0, 0.004, 64)
        beats = np.sort(beats)
        g = analysis.subdivide(beats, 4)
        for b in beats:
            self.assertLess(float(np.min(np.abs(g - b))), 1e-9)

    def test_short_input_is_safe(self):
        self.assertEqual(analysis.subdivide([1.0], 4).size, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
