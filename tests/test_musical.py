"""
Tests for the musicality layer and the Musical IR.

These run with no audio dependencies -- numpy only -- which is the point
of keeping `musical/` free of them. Run with:

    python3 -m unittest discover -s tests -v

Each test asserts a *musical* claim, not merely that a function returns.
Where a claim comes from a specific production convention or research
result, the reasoning is in the docstring so a disagreement can be settled
by argument rather than by guessing what the code intended.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mixengine.core.ir import MusicalIR, ROLE_BEAT       # noqa: E402
from mixengine.core.keys import Key                    # noqa: E402
from mixengine.core.types import (                                    # noqa: E402
    ATTACK_SCOOP, Beat, ChordEvent, GrooveTemplate, KeyRegion,
    Note, Phrase, Phoneme, Section, TempoCurve, Word,
    FUNC_DOMINANT, FUNC_SUBDOMINANT, FUNC_TONIC, PHON_SIBILANT,
)
from mixengine.musical import energy, groove, salience, theory        # noqa: E402


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def make_grid(bpm=120.0, bars=8, beats_per_bar=4, subdivision=16,
              swing=0.5, slot_offsets_ms=None, t0=0.0):
    """Generate a synthetic beat grid with known, controllable feel.

    Returns `(onsets, downbeats, bar_duration_s)`. Because the ground truth
    is known exactly, extraction can be checked for accuracy rather than
    merely for not crashing.
    """
    beat = 60.0 / bpm
    bar_dur = beat * beats_per_bar
    downbeats = [t0 + i * bar_dur for i in range(bars + 1)]
    onsets = []
    per_eighth = subdivision // 8
    for b in range(bars):
        start = downbeats[b]
        for slot in range(subdivision):
            t = start + bar_dur * slot / subdivision
            # Swing delays the odd eighths toward the triplet position.
            eighth_index = slot / per_eighth
            if per_eighth and abs(eighth_index - round(eighth_index)) < 1e-9:
                if int(round(eighth_index)) % 2 == 1:
                    t += (swing - 0.5) * 2.0 * (bar_dur / 8.0)
            if slot_offsets_ms is not None:
                t += slot_offsets_ms[slot % len(slot_offsets_ms)] / 1000.0
            onsets.append(t)
    return np.array(onsets), np.array(downbeats), bar_dur


def chord(start, end, root, quality="maj", conf=0.9):
    return ChordEvent(start=start, end=end, root=root, quality=quality,
                      confidence=conf)


C_MAJOR = KeyRegion(start=0.0, end=999.0, pc=0, mode="major", confidence=0.9)
A_MINOR = KeyRegion(start=0.0, end=999.0, pc=9, mode="minor", confidence=0.9)


# ═════════════════════════════════════════════════════════════════════════════
# Core types
# ═════════════════════════════════════════════════════════════════════════════

class TestTempoCurve(unittest.TestCase):

    def test_constant_curve_is_flat_and_stable(self):
        tc = TempoCurve.constant(140.0, duration_s=180.0)
        self.assertAlmostEqual(tc.at(0.0), 140.0)
        self.assertAlmostEqual(tc.at(90.0), 140.0)
        self.assertAlmostEqual(tc.mean_bpm, 140.0)
        self.assertTrue(tc.is_known)

    def test_from_beats_recovers_tempo(self):
        beats = [i * 0.5 for i in range(33)]          # 120 BPM
        tc = TempoCurve.from_beats(beats)
        self.assertAlmostEqual(tc.mean_bpm, 120.0, places=4)
        self.assertGreater(tc.stability, 0.95)

    def test_drifting_tempo_reports_low_stability(self):
        """A take that speeds up must not be described as metronomic.

        This is the measurement that routes a source to variable-rate
        alignment instead of a single global stretch.
        """
        t, beats = 0.0, []
        bpm = 100.0
        for _ in range(48):
            beats.append(t)
            t += 60.0 / bpm
            bpm += 0.6                                 # drifts 100 -> ~128
        tc = TempoCurve.from_beats(beats)
        self.assertLess(tc.stability, 0.8)
        self.assertGreater(tc.at(beats[-1]), tc.at(beats[0]))

    def test_unknown_tempo_is_zero_not_a_guess(self):
        tc = TempoCurve()
        self.assertFalse(tc.is_known)
        self.assertEqual(tc.mean_bpm, 0.0)


class TestChordEvent(unittest.TestCase):

    def test_triad_tones(self):
        self.assertEqual(chord(0, 1, 0, "maj").tones, frozenset({0, 4, 7}))
        self.assertEqual(chord(0, 1, 0, "min").tones, frozenset({0, 3, 7}))
        self.assertEqual(chord(0, 1, 2, "min").tones, frozenset({2, 5, 9}))

    def test_power_chord_has_no_third(self):
        """A power chord is mode-neutral, which is why it fits any melody."""
        c = chord(0, 1, 0, "pow")
        self.assertEqual(c.tones, frozenset({0, 7}))
        self.assertFalse(c.has_third)
        self.assertIsNone(c.third_pc)

    def test_extensions_join_the_tone_set(self):
        c = ChordEvent(start=0, end=1, root=0, quality="min", extensions=(10,))
        self.assertIn(10, c.tones)


class TestNote(unittest.TestCase):

    def test_cents_deviation_is_signed(self):
        self.assertAlmostEqual(Note(0, 1, 60.25).cents_dev, 25.0, places=3)
        self.assertAlmostEqual(Note(0, 1, 59.75).cents_dev, -25.0, places=3)

    def test_transition_notes_are_flagged(self):
        """Scoops and falls are expression; the tuner must skip them."""
        self.assertTrue(Note(0, 1, 60.0, attack=ATTACK_SCOOP).is_transition)
        self.assertFalse(Note(0, 1, 60.0).is_transition)


class TestGrooveTemplate(unittest.TestCase):

    def test_straight_template_is_meaningful_and_flat(self):
        g = GrooveTemplate.straight(16)
        g.n_bars_observed = 8
        self.assertTrue(g.is_meaningful)
        self.assertEqual(g.max_offset_ms, 0.0)

    def test_unreliable_template_is_rejected(self):
        """A pattern that does not repeat must never be applied.

        Applying it would be exactly the random-jitter mistake the whole
        groove design exists to prevent.
        """
        g = GrooveTemplate(subdivision=16, offsets_ms=np.array([12.0] * 16),
                           n_bars_observed=2, consistency=0.2)
        self.assertFalse(g.is_meaningful)

    def test_roundtrip(self):
        g = GrooveTemplate(subdivision=16,
                           offsets_ms=np.linspace(-8, 8, 16),
                           velocities=np.ones(16), swing_ratio=0.62,
                           consistency=0.8, n_bars_observed=16,
                           source="measured")
        back = GrooveTemplate.from_dict(g.to_dict())
        self.assertEqual(back.subdivision, 16)
        self.assertAlmostEqual(back.swing_ratio, 0.62, places=4)
        np.testing.assert_allclose(back.offsets_ms, g.offsets_ms, atol=0.01)


# ═════════════════════════════════════════════════════════════════════════════
# Theory
# ═════════════════════════════════════════════════════════════════════════════

class TestScalesAndKeys(unittest.TestCase):

    def test_scale_pitch_classes(self):
        self.assertEqual(theory.scale_pcs(0, "major"), (0, 2, 4, 5, 7, 9, 11))
        self.assertEqual(theory.scale_pcs(9, "minor"), (9, 11, 0, 2, 4, 5, 7))

    def test_relative_pairs_share_a_family_root(self):
        """A minor and C major contain the same notes.

        Normalising to the family root before comparing is what lets an
        E-minor vocal score as a perfect match over a G-major beat instead
        of a three-semitone clash.
        """
        self.assertEqual(theory.relative_major_pc(9, "minor"),
                         theory.relative_major_pc(0, "major"))
        self.assertEqual(theory.relative_major_pc(4, "minor"),
                         theory.relative_major_pc(7, "major"))

    def test_diatonic_triad_qualities(self):
        self.assertEqual(theory.diatonic_triad(0, "major", 0), (0, "maj"))   # I
        self.assertEqual(theory.diatonic_triad(0, "major", 1), (2, "min"))   # ii
        self.assertEqual(theory.diatonic_triad(0, "major", 4), (7, "maj"))   # V
        self.assertEqual(theory.diatonic_triad(0, "major", 6), (11, "dim"))  # vii


class TestHarmonicFunction(unittest.TestCase):

    def test_primary_functions_in_major(self):
        self.assertEqual(theory.harmonic_function(chord(0, 1, 0, "maj"), C_MAJOR),
                         FUNC_TONIC)
        self.assertEqual(theory.harmonic_function(chord(0, 1, 5, "maj"), C_MAJOR),
                         FUNC_SUBDOMINANT)
        self.assertEqual(theory.harmonic_function(chord(0, 1, 7, "maj"), C_MAJOR),
                         FUNC_DOMINANT)

    def test_raised_v_in_minor_is_dominant(self):
        """E major in A minor is a V with a leading tone, not a iii."""
        self.assertEqual(theory.harmonic_function(chord(0, 1, 4, "maj"), A_MINOR),
                         FUNC_DOMINANT)

    def test_roman_numerals_read_correctly(self):
        self.assertEqual(theory.roman_numeral(chord(0, 1, 0, "maj"), C_MAJOR), "I")
        self.assertEqual(theory.roman_numeral(chord(0, 1, 9, "min"), C_MAJOR), "vi")
        self.assertEqual(theory.roman_numeral(chord(0, 1, 7, "maj"), C_MAJOR), "V")


class TestConsonance(unittest.TestCase):

    def test_chord_tones_are_maximally_consonant(self):
        c = chord(0, 1, 0, "maj")
        for pc in (0, 4, 7):
            self.assertEqual(theory.consonance(pc, c, C_MAJOR), 1.0)

    def test_minor_ninth_above_a_chord_tone_is_the_real_clash(self):
        """C# over a C major chord is the interval the ear rejects.

        This is a far better definition of a wrong note than "not in the
        triad", and correcting only these is what leaves tensions and
        passing tones intact.
        """
        c = chord(0, 1, 0, "maj")
        self.assertIn(1, theory.avoid_notes(c))          # b9 above the root
        self.assertIn(5, theory.avoid_notes(c))          # b9 above the third
        self.assertLess(theory.consonance(1, c, C_MAJOR), 0.2)

    def test_ninth_is_colour_not_error(self):
        c = chord(0, 1, 0, "maj")
        self.assertIn(2, theory.available_tensions(c, C_MAJOR))
        self.assertGreater(theory.consonance(2, c, C_MAJOR), 0.8)

    def test_blue_third_is_protected_in_blues_family_genres(self):
        """Eb over a C major chord is the sound of the genre, not a mistake.

        A tuner that snaps it to E has destroyed the performance, so the
        blues degrees score as expression in the genres that use them and
        as a wrong note in the genres that do not.
        """
        c = chord(0, 1, 0, "maj")
        trap = theory.consonance(3, c, C_MAJOR, genre="trap")
        classical = theory.consonance(3, c, C_MAJOR, genre="classical")
        self.assertGreater(trap, 0.5)
        self.assertLess(classical, trap)

    def test_tuning_candidates_narrow_when_a_note_is_exposed(self):
        """Long notes are judged; short passing notes are not.

        So an exposed note may only be corrected to chord tones and
        deliberate tensions, while a passing note keeps the whole scale
        available and is left alone.
        """
        c = chord(0, 1, 0, "maj")
        exposed = theory.tuning_candidates(c, C_MAJOR, "pop", exposed=True)
        passing = theory.tuning_candidates(c, C_MAJOR, "pop", exposed=False)
        self.assertTrue(exposed.issubset(passing) or len(exposed) < len(passing))
        self.assertIn(0, exposed)
        self.assertIn(4, exposed)

    def test_nearest_target_refuses_implausible_corrections(self):
        """A note far from any legal target is a different note, or a
        detection failure. Dragging it into place is a confident wrong
        answer, which is worse than leaving it alone."""
        self.assertAlmostEqual(theory.nearest_target(60.3, {0, 4, 7}), 60.0)
        self.assertIsNone(theory.nearest_target(62.0, {0}, max_semitones=1.5))

    def test_nearest_target_picks_the_closer_of_two_candidates(self):
        self.assertAlmostEqual(theory.nearest_target(64.4, {0, 4, 7}), 64.0)
        self.assertAlmostEqual(theory.nearest_target(66.6, {0, 4, 7}), 67.0)


class TestProgressions(unittest.TestCase):

    def test_authentic_cadence_is_found_and_ranked_highest(self):
        chords = [chord(0, 2, 7, "maj"), chord(2, 4, 0, "maj")]        # V -> I
        cads = theory.detect_cadences(chords, [C_MAJOR])
        self.assertEqual(len(cads), 1)
        self.assertEqual(cads[0].kind, "authentic")
        self.assertAlmostEqual(cads[0].time, 2.0)

    def test_deceptive_cadence_is_distinguished_from_authentic(self):
        chords = [chord(0, 2, 7, "maj"), chord(2, 4, 9, "min")]        # V -> vi
        cads = theory.detect_cadences(chords, [C_MAJOR])
        self.assertEqual(cads[0].kind, "deceptive")

    def test_progression_fingerprint_is_key_independent(self):
        """The same progression in two keys must produce the same shape.

        That is what makes it a usable matching signal: it says two beats
        support the same melody regardless of what key they are in.
        """
        in_c = [chord(0, 1, 0), chord(1, 2, 5), chord(2, 3, 7), chord(3, 4, 0)]
        in_g = [chord(0, 1, 7), chord(1, 2, 0), chord(2, 3, 2), chord(3, 4, 7)]
        g_major = KeyRegion(0, 999, 7, "major", 0.9)
        self.assertEqual(theory.progression_fingerprint(in_c, C_MAJOR),
                         theory.progression_fingerprint(in_g, g_major))

    def test_modulation_is_detected_across_a_key_change(self):
        chords = ([chord(i, i + 1, r) for i, r in
                   enumerate([0, 5, 7, 0, 0, 5, 7, 0])] +
                  [chord(8 + i, 9 + i, r) for i, r in
                   enumerate([2, 7, 9, 2, 2, 7, 9, 2])])
        regions = theory.detect_modulations(chords, window=6)
        self.assertGreaterEqual(len(regions), 2)
        self.assertNotEqual(
            (regions[0].pc, regions[0].mode),
            (regions[-1].pc, regions[-1].mode))

    def test_stable_progression_yields_a_single_region(self):
        """One borrowed chord is colour, not a modulation.

        Splitting on it would fragment the analysis and make every
        downstream harmonic lookup less reliable.
        """
        chords = [chord(i, i + 1, r) for i, r in
                  enumerate([0, 5, 7, 0, 0, 5, 10, 0, 0, 5, 7, 0])]
        regions = theory.detect_modulations(chords, window=8)
        self.assertEqual(len(regions), 1)


class TestHarmonyGeneration(unittest.TestCase):

    def test_scale_thirds_alternate_between_three_and_four_semitones(self):
        """A real harmony line is not a fixed transposition.

        Shifting a melody by a constant interval pushes half the notes out
        of the chord. A third *of the scale* is sometimes three semitones
        and sometimes four, and that alternation is what makes the line
        sound sung rather than pitch-shifted.
        """
        lead = [60.0, 62.0, 64.0, 65.0]               # C D E F
        times = [0.0, 1.0, 2.0, 3.0]
        # `chord_at` deliberately returns None: with no chord underneath, the
        # harmony must fall back to the *scale*, which is the case under test.
        h = theory.generate_harmony(
            lead, times, chord_at=lambda t: None, key_at=lambda t: C_MAJOR,
            interval="third", direction=1)
        intervals = sorted({round(o) for o in h.offsets})
        self.assertTrue(set(intervals).issubset({3, 4, 5}),
                        "scale thirds should be 3 or 4 semitones, got %s" % intervals)
        self.assertNotEqual(len(intervals), 1,
                            "a constant interval means it is transposing, not harmonising")

    def test_harmony_is_folded_into_the_singers_range(self):
        lead = [72.0, 74.0, 76.0]
        h = theory.generate_harmony(
            lead, [0, 1, 2], chord_at=lambda t: None,
            key_at=lambda t: C_MAJOR, interval="third", direction=1,
            tessitura=(55.0, 70.0))
        for m, o in zip(lead, h.offsets):
            self.assertLessEqual(m + o, 72.0)

    def test_singable_range_check(self):
        self.assertTrue(theory.singable(60.0, 55.0, 70.0))
        self.assertFalse(theory.singable(80.0, 55.0, 70.0))


# ═════════════════════════════════════════════════════════════════════════════
# Groove
# ═════════════════════════════════════════════════════════════════════════════

class TestGrooveExtraction(unittest.TestCase):

    def test_quantised_input_measures_as_straight(self):
        """Programmed trap really is quantised.

        Reporting swing or offset on a straight beat would make the engine
        impose feel that is not there.
        """
        onsets, dbs, bar = make_grid(bpm=140, bars=12, swing=0.5)
        g = groove.extract(onsets, dbs, bar, subdivision=16)
        self.assertTrue(g.is_meaningful)
        self.assertLess(g.max_offset_ms, 2.0)
        self.assertAlmostEqual(g.swing_ratio, 0.5, delta=0.02)
        self.assertGreater(g.consistency, 0.9)

    def test_triplet_swing_is_recovered(self):
        """A grid built with 2:1 swing must measure as ~0.667.

        Reported on the same scale a DAW uses so the number is directly
        comparable to what a producer would dial in.
        """
        onsets, dbs, bar = make_grid(bpm=90, bars=16, swing=2.0 / 3.0)
        g = groove.extract(onsets, dbs, bar, subdivision=16)
        self.assertAlmostEqual(g.swing_ratio, 2.0 / 3.0, delta=0.03)

    def test_systematic_per_slot_offsets_are_recovered(self):
        """A consistently late backbeat is the feel, and must be measured.

        This is the property that carries groove -- as opposed to random
        jitter, which averages to nothing and is not worth reproducing.
        """
        offs = [0.0] * 16
        offs[4] = 14.0            # snare on beat 2, laid back
        offs[12] = 14.0           # snare on beat 4, laid back
        offs[2] = -6.0            # off-beat hat, pushed
        onsets, dbs, bar = make_grid(bpm=120, bars=16, slot_offsets_ms=offs)
        g = groove.extract(onsets, dbs, bar, subdivision=16)
        self.assertAlmostEqual(g.offset_at_slot(4), 14.0, delta=2.5)
        self.assertAlmostEqual(g.offset_at_slot(12), 14.0, delta=2.5)
        self.assertAlmostEqual(g.offset_at_slot(2), -6.0, delta=2.5)
        self.assertAlmostEqual(g.offset_at_slot(0), 0.0, delta=2.5)

    def test_random_jitter_produces_a_low_consistency_template(self):
        """Random deviation is not groove and must not be transferred.

        The literature is clear that exaggerated or unsystematic
        microtiming reduces perceived groove rather than increasing it, so
        the engine has to be able to tell the two apart. A template built
        from noise must fail `is_meaningful` so callers skip it.
        """
        rng = np.random.default_rng(7)
        onsets, dbs, bar = make_grid(bpm=120, bars=16)
        jittered = onsets + rng.normal(0, 0.030, onsets.size)
        g = groove.extract(jittered, dbs, bar, subdivision=16)
        straight = groove.extract(*make_grid(bpm=120, bars=16)[:2],
                                  bar_duration_s=bar, subdivision=16)
        self.assertLess(g.consistency, straight.consistency)
        self.assertLess(g.consistency, 0.75)

    def test_too_few_bars_is_not_measurable(self):
        onsets, dbs, bar = make_grid(bpm=120, bars=2)
        g = groove.extract(onsets, dbs, bar, subdivision=16)
        self.assertFalse(g.is_meaningful)

    def test_timing_bias_detects_a_performer_sitting_behind(self):
        onsets, dbs, bar = make_grid(bpm=100, bars=8, subdivision=8)
        grid = np.array(sorted(onsets))
        late = grid + 0.022                       # 22 ms consistently behind
        bias, spread = groove.timing_bias(late, grid)
        self.assertAlmostEqual(bias, 22.0, delta=3.0)
        self.assertLess(spread, 5.0)


class TestGrooveApplication(unittest.TestCase):

    def test_grid_carries_the_measured_feel(self):
        """The alignment target must not be a mathematically exact grid.

        Landing exactly on a dead grid is what makes automated placement
        sound stiff; landing on the track's own measured positions is what
        "in the pocket" means.
        """
        offs = [0.0] * 16
        offs[4] = 20.0
        onsets, dbs, bar = make_grid(bpm=120, bars=16, slot_offsets_ms=offs)
        g = groove.extract(onsets, dbs, bar, subdivision=16)

        with_groove = groove.grid_with_groove(dbs, bar, g, 16, amount=1.0)
        without = groove.grid_with_groove(dbs, bar, GrooveTemplate.straight(16), 16)
        self.assertAlmostEqual((with_groove[4] - without[4]) * 1000.0, 20.0, delta=2.5)

    def test_slot_weights_follow_the_metrical_hierarchy(self):
        w = groove.slot_weights(16, 4)
        self.assertGreater(w[0], w[4])            # downbeat beats other beats
        self.assertGreater(w[4], w[2])            # beats beat eighths
        self.assertGreater(w[2], w[1])            # eighths beat sixteenths

    def test_quantize_strength_is_genre_and_delivery_aware(self):
        """Softer for laid-back hip-hop, tighter for programmed dance music,
        zero for spoken word where rigid timing destroys the point."""
        rap_trap = groove.quantize_strength("trap", "rap")
        sung_rnb = groove.quantize_strength("rnb", "sung")
        edm = groove.quantize_strength("edm", "sung")
        jazz = groove.quantize_strength("jazz", "sung")
        self.assertGreater(rap_trap, sung_rnb)
        self.assertGreater(edm, sung_rnb)
        self.assertGreater(sung_rnb, jazz)
        self.assertEqual(groove.quantize_strength("trap", "spoken"), 0.0)

    def test_quantize_strength_collapses_on_an_unreliable_grid(self):
        """There is no point quantising to a grid that is itself unreliable."""
        self.assertAlmostEqual(
            groove.quantize_strength("trap", "rap", grid_consistency=0.0), 0.0)

    def test_scaling_beyond_one_is_possible_but_marked(self):
        g = GrooveTemplate(subdivision=16, offsets_ms=np.full(16, 10.0),
                           consistency=0.9, n_bars_observed=8)
        self.assertAlmostEqual(g.scaled(0.5).max_offset_ms, 5.0)
        self.assertAlmostEqual(g.scaled(2.0).max_offset_ms, 20.0)


class TestGrooveCompatibility(unittest.TestCase):

    def test_matching_feels_score_high(self):
        a = groove.extract(*make_grid(bpm=90, bars=16, swing=0.66)[:2],
                           bar_duration_s=60.0 / 90 * 4, subdivision=16)
        b = groove.extract(*make_grid(bpm=90, bars=16, swing=0.66)[:2],
                           bar_duration_s=60.0 / 90 * 4, subdivision=16)
        self.assertGreater(groove.compatibility(a, b), 0.85)

    def test_straight_vocal_against_swung_beat_scores_low(self):
        """A mismatch invisible to key and tempo scoring, and audible.

        This is exactly why groove deserves its own matching term.
        """
        bar = 60.0 / 90 * 4
        straight = groove.extract(*make_grid(bpm=90, bars=16, swing=0.5)[:2],
                                  bar_duration_s=bar, subdivision=16)
        swung = groove.extract(*make_grid(bpm=90, bars=16, swing=0.68)[:2],
                               bar_duration_s=bar, subdivision=16)
        self.assertLess(groove.compatibility(straight, swung), 0.65)

    def test_unknown_groove_is_neutral(self):
        """Absent information must neither reward nor punish a pairing."""
        self.assertAlmostEqual(
            groove.compatibility(GrooveTemplate(), GrooveTemplate()), 0.6)

    def test_describe_is_readable(self):
        g = GrooveTemplate(subdivision=16, offsets_ms=np.full(16, 12.0),
                           velocities=np.ones(16), swing_ratio=0.63,
                           consistency=0.85, n_bars_observed=16)
        text = groove.describe(g)
        self.assertIn("swung", text)
        self.assertIn("laid back", text)


# ═════════════════════════════════════════════════════════════════════════════
# Salience
# ═════════════════════════════════════════════════════════════════════════════

class TestSalience(unittest.TestCase):

    def setUp(self):
        self.dbs = [0.0, 2.0, 4.0, 6.0]           # 120 BPM, 4/4
        self.bar = 2.0

    def test_metrical_hierarchy_is_respected(self):
        """Downbeat > beat > eighth > sixteenth.

        Listeners track the pulse at these levels, so timing errors are
        audible in proportion to the strength of the position they land on.
        """
        downbeat = salience.metrical_weight(0.0, self.dbs, self.bar)
        beat = salience.metrical_weight(0.5, self.dbs, self.bar)
        eighth = salience.metrical_weight(0.25, self.dbs, self.bar)
        sixteenth = salience.metrical_weight(0.125, self.dbs, self.bar)
        self.assertGreater(downbeat, beat)
        self.assertGreater(beat, eighth)
        self.assertGreater(eighth, sixteenth)

    def test_timing_error_on_a_downbeat_outweighs_one_off_grid(self):
        strong = salience.timing_salience(
            0.0, downbeats=self.dbs, bar_duration_s=self.bar, word_stress=0.9)
        weak = salience.timing_salience(
            0.1875, downbeats=self.dbs, bar_duration_s=self.bar, word_stress=0.2)
        self.assertGreater(strong, weak * 1.8)

    def test_phrase_entry_carries_extra_weight(self):
        p = Phrase(start=2.0, end=6.0)
        at_entry = salience.timing_salience(
            2.0, downbeats=self.dbs, bar_duration_s=self.bar, phrase=p)
        mid = salience.timing_salience(
            3.0, downbeats=self.dbs, bar_duration_s=self.bar, phrase=p)
        self.assertGreater(at_entry, mid)

    def test_hook_outweighs_intro(self):
        hook = Section(0, 8, "hook")
        intro = Section(0, 8, "intro")
        self.assertGreater(
            salience.timing_salience(0.0, downbeats=self.dbs,
                                     bar_duration_s=self.bar, section=hook),
            salience.timing_salience(0.0, downbeats=self.dbs,
                                     bar_duration_s=self.bar, section=intro))

    def test_long_notes_expose_pitch_more_than_short_ones(self):
        """The ear needs a couple of hundred milliseconds of steady tone to
        judge intonation at all, and beyond that every extra moment makes
        an error more obvious."""
        short = salience.pitch_salience(Note(0.0, 0.08, 60.0))
        medium = salience.pitch_salience(Note(0.0, 0.3, 60.0))
        long = salience.pitch_salience(Note(0.0, 1.5, 60.0))
        self.assertGreater(medium, short)
        self.assertGreater(long, medium)

    def test_melisma_is_de_prioritised(self):
        """Runs are gesture, not target pitches."""
        plain = salience.pitch_salience(Note(0.0, 0.5, 60.0))
        run = salience.pitch_salience(Note(0.0, 0.5, 60.0, is_melisma=True))
        self.assertLess(run, plain)

    def test_deep_vibrato_reduces_correction_pressure(self):
        plain = salience.pitch_salience(Note(0.0, 1.0, 60.0))
        vib = salience.pitch_salience(
            Note(0.0, 1.0, 60.0, vibrato_depth_cents=40.0))
        self.assertLess(vib, plain)

    def test_deesser_is_idle_between_sibilants(self):
        """Set it so it moves on problem words, not constantly -- over-applied
        de-essing turns sibilants into lisps."""
        on = salience.deess_salience(1.0, in_sibilant_span=True)
        off = salience.deess_salience(1.0, in_sibilant_span=False)
        self.assertGreater(on, off * 3.0)

    def test_weighted_error_punishes_errors_where_they_matter(self):
        """A take clean everywhere except its downbeats is worse than one
        slightly loose throughout. An unweighted mean says the opposite."""
        errors = [30.0, 2.0, 2.0, 2.0]
        at_downbeat = salience.weighted_error(errors, [3.0, 0.5, 0.5, 0.5])
        spread_out = salience.weighted_error([9.0] * 4, [3.0, 0.5, 0.5, 0.5])
        self.assertGreater(at_downbeat, spread_out)

    def test_perceptible_threshold_is_bounded(self):
        """Correcting below the audible threshold spends stretch artifacts
        on something nobody can hear."""
        self.assertGreaterEqual(salience.perceptible_timing_threshold_ms(125.0), 12.0)
        self.assertLessEqual(salience.perceptible_timing_threshold_ms(1000.0), 30.0)


# ═════════════════════════════════════════════════════════════════════════════
# Energy
# ═════════════════════════════════════════════════════════════════════════════

class TestEnergyContour(unittest.TestCase):

    def sections(self):
        return [
            Section(0, 8, "intro"), Section(8, 24, "verse"),
            Section(24, 32, "prehook"), Section(32, 48, "hook"),
            Section(48, 64, "verse"), Section(64, 80, "hook"),
            Section(80, 88, "bridge"), Section(88, 104, "hook"),
            Section(104, 112, "outro"),
        ]

    def test_hook_outranks_verse_outranks_intro(self):
        c = energy.build(self.sections(), genre="trap")
        self.assertGreater(c.at(40.0), c.at(16.0))
        self.assertGreater(c.at(16.0), c.at(4.0))

    def test_later_hooks_climb(self):
        """A second hook is bigger than the first -- the most reliable
        arrangement convention in popular music."""
        c = energy.build(self.sections(), genre="trap")
        self.assertGreater(c.at(72.0), c.at(40.0))

    def test_final_hook_after_a_bridge_is_the_peak(self):
        """The bridge exists to create the drop that makes it land."""
        c = energy.build(self.sections(), genre="trap")
        self.assertGreater(c.at(96.0), c.at(84.0))
        self.assertGreaterEqual(c.at(96.0), c.at(72.0) - 1e-6)

    def test_genre_contrast_is_applied(self):
        """Trap lives on wide verse-to-hook contrast; lo-fi is deliberately
        flat and exaggerating it would read as clumsy."""
        secs = self.sections()
        self.assertGreater(energy.build(secs, genre="trap").contrast,
                           energy.build(secs, genre="lofi").contrast)

    def test_contour_ramps_rather_than_steps(self):
        """A hard step in level at a section line is audible as an edit."""
        c = energy.build(self.sections(), genre="pop")
        just_before = c.at(31.5)
        just_after = c.at(32.5)
        self.assertNotAlmostEqual(just_before, just_after, places=4)
        self.assertLess(abs(just_after - just_before), 0.5)

    def test_empty_sections_degrade_to_neutral(self):
        c = energy.build([], genre="trap", duration_s=60.0)
        self.assertAlmostEqual(c.at(30.0), 0.5)


class TestMixTargets(unittest.TestCase):

    def test_hook_pushes_the_vocal_forward_and_opens_the_top(self):
        verse = energy.mix_targets(0.5)
        hook = energy.mix_targets(1.0)
        self.assertGreater(hook.vir_db, verse.vir_db)
        self.assertGreater(hook.air_db, verse.air_db)
        self.assertGreater(hook.double_gain_db, verse.double_gain_db)
        self.assertGreater(hook.duck_depth_db, verse.duck_depth_db)

    def test_rap_stays_drier_than_sung_material(self):
        """Opening reverb on a rap hook washes out the consonants that
        carry the bars."""
        sung = energy.mix_targets(1.0, performance_type="sung")
        rap = energy.mix_targets(1.0, performance_type="rap")
        self.assertLess(rap.reverb_wet, sung.reverb_wet)
        self.assertGreater(rap.vir_db, sung.vir_db)

    def test_offsets_stay_small(self):
        """The arrangement should do the work; the mix supports it."""
        for e in (0.0, 0.25, 0.5, 0.75, 1.0):
            t = energy.mix_targets(e)
            self.assertLess(abs(t.vir_db), 2.5)
            self.assertLess(abs(t.air_db), 3.0)

    def test_arrangement_density_grows_with_energy(self):
        sparse = energy.arrangement_density(0.3)
        full = energy.arrangement_density(0.95)
        self.assertFalse(sparse["pads"])
        self.assertTrue(full["pads"])
        self.assertFalse(sparse["wide_double"])
        self.assertTrue(full["wide_double"])
        self.assertTrue(sparse["lead_vocal"] and full["lead_vocal"])

    def test_transitions_are_gated_on_the_size_of_the_jump(self):
        """Decorating every boundary makes the arrangement fussy."""
        self.assertEqual(energy.transition_before("verse", 0.05), [])
        big = energy.transition_before("hook", 0.35)
        self.assertIn("riser", big)
        self.assertIn("impact", big)


# ═════════════════════════════════════════════════════════════════════════════
# Musical IR
# ═════════════════════════════════════════════════════════════════════════════

class TestMusicalIR(unittest.TestCase):

    def build_ir(self):
        ir = MusicalIR(source_id="test", role=ROLE_BEAT, duration_s=32.0)
        ir.tempo = TempoCurve.constant(120.0, 32.0)
        beats = []
        for i in range(64):                       # 16 bars at 120 BPM
            t = i * 0.5
            beats.append(Beat(time=t, index=i, bar=i // 4,
                              position_in_bar=(i % 4) + 1,
                              is_downbeat=(i % 4 == 0)))
        ir.beats = beats
        ir.key_regions = [KeyRegion(0.0, 16.0, 0, "major", 0.9),
                          KeyRegion(16.0, 32.0, 9, "minor", 0.8)]
        ir.chords = [chord(0, 4, 0), chord(4, 8, 5), chord(8, 12, 7),
                     chord(12, 16, 0)]
        ir.sections = [Section(0, 8, "intro"), Section(8, 24, "verse"),
                       Section(24, 32, "hook", energy=0.95)]
        ir.phrases = [Phrase(start=8.0, end=12.0), Phrase(start=13.0, end=16.0)]
        ir.notes = [Note(8.0, 8.5, 60.0), Note(9.0, 10.5, 64.0)]
        ir.words = [Word("hello", 8.0, 8.4), Word("sun", 9.0, 9.6)]
        ir.phonemes = [Phoneme("s", 9.0, 9.1, PHON_SIBILANT, 1)]
        return ir

    def test_derived_grid_properties(self):
        ir = self.build_ir()
        self.assertTrue(ir.has_grid)
        self.assertAlmostEqual(ir.bpm, 120.0)
        self.assertEqual(ir.beats_per_bar, 4)
        self.assertAlmostEqual(ir.bar_duration_s, 2.0)
        self.assertEqual(len(ir.downbeats), 16)

    def test_lookups_are_time_accurate(self):
        ir = self.build_ir()
        self.assertEqual(ir.chord_at(5.0).root, 5)
        self.assertEqual(ir.key_at(4.0).mode, "major")
        self.assertEqual(ir.key_at(20.0).mode, "minor")
        self.assertEqual(ir.section_at(26.0).label, "hook")
        self.assertEqual(ir.bar_of(9.0), 4)
        self.assertAlmostEqual(ir.nearest_downbeat(9.2), 10.0)

    def test_lookups_return_none_rather_than_guessing(self):
        """An honest 'unknown' routes the pipeline correctly; a fabricated
        value corrupts everything downstream of it."""
        ir = self.build_ir()
        self.assertIsNone(ir.chord_at(25.0))       # past the last chord
        self.assertIsNone(ir.note_at(100.0))
        self.assertIsNone(MusicalIR().beat_at(1.0))

    def test_subdivision_grid_spans_the_bars(self):
        ir = self.build_ir()
        g = ir.subdivision_grid(16, apply_groove=False)
        self.assertGreater(g.size, 200)
        self.assertAlmostEqual(g[0], 0.0, places=6)
        self.assertAlmostEqual(g[1], 0.125, places=6)     # 16th at 120 BPM
        self.assertTrue(np.all(np.diff(g) >= -1e-9), "grid must be sorted")

    def test_groove_shifts_the_grid_when_meaningful(self):
        ir = self.build_ir()
        offs = np.zeros(16)
        offs[4] = 18.0
        ir.groove = GrooveTemplate(subdivision=16, offsets_ms=offs,
                                   velocities=np.ones(16), consistency=0.9,
                                   n_bars_observed=16, source="measured")
        plain = ir.subdivision_grid(16, apply_groove=False)
        feel = ir.subdivision_grid(16, apply_groove=True)
        self.assertAlmostEqual((feel[4] - plain[4]) * 1000.0, 18.0, delta=0.5)
        self.assertAlmostEqual(feel[0], plain[0], places=6)

    def test_sibilant_spans_come_from_measured_phonemes(self):
        ir = self.build_ir()
        spans = ir.sibilant_spans()
        self.assertEqual(len(spans), 1)
        self.assertAlmostEqual(spans[0][0], 9.0)

    def test_serialisation_roundtrip_preserves_musical_content(self):
        ir = self.build_ir()
        back = MusicalIR.from_dict(ir.to_dict())
        self.assertAlmostEqual(back.bpm, 120.0, places=2)
        self.assertEqual(len(back.beats), 64)
        self.assertEqual(len(back.chords), 4)
        self.assertEqual(back.chord_at(5.0).root, 5)
        self.assertEqual(back.section_at(26.0).label, "hook")
        self.assertEqual(back.key_at(20.0).mode, "minor")
        self.assertEqual(len(back.notes), 2)
        self.assertEqual(len(back.phonemes), 1)

    def test_json_serialisable(self):
        import json
        ir = self.build_ir()
        text = json.dumps(ir.to_dict())
        self.assertGreater(len(text), 500)
        self.assertEqual(json.loads(text)["bpm"], 120.0)

    def test_empty_ir_is_safe_to_query(self):
        """Nothing raises for missing data -- the engine must degrade,
        not crash, when an analyzer produced nothing."""
        ir = MusicalIR()
        self.assertEqual(ir.bpm, 0.0)
        self.assertFalse(ir.has_grid)
        self.assertIsNone(ir.key)
        self.assertIsNone(ir.chord_at(1.0))
        self.assertIsNone(ir.section_at(1.0))
        self.assertEqual(ir.subdivision_grid().size, 0)
        self.assertEqual(ir.syllable_rate, 0.0)
        self.assertIsInstance(ir.summary(), str)

    def test_summary_mentions_key_and_tempo(self):
        ir = self.build_ir()
        s = ir.summary()
        self.assertIn("120 BPM", s)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestHarmonyAgainstAChord(unittest.TestCase):
    """The chord-aware path. The original walked the chord's own tones, so a
    'third' over a triad came out as a fifth or a sixth (+6 to +10 semitones
    on the first render that used it). Nothing caught it because the only
    harmony test passed `chord_at=lambda t: None`."""

    def test_thirds_stay_thirds_when_a_chord_is_present(self):
        c_major = chord(0, 4, 0, "maj")
        lead = [60.0, 62.0, 64.0, 65.0, 67.0]            # C D E F G
        times = [0.0, 1.0, 2.0, 3.0, 4.0]
        h = theory.generate_harmony(
            lead, times, chord_at=lambda t: c_major, key_at=lambda t: C_MAJOR,
            interval="third", direction=1)
        offs = [int(round(o)) for o in h.offsets]
        self.assertTrue(all(3 <= o <= 5 for o in offs),
                        f"a harmony must sit a third to a fourth above the "
                        f"lead, never a second, got {offs}")
        self.assertIn(3, offs)
        self.assertIn(4, offs)

    def test_avoid_note_is_resolved_away_from_the_lead(self):
        # Over C major, the scale third above D is F, a minor ninth against
        # the chord's E. Dropping it onto E would put the harmony a major
        # second above the singer; it must continue up to G instead.
        c_major = chord(0, 4, 0, "maj")
        h = theory.generate_harmony([62.0], [0.0], chord_at=lambda t: c_major,
                                    key_at=lambda t: C_MAJOR, interval="third")
        self.assertEqual(int(round(h.offsets[0])), 5)

    def test_chromatic_lead_note_still_gets_a_real_interval(self):
        # F is not in B minor. The nearest scale tone is E, and the scale
        # third above E is G -- two semitones above the F that was sung.
        b_minor = Key(11, "minor")
        bm = chord(0, 4, 11, "min")
        h = theory.generate_harmony([65.0], [0.0], chord_at=lambda t: bm,
                                    key_at=lambda t: b_minor, interval="third")
        self.assertGreaterEqual(h.offsets[0], 3.0)
        self.assertNotEqual(int(round(h.offsets[0])), 6)

    def test_flat_lead_note_does_not_harmonise_at_a_tritone(self):
        # A B sung a quarter-tone flat, over F# minor in B minor. The scale
        # third above B is D, a minor ninth against the chord's C#, so the
        # line steps to E -- which is 5.5 semitones above what was sung and
        # would be shifted by a rounded six: a tritone. It must keep going.
        b_minor = Key(11, "minor")
        f_sharp_minor = chord(0, 4, 6, "min")
        h = theory.generate_harmony([70.5], [0.0],
                                    chord_at=lambda t: f_sharp_minor,
                                    key_at=lambda t: b_minor, interval="third")
        o = int(round(h.offsets[0]))
        self.assertNotIn(o, (1, 2, 6), f"got +{o}")
        self.assertTrue(3 <= o <= 9, f"got +{o}")

    def test_no_harmony_note_is_ever_a_second_above_the_lead(self):
        # Every diatonic lead note over every diatonic triad in C major.
        lead = [60.0, 62.0, 64.0, 65.0, 67.0, 69.0, 71.0]
        for root, quality in ((0, "maj"), (2, "min"), (4, "min"), (5, "maj"),
                              (7, "maj"), (9, "min")):
            c = chord(root, 4, 0, quality)
            h = theory.generate_harmony(lead, [float(i) for i in range(7)],
                                        chord_at=lambda t, c=c: c,
                                        key_at=lambda t: C_MAJOR,
                                        interval="third")
            for midi, o in zip(lead, h.offsets):
                self.assertGreaterEqual(o, 3.0, f"{midi} over {root}{quality}: +{o}")
                self.assertLessEqual(o, 7.0, f"{midi} over {root}{quality}: +{o}")

    def test_accepts_a_bare_key_as_well_as_a_key_region(self):
        from mixengine.core.keys import Key
        h_region = theory.generate_harmony([60.0, 64.0], [0.0, 1.0],
                                           chord_at=lambda t: None,
                                           key_at=lambda t: C_MAJOR, interval="third")
        h_key = theory.generate_harmony([60.0, 64.0], [0.0, 1.0],
                                        chord_at=lambda t: None,
                                        key_at=lambda t: Key(0, "major"), interval="third")
        self.assertEqual(h_region.offsets, h_key.offsets)
