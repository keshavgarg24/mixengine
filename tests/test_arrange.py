"""
Tests for the arrangement stage.

The structure tests use synthesised phrases with a known form, because the
question is whether the classifier recovers a form that was actually put
there -- a real take has no ground truth to check against, only an opinion.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mixengine.arrange import automation, layers, plan as plan_mod        # noqa: E402
from mixengine.arrange import structure as structure_mod                  # noqa: E402
from mixengine.core.types import Section                                  # noqa: E402

SR = 22050

VERSE_A = [60, 62, 64, 62]
VERSE_B = [60, 59, 57, 59]
HOOK = [67, 69, 71, 69]
BRIDGE = [65, 64, 62, 60]
ADLIB = [72]


def tone_phrase(notes, dur=2.6, sr=SR):
    out = []
    for m in notes:
        f = 440.0 * 2 ** ((m - 69) / 12.0)
        n = int(sr * dur / len(notes))
        t = np.arange(n) / sr
        sig = sum(0.4 / (k + 1) * np.sin(2 * np.pi * f * (k + 1) * t)
                  for k in range(4))
        out.append((sig * np.hanning(n)).astype(np.float32))
    return np.concatenate(out)


def build_take(order, gap_s=0.7, sr=SR):
    gap = np.zeros(int(sr * gap_s), dtype=np.float32)
    parts, regions, pos = [], [], 0
    for _label, notes, dur in order:
        seg = tone_phrase(notes, dur, sr)
        parts.append(seg)
        regions.append((pos, pos + len(seg)))
        pos += len(seg)
        parts.append(gap)
        pos += len(gap)
    return np.concatenate(parts), regions


SONG = [("verse", VERSE_A, 2.6), ("prehook", VERSE_B, 2.6),
        ("hook", HOOK, 2.6), ("adlib", ADLIB, 0.7),
        ("verse", VERSE_A, 2.6), ("hook", HOOK, 2.6),
        ("bridge", BRIDGE, 2.6), ("hook", HOOK, 2.6)]


class TestStructure(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.y, cls.regions = build_take(SONG)
        cls.res = structure_mod.analyze(cls.y, SR, cls.regions,
                                        performance_type="sung")

    def test_finds_every_hook(self):
        truth = [i for i, (lab, _, _) in enumerate(SONG) if lab == "hook"]
        found = [i for i, lab in enumerate(self.res.labels) if lab == "hook"]
        self.assertEqual(found, truth)

    def test_repeats_cluster_together(self):
        g = self.res.groups
        self.assertEqual(g[2], g[5])
        self.assertEqual(g[5], g[7])
        self.assertNotEqual(g[0], g[2])

    def test_finds_the_bridge(self):
        self.assertEqual(self.res.labels[6], "bridge")

    def test_finds_the_adlib(self):
        self.assertEqual(self.res.labels[3], "adlib")

    def test_short_phrases_are_not_all_adlibs(self):
        """Testing duration alone turned every phrase of a short-lined song
        into an ad-lib: on 1.2-second phrases the verses and the bridge were
        all relabelled, and only the hooks survived because repetition ran
        first. An ad-lib must also be unique and tucked against a phrase."""
        order = [(lab, notes, 1.2) for lab, notes, _ in SONG if lab != "adlib"]
        y, regions = build_take(order, gap_s=0.5)
        res = structure_mod.analyze(y, SR, regions)
        self.assertLess(res.labels.count("adlib"), len(order) // 2)

    def test_no_repetition_declines_rather_than_inventing_a_hook(self):
        order = [("verse", VERSE_A, 2.6), ("verse", VERSE_B, 2.6),
                 ("verse", BRIDGE, 2.6)]
        y, regions = build_take(order)
        res = structure_mod.analyze(y, SR, regions)
        self.assertEqual(res.hook_group, -1)
        self.assertNotIn("hook", res.labels)
        self.assertIn("hook", res.note)

    def test_sections_cover_the_take_without_gaps(self):
        total = len(self.y) / SR
        secs = structure_mod.sections_from_labels(
            self.res.phrases, self.res.labels, total)
        self.assertAlmostEqual(secs[0].start, 0.0, places=6)
        self.assertGreaterEqual(secs[-1].end, total - 1e-6)
        for a, b in zip(secs, secs[1:]):
            self.assertAlmostEqual(a.end, b.start, places=6)


class TestPlan(unittest.TestCase):

    def setUp(self):
        self.y, self.regions = build_take(SONG)
        self.total = len(self.y) / SR

    def test_hook_regions_match_the_hook_phrases(self):
        p = plan_mod.build(self.y, SR, self.regions, genre="pop")
        self.assertEqual(len(p.hook_regions), 3)

    def test_energy_rises_on_the_hook(self):
        p = plan_mod.build(self.y, SR, self.regions, genre="pop")
        hook_e = [p.energy_at((s + e) / 2 / SR) for s, e in p.hook_regions]
        verse = p.structure.phrases[0]
        verse_e = p.energy_at((verse.start + verse.end) / 2)
        self.assertGreater(min(hook_e), verse_e)

    def test_layers_land_where_the_energy_is(self):
        """Layers follow the contour, not the labels. A pre-hook at 0.76
        energy getting a tight double is right -- that lift is where a
        producer would put one -- so the test is that nothing lands on a
        *low* section, not that everything lands on a hook."""
        p = plan_mod.build(self.y, SR, self.regions, genre="pop")
        self.assertTrue(p.layer_regions, "no layers were planned at all")
        for name, regions in p.layer_regions.items():
            for s, e in regions:
                energy = p.energy_at((s + e) / 2 / SR)
                self.assertGreater(energy, 0.6,
                                   f"{name} was placed at energy {energy:.2f}")

    def test_hooks_get_at_least_as_many_layers_as_anything_else(self):
        p = plan_mod.build(self.y, SR, self.regions, genre="pop")
        hooks = set(p.hook_regions)
        on_hook = sum(1 for rs in p.layer_regions.values()
                      for r in rs if r in hooks)
        off_hook = sum(1 for rs in p.layer_regions.values()
                       for r in rs if r not in hooks)
        self.assertGreaterEqual(on_hook, off_hook)

    def test_flat_plan_when_there_is_no_structure(self):
        y, regions = build_take([("verse", VERSE_A, 2.6)])
        p = plan_mod.build(y, SR, regions, genre="pop")
        self.assertTrue(p.note)
        self.assertEqual(p.layer_regions, {})

    def test_automation_curves_are_sample_length(self):
        p = plan_mod.build(self.y, SR, self.regions, genre="pop")
        n = len(self.y)
        c = p.curve("vir_db", n, SR)
        self.assertEqual(c.size, n)
        self.assertTrue(np.all(np.isfinite(c)))

    def test_unknown_curve_is_zero_not_an_error(self):
        p = plan_mod.build(self.y, SR, self.regions)
        c = p.curve("no_such_parameter", 1000, SR)
        self.assertEqual(c.size, 1000)
        self.assertEqual(float(np.abs(c).max()), 0.0)

    def test_beat_sections_move_the_boundaries(self):
        beat_sections = [Section(start=0.0, end=6.0, label="intro"),
                         Section(start=6.0, end=self.total, label="chorus")]
        p = plan_mod.build(self.y, SR, self.regions, genre="pop",
                           beat_sections=beat_sections)
        edges = {round(s.start, 3) for s in p.sections}
        self.assertTrue(edges & {6.0}, f"no boundary snapped to 6.0: {edges}")


class TestLayers(unittest.TestCase):

    def setUp(self):
        self.y, self.regions = build_take(SONG)
        self.v = self.y[:, None]

    def test_a_double_is_not_a_delayed_copy(self):
        """A delayed copy of a signal is a comb filter, not a double. The
        generated layer must differ from the lead in time *and* pitch, or
        summing it with the lead cancels rather than thickens."""
        built, _ = layers.build(self.v, SR,
                                wanted={"tight_double": self.regions})
        self.assertTrue(built)
        d = built[0].audio[:, 0]
        lead = self.v[:len(d), 0]
        n = min(len(d), len(lead))
        # Identical up to a delay would show a near-perfect correlation at
        # *some* lag; a real double does not.
        best = 0.0
        for lag in range(0, int(SR * 0.05), 64):
            a, b = lead[:n - lag], d[lag:n]
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            if denom > 0:
                best = max(best, abs(float(np.dot(a, b)) / denom))
        self.assertLess(best, 0.97, "the double is a delayed copy of the lead")

    def test_layers_are_silent_outside_their_regions(self):
        region = [self.regions[2]]
        built, _ = layers.build(self.v, SR, wanted={"tight_double": region})
        a = built[0].audio[:, 0]
        before = a[:max(0, region[0][0] - int(SR * 0.2))]
        if before.size:
            self.assertLess(float(np.abs(before).max()), 0.02)

    def test_same_seed_gives_the_same_layer(self):
        """A mix that cannot be reproduced cannot be debugged."""
        a, _ = layers.build(self.v, SR, wanted={"tight_double": self.regions},
                            seed=7)
        b, _ = layers.build(self.v, SR, wanted={"tight_double": self.regions},
                            seed=7)
        np.testing.assert_allclose(a[0].audio, b[0].audio, atol=1e-6)

    def test_different_seeds_give_different_layers(self):
        a, _ = layers.build(self.v, SR, wanted={"tight_double": self.regions},
                            seed=1)
        b, _ = layers.build(self.v, SR, wanted={"tight_double": self.regions},
                            seed=2)
        n = min(len(a[0].audio), len(b[0].audio))
        self.assertGreater(
            float(np.abs(a[0].audio[:n] - b[0].audio[:n]).max()), 1e-4)

    def test_wide_pair_survives_a_mono_fold(self):
        """Two copies of one signal panned apart cancel in mono. Two
        different performances do not, which is the point of generating
        them independently."""
        built, _ = layers.build(self.v, SR, wanted={"wide_double": self.regions})
        self.assertEqual(len(built), 2)
        mixed = layers.sum_layers(built, len(self.v))
        mono = mixed.mean(axis=1)
        stereo_rms = float(np.sqrt(np.mean(mixed ** 2)))
        mono_rms = float(np.sqrt(np.mean(mono ** 2)))
        self.assertGreater(mono_rms, stereo_rms * 0.3)

    def test_every_builder_actually_produces_a_layer(self):
        """The builder loop catches exceptions and reports a skip, which is
        right for a render that must not die over one layer — but it also
        means a broken builder ships silently. `whisper_layer` passed the
        `(audio, gain_reduction)` tuple from `dsp.compressor` straight on
        and raised `'tuple' object has no attribute 'astype'` on every
        render for as long as it existed; the only trace was one line in
        the layer report. This asserts that nothing is quietly missing."""
        wanted = dict.fromkeys(("tight_double", "wide_double", "octave_down", "harmony", "adlibs", "whisper_layer"), self.regions)
        built, rep = layers.build(self.v, SR, wanted=wanted, seed=1)
        self.assertEqual(rep["skipped"], {},
                         f"builders failed: {rep['skipped']}")
        names = {L.name for L in built}
        for expected in ("tight_double", "octave_down", "harmony",
                         "adlibs", "whisper_layer"):
            self.assertIn(expected, names)
        self.assertIn("wide_double_l", names)
        self.assertIn("wide_double_r", names)

    def test_unknown_layer_is_reported_not_raised(self):
        built, rep = layers.build(self.v, SR,
                                  wanted={"not_a_layer": self.regions})
        self.assertEqual(built, [])
        self.assertIn("not_a_layer", rep["skipped"])

    def test_empty_regions_build_nothing(self):
        built, rep = layers.build(self.v, SR, wanted={"tight_double": []})
        self.assertEqual(built, [])
        self.assertEqual(rep["skipped"]["tight_double"], "no regions")


class TestAutomation(unittest.TestCase):

    def test_flat_contour_changes_nothing(self):
        y, regions = build_take([("verse", VERSE_A, 2.6)])
        p = plan_mod.build(y, SR, regions)
        out, rep = automation.apply_vocal(y[:, None], SR, p)
        self.assertEqual(rep["applied"], [])
        np.testing.assert_allclose(out, y[:, None], atol=1e-6)

    def test_level_rides_with_the_contour(self):
        y, regions = build_take(SONG)
        p = plan_mod.build(y, SR, regions, genre="pop")
        out, rep = automation.apply_vocal(y[:, None], SR, p)
        if not rep["applied"]:
            self.skipTest("contour is flat on this fixture")
        self.assertIn("vir_db", rep["applied"])
        hook = p.hook_regions[0]
        verse = p.structure.phrases[0]
        v_s, v_e = int(verse.start * SR), int(verse.end * SR)

        def gain(a, b):
            src = float(np.sqrt(np.mean(y[a:b] ** 2))) or 1e-9
            dst = float(np.sqrt(np.mean(out[a:b, 0] ** 2)))
            return dst / src

        self.assertGreater(gain(*hook), gain(v_s, v_e))

    def test_ride_never_exceeds_its_limit(self):
        y, regions = build_take(SONG)
        p = plan_mod.build(y, SR, regions, genre="trap")
        out, rep = automation.apply_vocal(y[:, None], SR, p)
        if "ride_db" in rep:
            self.assertLessEqual(abs(rep["ride_db"]["max"]),
                                 automation.MAX_RIDE_DB + 1e-6)
            self.assertLessEqual(abs(rep["ride_db"]["min"]),
                                 automation.MAX_RIDE_DB + 1e-6)

    def test_none_plan_is_safe(self):
        y = np.zeros((SR * 2, 1), dtype=np.float32)
        out, rep = automation.apply_vocal(y, SR, None)
        self.assertEqual(rep["applied"], [])
        np.testing.assert_allclose(out, y)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestTransitions(unittest.TestCase):
    """The section-change devices. Each is checked for *where* it lands
    rather than what it sounds like: a riser that runs past the downbeat or
    a dropout that mutes the wrong bar is wrong however good it sounds."""

    def setUp(self):
        from mixengine.arrange import transitions
        self.T = transitions
        self.sr = SR
        self.bar = 2.0
        self.beat = 0.5
        n = int(self.sr * 12)
        rng = np.random.default_rng(3)
        # A vocal with a phrase ending 0.4 s before the boundary at 6.0 s.
        self.vocal = np.zeros((n, 1), dtype=np.float32)
        t = np.arange(int(self.sr * 1.5)) / self.sr
        seg = (0.4 * np.sin(2 * np.pi * 220 * t) * np.hanning(t.size)).astype(np.float32)
        self.vocal[int(self.sr * 4.1):int(self.sr * 4.1) + seg.size, 0] = seg
        self.beat_audio = (rng.normal(0, 0.1, (n, 1))).astype(np.float32)
        self.boundary = 6.0

    def _build(self, effects):
        return self.T.build(self.vocal, self.beat_audio, self.sr,
                            [(self.boundary, effects)],
                            bar_s=self.bar, beat_s=self.beat, seed=1)

    def test_nothing_planned_builds_nothing(self):
        fx, gain, rep = self.T.build(self.vocal, self.beat_audio, self.sr, [],
                                     bar_s=self.bar, beat_s=self.beat)
        self.assertEqual(float(np.abs(fx).max()), 0.0)
        self.assertTrue(np.all(gain == 1.0))
        self.assertEqual(rep["built"], [])

    def test_riser_ends_on_the_downbeat(self):
        fx, _, rep = self._build(["riser"])
        at = int(self.boundary * self.sr)
        before = float(np.abs(fx[at - int(self.sr * 0.05):at]).max())
        after = float(np.abs(fx[at + int(self.sr * 0.02):at + self.sr]).max())
        self.assertGreater(before, 0.01)
        self.assertEqual(after, 0.0, "the riser must be cut dead at the boundary")
        self.assertEqual(rep["built"][0]["effect"], "riser")

    def test_riser_swells_rather_than_starting_loud(self):
        fx, _, _ = self._build(["riser"])
        at = int(self.boundary * self.sr)
        start = int(at - self.bar * self.sr)
        early = float(np.sqrt(np.mean(fx[start:start + self.sr // 4, 0] ** 2)))
        late = float(np.sqrt(np.mean(fx[at - self.sr // 4:at, 0] ** 2)))
        self.assertGreater(late, early * 4)

    def test_impact_lands_on_the_downbeat_and_decays(self):
        fx, _, _ = self._build(["impact"])
        at = int(self.boundary * self.sr)
        self.assertEqual(float(np.abs(fx[:at - 4]).max()), 0.0)
        self.assertGreater(float(np.abs(fx[at:at + self.sr // 10]).max()), 0.05)
        self.assertLess(float(np.abs(fx[at + self.sr // 2:at + self.sr]).max()), 0.01)

    def test_reverse_tail_uses_the_last_voiced_material(self):
        """The audio at the boundary itself is silence; the tail must be
        found by searching back to the last phrase."""
        fx, _, rep = self._build(["reverse_vocal_tail"])
        self.assertEqual(rep["built"][0]["effect"], "reverse_vocal_tail")
        self.assertAlmostEqual(rep["built"][0]["source_end_s"], 5.6, delta=0.15)
        at = int(self.boundary * self.sr)
        self.assertGreater(float(np.abs(fx[at - self.sr // 10:at]).max()), 0.01)
        self.assertEqual(float(np.abs(fx[at + 8:]).max()), 0.0)

    def test_reverse_tail_declines_without_voice(self):
        silent = np.zeros_like(self.vocal)
        _, _, rep = self.T.build(silent, self.beat_audio, self.sr,
                                 [(self.boundary, ["reverse_vocal_tail"])],
                                 bar_s=self.bar, beat_s=self.beat)
        self.assertEqual(rep["built"], [])
        self.assertEqual(rep["skipped"][0]["effect"], "reverse_vocal_tail")

    def test_drum_fill_is_reported_honestly(self):
        _, _, rep = self._build(["drum_fill"])
        b = rep["built"][0]
        self.assertEqual(b["effect"], "drum_fill")
        self.assertEqual(b["delivered_as"], "reverse_swell")

    def test_dropout_mutes_exactly_one_bar(self):
        _, gain, rep = self._build(["beat_dropout_1bar"])
        at = int(self.boundary * self.sr)
        start = at - int(self.bar * self.sr)
        self.assertTrue(np.all(gain[start + 16:at - 16] == 0.0))
        self.assertTrue(np.all(gain[:start - int(self.sr * 0.02)] == 1.0))
        self.assertTrue(np.all(gain[at + int(self.sr * 0.02):] == 1.0))
        self.assertAlmostEqual(rep["beat_muted_s"], self.bar, delta=0.01)

    def test_boundary_at_the_edge_is_skipped(self):
        _, _, rep = self.T.build(self.vocal, self.beat_audio, self.sr,
                                 [(0.05, ["riser"])],
                                 bar_s=self.bar, beat_s=self.beat)
        self.assertEqual(rep["built"], [])
        self.assertIn("edge", rep["skipped"][0]["reason"])


class TestChordAwareHarmony(unittest.TestCase):

    def setUp(self):
        from mixengine.core.types import ChordEvent, KeyRegion, Note
        self.Note = Note
        self.y, self.regions = build_take(SONG)
        self.v = self.y[:, None]
        self.key_at = lambda t: KeyRegion(start=0.0, end=1e9, pc=0, mode="major",
                                          confidence=1.0)
        # C major throughout: every third above a scale note is diatonic.
        self.chord_at = lambda t: ChordEvent(start=0.0, end=1e9, root=0,
                                             quality="maj", confidence=1.0)

    def _notes(self):
        # One note per synthesised tone: SONG phrases are four notes each.
        out = []
        pos = 0.0
        for _label, notes, dur in SONG:
            per = dur / len(notes)
            for m in notes:
                out.append(self.Note(start=pos, end=pos + per, midi=float(m)))
                pos += per
            pos += 0.7
        return out

    def test_uses_scale_thirds_not_a_fixed_interval(self):
        built, rep = layers.build(self.v, SR, wanted={"harmony": self.regions},
                                  notes=self._notes(), chord_at=self.chord_at,
                                  key_at=self.key_at, seed=1)
        self.assertEqual(len(built), 1)
        note = built[0].note
        self.assertIn("chord-aware", note)
        # A diatonic third in C major is 3 semitones from E, 4 from C, D, F.
        self.assertIn("+3st", note)
        self.assertIn("+4st", note)

    def test_falls_back_and_says_so_without_notes(self):
        built, _ = layers.build(self.v, SR, wanted={"harmony": self.regions}, seed=1)
        self.assertIn("diatonic third", built[0].note)

    def test_harmony_is_silent_where_the_lead_is(self):
        built, _ = layers.build(self.v, SR, wanted={"harmony": [self.regions[2]]},
                                notes=self._notes(), chord_at=self.chord_at,
                                key_at=self.key_at, seed=1)
        a = built[0].audio[:, 0]
        gap = a[self.regions[0][1] + 1000:self.regions[1][0] - 1000]
        self.assertLess(float(np.abs(gap).max()), 1e-3)



class TestContourDistance(unittest.TestCase):
    """The phrase-contour comparison tolerates a one-point slide."""

    def test_a_slid_copy_of_the_same_melody_scores_as_identical(self):
        from mixengine.arrange.structure import _contour_distance
        melody = np.array([0, 2, 4, 5, 7, 5, 4, 2, 0, -1, 0, 2, 4, 2, 0, 0], dtype=float)
        slid = np.concatenate([[melody[0]], melody[:-1]])
        self.assertGreater(float(np.mean(np.abs(melody - slid))), 1.0)
        self.assertEqual(_contour_distance(melody, slid), 0.0)

    def test_a_different_melody_gains_little_from_the_slide(self):
        from mixengine.arrange.structure import _contour_distance
        rising = np.arange(16, dtype=float)
        falling = rising[::-1].copy()
        pointwise = float(np.mean(np.abs(rising - falling)))       # 8 semitones
        d = _contour_distance(rising, falling)
        self.assertLessEqual(d, pointwise)
        self.assertGreaterEqual(d, 0.9 * pointwise, f"slide gave {d:.2f} vs {pointwise:.2f}")

    def test_the_slide_is_never_more_than_one_point(self):
        from mixengine.arrange.structure import _contour_distance
        melody = np.array([0, 2, 4, 5, 7, 5, 4, 2, 0, -1, 0, 2, 4, 2, 0, 0], dtype=float)
        two = np.concatenate([melody[:2], melody[:-2]])
        self.assertGreater(_contour_distance(melody, two), 0.0)
