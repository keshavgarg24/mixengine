"""
Regression tests for the Phase 0 analysis fixes.

These cover the two failures that made the engine's only recorded real run
useless: every catalog beat was classified as drum-only, disabling the
entire harmonic layer, and a 170-second vocal collapsed into four phrases,
one of them 57 seconds long.

Both are tested here against synthetic material whose ground truth is
known, and both reproduce the original failure when the old rule is
applied -- so these are genuine regression tests, not restatements of the
new implementation.

Runs on numpy + scipy alone; librosa is not required because the fixed
logic is separated from the feature extraction that needs it.
"""

from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mixengine.analysis.analysis import (                              # noqa: E402
    atonality_score, chroma_concentration, detect_phrases,
    voice_activity_threshold, _split_overlong,
)
from mixengine.config import CFG                                       # noqa: E402

SR = 22050


# ═════════════════════════════════════════════════════════════════════════════
# Synthetic material
# ═════════════════════════════════════════════════════════════════════════════

def tonal_chroma(n_frames=400,
                 chords=((0, 4, 7), (5, 9, 0), (7, 11, 2), (9, 0, 4)),
                 seed=0, floor=0.35):
    """Chroma for a tonal track: each frame holds one triad, chords change.

    The `floor` is not decoration and is the reason this reproduces the
    real failure. CQT chroma of an actual mix never has empty bins: drums
    are broadband, every pitched note leaks into its neighbours' bins, and
    librosa normalises each frame so the strongest bin reads 1.0. A
    synthetic chromagram with near-zero non-chord bins is far cleaner than
    anything the analyser will ever see, and testing against it would prove
    nothing about the code's behaviour on real audio.

    With a realistic floor, time-averaging this over a four-chord loop is
    almost perfectly flat -- which is exactly why the old flatness test
    reported ordinary tonal music as drum-only.
    """
    rng = np.random.default_rng(seed)
    c = rng.uniform(floor * 0.85, floor * 1.15, size=(12, n_frames))
    per = max(1, n_frames // len(chords))
    for i in range(n_frames):
        triad = chords[(i // per) % len(chords)]
        for pc in triad:
            c[pc, i] = rng.uniform(0.85, 1.0)
            # Harmonic leakage into the fifth above, as a real CQT shows.
            c[(pc + 7) % 12, i] = max(c[(pc + 7) % 12, i], rng.uniform(0.4, 0.55))
    return c


def drum_chroma(n_frames=400, seed=0):
    """Chroma for drum-only material: energy spread across all classes."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.35, 0.65, size=(12, n_frames))


def old_flatness_rule(chroma, harmonic_ratio):
    """The rule that shipped, reproduced so the regression is demonstrable."""
    pcp = chroma.mean(axis=1)
    flatness = float(np.exp(np.mean(np.log(pcp + 1e-9))) / (np.mean(pcp) + 1e-9))
    return harmonic_ratio < 0.25 or flatness > 0.92


def make_vocal(sr=SR, phrase_spans=((1.0, 4.0), (5.0, 8.5), (10.0, 13.0)),
               total_s=15.0, noise_db=-60.0, rt60=0.0, seed=1):
    """A synthetic vocal: tone bursts in phrases, with optional room tail.

    `rt60 > 0` adds an exponentially decaying tail after each phrase, which
    is what fills the gaps on a reverberant recording and defeats a fixed
    peak-relative silence threshold.
    """
    rng = np.random.default_rng(seed)
    n = int(total_s * sr)
    t = np.arange(n) / sr
    y = rng.normal(0, 10 ** (noise_db / 20.0), n)

    for (a, b) in phrase_spans:
        i0, i1 = int(a * sr), int(b * sr)
        seg = np.zeros(i1 - i0)
        # Sum a few harmonics with per-syllable amplitude, so the envelope
        # looks like speech rather than a steady tone.
        f0 = 180.0
        for h in (1, 2, 3, 4):
            seg += (0.5 / h) * np.sin(2 * np.pi * f0 * h * t[i0:i1])
        syl = 0.5 + 0.5 * np.abs(np.sin(2 * np.pi * 3.5 * t[i0:i1]))
        y[i0:i1] += seg * syl * 0.4

        if rt60 > 0:
            tail_n = int(min(rt60 * 1.5, total_s - b) * sr)
            if tail_n > 0 and i1 + tail_n <= n:
                decay = 10 ** (-3.0 * np.arange(tail_n) / (rt60 * sr))
                y[i1:i1 + tail_n] += seg[-tail_n:] * decay * 0.4 \
                    if tail_n <= len(seg) else 0.0
    return y.astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# Atonality
# ═════════════════════════════════════════════════════════════════════════════

class TestAtonality(unittest.TestCase):

    def test_old_rule_misclassified_tonal_music(self):
        """Demonstrates the shipped bug rather than asserting around it.

        Every one of the five top matches in the engine's recorded run was
        labelled 'drum-based beat - works in any key', which pinned the
        harmonic sub-score at a flat constant and disabled key matching,
        the Camelot wheel and the dissonance penalty for the whole catalog.
        """
        self.assertTrue(old_flatness_rule(tonal_chroma(), harmonic_ratio=0.65),
                        "expected the old rule to wrongly call tonal music atonal")

    def test_per_frame_concentration_separates_tonal_from_drums(self):
        """Per frame, a triad concentrates energy; a drum hit does not.

        The gap is narrower than intuition suggests -- roughly 0.47 against
        0.31 on realistic chroma -- because leakage and per-frame
        normalisation lift the non-chord bins. It is still a clean,
        consistent separation, and unlike the time-averaged flatness it
        does not vanish as a song visits more pitch classes.
        """
        tonal = float(np.median(chroma_concentration(tonal_chroma(), top_k=3)))
        drums = float(np.median(chroma_concentration(drum_chroma(), top_k=3)))
        self.assertGreater(tonal, 0.42)
        self.assertLess(drums, 0.35)
        self.assertGreater(tonal - drums, 0.10)

    def test_tonal_track_scores_as_tonal(self):
        score = atonality_score(0.65, chroma_concentration(tonal_chroma()))
        self.assertLess(score, 0.4)

    def test_drum_only_track_scores_as_atonal(self):
        score = atonality_score(0.12, chroma_concentration(drum_chroma()))
        self.assertGreater(score, 0.7)

    def test_both_kinds_of_evidence_are_required(self):
        """The old rule OR-ed its two tests, so one over-sensitive test
        could veto the harmonic layer alone. Averaging means a single
        ambiguous signal cannot produce a confident verdict."""
        # Percussive energy but clearly pitched content: not atonal.
        mixed = atonality_score(0.20, chroma_concentration(tonal_chroma()))
        self.assertLess(mixed, 0.62)
        # Harmonic energy but no pitch focus: also not a confident verdict.
        mixed2 = atonality_score(0.70, chroma_concentration(drum_chroma()))
        self.assertLess(mixed2, 0.62)

    def test_no_evidence_is_not_a_verdict(self):
        self.assertLess(atonality_score(0.45, np.zeros(0)), 0.62)

    def test_score_is_bounded(self):
        for hr in (0.0, 0.25, 0.5, 1.0):
            for ch in (tonal_chroma(50), drum_chroma(50)):
                s = atonality_score(hr, chroma_concentration(ch))
                self.assertGreaterEqual(s, 0.0)
                self.assertLessEqual(s, 1.0)


# ═════════════════════════════════════════════════════════════════════════════
# Phrase detection
# ═════════════════════════════════════════════════════════════════════════════

class TestPhraseDetection(unittest.TestCase):

    def test_clean_take_splits_into_the_right_phrases(self):
        spans = ((1.0, 4.0), (5.0, 8.5), (10.0, 13.0))
        y = make_vocal(phrase_spans=spans, total_s=15.0)
        ph = detect_phrases(y, SR)
        self.assertEqual(len(ph), 3)
        for (s, e), (a, b) in zip(ph, spans):
            self.assertAlmostEqual(s / SR, a, delta=0.25)
            self.assertAlmostEqual(e / SR, b, delta=0.25)

    def test_reverberant_take_does_not_collapse_into_one_phrase(self):
        """The recorded failure: RT60 0.83 s, 170 s of rap, 4 phrases.

        A reverb tail sits well inside 38 dB of the peak, so a fixed
        peak-relative threshold reads the gaps as active and merges every
        phrase into one. The adaptive threshold has to find the real valley
        in the level distribution instead.
        """
        spans = ((1.0, 4.0), (5.5, 8.5), (10.0, 13.0))
        y = make_vocal(phrase_spans=spans, total_s=15.0, rt60=0.8, noise_db=-45.0)
        ph = detect_phrases(y, SR)
        self.assertGreaterEqual(len(ph), 2,
                                "reverberant take collapsed into one phrase")
        longest = max((e - s) / SR for s, e in ph)
        self.assertLess(longest, 12.5)

    def test_no_phrase_exceeds_the_musical_maximum(self):
        """A 57-second phrase is a detection failure, not a long phrase.

        Even when the level signal gives the detector nothing to work with,
        the output must stay usable by the stages that consume it.
        """
        y = make_vocal(phrase_spans=((0.5, 29.5),), total_s=30.0)
        ph = detect_phrases(y, SR)
        self.assertGreater(len(ph), 1)
        for s, e in ph:
            self.assertLessEqual((e - s) / SR, CFG.analysis.max_phrase_dur_s + 0.5)

    def test_breaths_do_not_split_a_phrase(self):
        """Short gaps are bridged; only real pauses separate phrases."""
        y = make_vocal(phrase_spans=((1.0, 3.0), (3.15, 5.0)), total_s=7.0)
        ph = detect_phrases(y, SR)
        self.assertEqual(len(ph), 1)

    def test_silence_yields_no_phrases(self):
        self.assertEqual(detect_phrases(np.zeros(SR * 3, dtype=np.float32), SR), [])

    def test_very_short_input_is_safe(self):
        self.assertEqual(detect_phrases(np.zeros(100, dtype=np.float32), SR), [])

    def test_phrases_are_sorted_and_non_overlapping(self):
        y = make_vocal(phrase_spans=((1.0, 3.0), (4.0, 6.0), (7.0, 9.0)),
                       total_s=11.0)
        ph = detect_phrases(y, SR)
        for i in range(len(ph) - 1):
            self.assertLessEqual(ph[i][0], ph[i + 1][0])
            self.assertLess(ph[i][0], ph[i][1])


class TestVoiceActivityThreshold(unittest.TestCase):

    def test_finds_the_valley_in_a_bimodal_distribution(self):
        quiet = np.random.default_rng(0).normal(-55, 3, 400)
        loud = np.random.default_rng(1).normal(-15, 4, 400)
        open_db, close_db = voice_activity_threshold(np.concatenate([quiet, loud]))
        self.assertGreater(open_db, close_db)
        self.assertGreater(open_db, -50.0)
        self.assertLess(open_db, -20.0)

    def test_adapts_when_the_gap_is_small(self):
        """A noisy recording has a narrow gap; a fixed 38 dB offset would
        put the threshold below the noise floor and mark everything active."""
        quiet = np.random.default_rng(0).normal(-32, 2, 400)
        loud = np.random.default_rng(1).normal(-14, 3, 400)
        r_db = np.concatenate([quiet, loud])
        open_db, _ = voice_activity_threshold(r_db)
        naive = float(np.percentile(r_db, 95)) - CFG.analysis.silence_rel_db
        self.assertGreater(open_db, naive,
                           "adaptive threshold must sit above the naive one here")

    def test_constant_level_degrades_to_all_active(self):
        r_db = np.full(200, -20.0)
        open_db, close_db = voice_activity_threshold(r_db)
        self.assertLess(open_db, -20.0)
        self.assertLess(close_db, open_db)

    def test_tiny_input_is_safe(self):
        self.assertEqual(len(voice_activity_threshold(np.zeros(2))), 2)


class TestSplitOverlong(unittest.TestCase):

    def test_splits_at_the_quietest_interior_point(self):
        # max_frames must exceed half the region, or a single split cannot
        # satisfy it and the function correctly recurses further.
        r_db = np.full(1000, -10.0)
        r_db[600] = -60.0                      # the breath
        out = _split_overlong([(0, 1000)], r_db, max_frames=700, min_frames=50)
        self.assertEqual(len(out), 2)
        self.assertAlmostEqual(out[0][1], 600, delta=5)

    def test_leaves_acceptable_regions_alone(self):
        r_db = np.full(300, -10.0)
        self.assertEqual(_split_overlong([(0, 300)], r_db, 500, 50), [(0, 300)])

    def test_recurses_until_every_region_fits(self):
        r_db = np.random.default_rng(3).normal(-20, 5, 4000)
        out = _split_overlong([(0, 4000)], r_db, max_frames=500, min_frames=50)
        self.assertGreater(len(out), 4)
        for s, e in out:
            self.assertLessEqual(e - s, 500)

    def test_never_produces_slivers(self):
        r_db = np.full(1000, -10.0)
        r_db[55] = -80.0                       # a dip very close to the edge
        out = _split_overlong([(0, 1000)], r_db, max_frames=400, min_frames=50)
        for s, e in out:
            self.assertGreaterEqual(e - s, 50)


if __name__ == "__main__":
    unittest.main(verbosity=2)


from mixengine.analysis import analysis                                # noqa: E402


class TestFallbackSections(unittest.TestCase):
    """The terminal rung of the structure ladder: fixed 8-bar blocks.

    Only the times used to be clipped at the end of the file, so a beat
    three seconds long was described as an eight-bar section.
    """

    def sections(self, duration_s, bar=2.5):
        y = np.zeros(int(duration_s * 8000), dtype=np.float32)
        downbeats = np.arange(0.0, duration_s, bar)
        return analysis._fallback_sections(y, 8000, downbeats)

    def test_a_beat_shorter_than_a_block_claims_only_the_bars_it_has(self):
        (only,) = self.sections(3.0)
        self.assertEqual((only["start_bar"], only["end_bar"]), (0, 1))
        self.assertEqual(only["end"], 3.0)

    def test_whole_blocks_are_eight_bars_as_before(self):
        secs = self.sections(40.0)
        self.assertEqual([(s["start_bar"], s["end_bar"]) for s in secs[:2]],
                         [(0, 8), (8, 16)])

    def test_every_block_claims_the_bars_its_audio_covers(self):
        for duration in (3.0, 17.5, 40.0, 63.0):
            for s in self.sections(duration):
                self.assertAlmostEqual((s["end_bar"] - s["start_bar"]) * 2.5,
                                       s["end"] - s["start"], delta=1.3,
                                       msg="%.1f s file, section %r"
                                           % (duration, s))


class TestTempoAgreement(unittest.TestCase):
    """Two independent estimators; their agreement is the confidence."""

    def _onsets(self, bpm, n=120, jitter=0.010, seed=1, subdiv=2):
        rng = np.random.default_rng(seed)
        step = 60.0 / bpm / subdiv
        base = np.arange(n) * step
        keep = rng.random(n) > 0.25             # drop a quarter: real phrasing
        return np.sort(base[keep] + rng.normal(0, jitter, keep.sum()))

    def test_agreement_raises_confidence_above_the_histogram_alone(self):
        d = analysis.estimate_vocal_tempo_detailed(self._onsets(128.0))
        self.assertIn(d["agreement"], ("exact", "octave"))
        h_bpm, h_conf, _ = analysis._tempo_histogram(self._onsets(128.0))
        self.assertGreaterEqual(d["confidence"], h_conf)
        self.assertTrue(analysis._octave_related(d["bpm"], 128.0))

    def test_rubato_is_reported_with_low_confidence(self):
        rng = np.random.default_rng(4)
        onsets = np.sort(rng.uniform(0, 40, 90))
        d = analysis.estimate_vocal_tempo_detailed(onsets)
        self.assertLess(d["confidence"], 0.5)

    def test_both_readings_are_surfaced_on_disagreement(self):
        d = analysis.estimate_vocal_tempo_detailed(self._onsets(128.0))
        self.assertIn("histogram_bpm", d)
        self.assertIn("autocorr_bpm", d)
        self.assertGreater(d["autocorr_bpm"], 0)


class TestReferenceBeatTempo(unittest.TestCase):
    """A take recorded over a known beat has that beat's tempo, full stop."""

    def test_reference_beat_sets_the_tempo_and_is_verified(self):
        from mixengine.analysis import vocal_dna
        import inspect
        sig = inspect.signature(vocal_dna.extract)
        self.assertIn("reference_beat_dna", sig.parameters)

    def test_verify_tempo_adopts_the_beat_when_onsets_fit_it(self):
        bpm = 140.0
        beats = np.arange(0, 30, 60.0 / bpm)
        grid = analysis.subdivide(beats, 4)
        onsets = grid[::3][:80] + 0.004
        v_bpm, src, err = analysis.verify_tempo(onsets, 133.0, grid)
        self.assertEqual(src, "verified_beat")
        self.assertAlmostEqual(v_bpm, bpm, delta=0.5)
        self.assertLess(err, 0.01)


class TestCrepeDecoder(unittest.TestCase):
    """The dither-free decoder the engine hands to torchcrepe.

    torchcrepe's `viterbi` returns bin centres plus up to twenty cents of
    random dither; ours returns the probability-weighted mean around the
    Viterbi bin, so a pitch between two bins is recovered, and the same
    input always gives the same answer.
    """

    def setUp(self):
        try:
            import torch, torchcrepe  # noqa: F401
        except ImportError:
            self.skipTest("torch and torchcrepe are not installed")

    @staticmethod
    def _logits(centre_cents, frames=8, width_bins=1.5):
        import torch, torchcrepe
        idx = torch.arange(360).float()
        centres = torchcrepe.CENTS_PER_BIN * idx + 1997.3794084376191
        z = (centres - centre_cents) / (width_bins * torchcrepe.CENTS_PER_BIN)
        probs = torch.exp(-0.5 * z ** 2).clamp(1e-6, 1 - 1e-6)
        logits = torch.log(probs / (1 - probs))
        return logits[None, :, None].repeat(1, 1, frames)

    def test_a_pitch_between_two_bins_is_recovered(self):
        import torchcrepe
        from mixengine.analysis.analysis import _decode_viterbi_precise
        a440 = 1200.0 * math.log2(440.0 / 10.0)          # 13.9 cents off a bin centre
        _, hz = _decode_viterbi_precise(self._logits(a440))
        err = 1200.0 * abs(math.log2(float(hz[0, 0]) / 440.0))
        self.assertLess(err, 1.0, f"{float(hz[0, 0]):.2f} Hz is {err:.1f} cents off")
        # The stock decoder dithers randomly, so its error is a draw and
        # not a fact about either decoder. Seeded, because on roughly one
        # run in a hundred the dither lands nearer the true pitch than our
        # 0.15-cent error and fails a comparison that is otherwise sound.
        import torch
        torch.manual_seed(0)
        _, stock = torchcrepe.decode.viterbi(self._logits(a440))
        self.assertGreater(1200.0 * abs(math.log2(float(stock[0, 0]) / 440.0)), err)

    def test_the_same_input_gives_the_same_answer(self):
        from mixengine.analysis.analysis import _decode_viterbi_precise
        logits = self._logits(6000.0)
        _, a = _decode_viterbi_precise(logits.clone())
        _, b = _decode_viterbi_precise(logits.clone())
        self.assertTrue(bool((a == b).all()))
        self.assertTrue(bool((a[0, 0] == a[0]).all()), "identical frames must agree")
