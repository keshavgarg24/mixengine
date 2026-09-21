"""
Regression tests for the master chain's loudness and true-peak behaviour.

Three defects are locked down here, all found by running the engine on real
audio rather than by reading it:

  1. The limiter did not hold its own ceiling. A zero-phase smoothing pass
     applied to the gain curve raised it *above* the required reduction
     (measured overshoot: 0.068 linear), and the audio was additionally
     delayed against a max filter that was already centred. Result: +0.08
     dBTP output against a -1.0 dBTP ceiling.

  2. Because the ceiling was breached, a corrective trim ran afterwards --
     and that trim undid the loudness targeting one-for-one. A -5.05 dB
     peak trim produced a -5.05 dB loudness miss, delivering -13.6 LUFS
     against a -8.5 target.

  3. There was no honest reporting of a missed target. The chain returned a
     number and said nothing about whether it had been reached.

These require numpy/scipy only; `pyloudnorm` is optional and the LUFS
fallback keeps the relative assertions meaningful without it.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mixengine.audio import dsp, master                                # noqa: E402
from mixengine.config import CFG                                       # noqa: E402

SR = 44100


def transient_signal(seconds=3.0, sr=SR, seed=0):
    """Tone plus noise plus sharp spikes.

    The spikes matter: inter-sample overshoot is largest on sharp
    transients, which is exactly where a sample-domain limiter lets true
    peaks through.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    y = 0.6 * np.sin(2 * np.pi * 220 * t) + 0.3 * rng.normal(0, 1, t.size)
    y[::5000] += 1.2
    return np.stack([y, y], axis=1).astype(np.float32)


def dense_signal(seconds=6.0, sr=SR, seed=1):
    """Loud, dense, low-crest material -- a mastered-style mix."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    y = np.tanh(3.0 * (0.5 * np.sin(2 * np.pi * 110 * t)
                       + 0.3 * np.sin(2 * np.pi * 220 * t)
                       + 0.2 * rng.normal(0, 1, t.size)))
    return np.stack([y, y * 0.98], axis=1).astype(np.float32)


class TestLimiterHoldsCeiling(unittest.TestCase):

    def test_true_peak_never_exceeds_the_ceiling(self):
        """A limiter that breaches its own ceiling is a delivery defect.

        Encoders and D/A converters see the inter-sample peak, which is why
        -1 dBTP is the delivery standard rather than -1 dBFS.
        """
        y = transient_signal()
        self.assertGreater(dsp.true_peak_db(y), 0.0, "fixture should start hot")
        for ceiling in (-1.0, -2.0, -0.3):
            out = master._lookahead_limiter(y, SR, ceiling)
            tp = dsp.true_peak_db(out)
            self.assertLessEqual(tp, ceiling + 0.05,
                                 "true peak %.2f exceeded ceiling %.2f" % (tp, ceiling))

    def test_smoothing_can_only_lower_gain(self):
        """The specific mechanism of the original breach.

        Zero-phase smoothing of a gain curve that must be an upper bound
        can lift it at a dip. Clamping after smoothing is what guarantees
        the ceiling.
        """
        y = transient_signal()
        y2 = dsp.as_2d(y).astype(np.float64)
        ceiling = dsp.db_to_lin(-1.0)
        la = max(1, int(0.005 * SR))
        peak = master._true_peak_envelope(y2)
        from scipy.ndimage import maximum_filter1d
        padded = np.pad(peak, (la, la), mode="edge")
        running = maximum_filter1d(padded, size=la * 2 + 1)[la:la + len(peak)]
        required = np.minimum(ceiling / np.maximum(running, 1e-9), 1.0)

        naive = dsp.smooth(required, SR, 0.002)
        self.assertGreater(float(np.max(naive - required)), 0.0,
                           "expected unclamped smoothing to overshoot")
        clamped = np.minimum(naive, required)
        self.assertLessEqual(float(np.max(clamped - required)), 1e-12)

    def test_true_peak_envelope_detects_inter_sample_overshoot(self):
        y = transient_signal()
        y2 = dsp.as_2d(y).astype(np.float64)
        env = master._true_peak_envelope(y2)
        self.assertEqual(env.size, len(y2))
        sample_peak = np.max(np.abs(y2), axis=1)
        self.assertGreaterEqual(float(np.max(env)), float(np.max(sample_peak)) - 1e-9)

    def test_limiter_is_gain_aligned_with_the_signal(self):
        """The max filter is centred, so delaying the audio misaligns it.

        A misaligned gain curve both breaches the ceiling and audibly
        ducks material that did not need it.
        """
        y = np.zeros((SR, 2), dtype=np.float32)
        y[SR // 2] = 1.5                      # one isolated spike
        out = master._lookahead_limiter(y, SR, -1.0)
        loudest = int(np.argmax(np.abs(out[:, 0])))
        self.assertLess(abs(loudest - SR // 2), int(0.008 * SR),
                        "peak moved by more than the look-ahead window")

    def test_silence_is_untouched(self):
        y = np.zeros((SR, 2), dtype=np.float32)
        out = master._lookahead_limiter(y, SR, -1.0)
        self.assertLess(float(np.max(np.abs(out))), 1e-6)


class TestLoudnessTargeting(unittest.TestCase):

    def test_no_corrective_trim_is_needed_after_limiting(self):
        """The trim is the symptom; its absence is the proof of the fix.

        Before, the trim was -5.05 dB and the loudness miss was -5.05 dB --
        the same number, because a linear trim cannot satisfy a LUFS target
        and a dBTP ceiling at once.
        """
        _, rep = master.master(dense_signal(), SR, CFG.genre("trap"))
        self.assertLessEqual(abs(float(rep.get("true_peak_trim_db", 0.0))), 0.1)

    def test_targets_are_met_when_the_budget_allows(self):
        y = dense_signal()
        for genre in ("rnb", "lofi", "hip_hop"):
            prof = CFG.genre(genre)
            _, rep = master.master(y, SR, prof)
            if rep.get("loudness_converged"):
                self.assertLess(abs(rep["output_lufs"] - prof.lufs_target), 0.7,
                                "%s: converged but %.2f off target"
                                % (genre, rep["output_lufs"] - prof.lufs_target))

    def test_output_respects_the_true_peak_standard(self):
        for genre in ("trap", "drill", "rnb", "lofi"):
            out, rep = master.master(dense_signal(), SR, CFG.genre(genre))
            self.assertLessEqual(rep["output_true_peak_db"],
                                 CFG.mix.true_peak_db + 0.05, genre)

    def test_a_missed_target_is_reported_not_hidden(self):
        """Loudness is always reachable with enough limiting; the engine
        stops at a musical budget instead and must say that it did."""
        quiet = dense_signal() * 0.02
        _, rep = master.master(quiet, SR, CFG.genre("drill"))
        self.assertIn("loudness_converged", rep)
        self.assertIn("loudness_error_db", rep)
        if not rep["loudness_converged"]:
            self.assertIn("loudness_note", rep)

    def test_limiting_budget_is_respected(self):
        """Charged against gain beyond the ceiling, not total gain -- a
        quiet mix must not spend its budget on makeup gain."""
        _, rep = master.master(dense_signal() * 0.05, SR, CFG.genre("drill"))
        self.assertLessEqual(rep["gain_into_limiter_db"], CFG.mix.max_limiting_db + 0.01)
        self.assertGreater(rep["free_gain_db"], 0.0)

    def test_louder_genre_targets_produce_louder_masters(self):
        y = dense_signal()
        lufs = {}
        for genre in ("lofi", "rnb", "hip_hop"):
            _, rep = master.master(y, SR, CFG.genre(genre))
            lufs[genre] = rep["output_lufs"]
        self.assertGreater(lufs["rnb"], lufs["lofi"])
        self.assertGreater(lufs["hip_hop"], lufs["lofi"])

    def test_report_is_json_safe(self):
        import json
        _, rep = master.master(dense_signal(), SR, CFG.genre("trap"))
        json.dumps(rep)


class TestKeyConfidenceGate(unittest.TestCase):
    """A pitch shift must not be justified by a weak key estimate.

    On a real render the engine detected the vocal key at 0.49 confidence
    and transposed by 2 semitones on the strength of it. Unlike a wrong
    tempo, a wrong transposition cannot be recovered from downstream.
    """

    def _pair(self, vocal_conf):
        from mixengine.core.keys import Key
        v = {"key": Key(4, "major").to_dict(), "bpm": 134.0, "syllable_rate": 3.2,
             "key_confidence": vocal_conf, "performance_type": "sung",
             "beat_requirements": {"genres": ["rnb"]}}
        b = {"key": Key(11, "minor").to_dict(), "bpm": 140.0, "status": "ok",
             "beat_id": "b1", "genre": "trap", "pocket_score": 0.78,
             "key_confidence": 1.0, "grid_stability": 0.97}
        from mixengine.analysis import matching
        return matching.score_pair(v, b)

    def test_low_confidence_suppresses_the_shift(self):
        for conf in (0.0, 0.3, 0.49):
            m = self._pair(conf)
            self.assertEqual(m.semitone_shift, 0, "shift applied at conf=%.2f" % conf)
            self.assertTrue(any("uncertain" in w for w in m.warnings))

    def test_high_confidence_still_allows_the_shift(self):
        m = self._pair(0.95)
        self.assertNotEqual(m.semitone_shift, 0)
        self.assertFalse(any("uncertain" in w for w in m.warnings))

    def test_harmonic_score_blends_toward_neutral_when_uncertain(self):
        """An uncertain key should let the other dimensions decide rather
        than assert a confident harmonic verdict."""
        low = self._pair(0.0).sub_scores["harmonic"]
        high = self._pair(0.95).sub_scores["harmonic"]
        self.assertLess(abs(low - 0.5), abs(high - 0.5))


if __name__ == "__main__":
    unittest.main(verbosity=2)
