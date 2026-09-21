"""
The critic's loudness gate.

A master that stops short of its loudness target because the limiter is
already the binding constraint is following the engine's own policy
(`MasterConfig.max_limiting_db`), and the critic used to flag exactly that
as a warning -- the engine contradicting itself on every trap render whose
drums have any crest factor. The gate now recognises the situation from the
audio alone: true peak on the ceiling, loudness a little under target.
"""

import os
import sys
import unittest
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mixengine.audio import critic, dsp                          # noqa: E402
from mixengine.config import CFG, GENRE_PROFILES                 # noqa: E402
from mixengine.core import audio_io                              # noqa: E402

SR = 44100


def bursty_noise(seconds: float = 6.0, peak_db: float = -1.0, seed: int = 3):
    """Stereo noise with a high crest factor, peak placed exactly."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    y = rng.standard_normal((n, 2))
    gate = (np.arange(n) % int(0.2 * SR)) < int(0.05 * SR)      # 25% duty
    y *= gate[:, None]
    y *= dsp.db_to_lin(peak_db) / np.abs(y).max()
    return y.astype(np.float64)


def loudness_gate(y, target_lufs):
    profile = replace(GENRE_PROFILES["default"], lufs_target=target_lufs)
    rep = critic.evaluate(y, SR, "test", profile)
    return next(g for g in rep.gates if g.name == "loudness")


class TestLoudnessGate(unittest.TestCase):

    def test_within_tolerance_passes(self):
        y = bursty_noise()
        lufs = audio_io.integrated_lufs(y, SR)
        g = loudness_gate(y, lufs + 1.0)
        self.assertTrue(g.passed)
        self.assertEqual(g.severity, "warning")

    def test_short_of_target_with_peak_on_the_ceiling_is_information(self):
        y = bursty_noise(peak_db=-1.0)
        lufs = audio_io.integrated_lufs(y, SR)
        g = loudness_gate(y, lufs + 2.2)         # past the 1.5 dB tolerance
        self.assertTrue(g.passed, g.message)
        self.assertEqual(g.severity, "info")
        self.assertIn("ceiling", g.message)
        self.assertIn("ceiling", g.to_dict()["message"],
                      "an info gate's message must survive serialisation")

    def test_same_shortfall_with_headroom_left_is_a_warning(self):
        y = bursty_noise(peak_db=-7.0)           # limiter never touched it
        lufs = audio_io.integrated_lufs(y, SR)
        g = loudness_gate(y, lufs + 2.2)
        self.assertFalse(g.passed)
        self.assertEqual(g.severity, "warning")

    def test_far_short_of_target_is_a_warning_even_at_the_ceiling(self):
        y = bursty_noise(peak_db=-1.0)
        lufs = audio_io.integrated_lufs(y, SR)
        grace = CFG.critic.lufs_tolerance_db + CFG.critic.lufs_ceiling_grace_db
        g = loudness_gate(y, lufs + grace + 1.0)
        self.assertFalse(g.passed)

    def test_passed_gate_messages_do_not_read_as_failures(self):
        y = bursty_noise()
        lufs = audio_io.integrated_lufs(y, SR)
        rep = critic.evaluate(y, SR, "test",
                              replace(GENRE_PROFILES["default"], lufs_target=lufs))
        for g in rep.to_dict()["gates"]:
            if g["passed"] and g["severity"] != "info":
                self.assertNotIn("exceeds", g["message"])
                self.assertIn("within limit", g["message"])


if __name__ == "__main__":
    unittest.main()
