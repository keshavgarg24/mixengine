"""
The critic must not mark the engine down for the engine's own policy.

The master is allowed to stop short of its loudness target when the last
decibels would only come from crushing the dynamics, and it records that
it did. The critic was re-deriving that decision from a fixed window
instead of reading it, so the two numbers had to be kept in step by hand
and were not: lowering the limiting budget let masters land legitimately
further under target than the window allowed, and correct masters started
failing. A dynamic take delivered at -15.8 LUFS with DR 16.5 -- exactly as
intended -- scored 80% for obeying instructions.
"""

import unittest

import numpy as np

from mixengine.audio import critic
from mixengine.config import CFG

SR = 44100


def _master(lufs_ish: float = 0.06, seconds: float = 12.0) -> np.ndarray:
    """A mix that peaks on the ceiling but sits well below target loudness.

    Sparse loud transients over a quiet bed, which is what a dynamic
    master looks like to a loudness meter: true peak at the limit, low
    integrated loudness.
    """
    n = int(SR * seconds)
    rng = np.random.default_rng(3)
    t = np.arange(n) / SR
    bed = lufs_ish * np.sin(2 * np.pi * 220.0 * t)
    bed += 0.2 * lufs_ish * rng.standard_normal(n)
    y = bed.astype(np.float32)
    for k in range(int(seconds)):
        a = int(k * SR)
        b = a + int(0.05 * SR)
        y[a:b] += (0.85 * np.hanning(b - a)).astype(np.float32)
    y = np.clip(y, -0.891, 0.891)          # about -1.0 dBFS
    return np.stack([y, y], axis=1)


class TestAShortfallTheMasterChose(unittest.TestCase):

    def setUp(self):
        self.y = _master()
        self.profile = CFG.genre("default")

    def _gate(self, master_report):
        r = critic.evaluate(self.y, SR, "clean", self.profile, {}, {},
                            master_report=master_report)
        gate = next(g for g in r.gates if g.name == "loudness")
        return gate, r

    def test_a_master_that_declined_to_limit_harder_passes(self):
        gate, r = self._gate({"loudness_converged": False})
        self.assertTrue(gate.passed)
        self.assertEqual(gate.severity, "info")

    def test_and_is_not_marked_down_for_it(self):
        """Passing the gate is not enough: the sub-score must not punish
        the same decision, or a correct master still cannot score well."""
        _, r = self._gate({"loudness_converged": False})
        self.assertGreaterEqual(r.sub_scores["loudness"], 0.99)

    def test_a_master_that_believed_it_converged_still_fails(self):
        """Same audio, same shortfall. If the master thought it hit the
        target and did not, something went wrong and the critic must say
        so -- this is what stops the fix from excusing every miss."""
        gate, r = self._gate({"loudness_converged": True})
        self.assertFalse(gate.passed)
        self.assertLess(r.sub_scores["loudness"], 0.5)

    def test_being_over_target_is_always_a_miss(self):
        """Nothing in the chain intends to overshoot, so no report
        excuses it."""
        loud = np.clip(self.y * 12.0, -0.891, 0.891).astype(np.float32)
        r = critic.evaluate(loud, SR, "clean", self.profile, {}, {},
                            master_report={"loudness_converged": False})
        lufs_gate = next(g for g in r.gates if g.name == "loudness")
        if lufs_gate.value > self.profile.lufs_target + CFG.critic.lufs_tolerance_db:
            self.assertFalse(lufs_gate.passed)

    def test_without_a_report_it_falls_back_to_measuring(self):
        """A caller that passes no report gets the old inference, which is
        all there is to go on."""
        gate, _ = self._gate(None)
        self.assertIn(gate.severity, ("info", "warning"))


class TestPerceptualScoring(unittest.TestCase):
    """The learned scorer was installed, called wrongly, and silent.

    It was handed a numpy array where torchaudio's resampler needs a torch
    tensor, so every call raised inside a bare `except: pass`. A scorer
    that is present and broken looked exactly like one that is absent.
    """

    def setUp(self):
        from mixengine.core.capabilities import CAPS
        if not CAPS.audiobox:
            self.skipTest("audiobox_aesthetics is not installed")

    def test_it_returns_scores_rather_than_nothing(self):
        out = critic._perceptual_scores(_master(seconds=11.0), SR)
        self.assertIn("production_quality", out)
        self.assertGreater(out["production_quality"], 0.0)
        self.assertLessEqual(out["production_quality"], 1.0)

    def test_a_mono_take_is_accepted_too(self):
        """as_2d is what shapes the tensor, so a 1-D array must work."""
        mono = _master(seconds=11.0)[:, 0]
        out = critic._perceptual_scores(mono, SR)
        self.assertIn("production_quality", out)

    def test_absence_is_reported_rather_than_assumed(self):
        """The capability exists so `doctor` can name what is missing,
        which is how this one went unnoticed."""
        from mixengine.core.capabilities import CAPS
        self.assertIn("audiobox", CAPS.to_dict())


if __name__ == "__main__":
    unittest.main()
