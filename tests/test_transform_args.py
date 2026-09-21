"""
Regression tests for the Rubber Band command construction.

The bug these lock down was invisible from the outside. The engine passed
`rbargs={"--formant": "", "-F": ""}` to pyrubberband, which emits every
entry as a `key value` pair -- producing `--formant "" -F ""`. Rubber Band
reads the empty strings as extra positional filenames and exits 2. The
caller caught the exception and silently fell back to the librosa phase
vocoder, so formant preservation never ran once while `doctor` and the
README both reported it as active.

Verified against the real binary before writing these:

    rubberband -q --pitch 2              in.wav out.wav   -> rc=0
    rubberband -q --formant "" -F "" ... in.wav out.wav   -> rc=2
    rubberband -q --formant --pitch 2    in.wav out.wav   -> rc=0

`soundfile` is stubbed so these run without the audio dependencies; what
is under test is the argument vector, which is exactly where the defect
lived.
"""

from __future__ import annotations

import os
import sys
import types
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _install_soundfile_stub():
    """Minimal soundfile stand-in: records writes, returns a fixed buffer."""
    if "soundfile" in sys.modules:
        return
    stub = types.ModuleType("soundfile")
    stub.written = []

    def write(path, data, sr, subtype=None):
        stub.written.append((path, np.asarray(data).shape, sr, subtype))
        with open(path, "wb") as f:
            f.write(b"\0" * 64)

    def read(path, dtype="float32", always_2d=True):
        return np.zeros((1024, 1), dtype=np.float32), 48000

    stub.write, stub.read = write, read
    sys.modules["soundfile"] = stub


_install_soundfile_stub()

from mixengine.audio import transform                                  # noqa: E402


class _Result:
    def __init__(self, rc=0):
        self.returncode = rc
        self.stdout = ""
        self.stderr = ""


def capture_args(fn, *a, **kw):
    """Run `fn`, returning the argv Rubber Band would have been invoked with."""
    seen = {}

    def fake_run(cmd, **kwargs):
        if cmd and cmd[0] == "rubberband" and "--help" in cmd:
            r = _Result(0)
            r.stdout = "--formant\n--fine\n"
            return r
        seen["cmd"] = list(cmd)
        # Make the output file exist so the wrapper proceeds to read it.
        if len(cmd) >= 2:
            with open(cmd[-1], "wb") as f:
                f.write(b"\0" * 64)
        return _Result(0)

    with mock.patch.object(transform.subprocess, "run", side_effect=fake_run):
        transform._RB_FORMANT_OK = None
        fn(*a, **kw)
    return seen.get("cmd", [])


class TestRubberBandArguments(unittest.TestCase):

    def setUp(self):
        self.y = np.zeros((4096, 1), dtype=np.float32)
        transform._RB_FORMANT_OK = None

    def test_formant_is_a_bare_flag_with_no_value(self):
        """The whole bug in one assertion.

        `--formant` takes no argument. Any token following it that is not
        another flag means an empty value was emitted, which is what made
        Rubber Band exit 2.
        """
        cmd = capture_args(transform._rubberband, self.y, 48000,
                           ["--pitch", "2.0"], True, "test")
        self.assertIn("--formant", cmd)
        i = cmd.index("--formant")
        self.assertTrue(cmd[i + 1].startswith("-"),
                        "a value was emitted after --formant: %r" % cmd[i + 1:i + 2])
        self.assertNotIn("", cmd, "empty argument would be read as a filename")

    def test_no_formant_flag_when_not_requested(self):
        cmd = capture_args(transform._rubberband, self.y, 48000,
                           ["--pitch", "2.0"], False, "test")
        self.assertNotIn("--formant", cmd)

    def test_r3_engine_is_requested_when_available(self):
        """R3 almost always beats R2, and is markedly better on vocals and
        soft onsets. The extra CPU is the right trade when rendering
        offline."""
        cmd = capture_args(transform._rubberband, self.y, 48000,
                           ["--pitch", "2.0"], True, "test")
        self.assertIn("--fine", cmd)

    def test_stretch_passes_the_duration_ratio_directly(self):
        """`--time X` stretches to X times the original duration.

        Our ratio is already a duration multiplier, so it goes straight
        through. Any reciprocal here would invert every tempo decision in
        the engine while still producing plausible-sounding output, which
        is the kind of error that survives a listening test.
        """
        cmd = capture_args(transform.time_stretch, self.y, 48000, 1.25, True)
        self.assertIn("--time", cmd)
        self.assertAlmostEqual(float(cmd[cmd.index("--time") + 1]), 1.25, places=6)

    def test_pitch_passes_semitones_directly(self):
        cmd = capture_args(transform.pitch_shift, self.y, 48000, -3.0, True)
        self.assertIn("--pitch", cmd)
        self.assertAlmostEqual(float(cmd[cmd.index("--pitch") + 1]), -3.0, places=6)

    def test_input_and_output_paths_are_the_final_two_arguments(self):
        cmd = capture_args(transform._rubberband, self.y, 48000,
                           ["--pitch", "1.0"], True, "test")
        self.assertTrue(cmd[-2].endswith("in.wav"))
        self.assertTrue(cmd[-1].endswith("out.wav"))

    def test_nonzero_exit_returns_none_so_the_caller_can_fall_back(self):
        """Failure must be reported, not swallowed. Returning None lets the
        caller choose the phase vocoder explicitly and log that it did."""
        def failing_run(cmd, **kwargs):
            if "--help" in cmd:
                r = _Result(0)
                r.stdout = "--formant\n--fine\n"
                return r
            r = _Result(2)
            r.stderr = "usage error"
            return r

        with mock.patch.object(transform.subprocess, "run", side_effect=failing_run):
            out = transform._rubberband(self.y, 48000, ["--pitch", "2"],
                                        True, "test")
        self.assertIsNone(out)

    def test_temp_directory_is_cleaned_up_on_failure(self):
        before = set(os.listdir(transform.tempfile.gettempdir()))

        def failing_run(cmd, **kwargs):
            if "--help" in cmd:
                r = _Result(0)
                r.stdout = "--formant\n"
                return r
            return _Result(2)

        with mock.patch.object(transform.subprocess, "run", side_effect=failing_run):
            transform._rubberband(self.y, 48000, ["--pitch", "2"], True, "t")
        leaked = [d for d in set(os.listdir(transform.tempfile.gettempdir())) - before
                  if d.startswith("mixengine_rb_")]
        self.assertEqual(leaked, [])

    def test_capability_probe_is_cached(self):
        calls = {"n": 0}

        def counting_run(cmd, **kwargs):
            calls["n"] += 1
            r = _Result(0)
            r.stdout = "--formant\n--fine\n"
            return r

        transform._RB_FORMANT_OK = None
        with mock.patch.object(transform.subprocess, "run", side_effect=counting_run):
            transform._rubberband_supports_formant()
            first = calls["n"]
            transform._rubberband_supports_formant()
            transform._rubberband_supports_formant()
        self.assertEqual(calls["n"], first, "probe should run once and cache")


class TestNoOpShortCircuits(unittest.TestCase):
    """Transforms that would change nothing must not touch the audio.

    Round-tripping through an external process for a no-op costs time and,
    worse, costs a generation of resampling quality for no benefit.
    """

    def test_unit_stretch_is_a_passthrough(self):
        y = np.random.default_rng(0).normal(0, 0.1, (2048, 1)).astype(np.float32)
        with mock.patch.object(transform.subprocess, "run") as run:
            out = transform.time_stretch(y, 48000, 1.0)
        run.assert_not_called()
        np.testing.assert_allclose(out, y, atol=1e-6)

    def test_zero_semitone_shift_is_a_passthrough(self):
        y = np.random.default_rng(0).normal(0, 0.1, (2048, 1)).astype(np.float32)
        with mock.patch.object(transform.subprocess, "run") as run:
            out = transform.pitch_shift(y, 48000, 0.0)
        run.assert_not_called()
        np.testing.assert_allclose(out, y, atol=1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
