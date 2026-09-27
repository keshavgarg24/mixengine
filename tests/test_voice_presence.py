"""
Is there a voice in the take at all.

A beat in the vocal slot, a full song and a test tone each went through
as "the vocal" and came back scored in the nineties. The voice model can
tell them from a take; these pin that it does on the real fixture, and
that the check degrades to "not made" rather than to a wrong answer.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mixengine.analysis import analysis                           # noqa: E402
from mixengine.audio import mixer                                 # noqa: E402
from mixengine.core.capabilities import CAPS                      # noqa: E402

# The fixture vocal is a synthesised sine stack (see make_fixtures.py) and
# the model correctly hears no voice in it. A real take, when the machine
# has one, is the positive case.
FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "vocal.wav")
REAL_TAKE = os.path.join(os.path.dirname(__file__), "..", "data", "vocals",
                         "c971095b6d8d", "v1.wav")
SR = 16000


@unittest.skipUnless(CAPS.silero_vad, "needs silero-vad")
class TestSileroHearsAVoice(unittest.TestCase):

    @unittest.skipUnless(os.path.exists(REAL_TAKE), "needs a real take on disk")
    def test_a_real_take_is_a_voice(self):
        import soundfile as sf
        y, sr = sf.read(REAL_TAKE, dtype="float32", always_2d=True)
        v = analysis.voice_presence(y[:sr * 20], sr)
        self.assertIsNotNone(v)
        self.assertEqual(v["verdict"], "voice", v)
        self.assertGreater(v["speech_in_phrases"], 0.5)

    @unittest.skipUnless(os.path.exists(FIXTURE), "needs the fixture")
    def test_the_synthesised_fixture_is_not(self):
        import soundfile as sf
        y, sr = sf.read(FIXTURE, dtype="float32", always_2d=True)
        v = analysis.voice_presence(y[:sr * 20], sr)
        self.assertEqual(v["verdict"], "no_voice", v)

    def test_a_tone_and_noise_are_not(self):
        t = np.arange(SR * 8) / SR
        tone = (0.3 * np.sin(2 * np.pi * 220 * t)
                * (0.6 + 0.4 * np.sin(2 * np.pi * 2 * t))).astype(np.float32)
        noise = (0.1 * np.random.default_rng(3).standard_normal(SR * 8)
                 ).astype(np.float32)
        for name, y in (("tone", tone), ("noise", noise)):
            v = analysis.voice_presence(y, SR)
            self.assertIsNotNone(v)
            self.assertEqual(v["verdict"], "no_voice", (name, v))


class TestWithoutTheModel(unittest.TestCase):

    def test_no_model_means_no_check_not_a_wrong_answer(self):
        had = CAPS.silero_vad
        try:
            CAPS.silero_vad = False
            self.assertIsNone(analysis.voice_presence(
                np.zeros(SR * 2, np.float32), SR))
        finally:
            CAPS.silero_vad = had


class TestBalanceOnAQuietTake(unittest.TestCase):
    """The loudness meter gates a very quiet take out and reports nothing;
    the balance stage then applied no gain and the voice went out 70 dB
    under the beat."""

    def test_a_take_the_meter_cannot_see_is_still_brought_up(self):
        sr = 44100
        t = np.arange(sr * 6) / sr
        vocal = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32) * 0.0003
        beat = (0.3 * np.sin(2 * np.pi * 110 * t)).astype(np.float32)
        phrases = [(sr, sr * 5)]
        from mixengine.config import GENRE_PROFILES
        profile = GENRE_PROFILES["trap"]
        mix, report = mixer.balance_and_sum(vocal, beat, sr, phrases, profile)
        self.assertGreater(report["vocal_gain_db"], 20.0, report)
        self.assertIn("rms", report["method"])


if __name__ == "__main__":
    unittest.main()
