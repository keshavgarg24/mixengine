"""
Cached analyses must not outlive the backends that made them.

Analysis is cached by content hash and schema version. Installing a better
backend -- torchcrepe for pitch, demucs for stems -- changed neither, so
every cached beat and vocal kept serving the analysis made without it and
the better backend never reached a render. Each document now records the
backends it was made with, and every cache lookup asks whether this
machine could do better before trusting it.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mixengine.analysis import beat_dna, vocal_dna                     # noqa: E402
from mixengine.core.capabilities import CAPS, improvement_over          # noqa: E402

BASIC = {"rhythm": "librosa", "pitch": "pyin", "separation": "none"}
FULL = {"rhythm": "madmom", "pitch": "torchcrepe", "separation": "demucs"}

# What a take analysed today carries: the judgement the questions read.
JUDGED = {"noise": {"verdict": "clean"}, "performance_span": {},
          "voice": {"verdict": "voice"}, "lyrics": {"n_words": 12}}


class TestImprovementOver(unittest.TestCase):

    def test_same_backends_keep_the_cache(self):
        self.assertIsNone(improvement_over(BASIC, now=BASIC))
        self.assertIsNone(improvement_over(FULL, now=FULL))

    def test_a_better_backend_names_the_stage(self):
        why = improvement_over(BASIC, now={**BASIC, "pitch": "torchcrepe"})
        self.assertEqual(why, "pitch: pyin -> torchcrepe")

    def test_a_worse_machine_keeps_a_better_analysis(self):
        self.assertIsNone(improvement_over(FULL, now=BASIC))

    def test_separation_only_counts_when_wanted(self):
        now = {**BASIC, "separation": "demucs"}
        self.assertIsNone(improvement_over(BASIC, want_separation=False, now=now))
        self.assertIn("separation", improvement_over(BASIC, want_separation=True, now=now))

    def test_unrecorded_backends_refresh_once(self):
        self.assertTrue(improvement_over(None, now=BASIC))
        self.assertTrue(improvement_over({}, now=BASIC))

    def test_every_stage_lists_its_ranks(self):
        rec = CAPS.analysis_backends()
        self.assertEqual(set(rec), {"rhythm", "pitch", "separation"})
        self.assertIsNone(improvement_over(rec))       # this machine, right now
        self.assertEqual(CAPS.analysis_backends(separated=False)["separation"], "none")


class TestDocumentWrappers(unittest.TestCase):

    def test_beat_wrapper_respects_the_stems_request(self):
        doc = {"status": "ok", "analysis_backends": BASIC, "bar_anchor": {}}
        now = {**BASIC, "separation": "demucs"}
        self.assertIsNone(beat_dna.can_improve(doc, want_stems=False, now=now))
        self.assertTrue(beat_dna.can_improve(doc, want_stems=True, now=now))

    def test_a_beat_whose_bar_lines_were_never_checked_is_analysed_again(self):
        """Bar-ones are re-counted from the drop where the tracker slipped;
        a document from before that check may carry bar lines a beat off."""
        doc = {"status": "ok", "analysis_backends": BASIC}
        self.assertIn("drops", beat_dna.can_improve(doc, want_stems=False, now=BASIC))

    def test_vocal_wrapper_reads_whether_the_take_needed_separation(self):
        now = {**BASIC, "separation": "demucs"}
        clean = {"status": "ok", "needs_separation": False, "analysis_backends": BASIC,
                 "conditioned_path": __file__, **JUDGED}
        mixture = {"status": "ok", "needs_separation": True, "analysis_backends": BASIC,
                   "conditioned_path": __file__, **JUDGED}
        self.assertIsNone(vocal_dna.can_improve(clean, now=now))
        self.assertTrue(vocal_dna.can_improve(mixture, now=now))

    def test_a_vocal_never_checked_for_a_voice_is_analysed_again(self):
        """Only where the voice model is installed: without it the check
        cannot be made, so an old document is not sent back for it."""
        from mixengine.core.capabilities import CAPS
        doc = {"status": "ok", "needs_separation": False, "analysis_backends": BASIC,
               "conditioned_path": __file__, **JUDGED}
        del doc["voice"]
        had = CAPS.silero_vad
        try:
            CAPS.silero_vad = True
            self.assertIn("voice", vocal_dna.can_improve(doc, now=BASIC))
            CAPS.silero_vad = False
            self.assertIsNone(vocal_dna.can_improve(doc, now=BASIC))
        finally:
            CAPS.silero_vad = had

    def test_a_vocal_never_transcribed_is_analysed_again(self):
        """The hook, the bar phase and the intelligibility check all read
        the transcript, so a document made before a transcriber was
        installed is worth redoing -- unless the take has no voice in it,
        where there is nothing to transcribe."""
        from mixengine.core.capabilities import CAPS
        doc = {"status": "ok", "needs_separation": False,
               "analysis_backends": BASIC,
               "conditioned_path": __file__, **JUDGED}
        del doc["lyrics"]
        had = CAPS.whisper
        try:
            CAPS.whisper = True
            self.assertIn("transcribed", vocal_dna.can_improve(doc, now=BASIC))
            silent = dict(doc, voice={"verdict": "no_voice"})
            self.assertIsNone(vocal_dna.can_improve(silent, now=BASIC))
            CAPS.whisper = False
            self.assertIsNone(vocal_dna.can_improve(doc, now=BASIC))
        finally:
            CAPS.whisper = had

    def test_a_vocal_never_judged_for_noise_is_analysed_again(self):
        """The noise verdict and the performance span are what the person
        is asked about before a render; a document from before they
        existed cannot ask, so it is analysed once more."""
        doc = {"status": "ok", "needs_separation": False, "analysis_backends": BASIC,
               "conditioned_path": __file__}
        self.assertIn("judged", vocal_dna.can_improve(doc, now=BASIC))
        self.assertIsNone(vocal_dna.can_improve({**doc, **JUDGED}, now=BASIC))

    def test_a_vocal_whose_restored_take_is_gone_is_analysed_again(self):
        """The render reads the restored take from the cached path; an
        analysis made before one was written, or whose file was cleaned
        up, would render the noisy original."""
        doc = {"status": "ok", "needs_separation": False, "analysis_backends": BASIC}
        self.assertIn("restored take", vocal_dna.can_improve(doc))
        doc["conditioned_path"] = os.path.join(os.path.dirname(__file__), "missing.wav")
        self.assertIn("restored take", vocal_dna.can_improve(doc))

    def test_failed_documents_are_not_the_cache_layer_s_problem(self):
        self.assertIsNone(beat_dna.can_improve({"status": "failed"}, want_stems=True))
        self.assertIsNone(vocal_dna.can_improve({"status": "failed"}))


if __name__ == "__main__":
    unittest.main()
