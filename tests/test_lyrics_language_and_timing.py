"""
Language is declared or detected, never guessed -- and a transcript's times
are used only if they agree with the audio.

The transcriber used to answer "not sure what language this is" by forcing
English. Forced onto audio in another language, Whisper does not fail, it
translates: Hindi speech came back as fluent English whose word times no
longer belonged to the audio. Scoring each language on an excerpt was tried
instead and is worse (forced to Hindi, an English rap produced seventy
plausible words and outscored the English decode), so an unsure detection
now falls back to English and says so.

Separately, word times from the model on rap and sung material agree with
the audio's onsets only sometimes, and no decode setting changes that. The
document now carries a measured verdict and the timing accessors honour it.
"""

import os
import sys
import types
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mixengine.analysis import lyrics                                  # noqa: E402
from mixengine.core import questions                                   # noqa: E402
from mixengine.core.intents import Intents                             # noqa: E402

SR = lyrics.ASR_SR


class _Word:
    def __init__(self, text, start, end, p=0.9):
        self.word, self.start, self.end, self.probability = text, start, end, p


class _Seg:
    def __init__(self, text, words, start=0.0, end=1.0):
        self.text, self.words, self.start, self.end = text, words, start, end


class _Model:
    """Records how it was called; answers with one fixed segment."""

    def __init__(self, language="en", prob=0.95):
        self.detected = (language, prob, [])
        self.detect_calls = 0
        self.transcribe_kwargs = None

    def detect_language(self, audio):
        self.detect_calls += 1
        return self.detected

    def transcribe(self, audio, **kw):
        self.transcribe_kwargs = kw
        return iter([_Seg("a b", [_Word("a", 0.1, 0.4), _Word("b", 0.5, 0.9)])]), None


def _audio(seconds=3.0):
    rng = np.random.default_rng(0)
    return (0.05 * rng.standard_normal(int(seconds * SR))).astype(np.float32)


class TestNormalise(unittest.TestCase):

    def test_urdu_is_read_as_hindi(self):
        self.assertEqual(lyrics.normalise_language("ur"), "hi")

    def test_regional_codes_and_case_are_reduced(self):
        self.assertEqual(lyrics.normalise_language("Hi-IN"), "hi")
        self.assertEqual(lyrics.normalise_language("PA"), "pa")

    def test_auto_and_empty_mean_work_it_out(self):
        for v in (None, "", "auto", "AUTO", "  "):
            self.assertIsNone(lyrics.normalise_language(v))

    def test_indic_languages_get_the_larger_model(self):
        self.assertEqual(lyrics.model_size_for("hi"), lyrics.INDIC_MODEL_SIZE)
        self.assertEqual(lyrics.model_size_for("pa"), lyrics.INDIC_MODEL_SIZE)
        self.assertEqual(lyrics.model_size_for("en"), lyrics.MODEL_SIZE)


class TestResolve(unittest.TestCase):

    def _resolve(self, model, declared=None):
        with mock.patch.object(lyrics, "_model", return_value=model):
            return lyrics.resolve_language(_audio(), declared)

    def test_a_declared_language_is_used_and_nothing_is_detected(self):
        m = _Model("en", 0.99)
        lang, source, _ = self._resolve(m, "pa")
        self.assertEqual((lang, source), ("pa", "declared"))
        self.assertEqual(m.detect_calls, 0)

    def test_a_confident_detection_is_trusted(self):
        lang, source, ev = self._resolve(_Model("hi", 0.93))
        self.assertEqual((lang, source), ("hi", "detected"))
        self.assertAlmostEqual(ev["detected_probability"], 0.93)

    def test_an_unsure_detection_falls_back_to_english_and_says_so(self):
        # The case that started this: an English rap read as Punjabi at 0.41.
        lang, source, ev = self._resolve(_Model("pa", 0.41))
        self.assertEqual((lang, source), ("en", "default"))
        self.assertEqual(ev["detected"], "pa")

    def test_it_never_forces_a_language_on_a_confident_other_one(self):
        lang, source, _ = self._resolve(_Model("hi", 0.71))
        self.assertEqual(lang, "hi")

    def test_urdu_detection_is_transcribed_as_hindi(self):
        lang, _, _ = self._resolve(_Model("ur", 0.95))
        self.assertEqual(lang, "hi")


class TestTranscribe(unittest.TestCase):

    def _run(self, model, language=None):
        sizes = []

        def fake_model(size=lyrics.MODEL_SIZE):
            sizes.append(size)
            return model
        with mock.patch.object(lyrics, "_model", side_effect=fake_model), \
                mock.patch.object(lyrics, "CAPS", types.SimpleNamespace(whisper=True)):
            doc = lyrics.transcribe(_audio(), SR, language=language)
        return doc, sizes

    def test_the_decode_is_repeatable(self):
        m = _Model("en", 0.95)
        self._run(m)
        kw = m.transcribe_kwargs
        self.assertEqual(kw["temperature"], 0.0)
        self.assertIsNone(kw["compression_ratio_threshold"])
        self.assertIsNone(kw["no_speech_threshold"])

    def test_a_declared_hindi_take_uses_hindi_and_the_larger_model(self):
        m = _Model("en", 0.99)
        doc, sizes = self._run(m, "hi")
        self.assertEqual(m.transcribe_kwargs["language"], "hi")
        self.assertEqual(doc["language"], "hi")
        self.assertEqual(doc["language_source"], "declared")
        self.assertTrue(doc["language_confirmed"])
        self.assertEqual(doc["model"], lyrics.INDIC_MODEL_SIZE)
        self.assertIn(lyrics.INDIC_MODEL_SIZE, sizes)

    def test_an_unsure_english_take_is_marked_unconfirmed(self):
        doc, _ = self._run(_Model("pa", 0.41))
        self.assertEqual(doc["language"], "en")
        self.assertEqual(doc["language_source"], "default")
        self.assertFalse(doc["language_confirmed"])

    def test_a_confident_detection_is_not_marked_confirmed(self):
        # Hindi and Punjabi sound alike; only the person can say which.
        doc, _ = self._run(_Model("pa", 0.9))
        self.assertEqual(doc["language_source"], "detected")
        self.assertFalse(doc["language_confirmed"])


def _clicks(times, seconds=8.0):
    y = np.zeros(int(seconds * SR), dtype=np.float32)
    rng = np.random.default_rng(1)
    for t in times:
        i = int(t * SR)
        y[i:i + 400] = 0.8 * rng.standard_normal(400)
    return y


class TestTimingGate(unittest.TestCase):

    def setUp(self):
        self.beats = np.arange(0.5, 8.1, 0.5)            # 16, over the minimum
        self.audio = _clicks(self.beats, seconds=9.0)

    def test_words_that_start_on_the_onsets_are_reliable(self):
        c = lyrics.timing_check(self.audio, self.beats + 0.01)
        self.assertTrue(c["reliable"], c)
        self.assertGreaterEqual(c["lift"], lyrics.TIMING_MIN_LIFT)

    def test_words_placed_between_the_onsets_are_not(self):
        c = lyrics.timing_check(self.audio, self.beats + 0.25)
        self.assertFalse(c["reliable"], c)

    def test_too_few_words_is_not_enough_to_trust(self):
        c = lyrics.timing_check(self.audio, self.beats[:5])
        self.assertFalse(c["reliable"])
        self.assertIsNone(c["lift"])

    def test_silence_has_no_onsets_to_agree_with(self):
        c = lyrics.timing_check(np.zeros(8 * SR, dtype=np.float32), self.beats)
        self.assertFalse(c["reliable"])


def _doc(reliable=None):
    d = {"lines": [{"text": "a b", "start": 1.0, "end": 2.0, "words": [
        {"text": "a", "start": 1.0, "end": 1.2, "probability": 0.9},
        {"text": "b", "start": 1.3, "end": 1.6, "probability": 0.9}]}]}
    if reliable is not None:
        d["timing"] = {"reliable": reliable}
    return d


class TestAccessorsHonourTheGate(unittest.TestCase):

    def test_unreliable_times_place_nothing(self):
        self.assertEqual(lyrics.line_starts(_doc(False)).size, 0)
        self.assertEqual(lyrics.word_onsets(_doc(False)).size, 0)

    def test_reliable_times_are_used(self):
        self.assertEqual(lyrics.line_starts(_doc(True)).tolist(), [1.0])
        self.assertEqual(lyrics.word_onsets(_doc(True)).size, 2)

    def test_a_document_from_before_the_check_is_trusted_as_it_was(self):
        self.assertEqual(lyrics.line_starts(_doc(None)).tolist(), [1.0])

    def test_the_verdict_survives_shifting_the_times(self):
        shifted = lyrics.shift_times(_doc(False), scale=1.1, offset_s=0.5)
        self.assertFalse(lyrics.timing_usable(shifted))

    def test_phrase_attribution_waits_on_the_same_verdict(self):
        # Words are given to phrases by when they start; mis-timed words
        # would land in the wrong phrase and fake a hook.
        phrases = [(int(0.9 * SR), int(2.1 * SR))]
        tokens, _ = lyrics.phrase_lyrics(_doc(True), phrases, SR)[0]
        self.assertEqual(tokens, ["a", "b"])
        self.assertEqual(lyrics.phrase_lyrics(_doc(False), phrases, SR),
                         [([], 0.0)])


class TestTheLanguageQuestion(unittest.TestCase):

    def _vdna(self, **lyric):
        doc = {"n_words": 40, "language": "en", "language_source": "detected",
               "language_confirmed": False,
               "language_evidence": {"detected": "en", "detected_probability": 0.95}}
        doc.update(lyric)
        return {"lyrics": doc}

    def _ask(self, **lyric):
        return [q for q in questions.questions_for(self._vdna(**lyric))
                if q.id == "language"]

    def test_confident_english_is_not_asked_about(self):
        self.assertEqual(self._ask(), [])

    def test_hindi_is_asked_about_even_when_the_detector_is_sure(self):
        q = self._ask(language="hi", language_evidence={
            "detected": "hi", "detected_probability": 0.97})
        self.assertEqual(len(q), 1)
        self.assertEqual(q[0].default, "hi")
        self.assertEqual([o["value"] for o in q[0].options], ["en", "hi", "pa"])
        self.assertEqual(q[0].intent, "language")

    def test_punjabi_is_asked_about_too(self):
        self.assertEqual(len(self._ask(language="pa")), 1)

    def test_an_unsure_detection_is_asked_about_and_says_it_read_english(self):
        q = self._ask(language="en", language_source="default", language_evidence={
            "detected": "pa", "detected_probability": 0.41})
        self.assertEqual(len(q), 1)
        self.assertIn("English", q[0].text)
        self.assertEqual(q[0].default, "en")

    def test_a_declared_language_is_not_asked_again(self):
        self.assertEqual(self._ask(language="hi", language_source="declared",
                                   language_confirmed=True), [])

    def test_a_take_with_no_words_is_not_asked(self):
        self.assertEqual(self._ask(language="hi", n_words=0), [])

    def test_the_question_never_blocks_the_render(self):
        q = self._ask(language="pa")[0]
        self.assertNotEqual(q.severity, "block")
        self.assertEqual(questions.unanswered_blocks([q], Intents()), [])


class TestTheIntent(unittest.TestCase):

    def test_the_three_languages_are_accepted(self):
        for v in ("en", "hi", "pa"):
            self.assertEqual(Intents.from_dict({"language": v}).language, v)

    def test_auto_means_decide_for_me(self):
        self.assertIsNone(Intents.from_dict({"language": "auto"}).language)

    def test_an_unsupported_language_is_refused_with_the_choices(self):
        with self.assertRaises(ValueError) as cm:
            Intents.from_dict({"language": "fr"})
        self.assertIn("hi", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
