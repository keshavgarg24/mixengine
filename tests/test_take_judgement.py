"""
Judging a take before rendering it.

A take is not a file. It often starts before the performance does -- a
count-in, a breath, the room -- and it carries whatever noise the room
had. Left unjudged, the sound before the first line was the vocal's
"first phrase", the placement put it on the drop, and a take whose noise
was nearly as loud as the voice was rendered without a word said. These
tests pin the measurements that now judge a take (noise verdict,
performance span, harmonicity), the two restoration steps added for
noisy takes (hum notches, gap muting), the questions those judgements
turn into, and the render's use of the answers.

Synthetic audio at a low sample rate keeps every test under a second.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mixengine.analysis import analysis                           # noqa: E402
from mixengine.audio import dsp, pipeline, separation             # noqa: E402
from mixengine.core import questions                              # noqa: E402
from mixengine.core.intents import Intents                        # noqa: E402

SR = 8000
RNG = np.random.default_rng(11)


def tone(seconds, f0=220.0, amp=0.3, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * f0 * t)).astype(np.float32)


def noise(seconds, amp=0.3, sr=SR):
    return (amp * RNG.standard_normal(int(seconds * sr))).astype(np.float32)


def take(phrases_s, total_s, floor_amp=0.0, sr=SR):
    """Silence (or a noise floor) with tones where the phrases are."""
    y = noise(total_s, floor_amp, sr) if floor_amp > 0 \
        else np.zeros(int(total_s * sr), dtype=np.float32)
    for s, e in phrases_s:
        a, b = int(s * sr), int(e * sr)
        y[a:b] += tone((b - a) / sr, sr=sr)[:b - a]
    return y


def in_samples(phrases_s, sr=SR):
    return [(int(s * sr), int(e * sr)) for s, e in phrases_s]


class TestHarmonicity(unittest.TestCase):

    def test_a_tone_is_harmonic_and_noise_is_not(self):
        frame, hop = int(0.04 * SR), int(0.01 * SR)
        h_tone = dsp.harmonicity(tone(1.0), SR, frame, hop)
        h_noise = dsp.harmonicity(noise(1.0), SR, frame, hop)
        self.assertGreater(float(np.median(h_tone)), 0.8)
        self.assertLess(float(np.median(h_noise)), 0.5)
        self.assertGreater(float(np.median(h_tone)),
                           float(np.median(h_noise)) + 0.3)

    def test_short_input_is_empty_not_an_error(self):
        self.assertEqual(dsp.harmonicity(tone(0.01), SR, 800, 80).size, 0)


class TestNoiseVerdict(unittest.TestCase):

    def verdict(self, floor_amp):
        phrases = [(1.0, 3.0), (4.0, 6.0)]
        y = take(phrases, 7.0, floor_amp)
        return analysis.noise_verdict(y, SR, in_samples(phrases))

    def test_a_clean_take_is_clean(self):
        v = self.verdict(0.001)
        self.assertEqual(v["verdict"], "clean")
        self.assertGreater(v["snr_db"], 30)

    def test_noise_near_the_voice_is_severe(self):
        v = self.verdict(0.15)
        self.assertEqual(v["verdict"], "severe")
        self.assertLess(v["snr_db"], 14)

    def test_the_verdicts_are_ordered_by_floor(self):
        order = [self.verdict(a)["snr_db"] for a in (0.001, 0.02, 0.06, 0.15)]
        self.assertEqual(order, sorted(order, reverse=True))

    def test_a_buffer_too_small_to_measure_says_so_instead_of_severe(self):
        """Fifty milliseconds holds neither a word nor a gap. Calling that
        severe told someone their room was too noisy to record in."""
        v = analysis.noise_verdict(tone(0.05), SR, [])
        self.assertEqual(v["verdict"], "unknown")
        self.assertEqual(v["snr_db"], 0.0)
        asked = questions.questions_for({"noise": v, "duration_s": 30.0})
        self.assertNotIn("noise", [q.id for q in asked])


class TestPerformanceSpan(unittest.TestCase):
    """The repro-2 shape: a loud blob at 0-3.5 s, a blip at 5.6 s, and
    the rap from 10.9 s. The performance is the rap."""

    NOISY = [(0.0, 3.51), (5.62, 6.14), (10.93, 19.31), (19.6, 27.0),
             (27.4, 35.0), (35.3, 41.0)]

    def test_lead_in_before_the_performance_is_trimmed_by_default(self):
        span = analysis.performance_span(in_samples(self.NOISY), SR, 42.0)
        self.assertAlmostEqual(span["start_s"], 10.93, places=2)
        # lead_in_s is how much *sound* sits before the first line, not
        # how far in the line is: 3.51 s of blob plus a 0.52 s blip.
        self.assertAlmostEqual(span["lead_in_s"], 4.03, places=2)
        self.assertEqual(span["default"], "trim")

    def test_a_take_that_starts_on_its_first_line_has_no_lead_in(self):
        phrases = [(0.2, 4.0), (4.4, 8.0), (8.3, 12.0)]
        span = analysis.performance_span(in_samples(phrases), SR, 12.5)
        self.assertEqual(span["lead_in_s"], 0.0)
        self.assertEqual(span["default"], "keep")

    def test_an_ad_lib_close_to_the_verse_is_part_of_it(self):
        """A short phrase 1 s before the verse is a pickup, not a lead-in."""
        phrases = [(1.0, 1.6), (2.6, 8.0), (8.3, 14.0)]
        span = analysis.performance_span(in_samples(phrases), SR, 14.5)
        self.assertAlmostEqual(span["start_s"], 1.0, places=2)
        self.assertEqual(span["default"], "keep")

    def test_an_isolated_lead_in_is_trimmed_even_when_voiced(self):
        """Sound 4.8 s clear of the first line is not part of the
        performance whatever it sounds like; it is cut, and asked about."""
        harm = [0.9] * len(self.NOISY)
        span = analysis.performance_span(in_samples(self.NOISY), SR, 42.0,
                                         harmonicity=harm)
        self.assertEqual(span["default"], "trim")
        self.assertIn("a voice", span["reason"])

    def test_a_voiced_lead_in_near_the_verse_is_kept_by_default(self):
        """A phrase 2.5 s ahead of the verse, as harmonic as the verse, is
        singing, not the room: kept, and asked about."""
        phrases = [(0.0, 2.0), (4.5, 12.0), (12.3, 20.0)]
        voiced = analysis.performance_span(in_samples(phrases), SR, 20.5,
                                           harmonicity=[0.9, 0.9, 0.9])
        self.assertGreater(voiced["lead_in_s"], 0)
        self.assertEqual(voiced["default"], "keep")
        room = analysis.performance_span(in_samples(phrases), SR, 20.5,
                                         harmonicity=[0.1, 0.9, 0.9])
        self.assertEqual(room["default"], "trim")
        self.assertIn("noise, not a voice", room["reason"])

    def test_no_phrases_is_the_whole_file(self):
        span = analysis.performance_span([], SR, 10.0)
        self.assertEqual((span["start_s"], span["end_s"]), (0.0, 10.0))
        self.assertEqual(span["default"], "keep")


class TestClassifyPerformanceEx(unittest.TestCase):

    def test_no_notes_and_dense_onsets_is_rap_with_low_confidence(self):
        pitch = analysis.PitchResult()
        onsets = np.arange(0.0, 10.0, 0.25)
        label, conf, reason = analysis.classify_performance_ex(pitch, onsets, 10.0)
        self.assertEqual(label, "rap")
        self.assertLess(conf, questions.PERFORMANCE_ASK_BELOW)
        self.assertIn("syllables", reason)

    def test_no_notes_and_sparse_onsets_is_spoken(self):
        label, conf, _ = analysis.classify_performance_ex(
            analysis.PitchResult(), np.arange(0.0, 10.0, 1.0), 10.0)
        self.assertEqual(label, "spoken")

    def test_wrapper_still_returns_a_label(self):
        self.assertIsInstance(analysis.classify_performance(
            analysis.PitchResult(), np.zeros(0), 1.0), str)


class TestHum(unittest.TestCase):
    """A take with words and gaps, long enough that the quietest fifth
    of the frames is a real sample: the detector refuses to judge fewer
    than sixteen. Hum is looked for in the gaps, never in the words."""

    LONG = 48.0
    PHRASES = [(s, s + 2.0) for s in np.arange(1.0, 46.0, 4.0)]

    def with_floor(self, floor_amp=0.01):
        return take(self.PHRASES, self.LONG, floor_amp=floor_amp)

    def test_a_hum_line_is_found_and_notched(self):
        y = self.with_floor() + tone(self.LONG, f0=150.0, amp=0.1)
        lines = separation.find_hum(y, SR)
        self.assertTrue(lines, "no hum line found")
        self.assertLess(abs(lines[0][0] - 150.0), 3.0)
        out, applied = separation.remove_hum(y, SR)
        self.assertEqual(len(applied), len(lines))
        spec_in = np.abs(np.fft.rfft(y))
        spec_out = np.abs(np.fft.rfft(out))
        k = int(round(150.0 * len(y) / SR))
        self.assertLess(spec_out[k], spec_in[k] * 0.1)

    def test_the_words_are_never_the_line(self):
        """The 220 Hz of every phrase is the loudest thing in the file
        and must not be reported: it is absent from the gaps."""
        lines = separation.find_hum(self.with_floor(), SR)
        self.assertFalse([f for f, _ in lines if abs(f - 220.0) < 3.0], lines)

    def test_a_clean_take_has_no_lines(self):
        for _ in range(3):
            self.assertEqual(separation.find_hum(self.with_floor(), SR), [])
        y = self.with_floor()
        out, lines = separation.remove_hum(y, SR)
        self.assertEqual(lines, [])
        np.testing.assert_array_equal(out, y)

    def test_a_take_without_gaps_is_not_judged(self):
        """A drone has no gaps to measure against; a held note would
        read as hum. Nothing is reported."""
        self.assertEqual(separation.find_hum(tone(self.LONG), SR), [])

    def test_a_short_take_is_left_alone(self):
        y = take([(1.0, 2.0)], 4.0, 0.01) + tone(4.0, f0=150.0, amp=0.1)
        self.assertEqual(separation.find_hum(y, SR), [])


class TestGapMute(unittest.TestCase):

    def test_gaps_are_silenced_and_phrases_untouched(self):
        phrases = [(1.0, 3.0), (4.0, 6.0)]
        y = take(phrases, 7.0, floor_amp=0.05)
        out = separation.mute_between_phrases(y, SR, in_samples(phrases))
        self.assertEqual(out.shape, y.shape)
        self.assertEqual(float(np.abs(out[:int(0.9 * SR)]).max()), 0.0)
        self.assertEqual(float(np.abs(out[int(3.1 * SR):int(3.9 * SR)]).max()), 0.0)
        self.assertEqual(float(np.abs(out[int(6.1 * SR):]).max()), 0.0)
        mid = slice(int(1.2 * SR), int(2.8 * SR))
        np.testing.assert_array_equal(out[mid], y[mid])

    def test_short_gaps_are_left_alone(self):
        phrases = [(1.0, 3.0), (3.1, 5.0)]
        y = take(phrases, 5.5, floor_amp=0.05)
        out = separation.mute_between_phrases(y, SR, in_samples(phrases))
        gap = slice(int(3.0 * SR), int(3.1 * SR))
        np.testing.assert_array_equal(out[gap], y[gap])


class TestQuestions(unittest.TestCase):

    SEVERE = {"noise": {"verdict": "severe", "snr_db": 11.0, "input_snr_db": 6.0},
              "performance_span": {"lead_in_s": 10.9, "tail_s": 0.0,
                                   "start_s": 10.9, "end_s": 50.0,
                                   "default": "trim", "reason": "room"},
              "performance_type": "rap", "performance_confidence": 0.3,
              "performance_reason": "no sustained pitch"}

    def test_a_severe_take_blocks_and_asks_for_a_cleaner_one(self):
        qs = questions.questions_for(self.SEVERE)
        self.assertEqual([q.id for q in qs], ["noise", "start", "performance"])
        noise = qs[0]
        self.assertEqual(noise.severity, "block")
        self.assertEqual(noise.default, "rerecord")
        self.assertEqual(questions.unanswered_blocks(qs, Intents.AUTO), [noise])
        self.assertEqual(questions.unanswered_blocks(
            qs, Intents.from_dict({"noise": "accept"})), [])

    def test_a_heavy_take_warns_but_renders(self):
        vdna = dict(self.SEVERE, noise={"verdict": "heavy", "snr_db": 17.0})
        qs = questions.questions_for(vdna)
        self.assertEqual(qs[0].severity, "warn")
        self.assertEqual(qs[0].default, "accept")
        self.assertEqual(questions.unanswered_blocks(qs, Intents.AUTO), [])

    def test_a_clean_confident_take_asks_nothing(self):
        vdna = {"noise": {"verdict": "clean", "snr_db": 40.0},
                "performance_span": {"lead_in_s": 0.0, "tail_s": 0.0,
                                     "default": "keep"},
                "performance_type": "sung", "performance_confidence": 0.85,
                "key": {"name": "F minor"}, "key_confidence": 0.8,
                "bpm": 100.0, "bpm_confidence": 0.9}
        self.assertEqual(questions.questions_for(vdna), [])

    def test_a_short_intro_on_the_beat_offers_the_entry(self):
        bdna = {"sections": [{"label": "intro", "start": 0.0, "end": 9.6,
                              "start_bar": 0, "end_bar": 4},
                             {"label": "verse", "start": 9.6, "end": 48.0}],
                "bar_s": 2.4}
        qs = questions.questions_for({}, bdna)
        self.assertEqual([q.id for q in qs], ["entry"])
        self.assertEqual(qs[0].default, "section")

    SHORT = {"duration_s": 1.5, "noise": {"verdict": "clean", "snr_db": 40.0},
             "performance_span": {"lead_in_s": 0.0, "tail_s": 0.0,
                                  "default": "keep"},
             "performance_type": "rap", "performance_confidence": 0.85}

    def test_a_take_too_short_for_a_song_blocks_and_says_how_long_it_is(self):
        """The render is cut to the length of the take, so a 1.5 s upload
        used to come back as a 1.5 s song with nothing said about it."""
        qs = questions.questions_for(self.SHORT)
        self.assertEqual([q.id for q in qs], ["length"])
        q = qs[0]
        self.assertEqual((q.severity, q.default, q.intent),
                         ("block", "rerecord", "length"))
        self.assertIn("1.5 s", q.text)
        self.assertEqual(questions.unanswered_blocks(qs, Intents.AUTO), [q])
        self.assertEqual(questions.unanswered_blocks(
            qs, Intents.from_dict({"length": "accept"})), [])

    def test_a_take_long_enough_to_perform_is_not_asked_about_length(self):
        for duration in (questions.MIN_TAKE_S, 36.5, 0.0, None):
            vdna = dict(self.SHORT, duration_s=duration)
            self.assertNotIn("length", [q.id for q in
                                        questions.questions_for(vdna)],
                             "asked about a %r-second take" % duration)

    def test_a_take_both_short_and_severe_is_asked_about_both(self):
        vdna = dict(self.SHORT, noise={"verdict": "severe", "snr_db": 10.0,
                                       "input_snr_db": 6.0})
        qs = questions.questions_for(vdna)
        self.assertEqual(sorted(q.id for q in qs), ["length", "noise"])
        self.assertEqual(len(questions.unanswered_blocks(qs, Intents.AUTO)), 2)

    def test_saying_a_better_take_is_coming_does_not_unblock_the_render(self):
        """"I'll upload a cleaner take" is a real answer -- it is parsed and
        kept -- but it is the opposite of permission to render."""
        qs = questions.questions_for(self.SEVERE)
        refused = Intents.from_dict({"noise": "rerecord"})
        self.assertEqual(refused.noise, "rerecord")
        self.assertEqual([q.id for q in
                          questions.unanswered_blocks(qs, refused)], ["noise"])

    def test_every_blocking_question_offers_a_refusal_and_a_way_past(self):
        for vdna in (self.SEVERE, self.SHORT):
            for q in questions.questions_for(vdna):
                if q.severity != "block":
                    continue
                values = [o["value"] for o in q.options]
                self.assertIn("accept", values, q.id)
                self.assertEqual(q.default, "rerecord", q.id)
                self.assertTrue(set(values) & set(questions.REFUSALS), q.id)

    def test_every_question_serialises(self):
        for q in questions.questions_for(self.SEVERE):
            d = q.to_dict()
            self.assertIn(d["default"], [o["value"] for o in d["options"]])
            self.assertIn(d["severity"], questions.SEVERITIES)


class TestIntentAnswers(unittest.TestCase):

    def test_answers_are_read_and_validated(self):
        i = Intents.from_dict({"performance": "melodic_rap", "lead_in": "keep",
                               "entry": "top", "noise": "accept"})
        self.assertEqual((i.performance, i.lead_in, i.entry, i.noise),
                         ("melodic_rap", "keep", "top", "accept"))
        self.assertFalse(i.is_all_auto)
        for bad in ({"performance": "opera"}, {"lead_in": "maybe"},
                    {"entry": "middle"}, {"noise": "ignore"},
                    {"length": "maybe"}):
            with self.assertRaises(ValueError):
                Intents.from_dict(bad)

    def test_the_value_a_blocking_question_offers_is_a_value_it_accepts(self):
        """A client that posts the question's own default must not get a
        422: every option value the engine offers has to parse."""
        vdna = {"duration_s": 1.5,
                "noise": {"verdict": "severe", "snr_db": 10.0,
                          "input_snr_db": 6.0}}
        for q in questions.questions_for(vdna):
            for option in q.options:
                Intents.from_dict({q.intent: option["value"]})


class TestKeepPerformance(unittest.TestCase):

    SPAN = {"start_s": 4.0, "end_s": 9.0, "lead_in_s": 4.0, "tail_s": 1.0,
            "default": "trim", "reason": "room"}

    def setUp(self):
        self.v = dsp.as_2d(take([(0.5, 3.0), (4.0, 9.0)], 10.0, floor_amp=0.02))

    def test_default_trim_silences_the_lead_in_and_tail(self):
        out, info = pipeline._keep_performance(self.v, SR, self.SPAN, None)
        self.assertEqual(out.shape, self.v.shape)
        self.assertEqual(float(np.abs(out[:int(3.9 * SR)]).max()), 0.0)
        self.assertEqual(float(np.abs(out[int(9.1 * SR):]).max()), 0.0)
        mid = slice(int(4.2 * SR), int(8.8 * SR))
        np.testing.assert_array_equal(out[mid], self.v[mid])
        self.assertEqual(info["decision"], "trim")
        self.assertEqual(info["source"], "analysis")

    def test_the_person_can_keep_it(self):
        out, info = pipeline._keep_performance(self.v, SR, self.SPAN, "keep")
        np.testing.assert_array_equal(out, self.v)
        self.assertEqual(info["source"], "user")

    def test_the_person_can_cut_what_the_analysis_kept(self):
        span = dict(self.SPAN, default="keep")
        out, _ = pipeline._keep_performance(self.v, SR, span, "trim")
        self.assertEqual(float(np.abs(out[:int(3.9 * SR)]).max()), 0.0)

    def test_nothing_to_cut_is_a_no_op(self):
        span = {"start_s": 0.0, "end_s": 10.0, "lead_in_s": 0.0, "tail_s": 0.0}
        out, info = pipeline._keep_performance(self.v, SR, span, None)
        self.assertIs(out, self.v)
        self.assertEqual(info, {})


if __name__ == "__main__":
    unittest.main()
