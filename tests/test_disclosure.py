"""
What the person is told about their files.

A take used to be flipped, denoised, cut, moved and stretched, and the
person got a song and no word about any of it. These pin the sentences
the engine now writes from each stage's report, and that both briefs
carry the loader's repairs to the interface.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mixengine.api import service                                  # noqa: E402
from mixengine.audio import pipeline                               # noqa: E402


class TestRenderNotes(unittest.TestCase):

    VDNA = {"repairs": ["a DC offset of +0.300 was removed"],
            "restoration": {"hum_lines_hz": [150.4], "separator_denoise": True,
                            "dereverb": True, "gap_gate": True},
            "performance_type": "melodic_rap", "performance_source": "user"}
    TINFO = {"performance_span": {"decision": "trim",
                                  "note": "silenced 3.20 s before the first "
                                          "line and 0.00 s after the last "
                                          "(analysis)"},
             "time_stretch": {"ratio": 1.0028},
             "alignment": {"placement": {"method": "section_entry",
                                         "section": "chorus", "moved_s": 3.19,
                                         "entry_s": 12.82}},
             "beat_fit": {"action": "looped", "original_len_s": 20.0,
                          "loop_s": [4.96, 17.36], "result_len_s": 38.0}}

    def test_every_change_gets_a_sentence(self):
        notes = pipeline._render_notes(self.VDNA, self.TINFO)
        text = "\n".join(notes)
        for must in ("DC offset", "hum at 150 Hz", "separated from the "
                     "background noise", "reverb", "gaps between lines",
                     "silenced 3.20 s", "slowed by 0.28%", "moved +3.19 s",
                     "chorus", "shorter than the take", "5-17 s section",
                     "treated as melodic rap, as you said"):
            self.assertIn(must, text)

    def test_nothing_done_is_nothing_said(self):
        self.assertEqual(pipeline._render_notes({}, {}), [])
        kept = {"performance_span": {"decision": "keep",
                                     "note": "lead-in kept (user)"},
                "time_stretch": {"ratio": 1.0},
                "alignment": {"placement": {"method": "kept"}},
                "beat_fit": {"action": "none"}}
        self.assertEqual(pipeline._render_notes({}, kept), [])

    def test_a_trimmed_beat_and_a_plain_denoise_are_described(self):
        notes = pipeline._render_notes(
            {"restoration": {"denoise": True}},
            {"beat_fit": {"action": "trimmed_to_bar", "original_len_s": 153.6,
                          "result_len_s": 54.4}})
        text = "\n".join(notes)
        self.assertIn("noise was reduced", text)
        self.assertIn("trimmed on a bar line from 154 s to 54 s", text)


class TestBriefsCarryRepairs(unittest.TestCase):

    def test_the_vocal_brief_keeps_repairs_and_warnings(self):
        brief = service._vocal_brief({"repairs": ["flipped"],
                                      "warnings": ["dull"], "notes": "x"})
        self.assertEqual(brief["repairs"], ["flipped"])
        self.assertEqual(brief["warnings"], ["dull"])

    def test_the_beat_brief_reads_the_loader_report(self):
        brief = service._beat_brief({"quality": {"repairs": ["collapsed"],
                                                  "warnings": ["clipped"]}})
        self.assertEqual((brief["repairs"], brief["warnings"]),
                         (["collapsed"], ["clipped"]))
        self.assertEqual(service._beat_brief({})["repairs"], [])


if __name__ == "__main__":
    unittest.main()
