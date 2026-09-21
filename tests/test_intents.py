"""
Intents: what the user tells the engine that it cannot measure.

Every field defaults to None, meaning "decide for me". A field that is
set wins outright -- the engine never overrides a stated intent, because
the user is the only source of truth for provenance.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mixengine.core.intents import Intents                       # noqa: E402


class TestIntents(unittest.TestCase):

    def test_auto_has_every_field_none(self):
        for name, value in Intents.AUTO.to_dict().items():
            self.assertIsNone(value, f"{name} should default to None")

    def test_from_dict_treats_auto_string_as_none(self):
        i = Intents.from_dict({"vocal_state": "auto", "tune": "auto"})
        self.assertIsNone(i.vocal_state)
        self.assertIsNone(i.tune)

    def test_from_dict_reads_values(self):
        i = Intents.from_dict({"vocal_state": "finished",
                               "relationship": "locked",
                               "tune": "off",
                               "bpm": "108"})
        self.assertEqual(i.vocal_state, "finished")
        self.assertEqual(i.relationship, "locked")
        self.assertEqual(i.tune, 0.0)
        self.assertEqual(i.bpm, 108.0)

    def test_off_and_numeric_strength_both_become_floats(self):
        self.assertEqual(Intents.from_dict({"tune": "off"}).tune, 0.0)
        self.assertEqual(Intents.from_dict({"tune": 0.5}).tune, 0.5)

    def test_none_dict_is_all_auto(self):
        self.assertEqual(Intents.from_dict(None), Intents.AUTO)

    def test_rejects_unknown_vocal_state(self):
        with self.assertRaises(ValueError):
            Intents.from_dict({"vocal_state": "pristine"})

    def test_rejects_strength_out_of_range(self):
        with self.assertRaises(ValueError):
            Intents.from_dict({"tune": 1.5})

    def test_round_trip(self):
        i = Intents.from_dict({"vocal_state": "raw", "space": "keep"})
        self.assertEqual(Intents.from_dict(i.to_dict()), i)


if __name__ == "__main__":
    unittest.main()
