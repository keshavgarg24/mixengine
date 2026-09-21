### Task 1: Intents

**Files:**
- Create: `src/mixengine/core/intents.py`
- Test: `tests/test_intents.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Intents` frozen dataclass with fields `vocal_state`, `relationship`, `key`, `bpm`, `tune`, `timing`, `space`, `separate`, `loudness`, all `Optional`; `Intents.from_dict(d: Optional[dict]) -> Intents`; `Intents.to_dict() -> dict`; `Intents.AUTO` sentinel instance with every field `None`.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_intents.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mixengine.core.intents'`

- [ ] **Step 3: Write minimal implementation**

```python
"""
What the user tells the engine.

Measurement can establish that a vocal sits within 14 cents of a
semitone grid. It cannot establish that the singer meant it, or that
this beat is the one they recorded against. Those are facts about
provenance, and the only reliable source for them is the person who
made the recording.

Every field defaults to None, meaning "decide for me". A field that is
set wins outright. This mirrors what the reference tools do -- RoEx
requires stem role tags, Neutron asks which track is the focus, Nectar
makes key a manual field -- and it exists because the alternative,
guessing, is what produced a render that retuned a finished vocal
toward the wrong key.
"""

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional

VOCAL_STATES = ("raw", "tuned", "finished")
RELATIONSHIPS = ("locked", "free")
SPACES = ("keep", "match", "add")
SEPARATIONS = ("never", "always")


def _choice(value: Any, allowed: tuple, field_name: str) -> Optional[str]:
    if value is None or value == "auto" or value == "":
        return None
    text = str(value).strip().lower()
    if text not in allowed:
        raise ValueError(
            f"{field_name} must be one of {', '.join(('auto',) + allowed)}, "
            f"got {value!r}")
    return text


def _strength(value: Any, field_name: str) -> Optional[float]:
    """A strength is `auto`, `off`, or a number in [0, 1]."""
    if value is None or value == "auto" or value == "":
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "off":
            return 0.0
        try:
            value = float(text)
        except ValueError:
            raise ValueError(
                f"{field_name} must be auto, off, or 0-1, got {value!r}")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field_name} must be within 0-1, got {number}")
    return number


def _number(value: Any, field_name: str,
            low: float, high: float) -> Optional[float]:
    if value is None or value == "auto" or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be a number, got {value!r}")
    if not low <= number <= high:
        raise ValueError(
            f"{field_name} must be within {low}-{high}, got {number}")
    return number


@dataclass(frozen=True)
class Intents:
    """User-stated facts. None means "decide for me"."""

    vocal_state: Optional[str] = None      # raw | tuned | finished
    relationship: Optional[str] = None     # locked | free
    key: Optional[str] = None              # e.g. "f_minor"
    bpm: Optional[float] = None
    tune: Optional[float] = None           # 0 = off
    timing: Optional[float] = None         # 0 = off
    space: Optional[str] = None            # keep | match | add
    separate: Optional[str] = None         # never | always
    loudness: Optional[float] = None       # target LUFS

    AUTO: "Intents"

    @staticmethod
    def from_dict(d: Optional[Dict[str, Any]]) -> "Intents":
        if not d:
            return Intents()
        key = d.get("key")
        return Intents(
            vocal_state=_choice(d.get("vocal_state"), VOCAL_STATES,
                                "vocal_state"),
            relationship=_choice(d.get("relationship"), RELATIONSHIPS,
                                 "relationship"),
            key=(str(key).strip().lower()
                 if key not in (None, "", "auto") else None),
            bpm=_number(d.get("bpm"), "bpm", 40.0, 220.0),
            tune=_strength(d.get("tune"), "tune"),
            timing=_strength(d.get("timing"), "timing"),
            space=_choice(d.get("space"), SPACES, "space"),
            separate=_choice(d.get("separate"), SEPARATIONS, "separate"),
            loudness=_number(d.get("loudness"), "loudness", -30.0, -3.0),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def is_all_auto(self) -> bool:
        return all(getattr(self, f.name) is None for f in fields(self))


Intents.AUTO = Intents()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_intents.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Lint, type-check, commit**

```bash
.venv/bin/ruff check src/mixengine/core/intents.py tests/test_intents.py
.venv/bin/mypy src/mixengine/core/intents.py
git add src/mixengine/core/intents.py tests/test_intents.py
git commit -m "Add Intents: the facts measurement cannot establish

Provenance is not measurable. Whether a vocal was recorded to this
beat, and whether its pitch sits on the grid by intent, are things
only the person who made the recording knows. Intents carry them."
```

---

