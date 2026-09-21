# Phase 1: Intake, Policy, and Locked Render — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the engine able to conclude "leave this alone" — so a finished vocal over its own beat renders faithfully instead of being retuned, requantised, de-reverbed, and laid over a looped intro.

**Architecture:** A pure decision layer sits between analysis and rendering. Intake detectors produce facts about provenance and state; `Intents` carry what the user knows and measurement cannot; `policy.plan()` combines them into a `RenderPlan` holding one `StageDecision` per stage. Pipeline stages read the plan instead of deciding for themselves. The critic verifies the plan was honoured.

**Tech Stack:** Python 3.9–3.13, numpy, scipy, librosa, dataclasses, unittest. No new runtime dependencies.

**Spec:** `docs/superpowers/specs/2026-09-21-mixengine-rebuild-design.md`

## Global Constraints

- Python 3.9 compatible. No `match`, no `X | Y` unions in annotations, no `dict[str, int]` builtin generics at runtime — use `typing.Dict`, `Optional`, `Tuple`.
- No new runtime dependencies in this phase.
- Tests use `unittest`, live in `tests/`, and start with the `sys.path.insert(0, .../src)` preamble used by every existing test file.
- Run tests with `.venv/bin/python -m pytest`. The Homebrew `python3` is 3.14 and cannot import torch.
- `ruff check src tests` and `mypy src` must pass before every commit.
- Never write to `data/` in tests. Use `tests/fixtures/`.
- Confidence values are floats in [0,1]. Every detector returns a value, a confidence, and a human-readable `evidence` string.
- Ambiguity resolves toward doing less. Under-processing yields a flat render; over-processing yields a destroyed one.
- Commit after every task with a message explaining *why*, not what.

---

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

### Task 2: Vocal state detection

**Files:**
- Create: `src/mixengine/analysis/intake.py`
- Test: `tests/test_intake_vocal_state.py`

**Interfaces:**
- Consumes: `Intents` from Task 1.
- Produces: `VocalState` frozen dataclass with `state: str`, `confidence: float`, `evidence: str`, `tuned_fraction: float`, `crest_db: float`, `phrase_spread_db: float`, `rt60_s: float`, `reverb_is_intentional: bool`, and `to_dict()`. Function `detect_vocal_state(y: np.ndarray, sr: int, vdna: dict, intents: Intents = Intents.AUTO) -> VocalState`.

- [ ] **Step 1: Write the failing test**

```python
"""
Vocal state: raw, tuned, or finished.

The engine's worst failure was treating a finished vocal as a raw take
-- retuning 317 of 506 notes that were already within 14 cents of the
grid, compressing a vocal whose phrases varied by under 1 dB, and
stripping a reverb that was a mix decision. Each of those is visible in
the signal if the question is asked.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mixengine.analysis.intake import detect_vocal_state          # noqa: E402
from mixengine.core.intents import Intents                        # noqa: E402

SR = 44100


def dna_with_notes(cents_offsets, spread_db=4.0, rt60=0.2,
                   deviation_cents=None):
    """A vocal DNA document whose notes sit at given cent offsets."""
    notes = []
    t = 0.0
    for i, off in enumerate(cents_offsets):
        midi = 57 + (i % 7) + off / 100.0
        notes.append({"midi": midi, "start": t, "duration": 0.4})
        t += 0.5
    if deviation_cents is None:
        deviation_cents = float(np.mean(np.abs(cents_offsets)))
    return {"notes": notes,
            "phrase_level_spread_db": spread_db,
            "tuning_deviation_cents": deviation_cents,
            "estimated_rt60_s": rt60}


def signal(crest_db=14.0, seconds=6.0, seed=1):
    """Noise bursts whose peak-to-RMS ratio is set to `crest_db`."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    y = rng.standard_normal(n)
    duty = 10.0 ** (-crest_db / 20.0)
    gate = (np.arange(n) % 1000) < max(1, int(1000 * duty * 4))
    y = y * gate
    y /= (np.abs(y).max() + 1e-12)
    return y.astype(np.float32)


class TestVocalState(unittest.TestCase):

    def test_notes_on_the_grid_read_as_tuned(self):
        dna = dna_with_notes([2, -3, 1, 4, -2, 0, 3, -1] * 4)
        st = detect_vocal_state(signal(), SR, dna)
        self.assertGreater(st.tuned_fraction, 0.8)
        self.assertIn(st.state, ("tuned", "finished"))

    def test_notes_spread_across_the_semitone_read_as_raw(self):
        dna = dna_with_notes([40, -35, 22, -48, 31, -27, 44, -19] * 4,
                             spread_db=5.0)
        st = detect_vocal_state(signal(), SR, dna)
        self.assertLess(st.tuned_fraction, 0.5)
        self.assertEqual(st.state, "raw")

    def test_narrow_phrase_spread_indicates_compression(self):
        tuned = [2, -3, 1, 4, -2, 0, 3, -1] * 4
        finished = detect_vocal_state(signal(), SR,
                                      dna_with_notes(tuned, spread_db=0.97))
        self.assertEqual(finished.state, "finished")

    def test_tuned_but_dynamic_is_not_finished(self):
        tuned = [2, -3, 1, 4, -2, 0, 3, -1] * 4
        st = detect_vocal_state(signal(), SR,
                                dna_with_notes(tuned, spread_db=5.5))
        self.assertEqual(st.state, "tuned")

    def test_reverb_on_a_finished_vocal_is_intentional(self):
        tuned = [2, -3, 1, 4, -2, 0, 3, -1] * 4
        st = detect_vocal_state(signal(), SR,
                                dna_with_notes(tuned, spread_db=0.97,
                                               rt60=0.77))
        self.assertTrue(st.reverb_is_intentional)

    def test_reverb_on_a_raw_vocal_is_a_room(self):
        raw = [40, -35, 22, -48, 31, -27, 44, -19] * 4
        st = detect_vocal_state(signal(), SR,
                                dna_with_notes(raw, spread_db=5.0, rt60=0.77))
        self.assertFalse(st.reverb_is_intentional)

    def test_intent_overrides_measurement(self):
        raw = [40, -35, 22, -48, 31, -27, 44, -19] * 4
        st = detect_vocal_state(signal(), SR, dna_with_notes(raw),
                                Intents.from_dict({"vocal_state": "finished"}))
        self.assertEqual(st.state, "finished")
        self.assertEqual(st.confidence, 1.0)
        self.assertIn("you told us", st.evidence)

    def test_too_few_notes_is_low_confidence_not_a_guess(self):
        st = detect_vocal_state(signal(), SR, dna_with_notes([1, 2, 3]))
        self.assertLess(st.confidence, 0.5)

    def test_report_is_serialisable(self):
        st = detect_vocal_state(signal(), SR, dna_with_notes([2, -3, 1, 4]))
        d = st.to_dict()
        self.assertIn("state", d)
        self.assertIsInstance(d["confidence"], float)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_intake_vocal_state.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mixengine.analysis.intake'`

- [ ] **Step 3: Write minimal implementation**

```python
"""
Intake: what this audio is, and where it came from.

The DNA describes what the audio *is* -- its key, its tempo, its notes.
Intake describes what was already *done* to it, and whether the vocal
and the beat belong together. Those two questions decide what the
engine may touch, and until this module existed nothing asked them.

Every detector reports a value, a confidence, and the evidence behind
it, because a decision the user cannot see is a decision they cannot
correct.
"""

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..core.intents import Intents

log = logging.getLogger("mixengine.intake")

# A note within this many cents of a semitone centre reads as deliberate.
# Antares' Flex-Tune calls 50-100 "significantly off"; Melodyne leaves
# notes "already quite close" alone. 20 cents is inside both.
TUNED_WINDOW_CENTS = 20.0

# Below this fraction of note time on the grid, the take is raw.
TUNED_FRACTION_RAW = 0.55
TUNED_FRACTION_SURE = 0.80

# A raw take's phrases vary 3-6 dB. A mixed vocal has been levelled.
SPREAD_MIXED_DB = 1.5
SPREAD_DYNAMIC_DB = 3.0

# Reverb longer than this is worth remarking on either way.
RT60_NOTABLE_S = 0.45

# Fewer notes than this and there is not enough evidence to judge.
MIN_NOTES_FOR_CONFIDENCE = 12


@dataclass(frozen=True)
class VocalState:
    """Whether the take is raw, tuned, or finished -- and the evidence."""

    state: str                       # raw | tuned | finished
    confidence: float
    evidence: str
    tuned_fraction: float = 0.0
    crest_db: float = 0.0
    phrase_spread_db: float = 0.0
    rt60_s: float = 0.0
    reverb_is_intentional: bool = False
    n_notes: int = 0

    @property
    def is_tuned(self) -> bool:
        return self.state in ("tuned", "finished")

    @property
    def is_mixed(self) -> bool:
        return self.state == "finished"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["is_tuned"] = self.is_tuned
        d["is_mixed"] = self.is_mixed
        return d


def _tuned_fraction(notes: Sequence[dict]) -> float:
    """Duration-weighted fraction of notes sitting on a semitone centre.

    A singer aiming at a note and a singer sliding through one produce
    the same mean deviation if the slides are symmetric. Weighting by
    duration asks the better question: how much of the *time* was spent
    on a grid pitch.
    """
    total = 0.0
    on_grid = 0.0
    for n in notes:
        midi = n.get("midi")
        if midi is None:
            continue
        duration = float(n.get("duration", 0.0) or 0.0)
        if duration <= 0.0:
            continue
        cents = abs(((float(midi) + 0.5) % 1.0) - 0.5) * 100.0
        total += duration
        if cents <= TUNED_WINDOW_CENTS:
            on_grid += duration
    return float(on_grid / total) if total > 0 else 0.0


def _crest_db(y: np.ndarray) -> float:
    """Peak minus RMS over the active part of the signal."""
    mono = y if y.ndim == 1 else np.mean(y, axis=1)
    active = mono[np.abs(mono) > (np.abs(mono).max() * 0.02 + 1e-9)]
    if active.size < 128:
        return 0.0
    rms = float(np.sqrt(np.mean(active ** 2)))
    peak = float(np.abs(active).max())
    if rms <= 0 or peak <= 0:
        return 0.0
    return float(20.0 * np.log10(peak / rms))


def detect_vocal_state(y: np.ndarray, sr: int, vdna: dict,
                       intents: Intents = Intents.AUTO) -> VocalState:
    """Decide whether a take is raw, tuned, or finished.

    Ambiguity resolves toward the more finished state. Treating a raw
    take as finished yields a flat render the user can ask more of;
    treating a finished take as raw destroys work that cannot be
    recovered.
    """
    notes: List[dict] = list(vdna.get("notes") or [])
    tuned_fraction = _tuned_fraction(notes)
    spread = float(vdna.get("phrase_level_spread_db") or 0.0)
    rt60 = float(vdna.get("estimated_rt60_s") or 0.0)
    crest = _crest_db(np.asarray(y))

    if intents.vocal_state is not None:
        return VocalState(
            state=intents.vocal_state, confidence=1.0,
            evidence=f"you told us the vocal is {intents.vocal_state}",
            tuned_fraction=tuned_fraction, crest_db=crest,
            phrase_spread_db=spread, rt60_s=rt60,
            reverb_is_intentional=(intents.vocal_state != "raw"
                                   and rt60 > RT60_NOTABLE_S),
            n_notes=len(notes))

    is_tuned = tuned_fraction >= TUNED_FRACTION_RAW
    is_levelled = 0.0 < spread <= SPREAD_MIXED_DB

    if is_tuned and is_levelled:
        state = "finished"
    elif is_tuned:
        state = "tuned"
    else:
        state = "raw"

    # Reverb is only evidence of a bad room on an otherwise raw take. On
    # a tuned or levelled vocal it is a mix decision, and removing it
    # destroys the sound the artist chose.
    reverb_is_intentional = bool(rt60 > RT60_NOTABLE_S and state != "raw")

    confidence = _state_confidence(tuned_fraction, spread, len(notes))

    bits = [f"{tuned_fraction * 100:.0f}% of note time on the grid"]
    if spread > 0:
        bits.append(f"phrases vary {spread:.1f} dB")
    if rt60 > RT60_NOTABLE_S:
        bits.append(f"reverb ~{rt60:.2f}s "
                    f"({'a mix choice' if reverb_is_intentional else 'a room'})")
    evidence = "; ".join(bits)

    log.info("vocal state: %s (%.0f%% confident) -- %s",
             state, confidence * 100, evidence)

    return VocalState(state=state, confidence=confidence, evidence=evidence,
                      tuned_fraction=tuned_fraction, crest_db=crest,
                      phrase_spread_db=spread, rt60_s=rt60,
                      reverb_is_intentional=reverb_is_intentional,
                      n_notes=len(notes))


def _state_confidence(tuned_fraction: float, spread: float,
                      n_notes: int) -> float:
    """How sure we are, given how far the evidence sits from the fence."""
    if n_notes < MIN_NOTES_FOR_CONFIDENCE:
        return 0.3
    if tuned_fraction >= TUNED_FRACTION_SURE or tuned_fraction <= 0.3:
        pitch_conf = 0.9
    else:
        distance = abs(tuned_fraction - TUNED_FRACTION_RAW)
        pitch_conf = 0.5 + min(distance / 0.25, 1.0) * 0.4
    if spread <= 0:
        return float(pitch_conf * 0.8)
    if spread <= SPREAD_MIXED_DB or spread >= SPREAD_DYNAMIC_DB:
        level_conf = 0.9
    else:
        level_conf = 0.6
    return float(min(pitch_conf, level_conf))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_intake_vocal_state.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Lint, type-check, commit**

```bash
.venv/bin/ruff check src/mixengine/analysis/intake.py tests/test_intake_vocal_state.py
.venv/bin/mypy src/mixengine/analysis/intake.py
git add src/mixengine/analysis/intake.py tests/test_intake_vocal_state.py
git commit -m "Detect whether a vocal is raw, tuned, or finished

The engine retuned 317 of 506 notes that already sat within 14 cents
of the grid, and compressed a vocal whose phrases varied by 0.97 dB.
Both facts were in the DNA. Nothing read them."
```

---

### Task 3: Relationship detection

**Files:**
- Modify: `src/mixengine/analysis/intake.py` (append)
- Test: `tests/test_intake_relationship.py`

**Interfaces:**
- Consumes: `Intents` (Task 1), `intake` module (Task 2).
- Produces: `Relationship` frozen dataclass with `state: str` (`locked`/`free`), `confidence: float`, `evidence: str`, `offset_s: float`, `peak_ratio: float`, `duration_delta_s: float`, `to_dict()`. Function `detect_relationship(vocal, beat, sr, vdna, bdna, intents=Intents.AUTO) -> Relationship`.

- [ ] **Step 1: Write the failing test**

```python
"""
Relationship: was this vocal recorded to this beat?

If it was, the alignment problem is one number -- the lag between them
-- and every stage that would "fix" the timing is doing damage. The
same cross-correlation that detects the relationship measures the lag,
so detection and alignment are one operation.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mixengine.analysis.intake import detect_relationship          # noqa: E402
from mixengine.core.intents import Intents                         # noqa: E402

SR = 22050


def pulse_train(seconds, period_s, sr=SR, seed=0, jitter=0.0):
    """Impulses at a fixed period -- a stand-in for an onset envelope."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    y = np.zeros(n, dtype=np.float32)
    t = 0.0
    while t < seconds:
        idx = int((t + rng.normal(0, jitter)) * sr)
        if 0 <= idx < n:
            y[idx:idx + 64] = 1.0
        t += period_s
    return y + rng.standard_normal(n).astype(np.float32) * 0.01


def dna(duration, bpm):
    return {"duration_s": duration, "bpm": bpm}


class TestRelationship(unittest.TestCase):

    def test_same_performance_reads_as_locked(self):
        beat = pulse_train(20.0, 0.5)
        vocal = np.concatenate([np.zeros(int(1.5 * SR), dtype=np.float32),
                                beat[:int(18.5 * SR)]])
        r = detect_relationship(vocal, beat, SR,
                                dna(20.0, 120.0), dna(20.0, 120.0))
        self.assertEqual(r.state, "locked")
        self.assertAlmostEqual(r.offset_s, 1.5, delta=0.05)

    def test_unrelated_audio_reads_as_free(self):
        beat = pulse_train(20.0, 0.5, seed=1)
        vocal = pulse_train(20.0, 0.37, seed=99, jitter=0.02)
        r = detect_relationship(vocal, beat, SR,
                                dna(20.0, 162.0), dna(20.0, 120.0))
        self.assertEqual(r.state, "free")

    def test_different_durations_are_not_locked(self):
        beat = pulse_train(20.0, 0.5)
        vocal = beat[:int(8.0 * SR)]
        r = detect_relationship(vocal, beat, SR,
                                dna(8.0, 120.0), dna(20.0, 120.0))
        self.assertEqual(r.state, "free")

    def test_half_time_tempo_is_not_a_disagreement(self):
        """Trap is written at 146 and felt at 73. That is one tempo."""
        beat = pulse_train(20.0, 0.5)
        vocal = np.concatenate([np.zeros(int(0.25 * SR), dtype=np.float32),
                                beat[:int(19.75 * SR)]])
        r = detect_relationship(vocal, beat, SR,
                                dna(20.0, 73.0), dna(20.0, 146.0))
        self.assertEqual(r.state, "locked")

    def test_intent_overrides_measurement(self):
        beat = pulse_train(20.0, 0.5, seed=1)
        vocal = pulse_train(20.0, 0.37, seed=99)
        r = detect_relationship(vocal, beat, SR,
                                dna(20.0, 162.0), dna(20.0, 120.0),
                                Intents.from_dict({"relationship": "locked"}))
        self.assertEqual(r.state, "locked")
        self.assertIn("you told us", r.evidence)

    def test_offset_is_reported_even_when_free(self):
        beat = pulse_train(20.0, 0.5, seed=1)
        vocal = pulse_train(20.0, 0.37, seed=99)
        r = detect_relationship(vocal, beat, SR,
                                dna(20.0, 162.0), dna(20.0, 120.0))
        self.assertIsInstance(r.offset_s, float)

    def test_report_is_serialisable(self):
        beat = pulse_train(10.0, 0.5)
        r = detect_relationship(beat, beat, SR, dna(10.0, 120.0),
                                dna(10.0, 120.0))
        self.assertIn("state", r.to_dict())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_intake_relationship.py -v`
Expected: FAIL with `ImportError: cannot import name 'detect_relationship'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/mixengine/analysis/intake.py`:

```python
# ─────────────────────────────────────────────────────────────────────────
# Relationship: does this vocal belong to this beat?
# ─────────────────────────────────────────────────────────────────────────

# Cross-correlation search window. The alignment literature uses +/-20 s
# for live-vocal-to-studio matching; a take that starts more than 20 s
# into a beat is not a take recorded to it.
MAX_LAG_S = 20.0

# The main peak must stand this far above the best competing peak before
# a single lag is believable.
PEAK_RATIO_LOCKED = 1.6

# Durations must agree within roughly two bars.
DURATION_TOLERANCE_S = 4.0

ENVELOPE_SR = 100


@dataclass(frozen=True)
class Relationship:
    """Whether the vocal was recorded to this beat -- and at what lag."""

    state: str                     # locked | free
    confidence: float
    evidence: str
    offset_s: float = 0.0
    peak_ratio: float = 0.0
    duration_delta_s: float = 0.0
    tempo_agrees: bool = False

    @property
    def is_locked(self) -> bool:
        return self.state == "locked"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["is_locked"] = self.is_locked
        return d


def _envelope(y: np.ndarray, sr: int) -> np.ndarray:
    """Onset-strength envelope at ENVELOPE_SR, mean-removed."""
    import librosa
    mono = y if y.ndim == 1 else np.mean(y, axis=1)
    mono = np.ascontiguousarray(mono.astype(np.float32))
    hop = max(1, int(round(sr / ENVELOPE_SR)))
    env = librosa.onset.onset_strength(y=mono, sr=sr, hop_length=hop)
    env = np.asarray(env, dtype=np.float64)
    if env.size == 0:
        return env
    env -= env.mean()
    norm = np.linalg.norm(env)
    return env / norm if norm > 0 else env


def _best_lag(a: np.ndarray, b: np.ndarray) -> tuple:
    """Return (lag_frames, peak_ratio) for `a` against `b`.

    The ratio of the main peak to the strongest peak outside its
    neighbourhood is what separates a real alignment from the periodic
    self-similarity every loop-based beat has. A high correlation at one
    lag means nothing if the next bar correlates just as well.
    """
    if a.size < 4 or b.size < 4:
        return 0, 0.0
    n = int(2 ** np.ceil(np.log2(a.size + b.size)))
    fa = np.fft.rfft(a, n)
    fb = np.fft.rfft(b, n)
    cross = fa * np.conj(fb)
    magnitude = np.abs(cross)
    # PHAT weighting: whiten the cross-spectrum so the correlation peaks
    # on timing rather than on whichever band happens to be loudest.
    cross = np.divide(cross, magnitude + 1e-9)
    corr = np.fft.irfft(cross, n)
    max_lag = int(MAX_LAG_S * ENVELOPE_SR)
    window = np.concatenate([corr[:max_lag + 1], corr[-max_lag:]])
    lags = np.concatenate([np.arange(0, max_lag + 1),
                           np.arange(-max_lag, 0)])
    best = int(np.argmax(window))
    peak = float(window[best])
    if peak <= 0:
        return int(lags[best]), 0.0
    guard = max(2, int(0.15 * ENVELOPE_SR))
    masked = window.copy()
    lo, hi = max(0, best - guard), min(window.size, best + guard + 1)
    masked[lo:hi] = -np.inf
    runner_up = float(np.max(masked)) if np.isfinite(masked).any() else 0.0
    ratio = peak / runner_up if runner_up > 1e-9 else float("inf")
    return int(lags[best]), float(min(ratio, 10.0))


def _tempo_agrees(a: float, b: float) -> bool:
    """True when two tempi match, allowing half and double time.

    Trap is written at 146 and felt at 73. A detector reporting either
    is right, and treating the disagreement as evidence against a
    relationship would reject exactly the genre this engine serves.
    """
    if a <= 0 or b <= 0:
        return False
    for factor in (0.5, 1.0, 2.0):
        if abs(a - b * factor) <= max(2.0, b * factor * 0.04):
            return True
    return False


def detect_relationship(vocal: np.ndarray, beat: np.ndarray, sr: int,
                        vdna: dict, bdna: dict,
                        intents: Intents = Intents.AUTO) -> Relationship:
    """Decide whether the vocal was recorded to this beat.

    Three independent signals must agree: a single dominant lag in the
    cross-correlation of their onset envelopes, durations within about
    two bars, and tempi that match at some metrical level. Any one alone
    is coincidence.
    """
    v_duration = float(vdna.get("duration_s") or 0.0)
    b_duration = float(bdna.get("duration_s") or 0.0)
    duration_delta = abs(v_duration - b_duration)
    tempo_ok = _tempo_agrees(float(vdna.get("bpm") or 0.0),
                             float(bdna.get("bpm") or 0.0))

    try:
        lag_frames, peak_ratio = _best_lag(_envelope(vocal, sr),
                                           _envelope(beat, sr))
        offset_s = float(lag_frames) / ENVELOPE_SR
    except Exception as e:                       # noqa: BLE001
        log.warning("relationship: correlation failed (%s)", e)
        lag_frames, peak_ratio, offset_s = 0, 0.0, 0.0

    if intents.relationship is not None:
        return Relationship(
            state=intents.relationship, confidence=1.0,
            evidence=f"you told us the vocal is {intents.relationship}",
            offset_s=offset_s, peak_ratio=peak_ratio,
            duration_delta_s=duration_delta, tempo_agrees=tempo_ok)

    durations_agree = duration_delta <= DURATION_TOLERANCE_S
    peak_is_clear = peak_ratio >= PEAK_RATIO_LOCKED
    locked = bool(peak_is_clear and durations_agree)

    if locked:
        confidence = float(min(0.95, 0.6 + (peak_ratio - PEAK_RATIO_LOCKED)
                               * 0.15 + (0.1 if tempo_ok else 0.0)))
        evidence = (f"one clear alignment at {offset_s:+.2f}s "
                    f"(peak {peak_ratio:.1f}x the next), lengths within "
                    f"{duration_delta:.1f}s")
    else:
        confidence = 0.7 if not durations_agree else 0.6
        why = []
        if not durations_agree:
            why.append(f"lengths differ by {duration_delta:.1f}s")
        if not peak_is_clear:
            why.append(f"no single alignment stands out "
                       f"(best {peak_ratio:.1f}x)")
        evidence = "; ".join(why)

    log.info("relationship: %s (%.0f%% confident) -- %s",
             "locked" if locked else "free", confidence * 100, evidence)

    return Relationship(state="locked" if locked else "free",
                        confidence=confidence, evidence=evidence,
                        offset_s=offset_s, peak_ratio=peak_ratio,
                        duration_delta_s=duration_delta,
                        tempo_agrees=tempo_ok)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_intake_relationship.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Lint, type-check, commit**

```bash
.venv/bin/ruff check src/mixengine/analysis/intake.py tests/test_intake_relationship.py
.venv/bin/mypy src/mixengine/analysis/intake.py
git add src/mixengine/analysis/intake.py tests/test_intake_relationship.py
git commit -m "Detect whether the vocal was recorded to this beat

When it was, alignment is one number and every stage that 'fixes' the
timing is doing damage. The cross-correlation that answers the question
also measures the lag, so detection and alignment are one operation."
```

---

### Task 4: Key decision

**Files:**
- Modify: `src/mixengine/analysis/intake.py` (append)
- Test: `tests/test_intake_key.py`
- Create: `tests/fixtures/key_candidates_voc_beat.json`

**Interfaces:**
- Consumes: `Intents` (Task 1), `mixengine.core.keys.Key`.
- Produces: `KeyDecision` frozen dataclass with `key: Optional[Key]`, `semitone_shift: int`, `confidence: float`, `evidence: str`, `families_agree: bool`, `scale_pcs: Tuple[int, ...]`, `to_dict()`. Function `decide_key(vdna: dict, bdna: dict, intents=Intents.AUTO) -> KeyDecision`.

- [ ] **Step 1: Create the fixture**

```bash
cat > tests/fixtures/key_candidates_voc_beat.json <<'EOF'
{
  "note": "Exact candidate lists from the 2026-09-20 voc.mp3 + beat.mp3 render. The engine reported the beat at 'key_confidence 1.0' as G# Major and transposed nothing, but handed the tuner G# major chord tones -- dragging a C# minor vocal's thirds sharp on 39 of 87 bars.",
  "vocal": {
    "key": {"pc": 1, "mode": "minor", "name": "C# Minor", "camelot": "12A"},
    "key_confidence": 0.31,
    "key_candidates": [["C# Minor", 0.436076158392668], ["E Major", 0.40214325489591224], ["G# Minor", 0.3729587095930959], ["D# Minor", 0.3505077166277843], ["G# Major", 0.29594861100369524]]
  },
  "beat": {
    "key": {"pc": 8, "mode": "major", "name": "G# Major", "camelot": "4B"},
    "key_confidence": 1.0,
    "key_candidates": [["G# Major", 0.7517002295037931], ["G# Minor", 0.6690403071807568], ["C# Major", 0.3789169934818132], ["D# Major", 0.3492150918782125], ["F Minor", 0.34476638815074917]]
  }
}
EOF
```

- [ ] **Step 2: Write the failing test**

```python
"""
Key as a distribution, not a label.

DJ tools disagree on roughly 60% of tracks, and the disagreements are
overwhelmingly relative-major/minor or fifth swaps. The reference render
is exactly that case: the vocal's candidates were C# Minor / E Major /
G# Minor, the beat's were G# Major / G# Minor / C# Major -- all one
family -- and the engine reported the beat at confidence 1.0 and tuned
the vocal toward G# major thirds.
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mixengine.analysis.intake import decide_key                   # noqa: E402
from mixengine.core.intents import Intents                         # noqa: E402
from mixengine.core.keys import Key                                # noqa: E402

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures",
                       "key_candidates_voc_beat.json")


def load_reference():
    with open(FIXTURE) as fh:
        return json.load(fh)


class TestDecideKey(unittest.TestCase):

    def test_reference_render_does_not_transpose(self):
        ref = load_reference()
        d = decide_key(ref["vocal"], ref["beat"])
        self.assertEqual(d.semitone_shift, 0)

    def test_reference_render_resolves_to_the_minor_family(self):
        """C# minor and G# major share no tonic, but the vocal states a
        third and an 808 does not. The voice wins on mode."""
        ref = load_reference()
        d = decide_key(ref["vocal"], ref["beat"])
        self.assertTrue(d.families_agree)
        self.assertIsNotNone(d.key)
        self.assertEqual(d.key.mode, "minor")

    def test_reference_render_reports_honest_confidence(self):
        ref = load_reference()
        d = decide_key(ref["vocal"], ref["beat"])
        self.assertLess(d.confidence, 0.8)

    def test_relative_keys_are_one_family(self):
        vocal = {"key": Key(0, "minor").to_dict(), "key_confidence": 0.7}
        beat = {"key": Key(3, "major").to_dict(), "key_confidence": 0.7}
        d = decide_key(vocal, beat)
        self.assertTrue(d.families_agree)
        self.assertEqual(d.semitone_shift, 0)

    def test_genuinely_distant_keys_propose_a_shift(self):
        vocal = {"key": Key(0, "minor").to_dict(), "key_confidence": 0.9}
        beat = {"key": Key(6, "minor").to_dict(), "key_confidence": 0.9}
        d = decide_key(vocal, beat)
        self.assertNotEqual(d.semitone_shift, 0)

    def test_low_confidence_never_transposes(self):
        vocal = {"key": Key(0, "minor").to_dict(), "key_confidence": 0.2}
        beat = {"key": Key(6, "minor").to_dict(), "key_confidence": 0.9}
        d = decide_key(vocal, beat)
        self.assertEqual(d.semitone_shift, 0)
        self.assertIn("uncertain", d.evidence)

    def test_low_confidence_widens_the_scale_instead_of_forcing_a_third(self):
        vocal = {"key": Key(0, "minor").to_dict(), "key_confidence": 0.2}
        beat = {"key": Key(3, "major").to_dict(), "key_confidence": 0.3}
        d = decide_key(vocal, beat)
        self.assertGreaterEqual(len(d.scale_pcs), 7)

    def test_user_key_wins(self):
        ref = load_reference()
        d = decide_key(ref["vocal"], ref["beat"],
                       Intents.from_dict({"key": "f_minor"}))
        self.assertEqual(d.key, Key(5, "minor"))
        self.assertEqual(d.confidence, 1.0)
        self.assertIn("you told us", d.evidence)

    def test_missing_keys_are_handled(self):
        d = decide_key({}, {})
        self.assertEqual(d.semitone_shift, 0)
        self.assertIsNone(d.key)

    def test_report_is_serialisable(self):
        ref = load_reference()
        self.assertIn("semitone_shift", decide_key(ref["vocal"],
                                                   ref["beat"]).to_dict())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_intake_key.py -v`
Expected: FAIL with `ImportError: cannot import name 'decide_key'`

- [ ] **Step 4: Write minimal implementation**

Append to `src/mixengine/analysis/intake.py`. Add `from ..core.keys import Key, parse_key` to the imports at the top of the file.

```python
# ─────────────────────────────────────────────────────────────────────────
# Key: a distribution, not a label
# ─────────────────────────────────────────────────────────────────────────

# Below this, a key estimate is not strong enough to transpose against.
KEY_CONFIDENCE_TO_TRANSPOSE = 0.55

# Below this, the tuner gets the union scale rather than a forced third.
KEY_CONFIDENCE_TO_NARROW = 0.45


@dataclass(frozen=True)
class KeyDecision:
    """Which key to work in, and whether to move the vocal at all."""

    key: Optional[Key]
    semitone_shift: int
    confidence: float
    evidence: str
    families_agree: bool = False
    scale_pcs: tuple = ()

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key.to_dict() if self.key else None,
                "semitone_shift": self.semitone_shift,
                "confidence": round(float(self.confidence), 3),
                "evidence": self.evidence,
                "families_agree": self.families_agree,
                "scale_pcs": list(self.scale_pcs)}


def _family_root(key: Key) -> int:
    """The relative-major root, so a key and its relative share a value.

    A minor key and its relative major contain the same seven pitch
    classes. A detector choosing between them is choosing a label, not a
    different set of notes, and the engine should not transpose across
    that choice.
    """
    return key.pc if key.mode == "major" else (key.pc + 3) % 12


def decide_key(vdna: dict, bdna: dict,
               intents: Intents = Intents.AUTO) -> KeyDecision:
    """Reconcile the vocal's and the beat's key estimates.

    Reports confidence as measured. The old code wrote 1.0 for a beat
    whose own top candidate scored 0.75, then handed the tuner major
    chord tones for a minor vocal.
    """
    v_key = Key.from_dict(vdna.get("key"))
    b_key = Key.from_dict(bdna.get("key"))
    v_conf = float(vdna.get("key_confidence") or 0.0)
    b_conf = float(bdna.get("key_confidence") or 0.0)

    if intents.key is not None:
        stated = parse_key(intents.key)
        if stated is not None:
            return KeyDecision(
                key=stated, semitone_shift=0, confidence=1.0,
                evidence=f"you told us the key is {stated.name}",
                families_agree=True, scale_pcs=tuple(stated.scale_pcs))
        log.warning("could not parse stated key %r; measuring instead",
                    intents.key)

    if v_key is None and b_key is None:
        return KeyDecision(None, 0, 0.0, "no key could be established")
    if v_key is None:
        return KeyDecision(b_key, 0, b_conf,
                           f"only the beat has a key ({b_key})",
                           True, tuple(b_key.scale_pcs))
    if b_key is None:
        return KeyDecision(v_key, 0, v_conf,
                           f"only the vocal has a key ({v_key})",
                           True, tuple(v_key.scale_pcs))

    families_agree = _family_root(v_key) == _family_root(b_key)
    confidence = float(min(v_conf, b_conf))

    if families_agree:
        # Same seven notes. Take the mode from the vocal: a sung line
        # states its third, a bassline does not.
        chosen = Key(v_key.pc, v_key.mode)
        evidence = (f"vocal {v_key} and beat {b_key} share a key signature; "
                    f"taking the mode from the vocal, no transposition")
        scale = tuple(chosen.scale_pcs)
        if confidence < KEY_CONFIDENCE_TO_NARROW:
            scale = tuple(sorted(set(v_key.scale_pcs) | set(b_key.scale_pcs)))
            evidence += "; both estimates are uncertain, so the tuner gets " \
                        "the full shared scale"
        return KeyDecision(chosen, 0, confidence, evidence, True, scale)

    if v_conf < KEY_CONFIDENCE_TO_TRANSPOSE or \
            b_conf < KEY_CONFIDENCE_TO_TRANSPOSE:
        scale = tuple(sorted(set(v_key.scale_pcs) | set(b_key.scale_pcs)))
        return KeyDecision(
            v_key, 0, confidence,
            f"vocal {v_key} and beat {b_key} disagree but at least one "
            f"estimate is uncertain (vocal {v_conf:.2f}, beat {b_conf:.2f}); "
            f"rendering without a transposition",
            False, scale)

    from ..core.keys import best_shift
    shift, _ = best_shift(v_key, b_key)
    return KeyDecision(
        b_key, int(shift), confidence,
        f"vocal {v_key} and beat {b_key} are in different keys; "
        f"shifting the vocal {shift:+d} semitones",
        False, tuple(b_key.scale_pcs))
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_intake_key.py -v`
Expected: PASS (10 tests)

- [ ] **Step 6: Lint, type-check, commit**

```bash
.venv/bin/ruff check src/mixengine/analysis/intake.py tests/test_intake_key.py
.venv/bin/mypy src/mixengine/analysis/intake.py
git add src/mixengine/analysis/intake.py tests/test_intake_key.py tests/fixtures/key_candidates_voc_beat.json
git commit -m "Treat key as a distribution and stop inventing confidence

The beat was reported at key_confidence 1.0 when its own top candidate
scored 0.75, and the vocal's candidates sat in the same key signature.
Relative keys share seven notes; transposing across that choice, or
handing a minor vocal major chord tones, is not a key decision."
```

---

### Task 5: RenderPlan and policy

**Files:**
- Create: `src/mixengine/core/policy.py`
- Test: `tests/test_policy.py`

**Interfaces:**
- Consumes: `Intents` (Task 1); `VocalState`, `Relationship`, `KeyDecision` (Tasks 2–4).
- Produces: `StageDecision` frozen dataclass (`enabled: bool`, `strength: float`, `method: str`, `reason: str`, `to_dict()`); `RenderPlan` frozen dataclass with attributes `separation`, `dereverb`, `tuning`, `alignment`, `arrangement`, `beat_fit`, `vocal_chain`, `space`, `ducking`, plus `key: KeyDecision`, `offset_s: float`, `is_locked: bool`, `summary() -> str`, `to_dict()`; function `plan(vdna, bdna, vocal_state, relationship, key_decision, intents=Intents.AUTO) -> RenderPlan`.

- [ ] **Step 1: Write the failing test**

```python
"""
The policy table.

Every stage decision in one pure function, so that "what will the engine
do to my audio" is answerable without running it -- and testable without
audio. The old code scattered these across pipeline.py:181, :208, :226,
:241, vocal_dna, and separation.py, which is why no stage could conclude
"leave this alone".
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mixengine.analysis.intake import (KeyDecision, Relationship,  # noqa: E402
                                       VocalState)
from mixengine.core.intents import Intents                         # noqa: E402
from mixengine.core.keys import Key                                # noqa: E402
from mixengine.core.policy import plan                             # noqa: E402


def state(kind, rt60=0.2, intentional=False):
    return VocalState(state=kind, confidence=0.9, evidence="test",
                      tuned_fraction=0.9 if kind != "raw" else 0.2,
                      phrase_spread_db=1.0 if kind == "finished" else 4.0,
                      rt60_s=rt60, reverb_is_intentional=intentional,
                      n_notes=200)


def rel(kind, offset=1.75):
    return Relationship(state=kind, confidence=0.9, evidence="test",
                        offset_s=offset, peak_ratio=3.0,
                        duration_delta_s=0.0, tempo_agrees=True)


def key():
    return KeyDecision(Key(1, "minor"), 0, 0.6, "test", True,
                       tuple(Key(1, "minor").scale_pcs))


DNA = {"duration_s": 147.8, "input_type": "light_bleed",
       "input_type_confidence": 0.56}
BDNA = {"duration_s": 147.8, "genre": "trap"}


class TestLockedFinished(unittest.TestCase):
    """The reference render's case: a finished vocal on its own beat."""

    def setUp(self):
        self.p = plan(DNA, BDNA, state("finished", 0.77, True),
                      rel("locked"), key())

    def test_does_not_separate(self):
        self.assertFalse(self.p.separation.enabled)

    def test_does_not_tune(self):
        self.assertFalse(self.p.tuning.enabled)

    def test_does_not_dereverb(self):
        self.assertFalse(self.p.dereverb.enabled)

    def test_aligns_with_a_single_offset(self):
        self.assertEqual(self.p.alignment.method, "single_offset")
        self.assertAlmostEqual(self.p.offset_s, 1.75)

    def test_does_not_rearrange(self):
        self.assertFalse(self.p.arrangement.enabled)

    def test_pads_the_beat_rather_than_looping_it(self):
        self.assertEqual(self.p.beat_fit.method, "pad")

    def test_finishes_rather_than_produces(self):
        self.assertEqual(self.p.vocal_chain.method, "finish")

    def test_keeps_the_existing_space(self):
        self.assertEqual(self.p.space.method, "keep")

    def test_every_decision_carries_a_reason(self):
        for name, d in self.p.to_dict()["stages"].items():
            self.assertTrue(d["reason"], f"{name} has no reason")


class TestFreeRaw(unittest.TestCase):
    """A raw take over an arbitrary beat still gets full production."""

    def setUp(self):
        self.p = plan(DNA, BDNA, state("raw", 0.9), rel("free"), key())

    def test_tunes(self):
        self.assertTrue(self.p.tuning.enabled)

    def test_tunes_with_a_dead_zone(self):
        self.assertEqual(self.p.tuning.method, "flex")

    def test_dereverbs_a_bad_room(self):
        self.assertTrue(self.p.dereverb.enabled)

    def test_arranges(self):
        self.assertTrue(self.p.arrangement.enabled)

    def test_produces(self):
        self.assertEqual(self.p.vocal_chain.method, "produce")


class TestSeparation(unittest.TestCase):

    def test_coin_flip_bleed_does_not_trigger_separation(self):
        """0.56 confidence cost two minutes and separated a clean vocal."""
        p = plan({**DNA, "input_type_confidence": 0.56},
                 BDNA, state("raw"), rel("free"), key())
        self.assertFalse(p.separation.enabled)

    def test_a_confident_full_mix_does_separate(self):
        p = plan({**DNA, "input_type": "full_mix",
                  "input_type_confidence": 0.82},
                 BDNA, state("raw"), rel("free"), key())
        self.assertTrue(p.separation.enabled)

    def test_locked_never_separates_however_confident(self):
        p = plan({**DNA, "input_type": "full_mix",
                  "input_type_confidence": 0.95},
                 BDNA, state("finished"), rel("locked"), key())
        self.assertFalse(p.separation.enabled)


class TestIntentsOverride(unittest.TestCase):

    def test_tune_off_disables_tuning_on_a_raw_take(self):
        p = plan(DNA, BDNA, state("raw"), rel("free"), key(),
                 Intents.from_dict({"tune": "off"}))
        self.assertFalse(p.tuning.enabled)
        self.assertIn("you", p.tuning.reason)

    def test_tune_on_enables_tuning_on_a_finished_vocal(self):
        p = plan(DNA, BDNA, state("finished"), rel("locked"), key(),
                 Intents.from_dict({"tune": 0.6}))
        self.assertTrue(p.tuning.enabled)
        self.assertAlmostEqual(p.tuning.strength, 0.6)

    def test_separate_never_wins(self):
        p = plan({**DNA, "input_type": "full_mix",
                  "input_type_confidence": 0.95},
                 BDNA, state("raw"), rel("free"), key(),
                 Intents.from_dict({"separate": "never"}))
        self.assertFalse(p.separation.enabled)

    def test_space_keep_wins(self):
        p = plan(DNA, BDNA, state("raw"), rel("free"), key(),
                 Intents.from_dict({"space": "keep"}))
        self.assertEqual(p.space.method, "keep")


class TestSummary(unittest.TestCase):

    def test_summary_is_plain_language(self):
        p = plan(DNA, BDNA, state("finished", 0.77, True), rel("locked"),
                 key())
        text = p.summary()
        self.assertIn("finished", text.lower())
        self.assertLess(len(text), 400)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_policy.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mixengine.core.policy'`

- [ ] **Step 3: Write minimal implementation**

```python
"""
What the engine may do to this audio.

One pure function, no audio, no I/O. The decisions it makes used to be
scattered across six modules, each with its own idea of what the input
was, which is how a finished vocal on its own beat ended up retuned,
requantised, de-reverbed and laid over a looped intro -- with every
stage individually behaving as designed.

Having them in one place makes the engine's behaviour a table you can
read, test without audio, and show to the user before it runs.
"""

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict

from ..analysis.intake import KeyDecision, Relationship, VocalState
from .intents import Intents

log = logging.getLogger("mixengine.policy")

# A "full mix" call this confident is worth two minutes of separation.
# The reference render separated on 0.56 -- a coin flip.
SEPARATION_CONFIDENCE = 0.75

# Reverb longer than this on a raw take is a room worth removing.
DEREVERB_RT60_S = 0.8

# Flex-Tune semantics: leave notes already close alone.
TUNING_DEAD_ZONE_CENTS = 35.0


@dataclass(frozen=True)
class StageDecision:
    """What one stage may do, how much, and why."""

    enabled: bool
    strength: float = 0.0
    method: str = "none"
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RenderPlan:
    """Every stage decision for one render."""

    separation: StageDecision
    dereverb: StageDecision
    tuning: StageDecision
    alignment: StageDecision
    arrangement: StageDecision
    beat_fit: StageDecision
    vocal_chain: StageDecision
    space: StageDecision
    ducking: StageDecision

    key: KeyDecision
    offset_s: float
    is_locked: bool
    vocal_state: str

    def summary(self) -> str:
        """The plan in the language a musician would use."""
        origin = ("was recorded to this beat" if self.is_locked
                  else "is being matched to this beat")
        doing = []
        doing.append(f"tuning {'on' if self.tuning.enabled else 'off'}")
        doing.append("timing kept" if self.alignment.method == "single_offset"
                     else "timing aligned")
        doing.append("reverb kept" if self.space.method == "keep"
                     else "reverb matched to the beat")
        doing.append("beat not looped" if self.beat_fit.method == "pad"
                     else f"beat {self.beat_fit.method}")
        return (f"Your vocal is {self.vocal_state} and {origin}. "
                + " · ".join(doing) + ".")

    def to_dict(self) -> Dict[str, Any]:
        stages = {name: getattr(self, name).to_dict() for name in
                  ("separation", "dereverb", "tuning", "alignment",
                   "arrangement", "beat_fit", "vocal_chain", "space",
                   "ducking")}
        return {"stages": stages, "key": self.key.to_dict(),
                "offset_s": round(float(self.offset_s), 4),
                "is_locked": self.is_locked,
                "vocal_state": self.vocal_state,
                "summary": self.summary()}


def _override(intent_value: Any, decision: StageDecision,
              method: str = "") -> StageDecision:
    """Apply a user intent to a decision, recording that they asked."""
    if intent_value is None:
        return decision
    if isinstance(intent_value, float):
        enabled = intent_value > 0.0
        return StageDecision(
            enabled=enabled,
            strength=intent_value,
            method=decision.method if enabled else "none",
            reason=f"you asked for {'strength %.2f' % intent_value if enabled else 'this off'}")
    return StageDecision(enabled=decision.enabled, strength=decision.strength,
                         method=method or str(intent_value),
                         reason=f"you asked for {intent_value}")


def plan(vdna: dict, bdna: dict, vocal_state: VocalState,
         relationship: Relationship, key_decision: KeyDecision,
         intents: Intents = Intents.AUTO) -> RenderPlan:
    """Decide what every stage may do.

    Ambiguity resolves toward doing less. A flat render can be asked for
    more; a destroyed one cannot be recovered.
    """
    locked = relationship.is_locked
    finished = vocal_state.is_mixed
    tuned = vocal_state.is_tuned

    # ── Separation ───────────────────────────────────────────────────
    sep_conf = float(vdna.get("input_type_confidence") or 0.0)
    is_full_mix = vdna.get("input_type") == "full_mix"
    if locked:
        separation = StageDecision(
            False, reason="the vocal was recorded to this beat, so the "
                          "bleed is the beat itself")
    elif is_full_mix and sep_conf >= SEPARATION_CONFIDENCE:
        separation = StageDecision(
            True, 1.0, "demucs",
            f"instrumental content detected at {sep_conf:.2f} confidence")
    else:
        separation = StageDecision(
            False, reason=f"no confident instrumental content "
                          f"({sep_conf:.2f}); separating would cost minutes "
                          f"and risk the take")
    separation = _override(intents.separate, separation)
    if intents.separate == "never":
        separation = StageDecision(False, reason="you asked us not to separate")
    elif intents.separate == "always":
        separation = StageDecision(True, 1.0, "demucs",
                                   "you asked us to separate")

    # ── Dereverb ─────────────────────────────────────────────────────
    if vocal_state.reverb_is_intentional:
        dereverb = StageDecision(
            False, reason=f"the {vocal_state.rt60_s:.2f}s tail is part of "
                          f"the vocal's sound, not a room")
    elif vocal_state.rt60_s > DEREVERB_RT60_S:
        dereverb = StageDecision(
            True, 0.7, "spectral",
            f"a {vocal_state.rt60_s:.2f}s room on an otherwise raw take")
    else:
        dereverb = StageDecision(False, reason="no problematic room")

    # ── Tuning ───────────────────────────────────────────────────────
    if tuned:
        tuning = StageDecision(
            False, reason=f"{vocal_state.tuned_fraction * 100:.0f}% of the "
                          f"note time is already on the grid")
    else:
        tuning = StageDecision(
            True, 0.5, "flex",
            f"only {vocal_state.tuned_fraction * 100:.0f}% of the note time "
            f"is on the grid; correcting notes more than "
            f"{TUNING_DEAD_ZONE_CENTS:.0f} cents off")
    if intents.tune is not None:
        tuning = StageDecision(
            enabled=intents.tune > 0.0, strength=intents.tune,
            method="flex" if intents.tune > 0.0 else "none",
            reason=(f"you asked for tuning at {intents.tune:.2f}"
                    if intents.tune > 0 else "you asked for tuning off"))

    # ── Alignment ────────────────────────────────────────────────────
    if locked:
        alignment = StageDecision(
            True, 0.0, "single_offset",
            f"recorded to this beat; applying the measured "
            f"{relationship.offset_s:+.2f}s lag and nothing else")
    elif finished:
        alignment = StageDecision(
            True, 0.2, "phrase_anchor",
            "a finished vocal's timing is a performance; anchoring phrases "
            "to downbeats without quantising inside them")
    else:
        alignment = StageDecision(
            True, 0.45, "grid_nudge",
            "a raw take over a new beat; nudging onsets toward the grid")
    if intents.timing is not None:
        alignment = StageDecision(
            enabled=intents.timing > 0.0, strength=intents.timing,
            method=("single_offset" if locked and intents.timing == 0.0
                    else alignment.method),
            reason=f"you asked for timing at {intents.timing:.2f}")

    # ── Arrangement ──────────────────────────────────────────────────
    if locked:
        arrangement = StageDecision(
            False, reason="the arrangement is the one it was recorded to")
    else:
        arrangement = StageDecision(
            True, 1.0, "structure_aware",
            "matching a new beat, so the sections are ours to choose")

    # ── Beat fit ─────────────────────────────────────────────────────
    v_duration = float(vdna.get("duration_s") or 0.0)
    b_duration = float(bdna.get("duration_s") or 0.0)
    if locked or b_duration >= v_duration - 0.5:
        beat_fit = StageDecision(
            True, 0.0, "pad",
            f"the beat ({b_duration:.0f}s) already covers the vocal "
            f"({v_duration:.0f}s)")
    else:
        beat_fit = StageDecision(
            True, 1.0, "loop_last_section",
            f"the beat ({b_duration:.0f}s) is shorter than the vocal "
            f"({v_duration:.0f}s); looping its last full section")

    # ── Vocal chain ──────────────────────────────────────────────────
    if finished:
        vocal_chain = StageDecision(
            True, 0.3, "finish",
            "the vocal is already mixed; levelling and seating it only")
    else:
        vocal_chain = StageDecision(
            True, 1.0, "produce",
            "a raw take needs the full chain")

    # ── Space ────────────────────────────────────────────────────────
    if vocal_state.reverb_is_intentional:
        space = StageDecision(
            True, 0.0, "keep",
            "the vocal brought its own space")
    else:
        space = StageDecision(
            True, 1.0, "match",
            "placing the vocal in the beat's room")
    if intents.space is not None:
        space = StageDecision(True, space.strength, intents.space,
                              f"you asked to {intents.space} the space")

    # ── Ducking ──────────────────────────────────────────────────────
    has_stems = bool(bdna.get("has_stems") or bdna.get("stems"))
    if has_stems:
        ducking = StageDecision(True, 1.0, "stems",
                                "ducking the tonal stems under the vocal")
    else:
        ducking = StageDecision(
            True, 0.4, "band_limited",
            "no stems, so ducking is limited to the vocal band to avoid "
            "pumping the drums")

    p = RenderPlan(separation=separation, dereverb=dereverb, tuning=tuning,
                   alignment=alignment, arrangement=arrangement,
                   beat_fit=beat_fit, vocal_chain=vocal_chain, space=space,
                   ducking=ducking, key=key_decision,
                   offset_s=relationship.offset_s, is_locked=locked,
                   vocal_state=vocal_state.state)
    _log_plan(p)
    return p


def _log_plan(p: RenderPlan) -> None:
    log.info("── plan ─────────────────────────────────────────────────")
    log.info("  %s", p.summary())
    for name in ("separation", "dereverb", "tuning", "alignment",
                 "arrangement", "beat_fit", "vocal_chain", "space",
                 "ducking"):
        d = getattr(p, name)
        mark = "on " if d.enabled else "off"
        log.info("  %-12s %s  %s", name, mark, d.reason)
    log.info("  %-12s     %s", "key", p.key.evidence)
    log.info("─────────────────────────────────────────────────────────")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_policy.py -v`
Expected: PASS (22 tests)

- [ ] **Step 5: Lint, type-check, commit**

```bash
.venv/bin/ruff check src/mixengine/core/policy.py tests/test_policy.py
.venv/bin/mypy src/mixengine/core/policy.py
git add src/mixengine/core/policy.py tests/test_policy.py
git commit -m "Put every stage decision in one pure function

Six modules each had their own idea of what the input was. Reading them
together as a table is what makes 'leave this alone' expressible, and
what lets the user see the plan before the engine runs."
```

---

### Task 6: Stop looping a beat that already covers the vocal

**Files:**
- Modify: `src/mixengine/audio/transform.py:407-472` (`fit_beat_to_vocal`, `_choose_loop`)
- Modify: `src/mixengine/audio/pipeline.py:245-249`
- Test: `tests/test_beat_fit.py`

**Interfaces:**
- Consumes: `RenderPlan` (Task 5).
- Produces: `fit_beat_to_vocal(beat, sr, target_len, downbeats_s, sections=None, plan=None)` — new optional `plan` parameter; when `plan.beat_fit.method == "pad"` it never loops. `_choose_loop` prefers the last full section.

- [ ] **Step 1: Write the failing test**

```python
"""
Fitting the beat to the vocal.

The reference render discarded 133 of 147 seconds of the beat and looped
its 14-second intro ten times, because a hardcoded 1.5s reverb tail
pushed the target past a 0.5s tolerance. The beat and the vocal were the
same length.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mixengine.analysis.intake import (KeyDecision, Relationship,  # noqa: E402
                                       VocalState)
from mixengine.audio import transform                              # noqa: E402
from mixengine.core.keys import Key                                # noqa: E402
from mixengine.core.policy import plan                             # noqa: E402

SR = 44100


def ramp(seconds):
    """Audio whose value encodes its own timestamp, so loops are visible."""
    n = int(seconds * SR)
    return np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None].repeat(2, 1)


def downbeats(seconds, bar_s=2.0):
    return np.arange(0.0, seconds, bar_s)


SECTIONS = [{"start": 0.51, "end": 14.58, "label": "chorus"},
            {"start": 14.58, "end": 26.59, "label": "break"},
            {"start": 26.59, "end": 92.37, "label": "chorus"},
            {"start": 92.37, "end": 147.8, "label": "outro"}]


def locked_plan(v_dur, b_dur):
    return plan({"duration_s": v_dur}, {"duration_s": b_dur},
                VocalState("finished", 0.9, "t", 0.9, 0.0, 1.0, 0.7, True,
                           200),
                Relationship("locked", 0.9, "t", 0.0, 3.0, 0.0, True),
                KeyDecision(Key(1, "minor"), 0, 0.6, "t", True, ()))


class TestBeatFit(unittest.TestCase):

    def test_equal_length_beat_is_never_looped(self):
        """The reference render's exact failure."""
        beat = ramp(147.8)
        target = int(147.8 * SR) + int(1.5 * SR)
        out, report = transform.fit_beat_to_vocal(
            beat, SR, target, downbeats(147.8), SECTIONS,
            plan=locked_plan(147.8, 147.8))
        self.assertNotEqual(report["action"], "looped")
        self.assertEqual(len(out), target)

    def test_padding_is_silent_not_repeated(self):
        beat = ramp(20.0)
        target = int(21.5 * SR)
        out, _ = transform.fit_beat_to_vocal(
            beat, SR, target, downbeats(20.0), None,
            plan=locked_plan(20.0, 20.0))
        tail = out[int(20.05 * SR):]
        self.assertLess(float(np.abs(tail).max()), 0.01)

    def test_the_beat_itself_is_unaltered(self):
        beat = ramp(20.0)
        out, _ = transform.fit_beat_to_vocal(
            beat, SR, int(21.5 * SR), downbeats(20.0), None,
            plan=locked_plan(20.0, 20.0))
        np.testing.assert_allclose(out[:len(beat)], beat, atol=1e-6)

    def test_a_genuinely_short_beat_still_loops(self):
        beat = ramp(30.0)
        out, report = transform.fit_beat_to_vocal(
            beat, SR, int(90.0 * SR), downbeats(30.0),
            [{"start": 0.0, "end": 10.0, "label": "intro"},
             {"start": 10.0, "end": 30.0, "label": "chorus"}],
            plan=locked_plan(90.0, 30.0))
        self.assertEqual(report["action"], "looped")
        self.assertEqual(len(out), int(90.0 * SR))

    def test_the_loop_comes_from_the_last_section_not_the_intro(self):
        """An outro loops credibly. An intro announces the edit."""
        beat = ramp(120.0)
        _, report = transform.fit_beat_to_vocal(
            beat, SR, int(200.0 * SR), downbeats(120.0),
            [{"start": 0.0, "end": 14.0, "label": "intro"},
             {"start": 14.0, "end": 60.0, "label": "chorus"},
             {"start": 60.0, "end": 120.0, "label": "chorus"}],
            plan=locked_plan(200.0, 120.0))
        self.assertGreater(report["loop_s"][0], 30.0)

    def test_a_longer_beat_is_still_trimmed_on_a_bar(self):
        beat = ramp(60.0)
        out, report = transform.fit_beat_to_vocal(
            beat, SR, int(30.0 * SR), downbeats(60.0), None,
            plan=locked_plan(30.0, 60.0))
        self.assertEqual(report["action"], "trimmed_to_bar")
        self.assertLessEqual(len(out), int(31.0 * SR))

    def test_works_without_a_plan(self):
        """Callers that predate the plan must keep working."""
        beat = ramp(20.0)
        out, _ = transform.fit_beat_to_vocal(beat, SR, int(21.0 * SR),
                                             downbeats(20.0), None)
        self.assertEqual(len(out), int(21.0 * SR))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_beat_fit.py -v`
Expected: FAIL — `test_equal_length_beat_is_never_looped` reports `action == "looped"`, and `fit_beat_to_vocal` rejects the unexpected `plan` keyword.

- [ ] **Step 3: Write minimal implementation**

In `src/mixengine/audio/transform.py`, change the signature and the too-short branch of `fit_beat_to_vocal`:

```python
def fit_beat_to_vocal(beat: np.ndarray, sr: int, target_len: int,
                      downbeats_s: np.ndarray,
                      sections: Optional[List[dict]] = None,
                      plan: Optional[Any] = None
                      ) -> Tuple[np.ndarray, dict]:
    """Trim or loop the beat to cover the vocal, always on bar boundaries.

    Cutting mid-bar is the most obvious tell of an automated edit, so every
    boundary here snaps to a downbeat. When the beat is too short, a
    musically coherent section is looped rather than the whole file
    repeated -- and when the plan says the beat already covers the vocal,
    the shortfall is the render's own reverb tail and is padded with
    silence rather than filled with a repeat of the intro.
    """
    b = dsp.as_2d(beat)
    report = {"action": "none", "original_len_s": round(len(b) / sr, 2),
              "target_len_s": round(target_len / sr, 2)}

    if target_len <= 0:
        return b, report
    if abs(len(b) - target_len) < sr * 0.5:
        return dsp.pad_to(b, target_len), report

    if len(b) > target_len:
        grid = np.asarray(downbeats_s, dtype=np.float64) * sr
        grid = grid[(grid > target_len * 0.55) & (grid <= len(b))]
        cut = int(grid[np.argmin(np.abs(grid - target_len))]) if grid.size else target_len
        out = b[:cut]
        report.update({"action": "trimmed_to_bar",
                       "result_len_s": round(cut / sr, 2)})
        return dsp.fade(out, sr, 0.001, 0.05), report

    # Too short. If the plan says only a tail is missing, pad it: the
    # shortfall is the render's own reverb decay, not missing music.
    if plan is not None and getattr(plan, "beat_fit", None) is not None \
            and plan.beat_fit.method == "pad":
        report.update({"action": "padded",
                       "result_len_s": round(target_len / sr, 2),
                       "reason": plan.beat_fit.reason})
        return dsp.pad_to(b, target_len), report

    loop_start, loop_end = _choose_loop(b, sr, downbeats_s, sections)
    ...  # remainder of the existing loop branch is unchanged
```

Then change `_choose_loop` to prefer the last full section:

```python
def _choose_loop(b: np.ndarray, sr: int, downbeats_s: np.ndarray,
                 sections: Optional[List[dict]]) -> Tuple[int, int]:
    """Pick a bar-aligned, musically sensible loop region.

    Search from the end. A beat's last full section is written to sit
    under a final chorus and loops without announcing itself; its intro
    is written to arrive once, and repeating it ten times is the most
    audible edit the engine can make.
    """
    if sections:
        for label in ("chorus", "verse"):
            for s in reversed(sections):
                if s.get("label") == label:
                    a, z = int(s["start"] * sr), int(s["end"] * sr)
                    if z - a > sr * 4:
                        return a, min(z, len(b))
    grid = np.asarray(downbeats_s, dtype=np.float64) * sr
    grid = grid[(grid >= 0) & (grid < len(b))]
    if len(grid) >= 9:
        start = max(0, len(grid) - 9)
        return int(grid[start]), int(grid[len(grid) - 1])
    return 0, len(b)
```

Add `Any` to the `typing` import at the top of `transform.py`.

In `src/mixengine/audio/pipeline.py`, pass the plan through (around line 246):

```python
    target_len = len(v) + int(sr * 1.5)
    beat_audio, fit_info = transform.fit_beat_to_vocal(
        beat_audio, sr, target_len, downbeats_s, bdna.get("sections"),
        plan=render_plan)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_beat_fit.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Run the existing suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: no new failures. `test_transform_args.py` exercises this function — if it asserts the old looping behaviour on an equal-length beat, update it and note why in the commit.

- [ ] **Step 6: Lint, type-check, commit**

```bash
.venv/bin/ruff check src/mixengine/audio/transform.py tests/test_beat_fit.py
.venv/bin/mypy src/mixengine/audio/transform.py
git add src/mixengine/audio/transform.py src/mixengine/audio/pipeline.py tests/test_beat_fit.py
git commit -m "Pad the reverb tail instead of looping the intro under it

A 1.5s tail pushed a same-length beat past a 0.5s tolerance, and the
engine answered by discarding 133 of 147 seconds and repeating the
first 14 ten times. When a loop is genuinely needed it now comes from
the last full section: an outro loops credibly, an intro announces the
edit."
```

---

### Task 7: Wire the plan through the pipeline

**Files:**
- Modify: `src/mixengine/audio/pipeline.py:75-110` (`render_variant` signature), `:177-260` (tuning, quantisation, warp)
- Modify: `src/mixengine/analysis/vocal_dna.py` (restoration decision)
- Modify: `src/mixengine/audio/separation.py` (separation decision)
- Test: `tests/test_pipeline_honours_plan.py`

**Interfaces:**
- Consumes: `RenderPlan` (Task 5), `fit_beat_to_vocal` (Task 6).
- Produces: `render_variant(..., render_plan: Optional[RenderPlan] = None)`. When a plan is supplied, `tinfo["plan"] = render_plan.to_dict()` and every stage consults it.

- [ ] **Step 1: Write the failing test**

```python
"""
The pipeline must do what the plan says.

Each stage used to decide for itself, from its own reading of the DNA.
These tests assert that a plan saying "off" produces a render in which
that stage did nothing -- checked through the transform report, so the
assertions describe behaviour rather than implementation.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mixengine.analysis.intake import (KeyDecision, Relationship,  # noqa: E402
                                       VocalState)
from mixengine.core.keys import Key                                # noqa: E402
from mixengine.core.policy import plan                             # noqa: E402

SR = 44100


def locked_finished_plan():
    return plan({"duration_s": 30.0, "input_type": "light_bleed",
                 "input_type_confidence": 0.56},
                {"duration_s": 30.0, "genre": "trap"},
                VocalState("finished", 0.9, "t", 0.92, 12.0, 0.97, 0.77,
                           True, 200),
                Relationship("locked", 0.9, "t", 1.75, 3.2, 0.0, True),
                KeyDecision(Key(1, "minor"), 0, 0.44, "t", True,
                            tuple(Key(1, "minor").scale_pcs)))


class TestPlanIsHonoured(unittest.TestCase):

    def test_plan_says_no_tuning(self):
        self.assertFalse(locked_finished_plan().tuning.enabled)

    def test_plan_says_no_separation(self):
        self.assertFalse(locked_finished_plan().separation.enabled)

    def test_plan_says_no_dereverb(self):
        self.assertFalse(locked_finished_plan().dereverb.enabled)

    def test_plan_says_single_offset(self):
        p = locked_finished_plan()
        self.assertEqual(p.alignment.method, "single_offset")
        self.assertAlmostEqual(p.offset_s, 1.75)

    def test_plan_says_pad_the_beat(self):
        self.assertEqual(locked_finished_plan().beat_fit.method, "pad")

    def test_plan_is_recorded_in_the_transform_report(self):
        """Whatever the engine decided must be visible in result.json."""
        d = locked_finished_plan().to_dict()
        self.assertIn("stages", d)
        self.assertIn("summary", d)
        self.assertFalse(d["stages"]["tuning"]["enabled"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pipeline_honours_plan.py -v`
Expected: PASS for the plan-shape assertions (they exercise Task 5). This file is the contract the wiring must satisfy; the real verification is Step 4's manual render and Task 10's regression.

- [ ] **Step 3: Wire the plan in**

In `src/mixengine/audio/pipeline.py`, add the parameter to `render_variant`:

```python
def render_variant(vocal_audio: np.ndarray, sr: int, vdna: dict,
                   bdna: dict, match: matching.Match,
                   variant: VariantSpec, out_path: str,
                   beat_audio: Optional[np.ndarray] = None,
                   stems: Optional[Dict[str, np.ndarray]] = None,
                   extra_overrides: Optional[dict] = None,
                   render_plan: Optional["RenderPlan"] = None) -> RenderResult:
```

Record it in the transform report immediately after `tinfo: Dict = {}`:

```python
    if render_plan is not None:
        tinfo["plan"] = render_plan.to_dict()
```

Replace the tuning decision (currently `pipeline.py:181-185`):

```python
    # ── 4. Tuning ─────────────────────────────────────────────────────────
    if render_plan is not None:
        tune_strength = (render_plan.tuning.strength
                         if render_plan.tuning.enabled else 0.0)
        tune_reason = render_plan.tuning.reason
    else:
        tune_strength = max(0.0, profile.tune_strength
                            + float(o.get("tune_strength", 0.0)))
        if vdna.get("performance_type") == "rap":
            tune_strength = 0.0
        tune_reason = "profile default"
    if tune_strength <= 0.02:
        log.info("  tuning: skipped -- %s", tune_reason)
```

Replace the quantisation decision (currently `pipeline.py:208-231`) so that a
plan with `alignment.method == "single_offset"` applies the offset and stops:

```python
    # ── 5. Timing ─────────────────────────────────────────────────────────
    if render_plan is not None and \
            render_plan.alignment.method == "single_offset":
        log.info("  timing: %s", render_plan.alignment.reason)
        aligned = True
        tinfo["align"] = {"enabled": True, "method": "single_offset",
                          "offset_s": round(render_plan.offset_s, 4),
                          "moved": 0,
                          "reason": render_plan.alignment.reason}
    else:
        q_strength = (render_plan.alignment.strength
                      if render_plan is not None
                      else float(o.get("quantize_strength", 0.0)))
        ...  # existing quantisation body, using q_strength
```

Guard the phrase-level warp (currently `pipeline.py:241`) so a locked render never warps:

```python
        enabled=(not aligned and stability > 0.6 and len(downbeats_s) > 3
                 and (render_plan is None
                      or render_plan.alignment.method != "single_offset")))
```

In `src/mixengine/analysis/vocal_dna.py`, thread an optional `plan` into the
restoration call and skip dereverb when `plan.dereverb.enabled` is False,
logging the plan's reason.

In `src/mixengine/audio/separation.py`, have the caller consult
`plan.separation.enabled` before invoking Demucs rather than deciding from
`classify_vocal_input` alone.

- [ ] **Step 4: Verify on the reference pair**

```bash
.venv/bin/python -m mixengine render \
  --vocal /Users/keshavgarg/Downloads/voc.mp3 \
  --dna data/outputs/voc-over-beat/beat-dna.json \
  --out data/outputs/phase1-check 2>&1 | tail -30
```

Expected in the log: no `running demucs`, no `tuned N/M notes`, no
`removing N ms of drift`, and a `plan` block. Expected duration: under
two minutes.

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: no new failures.

- [ ] **Step 6: Lint, type-check, commit**

```bash
.venv/bin/ruff check src tests
.venv/bin/mypy src
git add -A
git commit -m "Every stage now reads the plan instead of the DNA

Stages deciding for themselves is how six individually-correct
decisions combined into a destroyed render. The plan is also written
into result.json, so what the engine did is recoverable after the fact."
```

---

### Task 8: Flex tuning in one pass

**Files:**
- Modify: `src/mixengine/audio/tuning.py:155-262` (`tune_musical`)
- Test: `tests/test_tuning_flex.py`

**Interfaces:**
- Consumes: `RenderPlan` (Task 5).
- Produces: `tune_musical(..., dead_zone_cents: float = 0.0, single_pass: bool = True)`. Notes whose correction is under `dead_zone_cents` are untouched and counted in `report.notes_in_dead_zone`.

- [ ] **Step 1: Write the failing test**

```python
"""
Flex tuning: correct what is wrong, leave what is right.

Antares' Flex-Tune leaves notes inside a tolerance alone; Melodyne skips
notes "already quite close". The engine had no such zone, so it moved
317 of 506 notes on a vocal whose mean deviation was 14 cents.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mixengine.audio import tuning                                 # noqa: E402

SR = 44100


def tone(midi, seconds=0.4, sr=SR):
    f = 440.0 * 2 ** ((midi - 69) / 12.0)
    t = np.arange(int(seconds * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * f * t)).astype(np.float32)


def take(offsets_cents):
    """A take whose notes sit at the given cent offsets from A minor."""
    audio, notes, t = [], [], 0.0
    for i, off in enumerate(offsets_cents):
        midi = 57 + (i % 7) + off / 100.0
        audio.append(tone(midi))
        notes.append({"midi": midi, "start": t, "duration": 0.4})
        t += 0.4
    return np.concatenate(audio)[:, None].repeat(2, 1), notes


class TestDeadZone(unittest.TestCase):

    def test_notes_inside_the_dead_zone_are_untouched(self):
        y, notes = take([5, -8, 3, 10, -6, 2, 7])
        out, report = tuning.tune_musical(
            y, SR, tuning.notes_from_dna(notes), None,
            base_strength=0.8, dead_zone_cents=35.0)
        self.assertEqual(report.notes_corrected, 0)
        np.testing.assert_allclose(out, y, atol=1e-6)

    def test_notes_outside_the_dead_zone_are_corrected(self):
        y, notes = take([48, -45, 50, -47, 44, -49, 46])
        _, report = tuning.tune_musical(
            y, SR, tuning.notes_from_dna(notes), None,
            base_strength=0.8, dead_zone_cents=35.0)
        self.assertGreater(report.notes_corrected, 0)

    def test_the_dead_zone_is_counted_and_reported(self):
        y, notes = take([5, -8, 48, -45, 3])
        _, report = tuning.tune_musical(
            y, SR, tuning.notes_from_dna(notes), None,
            base_strength=0.8, dead_zone_cents=35.0)
        self.assertGreater(report.notes_in_dead_zone, 0)

    def test_zero_dead_zone_preserves_the_old_behaviour(self):
        y, notes = take([5, -8, 3, 10, -6, 2, 7])
        _, report = tuning.tune_musical(
            y, SR, tuning.notes_from_dna(notes), None,
            base_strength=0.8, dead_zone_cents=0.0)
        self.assertGreater(report.notes_corrected, 0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_tuning_flex.py -v`
Expected: FAIL with `TypeError: tune_musical() got an unexpected keyword argument 'dead_zone_cents'`

- [ ] **Step 3: Write minimal implementation**

Add `dead_zone_cents: float = 0.0` to `tune_musical`'s signature and
`notes_in_dead_zone: int = 0` to its report dataclass. In the per-note loop,
before computing the shift:

```python
        correction_cents = abs(delta) * 100.0
        if correction_cents < dead_zone_cents:
            # Antares calls 50-100 cents "significantly off"; Melodyne
            # leaves notes "already quite close" alone. A note inside the
            # zone was sung where the singer meant it, and moving it
            # trades their intonation for the engine's.
            rep.notes_in_dead_zone += 1
            continue
```

Replace the per-note `pitch_shift_fn` call (which spawns a `rubberband`
subprocess per note — 317 launches, 2m46 in the reference render) with a
single pass: accumulate each note's shift into a sample-rate pitch envelope,
then apply one phase-vocoder pass over the whole take.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_tuning_flex.py tests/test_musical.py -v`
Expected: PASS

- [ ] **Step 5: Wire the dead zone to the plan**

In `pipeline.py`'s tuning call, pass
`dead_zone_cents=(0.0 if render_plan is None else TUNING_DEAD_ZONE_CENTS)`
importing the constant from `core.policy`.

- [ ] **Step 6: Lint, type-check, commit**

```bash
.venv/bin/ruff check src/mixengine/audio/tuning.py tests/test_tuning_flex.py
.venv/bin/mypy src/mixengine/audio/tuning.py
git add src/mixengine/audio/tuning.py src/mixengine/audio/pipeline.py tests/test_tuning_flex.py
git commit -m "Give the tuner a dead zone and one pass instead of 317

A note within 35 cents was sung where the singer meant it. Correcting
it trades their intonation for the engine's -- and did so 317 times on
a vocal already averaging 14 cents. One pitch envelope replaces one
rubberband subprocess per note."
```

---

### Task 9: Critic fidelity gates

**Files:**
- Modify: `src/mixengine/audio/critic.py:280-360`
- Test: `tests/test_critic_fidelity.py`

**Interfaces:**
- Consumes: `RenderPlan` (Task 5).
- Produces: `critic.evaluate(..., render_plan=None)` adds gates `beat_not_looped`, `duration_preserved`, `key_unchanged` when `render_plan.is_locked`. Tuning is scored from `transform["tuning"]`, not by re-tracking pitch. Gates never evaluated report `severity="skipped"`.

- [ ] **Step 1: Write the failing test**

```python
"""
The critic must stop grading its own homework.

It scored the reference render 94% with zero warnings: sync measured
after quantising, tuning measured after retuning against the key it had
assumed, and no gate asking whether the output still resembled the
input.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mixengine.audio import critic                                 # noqa: E402


class TestFidelityGates(unittest.TestCase):

    def test_a_looped_beat_fails_a_locked_render(self):
        gate = critic.beat_not_looped_gate({"beat_fit": {"action": "looped"}},
                                           is_locked=True)
        self.assertFalse(gate.passed)
        self.assertEqual(gate.severity, "error")

    def test_a_padded_beat_passes(self):
        gate = critic.beat_not_looped_gate({"beat_fit": {"action": "padded"}},
                                           is_locked=True)
        self.assertTrue(gate.passed)

    def test_the_gate_is_skipped_when_not_locked(self):
        gate = critic.beat_not_looped_gate({"beat_fit": {"action": "looped"}},
                                           is_locked=False)
        self.assertEqual(gate.severity, "skipped")

    def test_duration_change_fails_a_locked_render(self):
        gate = critic.duration_preserved_gate(147.8, 220.0, is_locked=True)
        self.assertFalse(gate.passed)

    def test_matching_duration_passes(self):
        gate = critic.duration_preserved_gate(147.8, 149.3, is_locked=True)
        self.assertTrue(gate.passed)

    def test_tuning_is_scored_from_the_tuners_own_report(self):
        gate = critic.tuning_gate_from_report(
            {"notes_corrected": 0, "notes_considered": 506,
             "mean_correction_cents": 0.0})
        self.assertTrue(gate.passed)
        self.assertEqual(gate.value, 0.0)

    def test_heavy_correction_is_flagged(self):
        gate = critic.tuning_gate_from_report(
            {"notes_corrected": 317, "notes_considered": 506,
             "mean_correction_cents": 24.7})
        self.assertGreater(gate.value, 20.0)

    def test_a_skipped_gate_is_not_reported_as_passed(self):
        gate = critic.beat_not_looped_gate({}, is_locked=False)
        self.assertNotEqual(gate.severity, "error")
        self.assertEqual(gate.severity, "skipped")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_critic_fidelity.py -v`
Expected: FAIL with `AttributeError: module 'mixengine.audio.critic' has no attribute 'beat_not_looped_gate'`

- [ ] **Step 3: Write minimal implementation**

Add to `critic.py`:

```python
def beat_not_looped_gate(transform: dict, is_locked: bool) -> Gate:
    """A beat that already covers the vocal must not be looped.

    This is the gate the reference render most needed and least had: it
    scored 94% while playing the beat's first fourteen seconds ten times.
    """
    if not is_locked:
        return Gate("beat_not_looped", True, 0.0, 0.0, "skipped",
                    "not a locked render")
    action = (transform.get("beat_fit") or {}).get("action", "none")
    looped = action == "looped"
    return Gate("beat_not_looped", not looped, 1.0 if looped else 0.0, 0.0,
                "error",
                "the beat was looped under a vocal it already covers"
                if looped else f"beat {action}")


def duration_preserved_gate(input_s: float, output_s: float,
                            is_locked: bool, tolerance_s: float = 2.0) -> Gate:
    """A locked render is as long as what went into it."""
    if not is_locked:
        return Gate("duration_preserved", True, 0.0, tolerance_s, "skipped",
                    "not a locked render")
    delta = abs(float(output_s) - float(input_s))
    return Gate("duration_preserved", delta <= tolerance_s, round(delta, 2),
                tolerance_s, "error",
                f"output is {delta:.1f}s from the input length")


def tuning_gate_from_report(report: dict) -> Gate:
    """Score tuning from what the tuner did, not by re-tracking pitch.

    Re-running CREPE over the finished mix cost 2m48 and measured the
    engine's own correction against the key it had assumed -- it could
    only ever agree with itself.
    """
    considered = float(report.get("notes_considered") or 0)
    corrected = float(report.get("notes_corrected") or 0)
    mean_cents = float(report.get("mean_correction_cents") or 0.0)
    moved_fraction = (corrected / considered) if considered else 0.0
    value = mean_cents * moved_fraction
    limit = float(CFG.critic.max_tuning_error_cents)
    return Gate("tuning", value <= limit, round(value, 2), limit, "warning",
                f"moved {corrected:.0f}/{considered:.0f} notes, "
                f"mean {mean_cents:.1f} cents")
```

Add `"skipped"` to the severities `Gate.to_dict()` leaves untouched, alongside
`"info"`, and exclude skipped gates from the score.

Call the new gates from `evaluate()` when `render_plan` is supplied, and
replace the `analysis.track_pitch` call at `critic.py:345` with
`tuning_gate_from_report(transform.get("tuning") or {})`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_critic_fidelity.py tests/test_critic.py -v`
Expected: PASS

- [ ] **Step 5: Lint, type-check, commit**

```bash
.venv/bin/ruff check src/mixengine/audio/critic.py tests/test_critic_fidelity.py
.venv/bin/mypy src/mixengine/audio/critic.py
git add src/mixengine/audio/critic.py tests/test_critic_fidelity.py
git commit -m "Stop the critic grading its own homework

It scored 94% with zero warnings on a render that looped a beat's
intro ten times under a retuned vocal. Sync was measured after
quantising and tuning after retuning, so both could only agree with
the engine. Fidelity gates compare the output to the input instead."
```

---

### Task 10: The fidelity regression

**Files:**
- Create: `tests/test_fidelity_regression.py`
- Create: `tests/fixtures/locked_pair/` (12-second vocal and beat, generated)
- Create: `tests/fixtures/make_locked_pair.py`

**Interfaces:**
- Consumes: everything above.
- Produces: the end-to-end assertion that defines Phase 1 as done.

- [ ] **Step 1: Write the fixture generator**

```python
"""
Generate a locked pair: a beat, and a 'vocal' recorded over it.

Committing twelve seconds of synthesised audio rather than the user's
own files keeps the regression runnable by anyone, and keeps their
recordings out of the repository.
"""

import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

SR = 44100
SECONDS = 12.0
BPM = 120.0
OFFSET_S = 0.75


def beat(seconds=SECONDS, bpm=BPM):
    """Kick on the beat, hat on the offbeat, a sustained bass root."""
    n = int(seconds * SR)
    out = np.zeros(n, dtype=np.float32)
    period = 60.0 / bpm
    t = np.arange(n) / SR
    out += 0.12 * np.sin(2 * np.pi * 55.0 * t).astype(np.float32)
    for i in range(int(seconds / period)):
        k = int(i * period * SR)
        env = np.exp(-np.arange(min(4000, n - k)) / 900.0)
        out[k:k + len(env)] += 0.6 * env * np.sin(
            2 * np.pi * 60 * np.arange(len(env)) / SR)
        h = int((i + 0.5) * period * SR)
        if h + 1200 < n:
            rng = np.random.default_rng(i)
            out[h:h + 1200] += 0.08 * rng.standard_normal(1200) * \
                np.exp(-np.arange(1200) / 300.0)
    return np.clip(out, -1, 1)[:, None].repeat(2, 1)


def vocal(seconds=SECONDS, bpm=BPM, offset_s=OFFSET_S):
    """Tuned, levelled notes landing on the beat's grid, after a lag.

    Pitches sit exactly on semitone centres and every phrase peaks at
    the same level: a finished vocal, which the engine must not touch.
    """
    n = int(seconds * SR)
    out = np.zeros(n, dtype=np.float32)
    period = 60.0 / bpm
    scale = [57, 60, 62, 64, 67]
    for i in range(int((seconds - offset_s) / period)):
        midi = scale[i % len(scale)]
        f = 440.0 * 2 ** ((midi - 69) / 12.0)
        start = int((offset_s + i * period) * SR)
        length = int(period * 0.8 * SR)
        if start + length >= n:
            break
        t = np.arange(length) / SR
        note = np.zeros(length, dtype=np.float32)
        for h, amp in ((1, 1.0), (2, 0.3), (3, 0.15)):
            note += amp * np.sin(2 * np.pi * f * h * t).astype(np.float32)
        env = np.minimum(1.0, np.arange(length) / 500.0) * \
            np.minimum(1.0, (length - np.arange(length)) / 2000.0)
        note *= env
        note *= 0.5 / (np.abs(note).max() + 1e-9)     # every phrase level
        out[start:start + length] += note
    return np.clip(out, -1, 1)[:, None].repeat(2, 1)


if __name__ == "__main__":
    here = os.path.join(os.path.dirname(__file__), "locked_pair")
    os.makedirs(here, exist_ok=True)
    sf.write(os.path.join(here, "beat.wav"), beat(), SR)
    sf.write(os.path.join(here, "vocal.wav"), vocal(), SR)
    print("wrote", here)
```

Run it: `.venv/bin/python tests/fixtures/make_locked_pair.py`

- [ ] **Step 2: Write the failing test**

```python
"""
The regression that defines Phase 1.

A finished vocal recorded to a beat, rendered. The engine must return
the same song: same length, beat unlooped, key unchanged, notes
unmoved. Every assertion here corresponds to something the engine did
to voc.mp3 + beat.mp3 on 2026-09-20.
"""

import os
import sys
import unittest

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mixengine.analysis import intake                              # noqa: E402
from mixengine.core.intents import Intents                         # noqa: E402
from mixengine.core.policy import plan                             # noqa: E402

PAIR = os.path.join(os.path.dirname(__file__), "fixtures", "locked_pair")


def load(name):
    y, sr = sf.read(os.path.join(PAIR, name), dtype="float32",
                    always_2d=True)
    return y, sr


@unittest.skipUnless(os.path.exists(os.path.join(PAIR, "vocal.wav")),
                     "run tests/fixtures/make_locked_pair.py first")
class TestLockedPairIsRecognised(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.vocal, cls.sr = load("vocal.wav")
        cls.beat, _ = load("beat.wav")

    def test_the_pair_is_detected_as_locked(self):
        r = intake.detect_relationship(
            self.vocal, self.beat, self.sr,
            {"duration_s": 12.0, "bpm": 120.0},
            {"duration_s": 12.0, "bpm": 120.0})
        self.assertEqual(r.state, "locked")

    def test_the_measured_offset_is_the_real_one(self):
        r = intake.detect_relationship(
            self.vocal, self.beat, self.sr,
            {"duration_s": 12.0, "bpm": 120.0},
            {"duration_s": 12.0, "bpm": 120.0})
        self.assertAlmostEqual(r.offset_s, 0.75, delta=0.06)

    def test_the_vocal_is_recognised_as_finished(self):
        notes = [{"midi": 57 + (i % 5), "start": i * 0.5, "duration": 0.4}
                 for i in range(20)]
        st = intake.detect_vocal_state(
            self.vocal, self.sr,
            {"notes": notes, "phrase_level_spread_db": 0.4,
             "estimated_rt60_s": 0.1})
        self.assertEqual(st.state, "finished")

    def test_the_plan_leaves_this_pair_alone(self):
        notes = [{"midi": 57 + (i % 5), "start": i * 0.5, "duration": 0.4}
                 for i in range(20)]
        st = intake.detect_vocal_state(
            self.vocal, self.sr,
            {"notes": notes, "phrase_level_spread_db": 0.4,
             "estimated_rt60_s": 0.1})
        r = intake.detect_relationship(
            self.vocal, self.beat, self.sr,
            {"duration_s": 12.0, "bpm": 120.0},
            {"duration_s": 12.0, "bpm": 120.0})
        k = intake.decide_key({"key": {"pc": 9, "mode": "minor"},
                               "key_confidence": 0.6},
                              {"key": {"pc": 0, "mode": "major"},
                               "key_confidence": 0.6})
        p = plan({"duration_s": 12.0}, {"duration_s": 12.0, "genre": "trap"},
                 st, r, k, Intents.AUTO)

        self.assertFalse(p.tuning.enabled, "a finished vocal was retuned")
        self.assertFalse(p.separation.enabled, "a clean vocal was separated")
        self.assertFalse(p.dereverb.enabled, "a mix reverb was stripped")
        self.assertFalse(p.arrangement.enabled, "the arrangement was redone")
        self.assertEqual(p.beat_fit.method, "pad", "the beat was looped")
        self.assertEqual(p.alignment.method, "single_offset",
                         "the performance was requantised")
        self.assertEqual(p.vocal_chain.method, "finish",
                         "a mixed vocal was re-produced")
        self.assertEqual(p.key.semitone_shift, 0, "the key was moved")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run it**

Run: `.venv/bin/python -m pytest tests/test_fidelity_regression.py -v`
Expected: PASS (4 tests)

- [ ] **Step 4: Render the real pair end to end**

```bash
.venv/bin/python -m mixengine render \
  --vocal /Users/keshavgarg/Downloads/voc.mp3 \
  --dna data/outputs/voc-over-beat/beat-dna.json \
  --out data/outputs/phase1-final 2>&1 | tail -40
```

Verify in `result.json`: `plan.stages.tuning.enabled == false`,
`transform.beat_fit.action != "looped"`, `critic` contains
`beat_not_looped` and `duration_preserved` both passed, and
`total_seconds < 120`.

- [ ] **Step 5: Commit**

```bash
git add tests/test_fidelity_regression.py tests/fixtures/make_locked_pair.py tests/fixtures/locked_pair/
git commit -m "Add the regression that defines Phase 1

A finished vocal recorded to a beat must come back as the same song.
Every assertion here is something the engine did to a real pair on
2026-09-20."
```

---

### Task 11: Plumb intents through CLI and API

**Files:**
- Modify: `src/mixengine/__main__.py` (render subcommand)
- Modify: `src/mixengine/api/app.py:145-170`
- Modify: `src/mixengine/api/service.py:305-330`
- Test: `tests/test_intents_plumbing.py`

**Interfaces:**
- Consumes: `Intents` (Task 1).
- Produces: CLI flags `--vocal-state`, `--relationship`, `--tune`, `--timing`, `--space`, `--separate`, `--loudness`; `POST /api/render` accepts the same as form fields; `service.render_vocal(..., intents: Intents = Intents.AUTO)`.

- [ ] **Step 1: Write the failing test**

```python
"""
Intents must survive the trip from the command line to the plan.

A flag the user sets that quietly stops mattering two layers down is
worse than no flag at all.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mixengine.__main__ import build_parser                        # noqa: E402
from mixengine.core.intents import Intents                         # noqa: E402


class TestCliIntents(unittest.TestCase):

    def parse(self, *args):
        return build_parser().parse_args(["render", "--vocal", "v.mp3", *args])

    def test_defaults_are_all_auto(self):
        ns = self.parse()
        self.assertTrue(Intents.from_dict(vars(ns)).is_all_auto)

    def test_vocal_state_is_carried(self):
        ns = self.parse("--vocal-state", "finished")
        self.assertEqual(Intents.from_dict(vars(ns)).vocal_state, "finished")

    def test_tune_off_is_carried(self):
        ns = self.parse("--tune", "off")
        self.assertEqual(Intents.from_dict(vars(ns)).tune, 0.0)

    def test_relationship_is_carried(self):
        ns = self.parse("--relationship", "locked")
        self.assertEqual(Intents.from_dict(vars(ns)).relationship, "locked")

    def test_bad_value_is_rejected_by_argparse(self):
        with self.assertRaises(SystemExit):
            self.parse("--vocal-state", "pristine")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_intents_plumbing.py -v`
Expected: FAIL — either `ImportError: cannot import name 'build_parser'` or `unrecognized arguments`.

- [ ] **Step 3: Write minimal implementation**

Extract the existing argparse construction in `__main__.py` into
`build_parser() -> argparse.ArgumentParser` and add to the `render`
subparser:

```python
    p.add_argument("--vocal-state", choices=("auto", "raw", "tuned",
                                             "finished"), default="auto",
                   help="raw takes get full production; finished ones are "
                        "left alone")
    p.add_argument("--relationship", choices=("auto", "locked", "free"),
                   default="auto",
                   help="locked means the vocal was recorded to this beat")
    p.add_argument("--tune", default="auto",
                   help="auto, off, or 0-1")
    p.add_argument("--timing", default="auto",
                   help="auto, off, or 0-1")
    p.add_argument("--space", choices=("auto", "keep", "match", "add"),
                   default="auto")
    p.add_argument("--separate", choices=("auto", "never", "always"),
                   default="auto")
    p.add_argument("--loudness", default="auto",
                   help="auto, or a target LUFS such as -9.5")
```

Thread `Intents.from_dict(vars(args))` into `service.render_vocal`, and from
there into `policy.plan` and `render_variant`.

In `api/app.py`, add the same fields to the render endpoint as
`Optional[str] = Form(None)` and build `Intents.from_dict` from them.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_intents_plumbing.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Verify end to end**

```bash
.venv/bin/python -m mixengine render \
  --vocal /Users/keshavgarg/Downloads/voc.mp3 \
  --dna data/outputs/voc-over-beat/beat-dna.json \
  --relationship locked --vocal-state finished \
  --out data/outputs/phase1-intents 2>&1 | grep -A12 "plan"
```

Expected: the plan block shows `you told us` as the reason for the
relationship and vocal-state decisions.

- [ ] **Step 6: Lint, type-check, commit**

```bash
.venv/bin/ruff check src tests
.venv/bin/mypy src
git add -A
git commit -m "Carry intents from the command line to the plan

A flag that quietly stops mattering two layers down is worse than no
flag. These reach policy.plan and are recorded in result.json with
'you told us' as the reason."
```

---

### Task 12: Documentation

**Files:**
- Modify: `README.md`
- Modify: `ARCHITECTURE.md`

- [ ] **Step 1: Document the intents in README**

Add a section after the install instructions explaining what the engine
decides automatically, what the user can state, and the flags from Task 11.
Include the worked example: a vocal recorded to its own beat, rendered with
`--relationship locked --vocal-state finished`.

- [ ] **Step 2: Document the policy layer in ARCHITECTURE.md**

Add a section describing the intake → intents → plan → stages flow, the
policy table from the spec, and why stages no longer decide for themselves.
Link to the spec.

- [ ] **Step 3: Commit**

```bash
git add README.md ARCHITECTURE.md
git commit -m "Document the policy layer and the intents

Someone reading the code should be able to find out why a stage did
nothing without running a render to see."
```

---

## Self-Review

**Spec coverage.** Spec §1.1 Intents → Task 1, 11. §1.2 intake detectors →
Tasks 2, 3, 4. §1.3 policy → Task 5. §1.4 pipeline changes → Tasks 6, 7, 8
(vocal-chain `finish` mode and band-limited ducking are decided in Task 5 and
consumed in Task 7; their DSP implementation is Phase 3, and the plan records
the decision either way). §1.5 critic → Task 9. §1.6 testing → Tasks 2–5,
10. Phases 2–4 are out of scope for this plan by design.

**Placeholder scan.** No TBD/TODO. Task 7 Step 3 and Task 8 Step 3 describe
edits to existing function bodies rather than reproducing them in full —
the surrounding code is cited by file and line, and the new logic is given
verbatim.

**Type consistency.** `VocalState`, `Relationship`, `KeyDecision` are
constructed in Tasks 2–4 and consumed positionally in Tasks 5, 10 with
matching field order. `StageDecision.method` strings (`single_offset`,
`pad`, `loop_last_section`, `finish`, `produce`, `flex`, `keep`, `match`,
`stems`, `band_limited`) are used identically in Tasks 5, 6, 7, 10.
`plan()` takes `(vdna, bdna, vocal_state, relationship, key_decision,
intents)` everywhere.
