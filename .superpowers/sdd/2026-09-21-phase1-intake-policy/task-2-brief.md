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

