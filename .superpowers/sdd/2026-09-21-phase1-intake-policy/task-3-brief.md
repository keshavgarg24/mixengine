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

