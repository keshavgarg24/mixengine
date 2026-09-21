# Task 2 report: vocal state detection

## What was implemented

- `src/mixengine/analysis/intake.py` (new) — module docstring, constants
  (`TUNED_WINDOW_CENTS`, `TUNED_FRACTION_RAW`, `TUNED_FRACTION_SURE`,
  `SPREAD_MIXED_DB`, `SPREAD_DYNAMIC_DB`, `RT60_NOTABLE_S`,
  `MIN_NOTES_FOR_CONFIDENCE`), the frozen `VocalState` dataclass
  (`state, confidence, evidence, tuned_fraction, crest_db,
  phrase_spread_db, rt60_s, reverb_is_intentional, n_notes` — this exact
  field order, since later tasks construct it positionally), and
  `detect_vocal_state(y, sr, vdna, intents=Intents.AUTO) -> VocalState`,
  plus helpers `_tuned_fraction`, `_crest_db`, `_state_confidence`.
- `tests/test_intake_vocal_state.py` — already present (untracked) at the
  start of this task, byte-identical to the brief's Step 1 code; verified
  with a diff, not modified.

Implementation is the brief's Step 3 code verbatim, with two changes:
removed `field` from the `dataclasses` import (per controller note — never
used in the brief's code) and removed `Optional` from the `typing` import
(also never used in the brief's code; ruff's F401 caught it — see below).
No other line differs from the brief; confirmed with a scripted diff
against the brief's fenced code blocks.

## Test results

Command: `.venv/bin/python -m pytest tests/test_intake_vocal_state.py -v`

- Before implementation (Step 2): 1 collection error —
  `ModuleNotFoundError: No module named 'mixengine.analysis.intake'`, as
  expected.
- After implementation (Step 4): `9 passed in 3.60s`, all 9 tests green
  on the first run of the real implementation (see below — no threshold
  retuning was needed).

## Ruff / mypy

- `.venv/bin/ruff check src/mixengine/analysis/intake.py tests/test_intake_vocal_state.py`
  — first run flagged `F401 typing.Optional imported but unused`. Removed
  the import (it was never referenced in the brief's code — none of the
  dataclass fields or function signatures use `Optional`). Re-run:
  `All checks passed!`
- `.venv/bin/mypy src/mixengine/analysis/intake.py` — `Success: no issues
  found in 1 source file`.

## Threshold / fixture conflicts

None. I hand-traced all 9 fixtures against the brief's constants before
running anything, then confirmed empirically — every test passed on the
first execution of the real implementation. Worth recording the one
fixture that looks like it could be a near-miss but isn't:

- `test_notes_spread_across_the_semitone_read_as_raw` uses offsets
  `[40, -35, 22, -48, 31, -27, 44, -19]`. Seven of the eight are outside
  `TUNED_WINDOW_CENTS` (20¢), but `-19` maps to 19¢ — just inside the
  window. That yields `tuned_fraction = 4/32 = 0.125` (one of eight
  offsets repeated 4x), comfortably under both the test's `< 0.5`
  assertion and the `TUNED_FRACTION_RAW` (0.55) fence, so the take still
  reads as `raw`. No adjustment needed, but it's a fixture that would
  flip to a higher fraction with only one or two more near-boundary
  offsets — worth knowing if Task 3/4 add cases near this fixture.

No constants were changed from the brief's stated values.

## Self-review findings

- Diffed `tests/test_intake_vocal_state.py` against the brief's Step 1
  block programmatically: identical.
- Diffed `src/mixengine/analysis/intake.py` against the brief's Step 3
  block programmatically: only the two import-line removals noted above;
  every constant, field name, docstring, and code path matches verbatim.
- Confirmed no other tracked file was touched (`git status --short` shows
  only the two new files; `git diff` against `core/intents.py` is empty —
  Task 1's file was not modified).
- Checked `VocalState`/`class ...State` names don't collide with anything
  else in `src/` or `tests/` — no collisions.
- Checked the field names the brief's `detect_vocal_state` reads from
  `vdna` (`notes`, `midi`, `duration`, `phrase_level_spread_db`,
  `tuning_deviation_cents`, `estimated_rt60_s`) against what
  `src/mixengine/analysis/vocal_dna.py` actually produces: `notes`,
  `midi`, `duration`, `phrase_level_spread_db`, and
  `tuning_deviation_cents` all match. `estimated_rt60_s` is not currently
  emitted by `vocal_dna.py` (grepped, no match) — `detect_vocal_state`
  degrades gracefully via `.get(...) or 0.0`, so this task's tests (which
  supply `vdna` directly) are unaffected, but real callers won't get RT60
  evidence until something populates that key. Flagging as a note, not a
  defect of this task — Task 2's job was the detector, not the DNA
  extractor, and the brief's interface takes `vdna: dict` generically.

No bugs found; no further changes made beyond the two import removals.

## Commit

Commit `33ac06c` on `rebuild/intake-policy-frontend` (2 files changed,
300 insertions, working tree clean afterward):

```
commit 33ac06c
Detect whether a vocal is raw, tuned, or finished

The engine retuned 317 of 506 notes that already sat within 14 cents
of the grid, and compressed a vocal whose phrases varied by 0.97 dB.
Both facts were in the DNA. Nothing read them.
```

## Concerns

- `estimated_rt60_s` is not yet produced anywhere in `vocal_dna.py` (see
  above). Not a blocker for this task, but a later task (or Task 3/4)
  will need to add it to the DNA extractor for `reverb_is_intentional`
  to mean anything on real audio rather than only in tests that hand-roll
  `vdna`.

## Fix round 1

Two review findings against `src/mixengine/analysis/intake.py`, both fixed.

### Finding 1: RT60 read from the wrong place

`detect_vocal_state` read `vdna.get("estimated_rt60_s")`, but
`vocal_dna.py:276` nests the whole `AudioQuality` payload under
`"quality"` (`"quality": quality_post.to_dict()`), and
`AudioQuality.to_dict()` is what actually carries `estimated_rt60_s`. On
real DNA documents the top-level key is absent, so `reverb_is_intentional`
was always `False` in production — only the brief's tests (which set the
key at the top level) ever saw it.

Added `_rt60_s(vdna: dict) -> float` in `intake.py`: returns the
top-level `estimated_rt60_s` when present (not `None`), otherwise reads
`vdna["quality"]["estimated_rt60_s"]` guarded by `isinstance(quality,
dict)`, defaulting to `0.0` if neither location has it.
`detect_vocal_state` now calls `rt60 = _rt60_s(vdna)` instead of reading
the top-level key directly. `vocal_dna.py` was not touched, per the
brief.

Covered by the new test `test_rt60_read_from_nested_quality_location`,
which builds a DNA document with `spread_db=0.97` and
`{"quality": {"estimated_rt60_s": 0.77}}` and no top-level RT60 key, and
asserts `reverb_is_intentional is True` and `rt60_s == 0.77`. Confirmed
red before the fix (`AssertionError: False is not true` on
`reverb_is_intentional`) and green after. All prior tests that set RT60
at the top level (e.g. `test_reverb_on_a_finished_vocal_is_intentional`,
`test_reverb_on_a_raw_vocal_is_a_room`) continue to pass unmodified, so
both read paths are proven.

### Finding 2: `_crest_db` should use the project's mono helper

`_crest_db` hand-rolled its downmix with
`mono = y if y.ndim == 1 else np.mean(y, axis=1)` instead of using the
house `dsp.to_mono` (= `dsp.as_2d(x).mean(axis=1)`, with `as_2d`
transposing channels-first input via a length heuristic).

Added `from ..audio import dsp` and replaced the downmix line with
`mono = dsp.to_mono(y)`. The `active.size < 128` guard and the rest of
the function are unchanged.

Covered by the new test `test_crest_db_handles_stereo_input`, which
builds a stereo `(n, 2)` array from two identical mono channels and
asserts `_crest_db` returns a value `> 1.0` (not the `0.0` guard return)
and matches the mono `_crest_db` result to 5 decimal places. Note for
the record: this test passed even before the fix — for a genuine
channels-last `(n, 2)` array with `n` far larger than 2, `dsp.as_2d`'s
length heuristic is a no-op, so `np.mean(y, axis=1)` and `dsp.to_mono(y)`
are numerically identical; verified this empirically before writing the
test. The finding is about consistency with the house pattern and
robustness to channels-first input (e.g. `librosa.load(..., mono=False)`
shape `(2, n)`), not a demonstrated bug for the `(n, 2)` shape itself.
The fix was still applied exactly as specified since it is the correct,
established pattern and guards against that channels-first case too.

### Test results

Command: `.venv/bin/python -m pytest tests/test_intake_vocal_state.py -v`

- Before fixes (tests added, source unchanged): `1 failed, 10 passed` —
  the failure was `test_rt60_read_from_nested_quality_location`
  (`AssertionError: False is not true`), the expected pre-fix failure for
  Finding 1. `test_crest_db_handles_stereo_input` was already green (see
  note above).
- After fixes: `11 passed in 3.37s` — all 9 original tests plus both new
  tests.

### Lint / type-check

- `.venv/bin/ruff check src/mixengine/analysis/intake.py tests/test_intake_vocal_state.py` → `All checks passed!`
- `.venv/bin/mypy src/mixengine/analysis/intake.py` → `Success: no issues found in 1 source file`

### Scope check

`VocalState`'s field order, the module constants, and the 9 original
test methods were not touched. `grep` for other importers of
`mixengine.analysis.intake` found only `tests/test_intake_vocal_state.py`
— no other code consumes this module yet, so there is no wider blast
radius to check.
