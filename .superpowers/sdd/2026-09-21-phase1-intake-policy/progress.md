# SDD ledger — plan: docs/superpowers/plans/2026-09-21-phase1-intake-policy.md

Spec: docs/superpowers/specs/2026-09-21-mixengine-rebuild-design.md (read)
Branch: rebuild/intake-policy-frontend (not main)
Baseline: d0f0fec

Ruling: work proceeds on branch `rebuild/intake-policy-frontend` rather than a
separate git worktree — the user is watching this directory and a worktree
elsewhere would hide the work from them. Branch isolation from `main` is
preserved. Cost if wrong: none material; `main` still holds the baseline.

## Pre-flight conflict scan

### Shared-file pairs

| Tasks | Shared | Produces → consumes | Finding |
|---|---|---|---|
| T2 → T3, T4 | `analysis/intake.py` | T2 creates module; T3, T4 append | Clean — sequential appends. T4 must add `Key, parse_key` to the top-level import (plan says so). |
| T5 → T6 | `core/policy.py` | `RenderPlan.beat_fit.method == "pad"` | Clean — T6's test imports `plan` from T5. |
| T6, T7, T8 | `audio/pipeline.py` | T6 passes `render_plan` to `fit_beat_to_vocal`; T7 introduces `render_plan` | **CONFLICT — see Ruling 4.** |
| T7 → T8 | `audio/pipeline.py` | T7 tuning branch; T8 passes `dead_zone_cents` | Clean — T8 edits the branch T7 created. |
| T8 → T9 | tuning report dict | T8 `notes_corrected/considered/mean_correction_cents`; T9 reads them | Clean — fields already exist in `TuningReport`. |
| T6 | `audio/transform.py` | sole writer | Clean. |
| T9 | `audio/critic.py` | sole writer | Clean. |
| T11 | `__main__.py`, `api/*` | sole writer | Clean. |

### Interface signatures (constructed in T2–T4, consumed positionally in T5, T6, T10)

| Type | Field order defined | Consumers match |
|---|---|---|
| `VocalState` | state, confidence, evidence, tuned_fraction, crest_db, phrase_spread_db, rt60_s, reverb_is_intentional, n_notes | T5, T6, T10 ✓ |
| `Relationship` | state, confidence, evidence, offset_s, peak_ratio, duration_delta_s, tempo_agrees | T5, T6, T10 ✓ |
| `KeyDecision` | key, semitone_shift, confidence, evidence, families_agree, scale_pcs | T5, T6, T10 ✓ |
| `plan()` | (vdna, bdna, vocal_state, relationship, key_decision, intents) | T6, T10 ✓ |
| `StageDecision.method` strings | single_offset, pad, loop_last_section, finish, produce, flex, keep, match, stems, band_limited | T5, T6, T7, T10 ✓ |

### Per-task self-consistency

| Task | Own text agrees with itself? |
|---|---|
| T1 | **NO — Ruling 1.** `AUTO: "Intents"` inside a `@dataclass` becomes a required field. |
| T2 | Yes. `_tuned_fraction` verified: midi 57.02 → 2 cents. `field` import is unused (ruff F401) — implementer to drop. |
| T3 | Yes. Lag sign convention is the implementer's to verify against the test. |
| T4 | **NO — Ruling 2.** Top-1-only family logic cannot produce the asserted result. |
| T5 | **NO — Ruling 3.** `_override` is dead code. |
| T6 | **NO — Ruling 4.** pipeline.py edit depends on a variable T7 introduces. |
| T7 | Step 2 expects PASS, not FAIL — it is a contract test over T5's output, not TDD. Acceptable; noted. |
| T8 | **NO — Rulings 5, 6.** Return type is a dict, and `context` is not Optional. |
| T9 | Yes. `Gate(name, passed, value, limit, severity, message)` verified against critic.py:39. |
| T10 | Yes. |
| T11 | Yes. |
| T12 | Yes — documentation only. |

## Rulings

Ruling 1 (T1): `AUTO` must be `typing.ClassVar["Intents"]`, not a bare
annotation. A bare annotation in a `@dataclass` declares a required field,
which would break `Intents()` and put `AUTO` into `asdict()` — failing the
plan's own `test_auto_has_every_field_none`. Cost if wrong: none; ClassVar is
the only construction that satisfies the stated tests.

Ruling 2 (T4): `decide_key` must reconcile across both **candidate lists**,
not top-1 keys alone, and the reference-case test must assert what is
load-bearing rather than `families_agree`. Checked by hand: C# minor
(pcs 1,3,4,6,8,9,11) and G# major (pcs 8,10,0,1,3,5,7) are genuinely
different keys, so the plan's `families_agree` assertion is unsatisfiable by
any correct implementation. But the beat's own #2 candidate was G# **minor**
at 0.669 — a dominant (fifth) relationship to the vocal's C# minor, which is
ordinary and needs no transposition. So: score every (vocal candidate, beat
candidate) pair, prefer pairs that are family-identical or fifth-adjacent,
and take the best-scoring pair. Replace
`test_reference_render_resolves_to_the_minor_family`'s `families_agree`
assertion with the two that matter — `semitone_shift == 0` and
`key.mode == "minor"`. Cost if wrong: a key decision that is musically
defensible but not the one a human would pick; caught by ear in Task 10's
end-to-end render.

Ruling 3 (T5): delete the `_override` helper. It is called once, for
`intents.separate`, and its result is immediately discarded by the explicit
`if intents.separate ==` branches below it; every other stage has its own
explicit override branch. Dead code that reviewers would flag. Cost if
wrong: none.

Ruling 4 (T6): Task 6 modifies `audio/transform.py` **only**. The
`pipeline.py` edit in its Step 3 moves to Task 7, which is where
`render_plan` enters `render_variant`'s signature. As written, Task 6 would
reference an undefined variable and break every render between T6 and T7.
T6's tests pass the plan to `fit_beat_to_vocal` directly and are unaffected.
Cost if wrong: none; this strictly removes a broken intermediate state.

Ruling 5 (T8): `tune_musical` returns `Tuple[np.ndarray, dict]` — it returns
`rep.to_dict()`. Task 8's tests must use `report["notes_corrected"]` and
`report["notes_in_dead_zone"]`, not attribute access. Cost if wrong: none;
verified against tuning.py:155-175.

Ruling 6 (T8): `tune_musical`'s `context` parameter is typed
`HarmonicContext`, not `Optional`. Task 8's tests must pass
`HarmonicContext()` rather than `None`. Cost if wrong: none; verified against
tuning.py:54-68, where every field has a default.

## Progress

Task 1: fix round 1/5 (2 addressed, 0 open — _strength TypeError leak; bool accepted as strength/number; commits d273151..096ada4)
Task 1: complete (commits a650fb6..096ada4, review clean) — Intents, 11/11 tests

Ruling 7 (T2): `detect_vocal_state` must read RT60 from `vdna["quality"]["estimated_rt60_s"]`
as well as the top level. The Task 2 implementer correctly flagged that
`estimated_rt60_s` is absent from the vocal DNA document's top level; verified
that `vocal_dna.py:276` nests the whole `AudioQuality` payload under `"quality"`,
and `AudioQuality.to_dict()` does contain `estimated_rt60_s`. Without this the
detector's `reverb_is_intentional` is always False in production, which silently
disables the spec's dereverb and space decisions (§1.3) — the exact failure that
stripped a mix reverb from the reference render. Fix in a Task 2 fix round rather
than deferring: it is the difference between the detector working and not.
Cost if wrong: none; a defensive two-location read.
Task 2: fix round 1/5 (2 addressed, 0 open — RT60 nested-location read; _crest_db to dsp.to_mono; commits 33ac06c..a9c24aa)
Task 2: complete (commits 096ada4..a9c24aa, review clean) — intake.py + detect_vocal_state, 11/11 tests
