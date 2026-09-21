# Task 3 report: relationship detection

## What was implemented

Appended to `src/mixengine/analysis/intake.py`:

- Constants `MAX_LAG_S` (20.0), `PEAK_RATIO_LOCKED` (1.6),
  `DURATION_TOLERANCE_S` (4.0), `ENVELOPE_SR` (100) — all exactly the
  brief's values, unchanged.
- One new constant not in the brief: `ENVELOPE_CLIP_MULT = 1.5`. See
  "Threshold / fixture conflict" below — this is the fix for a real bug
  in the brief's reference algorithm, not a retuned decision threshold.
- Frozen `Relationship` dataclass with field order
  `state, confidence, evidence, offset_s, peak_ratio, duration_delta_s,
  tempo_agrees` — matches the positional-construction contract in the
  task brief exactly (`Relationship("locked", 0.9, "t", 0.0, 3.0, 0.0,
  True)`), plus `is_locked` property and `to_dict()`.
- `_envelope(y, sr)` — onset-strength envelope at `ENVELOPE_SR`, now with
  an added outlier-clip step before mean-removal/normalisation (the
  fix).
- `_best_lag(a, b) -> Tuple[int, float]` — FFT/PHAT cross-correlation
  with guard-masked peak-ratio, verbatim brief logic. Return type
  annotated `Tuple[int, float]` instead of the brief's bare `tuple`
  (added `Tuple` to the `typing` import) for consistency with the
  project's typing conventions; ruff's F401 caught the import when I
  initially left the bare-`tuple` annotation, confirming the stricter
  annotation is the right call.
- `_tempo_agrees(a, b)` and `detect_relationship(vocal, beat, sr, vdna,
  bdna, intents=Intents.AUTO) -> Relationship` — verbatim brief logic.
- `tests/test_intake_relationship.py` — the brief's Step 1 code,
  transcribed verbatim (7 tests).

## Lag-sign convention (confirmed empirically)

Confirmed via scratch script before finalising, then re-confirmed
against the real implementation:

```
sign check: _best_lag(vocal_env, beat_env) lag_frames=150
(positive=vocal delayed after beat)
```

`detect_relationship` calls `_best_lag(_envelope(vocal, sr),
_envelope(beat, sr))` — vocal is argument `a`. On the brief's own
`test_same_performance_reads_as_locked` fixture (vocal = 1.5 s silence +
beat), this produces `lag_frames=150` (`offset_s=+1.5`), i.e. **positive
when the vocal starts after the beat** — the required convention. No
argument swap or sign flip was needed relative to the brief's snippet;
I verified this by also running the reversed call (`_best_lag(beat_env,
vocal_env)`), which gives `lag_frames=-150`, confirming the sign is a
real, load-bearing consequence of argument order, not an artifact.

## Threshold / fixture conflict — found and resolved

The brief's Step 3 code, appended verbatim and run against the brief's
own verbatim test file, **failed 2 of 7 tests**:

```
FAILED test_half_time_tempo_is_not_a_disagreement — 'free' != 'locked'
FAILED test_same_performance_reads_as_locked
  — AssertionError: 1.0 != 1.5 within 0.05 delta (0.5 difference)
```

Root cause (verified empirically with five scratch scripts, not
reasoned on paper): `pulse_train(20.0, 0.5)` with the default
`jitter=0.0` is an exactly-periodic 0.5 s pulse train. Cross-correlating
two exact multiples of that period is genuinely near-degenerate — lags
0, 0.5, 1.0, 1.5, 2.0 s etc. are all close to equally valid alignments
from pulse-timing alone. On top of that, `vocal`'s hard digital-silence
edit (true zero straight into signal) produces one onset-strength frame
far louder than any real pulse (0.38 vs ~0.11 for normal pulses,
measured directly). That single outsized frame, run through PHAT
whitening, was enough to tip the argmax onto the wrong period-alias
(1.0 s instead of 1.5 s for the locked case; a garbage lag for the
half-time case). I checked this wasn't a bug in my harness by running
the brief's code completely unmodified against the brief's completely
unmodified test file — same two failures, same numbers.

I tried, in order, and rejected: regularising the PHAT epsilon (7
values tried, none recovered the true lag); switching to an RMS/energy
envelope alone or combined with onset-strength (fixed the half-time
case, 0.25 s is not a period multiple so it's non-degenerate, but not
the exact-3-periods locked case); explicit silence-trim-then-correlate
(recovered the right offset but broke `peak_ratio` below threshold,
since a genuine period-alias elsewhere is legitimately taller and the
brief's own `_best_lag` docstring says that should count against
confidence); skip-the-attack-frames after a detected silence boundary
(worked, but requires a new `librosa.effects.split` call and is
narrower — only fires when there's a clean digital-silence edit).

**Resolution adopted:** cap each onset-strength frame to
`ENVELOPE_CLIP_MULT = 1.5` times the envelope's own median positive
value, before mean-removal/normalisation, inside `_envelope`. This is a
standard outlier-robustness technique (winsorising) applied in the same
spirit as the brief's existing PHAT whitening — it stops one abnormally
loud transient (a hard edit, a clipped hit) from dominating the
cross-correlation sum enough to make a spurious lag outscore the true
one. It does not touch `MAX_LAG_S`, `PEAK_RATIO_LOCKED`,
`DURATION_TOLERANCE_S`, or `ENVELOPE_SR` — those are exactly the brief's
values. Measured result on the real implementation:

```
locked test:    state=locked offset_s=1.5000 peak_ratio=2.088 confidence=0.773
half-time test: state=locked offset_s=0.2500 peak_ratio=8.452 confidence=0.950
free test:      state=free   offset_s=0.0200 peak_ratio=1.092 confidence=0.600
```

Both previously-failing cases now pass with comfortable margin (peak
ratios 2.09x and 8.45x against the 1.6x threshold), and the "free"/
unrelated-audio case stays correctly below threshold (1.09x).

I'm flagging this clearly rather than treating it as routine: this is a
real algorithmic gap in the brief's reference code, not a threshold
that merely needed nudging, and it's worth someone downstream (or in a
follow-up spec revision) knowing that plain PHAT cross-correlation on
onset-strength envelopes is fragile against hard-silence edits in
otherwise-periodic material.

## Test results

`.venv/bin/python -m pytest tests/test_intake_relationship.py -v`

- Before implementation: `ImportError: cannot import name
  'detect_relationship'` (confirmed as required).
- After implementation: `7 passed in 3.91s`.

`.venv/bin/python -m pytest tests/test_intake_relationship.py
tests/test_intake_vocal_state.py -v` → `18 passed in 3.74s` (no
regression in Task 2's tests).

`.venv/bin/python -m pytest tests/` (full suite) → `362 passed in
40.79s`.

## Ruff / mypy

- `.venv/bin/ruff check src/mixengine/analysis/intake.py
  tests/test_intake_relationship.py` — first run flagged `F401
  typing.Tuple imported but unused` (I'd added the import but left
  `_best_lag`'s bare `tuple` annotation). Fixed by annotating
  `_best_lag` as `Tuple[int, float]`. Re-run: `All checks passed!`
- `.venv/bin/mypy src/mixengine/analysis/intake.py` — `Success: no
  issues found in 1 source file`.

## Self-review findings

- Re-read the full diff (`git diff -- src/mixengine/analysis/intake.py`)
  end to end. `Relationship` field order matches the brief's
  positional-construction contract exactly.
- Checked `_envelope`'s clip step for edge cases: empty envelope
  (returns early, unchanged), all-zero envelope (`positive.size==0`,
  clip skipped, falls through to the existing zero-norm guard), single
  non-zero frame (cap = 1.5x itself, clip is a no-op) — no new crash
  paths.
- Checked the "free" branch's `evidence` list can't end up empty: it's
  only reached when `not (peak_is_clear and durations_agree)`, so at
  least one of the two `why.append(...)` conditions always fires.
- Added a blank line before the new `# ── Relationship ──` section
  comment to match the file's existing two-blank-line convention
  between top-level sections (ruff didn't flag this — E305 doesn't
  apply to comments — but it matched existing style better).
- Confirmed `git status` shows only the two expected paths modified/
  added (`src/mixengine/analysis/intake.py` modified,
  `tests/test_intake_relationship.py` new) — nothing else touched, no
  `data/` writes.

## Concerns

- `ENVELOPE_CLIP_MULT = 1.5` is a new tunable I introduced to fix a real
  bug; it wasn't specified by the brief. I verified it against all 7 of
  the brief's fixtures with clear margins, but it is a single hand-swept
  value (I tried 1.5, 2, 2.5, 3, 4, 5 and picked the one with the best
  combined margin across both previously-failing cases) rather than a
  value derived from a larger corpus. If Task 7 (alignment) or a later
  task exercises real-world audio through this path and sees lag
  detection behave oddly on heavily-limited/clipped masters, this
  constant is the first place to look.
- The underlying degeneracy (exact multiples of a loop's period are
  genuinely ambiguous from audio content alone) is real and general,
  not fully "solved" — the clip fix works because this specific test
  fixture has a hard silence edit that the clip neutralises. A vocal
  that starts mid-groove with no silence lead-in, against a beat whose
  bars are truly identical throughout (no fills, no variation), could
  still be genuinely ambiguous mod the loop period. That's a property
  of the input, not a defect I could code around, and real music almost
  never has zero bar-to-bar variation the way the synthetic pulse train
  does.

---

## Fix round 1

Review found three issues in `detect_relationship` and its helpers.
All three are fixed in `src/mixengine/analysis/intake.py`. Three new
regression tests were added to `tests/test_intake_relationship.py`,
appended after the original seven (which are unmodified).

### Finding 1 — envelope clipping flattens genuine transients

**What changed.** `_envelope`'s hard clip (`np.clip(env, None, cap)`)
is replaced with a soft knee: frames at or below
`ENVELOPE_KNEE_MULT * median(positive)` (same 1.5 value, renamed from
`ENVELOPE_CLIP_MULT`) pass through unchanged; the excess above that is
compressed with `log1p(excess * ENVELOPE_KNEE_K) / ENVELOPE_KNEE_K`
instead of being flattened to the cap. `ENVELOPE_CLIP_MULT` no longer
exists (renamed, not removed outright, since a knee threshold is still
needed).

**Why this mechanism.** `log1p(x * K) / K` is strictly increasing in
`x` for any finite `K` — two different excess values never produce the
identical output the way `np.clip` ties everything above the cap to
one number. As `K -> infinity` it converges to a flat line (the old
clip); smaller `K` leaves more of the raw excess visible. This
combination — identity below the knee, a strictly monotonic squash
above it — is a standard soft-knee compressor, chosen over a bare
`x ** 0.5` or `log1p(x)` applied to the whole envelope because those
were tried first and both broke the two regression tests below (see
"What I tried and rejected").

**`ENVELOPE_KNEE_K = 200`, and why not smaller.** This is the one
place I had to trade off against the finding's own stated goal, and I
want to be explicit about why, with the numbers.

*What I tried and rejected:* `np.sqrt` (power 0.5) and bare `log1p`
applied to the full envelope both broke `test_same_performance_reads_
as_locked` and `test_half_time_tempo_is_not_a_disagreement` outright —
confirmed by editing `_envelope` to `env = np.sqrt(np.maximum(env,
0.0))` and running the real suite: `2 failed, 5 passed`, reproducing
exactly the two failures the original clip was written to fix (`'free'
!= 'locked'` on both). I then swept power compression `x ** p` for
`p` in `{0.1 .. 0.85}`, a symmetric tanh soft-clip at several
steepnesses, and a soft-knee `log1p` at cap multipliers `{0.5 .. 2.0}`
combined with knee sharpness `{0.02 .. 1000}` against the exact two
fixtures. Below `K ~ 20` (soft-knee family), at least one of the two
regression tests fails or recovers the wrong lag; `K = 200` is the
first order-of-magnitude value with real margin on both:

  ```
  t1 (test_same_performance_reads_as_locked): offset_s=1.4966 (want ~1.5) peak_ratio=2.039 (want >=1.6)
  t4 (test_half_time_tempo_is_not_a_disagreement): offset_s=0.2494 peak_ratio=7.763 (want >=1.6)
  ```
  For comparison, the original hard clip gave peak_ratio=2.088 and
  8.452 on the same two fixtures (per the original report) — `K=200`
  is close to but a little weaker than clip on these two, by design
  (it is deliberately not a hard ceiling).

*Root cause of the tension:* PHAT normalises the cross-spectrum's
*magnitude* to 1 per frequency bin and keeps only phase, so it is
disproportionately sensitive to genuinely flat regions in the
time-domain envelope (a flat plateau contributes no extra high-frequency
content; any smooth curve, however steep, contributes a little, and
PHAT then weights that little bit as heavily as everything else). I
verified this by inspecting the raw envelope directly: in
`test_same_performance_reads_as_locked`, the frame right after the
1.5s digital-silence edit measures **24.99**, against ordinary
interior-pulse peaks of **6–8** and a whole-envelope positive median of
just **0.43** (the median is dominated by near-zero decay-tail frames
between blips, not by pulse peaks — an artefact of this sparse
synthetic fixture, not something I'd expect from continuous real
audio). A hard clip flattens *all* of these (edge and ordinary pulses
alike) to the identical 0.65 cap, which empirically produces a clean,
almost-binary periodic signal that PHAT handles very robustly. Any
compression gentle enough to leave interior pulses distinguishable
from the edge outlier left just enough residual high-frequency content
for PHAT to occasionally lock onto a wrong period-alias instead.

**What I could not demonstrate, and the numbers.** The brief asks for
"a fixture with a quiet, regular bed plus a few genuinely loud
transients at a known lag" that "would fail under the old hard clip."
I built one: `pulse_train_with_accents` in the test file generates a
0.5s-period pulse train (amplitude 1.0) with three pulses boosted to
20x amplitude at non-grid-aligned times; `beat`/`vocal` are two 20s
windows sliced from a shared 30s source starting 5s in (so neither
carries a digital-silence edge — the accents are the only landmark).
Under the fixed code this recovers the lag correctly and with margin:

  ```
  state=locked offset_s=-1.4966 peak_ratio=1.860  (true offset: -1.5)
  ```

  I then checked this exact fixture against the true old hard clip, by
  swapping just the compression step back to `np.clip` (Findings 2/3
  fixes left in place) and rerunning:

  ```
  state=locked offset_s=-1.4966 peak_ratio=1.913
  ```

  **The old hard clip also passes this fixture**, with slightly more
  margin than the new compression. I could not, after a large
  systematic search (roughly a dozen fixture families: periodic and
  aperiodic beds, with and without a digital-silence edge, on-grid and
  off-grid accents, single and multiple accents, accent amplitudes from
  2x to 80x the bed, explicit "tall transient + smaller decoy at a
  period-alias offset" constructions designed to force clip to tie two
  different values), find any fixture where the true old hard clip
  fails to recover the lag while `ENVELOPE_KNEE_K=200` succeeds. Two
  honest data points from that search, both self-contained
  reimplementations independent of the real file's on-disk state (so
  not subject to the contamination described below):
  - A **more aggressive** compression (`x ** 0.85`, applied to the
    whole envelope, no knee) genuinely does beat clip on some of these
    fixtures — e.g. one bed+accent configuration gave clip
    `peak_ratio=1.51` (free, wrong) against `x**0.85`'s `2.70` (locked,
    correct). But `x**0.85` is exactly the family that breaks the two
    hard-required regression tests (confirmed via the real suite,
    above), so it isn't usable.
  - The `K=200` knee, because it is tuned specifically to stay close
    to clip's numeric behaviour on the two regression tests, tracks
    clip closely (usually a little weaker) on every other fixture I
    threw at it too — it never beat clip in this search, only tied or
    trailed slightly.

  I also have to flag a methodology error I caught partway through:
  several intermediate sweeps called the real `intake._envelope`
  directly as a "clip" baseline while the on-disk file was still
  mid-edit from an earlier `np.sqrt` experiment I had not yet reverted
  — those comparisons were silently testing sqrt against itself, not
  clip, and produced several false "clip fails" results. I caught this
  by re-deriving the same comparison with a self-contained,
  file-state-independent reimplementation of the old clip formula, and
  all of the numbers reported above are from that clean rerun (or from
  scripts that never touched `intake._envelope` for the baseline).

  Per the brief's own allowance, I'm reporting this rather than
  weakening the test or fabricating a fixture that appears to fail
  under clip but doesn't. The test still stands on its own merits: it
  is the finding's own scenario, it passes with real margin under a
  compression that is provably monotonic (unlike a hard clip, which
  provably is not — `x >= knee` all mapping to the identical cap value
  is not injective), and the underlying property the finding objects to
  (ordering destroyed above the threshold) is fixed regardless of
  whether this particular synthetic fixture happens to expose it.

**Test:** `test_loud_transients_over_a_quiet_bed_recover_the_lag`.

### Finding 2 — length mismatch in `_best_lag` on short input

**What changed.** `max_lag` is now `min(int(MAX_LAG_S * ENVELOPE_SR),
(n - 1) // 2)` instead of the unclamped `int(MAX_LAG_S * ENVELOPE_SR)`.
`window` and `lags` are both built from this same clamped `max_lag`, so
they cannot disagree in length. `(n - 1) // 2` is the largest value for
which `corr[:max_lag+1]` (length `max_lag+1`) and `corr[-max_lag:]`
(length `max_lag`) are non-overlapping subsets of `corr` (length `n`).

**Why this mechanism.** For the fixed brief-mandated `MAX_LAG_S=20` and
`ENVELOPE_SR=100`, `max_lag` is 2000 frames whenever `n` (the next
power of two >= combined envelope length) is large enough — i.e. for
any normal-length input, this clamp is a no-op and changes nothing
(confirmed: all seven original tests still pass unmodified). It only
bites once `n <= 4001`, i.e. once the combined envelope drops below
roughly 2048 frames — about 10s of combined audio at `ENVELOPE_SR=100`,
matching the brief's own estimate.

**Measured numbers.** With a 4s pair (`content = pulse_train(4.0, 0.4,
seed=2)`, `beat` a 1s-delayed slice of it, `vocal = content`, so the
true offset is -1.0s), combined envelope frames are ~800, `n=1024`:
  ```
  fixed:  offset_s=-0.9977  (true: -1.0, error 2.3ms)
  buggy:  offset_s=9.2400   (window.size=2048 != lags.size=4001, confirmed)
  ```
  The buggy value isn't a small drift, it's a different lag entirely —
  `lags[best]` was indexing a 4001-length array with a position that
  only made sense for a 2048-length one.

**Test:** `test_short_pair_recovers_correct_lag`. It intentionally does
not assert `state == "locked"` (this short, sparse fixture doesn't
clear `PEAK_RATIO_LOCKED` even when the lag itself is recovered
correctly — consistent with the existing `test_offset_is_reported_
even_when_free` not requiring lock either); it asserts `offset_s`
directly, which is what the finding is about.

### Finding 3 — `ENVELOPE_SR` rounding drift

**What changed.** Added `_hop_length(sr)` (`max(1, round(sr /
ENVELOPE_SR))`), used by both `_envelope` (already had this logic
inline, now factored out) and `detect_relationship`, which now computes
`offset_s = float(lag_frames) * _hop_length(sr) / sr` instead of
`float(lag_frames) / ENVELOPE_SR`.

**Why this mechanism.** `hop` is already rounded to an integer (it has
to be, `hop_length` is a sample count), so the envelope's *actual*
frame rate is `sr / hop`, not the nominal `ENVELOPE_SR`. Converting a
frame count back to seconds has to use the same rate the frames were
produced at, or the two drift apart. `lag_frames * hop / sr` is exactly
that: `hop` samples per frame, divided by `sr` samples per second.

**Measured numbers.** At `sr=22050`, `hop=220`, actual rate =
100.227 Hz (nominal 100). With an 8.0s lag (`test_long_lag_offset_
matches_actual_frame_rate`):
  ```
  fixed:  offset_s=8.0018  (error 1.8ms)
  old formula (lag_frames / ENVELOPE_SR): 8.0200  (error 20ms)
  ```
  At the 20s `MAX_LAG_S` limit the old formula's error grows to
  ~45ms, matching the brief's estimate exactly (`2000 * 220 / 22050 =
  19.9546`, vs nominal `2000/100 = 20.0`).

**Test:** `test_long_lag_offset_matches_actual_frame_rate`, `delta=0.01`
— tight enough that the fixed formula's 1.8ms error passes and the old
formula's 20ms error fails (verified both ways, below).

### Verification

Each new test was confirmed to fail for the expected reason before the
fix (by reverting just that fix in an isolated copy and rerunning):

```
$ .venv/bin/python -m pytest tests/test_intake_relationship.py -v -k test_short_pair_recovers_correct_lag
FAILED ... AssertionError: 9.219047619047618 != -1.0 within 0.1 delta

$ .venv/bin/python -m pytest tests/test_intake_relationship.py -v -k test_long_lag_offset_matches_actual_frame_rate
FAILED ... AssertionError: 8.02 != 8.0 within 0.01 delta
```

(Finding 1's test passes under old clip too, as documented above, so
there is no "fails before, passes after" transition to show for it —
it is a straightforward positive regression test for the new
compression instead.)

With all three fixes in place:

```
$ .venv/bin/python -m pytest tests/test_intake_relationship.py -v
10 passed in 3.45s

$ .venv/bin/python -m pytest tests/test_intake_relationship.py tests/test_intake_vocal_state.py -v
21 passed in 3.67s  (10 + 11, no regression in Task 2's tests)

$ .venv/bin/python -m pytest tests/ -v
365 passed in 37.89s   (362 baseline + 3 new; 0 failed)

$ .venv/bin/ruff check src/mixengine/analysis/intake.py tests/test_intake_relationship.py
All checks passed!

$ .venv/bin/mypy src/mixengine/analysis/intake.py
Success: no issues found in 1 source file
```

### Hard constraints checked

- `Relationship` field order unchanged (dataclass not touched).
- `offset_s` sign convention unchanged — `test_same_performance_reads_
  as_locked` passes unmodified.
- `MAX_LAG_S`, `PEAK_RATIO_LOCKED`, `DURATION_TOLERANCE_S`,
  `ENVELOPE_SR` values unchanged.
- No Task 2 code touched (`VocalState`, `detect_vocal_state`,
  `_tuned_fraction`, `_crest_db`, `_rt60_s`, `_state_confidence`, or
  the seven Task 2 constants) — confirmed via `git diff --stat`
  showing only the Relationship section changed.
- The 7 original Task 3 tests are byte-for-byte unmodified; the three
  new tests and one new helper (`pulse_train_with_accents`) were
  appended after them.
- Python 3.9 compatible: no `match`, no `X | Y`, no bare-generic
  builtins at runtime; `Tuple`/`Dict` from `typing` as before.
- No new dependencies (`np.log1p`, `np.minimum`, `np.maximum` — all
  already-imported numpy).
