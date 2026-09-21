# mixengine rebuild — intake, policy, backends, and a single dashboard

Date: 2026-09-21
Status: approved for implementation

## Why

Given a finished vocal and the exact beat it was recorded to, the engine
produced an unusable render in 712 seconds. From that render's own
`result.json` and logs:

- It discarded 133 seconds of the beat and looped its 14-second intro ten
  times (`beat_fit: looped, loop_s [0.51, 14.58]`), because a hardcoded
  1.5 s reverb tail pushed the target past a 0.5 s tolerance.
- It moved 317 of 506 notes (mean 25 ¢, max 91 ¢) toward per-bar chord
  tones derived at 0.65 confidence from a beat whose key was reported as
  G# Major at "confidence 1.0" while the vocal's own candidates were
  C# Minor / E Major / G# Minor. The vocal arrived at 14.4 ¢ mean
  deviation — already tuned.
- It moved 520 of 678 onsets (up to 60 ms) against a 24 ms "error" that
  was the artist's pocket.
- It de-reverbed at 0.7 strength a reverb that was a mix decision, then
  failed to measure the beat's space (`confidence 0.00`) and added none
  back.
- It ran Demucs on a 0.56-confidence "light bleed" classification — a
  coin flip — costing 2 minutes.
- It scored itself 94% because every gate measures the output against
  the engine's own post-hoc assumptions. `n_warnings: 0`.

The root cause is single: **one pipeline with one assumption — raw take,
unknown key, unknown tempo, arbitrary beat — applied at full strength,
with no stage able to conclude "leave this alone."** The DNA already
contained the evidence to stop at nearly every step. Nothing reads it.

Research into how production systems solve this (RoEx, iZotope, Sonible,
Antares, Melodyne, LANDR, and the Reiss/De Man intelligent-music-production
literature) produced one consistent finding: **they ask the user for
semantic facts, and they default to not touching what is already
finished.** RoEx requires stem role/presence/reverb tags and ships a
`recombine` endpoint that skips separation entirely. Neutron makes the
user name the focus track. Nectar leaves pitch correction off and key
manual. Auto-Tune's Flex-Tune corrects only notes "significantly off."
Melodyne skips notes already close. KEAMS's listening study found the
failure mode of feature-only systems is exactly the missing high-level
information.

## Goals

1. A finished vocal over its own beat renders faithfully: same length,
   beat unlooped, key unchanged, no note moved more than 5 ¢, timing
   untouched — in under two minutes.
2. A raw take over an arbitrary beat still gets full production, and gets
   it better than today.
3. Every decision is explainable, overridable, and reported.
4. One dashboard, one pipeline run, no per-upload processing.
5. Renders are studio-ready: mastered against reference sub-targets, not
   a fixed LUFS.

## Non-goals

- ML mixing style transfer (FxNorm / Diff-MST). Reconsider after this
  lands; it needs training data and does not solve "what not to touch."
- Real-time pitch correction in the monitor path (see Phase 4).
- Multi-track stem mixing. Input remains one vocal, one beat.

## Architecture

Four phases. Each ships independently and leaves the system working.

```
Phase 1  intake + policy + locked render   ← stops the damage
Phase 2  backends                          ← speed and accuracy
Phase 3  mix and master quality            ← studio-ready
Phase 4  single dashboard                  ← one flow, one run
```

The spine is a new decision layer between analysis and rendering:

```
  DNA (facts about audio)
       +
  Intake (facts about provenance and state)
       +
  Intents (what the user says)
       ↓
  policy.plan()  → RenderPlan {per stage: enabled, strength, reason}
       ↓
  pipeline stages read the plan; none decides for itself
       ↓
  critic verifies the plan was honoured, not just that output sounds loud
```

`policy.plan()` is a pure function. No audio, no I/O. It is the most
heavily tested code in the system and can be reasoned about as a table.

---

## Phase 1 — Intake, policy, locked render

### 1.1 Intents

User-supplied semantic facts. Every field defaults to `auto`; the engine
reports what it chose. Carried through CLI, HTTP API, and UI.

| Field | Values | Meaning |
|---|---|---|
| `vocal_state` | `auto` `raw` `tuned` `finished` | `finished` = tuned and mixed |
| `relationship` | `auto` `locked` `free` | `locked` = recorded to this beat |
| `key` | `auto` or e.g. `f_minor` | user wins outright |
| `bpm` | `auto` or number | user wins outright |
| `tune` | `auto` `off` or 0–1 | intensity |
| `timing` | `auto` `off` or 0–1 | intensity |
| `space` | `auto` `keep` `match` `add` | `keep` = do not touch reverb |
| `separate` | `auto` `never` `always` | |
| `loudness` | `auto` or LUFS | `auto` = genre profile |

Represented as `core/intents.py::Intents` — a frozen dataclass with
`from_dict` / `to_dict` and validation. `auto` is `None` internally.

### 1.2 Intake detectors

New module `analysis/intake.py`. Three detectors. Each returns a value,
a confidence in [0,1], and a human-readable evidence string. All three
are reported in `result.json` and shown in the UI.

**`detect_vocal_state(y, sr, dna) -> VocalState`**

Analyses the loudest ~20 s where the signal allows (Ozone uses 8 s,
Sonible ~10 s, Nectar 20 s; a full-file pass is not needed to decide
this).

- *Tuned-ness*: histogram of note pitches modulo 100 ¢. A tuned vocal
  concentrates near 0 ¢; a raw one spreads. Score = fraction of note
  duration within ±20 ¢ of a semitone centre, weighted by note length.
  Combined with the existing `tuning_deviation_cents`. The literature
  offers a trained detector (Gohari 2024, 94.75%) — out of scope here;
  the histogram statistic is transparent and sufficient to *withhold*
  processing, which is the conservative direction.
- *Compression*: crest factor (peak minus RMS) over active regions, plus
  `phrase_level_spread_db`. A raw take spreads 3–6 dB between phrases;
  a mixed vocal sits under ~1.5 dB. (The reference render measured
  0.97 dB — unmistakably mixed, and the engine compressed it anyway.)
- *Reverb*: existing `estimated_rt60_s`, reinterpreted. Reverb is only
  evidence of a *bad room* when the vocal is otherwise raw. On a tuned
  or compressed vocal it is a mix decision and must be preserved.

Returns `raw` / `tuned` / `finished` with the evidence for each axis.
Ambiguity resolves toward the more finished state, because the cost of
under-processing is a flat render and the cost of over-processing is a
destroyed one.

**`detect_relationship(vocal, beat, sr, vdna, bdna) -> Relationship`**

Follows the vocal-to-instrumental alignment literature (GCC-PHAT on
accompaniment, ±20 s search):

1. Onset-strength envelope of both at a common rate.
2. Cross-correlate (GCC-PHAT) over ±20 s → best lag, and the ratio of
   the main peak to the highest side lobe.
3. Duration agreement within ±2 bars.
4. Tempo agreement, allowing half/double (trap is written 130–150 and
   felt 65–75; octave ambiguity is documented in the beat-tracking
   literature and must not count as disagreement).

`locked` requires a clear peak (ratio above threshold) **and** duration
agreement. The lag it returns *is* the alignment offset — the same
measurement serves detection and alignment.

**`decide_key(vdna, bdna, intents) -> KeyDecision`**

Key is a distribution, not a label. DJ tools disagree on ~60% of tracks,
overwhelmingly by relative-major/minor or fifth swaps — precisely the
G# Major / C# Minor / G# Minor spread in the reference render.

1. Take both candidate lists (already produced today).
2. Group into families by pitch-class-set identity: relative major/minor
   share a family; parallel and fifth-related are adjacent.
3. If vocal and beat top families agree → **no transposition**, and the
   mode is taken from the vocal's evidence (a voice states its third; an
   808 does not).
4. If they disagree and both are confident → propose a transposition and
   report it; never apply one when either side is below threshold. The
   current code already declines to transpose on low confidence — that
   behaviour is correct and is kept.
5. Below threshold → the tuner receives the **union scale** of the
   family, never a forced third.
6. `intents.key` overrides everything.

Replaces the `key_confidence: 1.0` fiction: confidence is reported as
measured (the beat's own top candidate scored 0.75).

### 1.3 Policy

New module `core/policy.py`. `plan(vdna, bdna, intake, intents) ->
RenderPlan`. Pure. `RenderPlan` holds a `StageDecision {enabled,
strength, method, reason}` per stage.

Base table by (relationship, vocal_state):

| Stage | locked + finished | locked + raw | free + finished | free + raw |
|---|---|---|---|---|
| separation | never | never | only if `full_mix` ≥ 0.75 | only if ≥ 0.75 |
| dereverb | off | only if RT60 > 0.8 s | off | only if RT60 > 0.8 s |
| tuning | off | flex (> 35 ¢ only) | off | flex |
| alignment | single offset | offset + drift check | phrase anchor | grid nudge ≤ strength |
| arrangement | none | none | structure-aware | structure-aware |
| beat fit | pad only | pad only | trim / loop last section | trim / loop last section |
| vocal chain | finish | produce | finish | produce |
| space | keep | match beat | keep | match beat |

Intents override any cell and the override is recorded as the reason.
The plan is logged as one screen before rendering and saved to
`result.json` under `plan`.

`tuning: flex` implements Flex-Tune / Melodyne semantics: a dead zone
where notes already close are untouched, correction ramping in only for
notes significantly off (default threshold 35 ¢), gestures and blue notes
preserved as today.

### 1.4 Pipeline changes

- `render_variant` takes a `RenderPlan` and consults it; the scattered
  decisions at `pipeline.py:181/208/226/241`, in `vocal_dna`, and in
  `separation` are removed.
- `fit_beat_to_vocal`: **never loop when the beat already covers the
  vocal.** The 1.5 s tail pads with silence. When a loop is genuinely
  needed, loop the *last* full section (an outro loops credibly; an intro
  does not) and cut on a downbeat as today.
- `align`: new `single_offset` method — apply one lag, verify grid fit
  improved, stop. DTW/warping only when the plan asks for it.
- Vocal chain gains a `finish` mode: HPF at most 0.6·f0_low (117 Hz on a
  137 Hz baritone was wrong), no formant notching (the three "resonances"
  at 312/495/646 Hz were his voice), de-ess only above a sibilance
  threshold, compress only to stabilise when spread > 3 dB, no air boost.
- Tuning: single-pass pitch-envelope shift instead of a `rubberband`
  subprocess per note (317 process launches, 2m46).
- Ducking: stems only. Without stems, cap at 1 dB within the vocal band
  rather than 2.5 dB across the whole beat.

### 1.5 Critic

The critic must stop grading its own homework.

- **Fidelity gates (locked mode)**: output duration equals input; beat
  not looped; key unchanged; no note moved more than 5 ¢; spectral
  distance to the level-matched sum of inputs within bound. These are
  errors, not warnings.
- **Tuning** is scored from the tuner's own report, not by re-running
  CREPE on the finished mix (2m48 of the 712 s).
- **Harmonic** is scored against the `KeyDecision`, with its confidence,
  not against an assumed key at a fabricated 1.0.
- Gates that were never evaluated report `skipped`, not `passed`.

### 1.6 Testing

- Unit: each detector against synthetic fixtures — pitch snapped to
  semitones vs. spread; compressed vs. dynamic; known-lag pairs.
- `decide_key` against the reference render's exact candidate lists,
  committed as a fixture. Must return "no transposition, minor family."
- Policy table as pure tests across the full intent matrix.
- **Fidelity regression**: the `voc.mp3` + `beat.mp3` pair. Assert
  duration equality, no loop, key unchanged, max note movement ≤ 5 ¢,
  spectral distance bound. This test is the definition of done for
  Phase 1.
- The existing suite stays green.

---

## Phase 2 — Backends

Measured against the current stack on this machine (Apple Silicon, MPS
available). Each swap is behind `capabilities.py` with the existing
backend as fallback, and `BACKEND_RANKS` already handles cache
invalidation when a better backend appears.

| Concern | Now | To | Why |
|---|---|---|---|
| Pitch | torchcrepe (~2.2× RT, F1 0.691) | RMVPE (~50× RT, F1 0.768) | ~20× faster *and* more accurate; the most noise-robust in the benchmark, which matters on separated vocals. FCPE (F1 0.728, same speed class) is the fallback if RMVPE will not run |
| Separation | demucs CPU (MPS fails every run) | demucs-mlx, fall back CPU | 52 s → 2.7 s on M4 for 3 min; stock Demucs' complex STFT/Wiener cannot run on MPS — stop attempting it every time |
| Rhythm | madmom | Beat This! | downbeat F1 75.5 vs 67.2, ~3.4 s for 3 min, CPU-fine, no DBN; keep madmom for tempo continuity (better CMLt/AMLt) |
| Key | chroma template | madmom CNN 24-class posterior | gives the distribution `decide_key` needs instead of a point estimate |

The MPS failure is fixed by *not asking*: `capabilities` records that
stock Demucs cannot run there and routes to MLX or CPU directly. The
existing `_UNUSABLE_DEVICES` memory becomes a static fact.

Phase 1 already removes ~5 minutes of the 712 s (separation skipped,
one-pass tuning, no critic re-track). Phase 2 targets the remainder: a
locked render should finish in well under a minute.

---

## Phase 3 — Mix and master quality

Mastering moves from a fixed genre LUFS to **reference sub-targets**, the
Ozone model: tonal balance curve, vocal-to-instrumental balance, stereo
width per band, micro-dynamics (short-term to very-short-term loudness
ratio), and integrated LUFS — learned from the loudest 8 s of a reference
and of the render. Genre profiles become the default reference when the
user supplies none; a user-supplied reference track overrides.

Reverb follows the estimate-before-add rule: measure the vocal's RT60/DRR
and the beat's space first; if the vocal is already wet, the send is
`none`. Never de-reverb unless the plan asks. When the beat's space
cannot be measured (`confidence 0.00`, as in the reference render), fall
back to the genre profile's reverb rather than silently adding none.

Dynamics follow the Pestana/Reiss finding: compress to stabilise an
erratic loudness range, not to a fixed ratio — which is why the `finish`
chain compresses only when phrase spread exceeds 3 dB.

Diagnose-then-act loops replace fixed constants: measure masking, adjust,
re-measure, stop when in range.

---

## Phase 4 — Single dashboard

### 4.1 What is wrong now

Four tabs (Catalog, Record, Vocal, Renders), each firing its own
analysis on upload. The user must understand the engine's internal
stages to operate it. There is no single place that says what will
happen.

### 4.2 The flow

**One page. One pipeline run. Uploads do not process anything.**

```
  ┌──────────────────────────────────────────────────┐
  │  mixengine                          tier · ready │
  ├──────────────────────────────────────────────────┤
  │                                                  │
  │   Start with a vocal        Start with a beat    │
  │   ┌────────────────┐        ┌────────────────┐   │
  │   │ record  upload │        │ record  upload │   │
  │   └────────────────┘        └────────────────┘   │
  │                                                  │
  ├──────────────────────────────────────────────────┤
  │  VOCAL   ▁▃▅▇▅▃▁ voc.mp3            2:27  ✕     │
  │  BEAT    ▁▅▇▃▅▇▁ beat.mp3           2:27  ✕     │
  ├──────────────────────────────────────────────────┤
  │  Treatment          all automatic ▾              │
  ├──────────────────────────────────────────────────┤
  │              [  Make the render  ]               │
  └──────────────────────────────────────────────────┘
```

Two slots. Either may be filled first — that is what "start with vocal
or beat" means; there is no mode, just whichever slot you fill. Each
slot offers **record** or **upload**. Filling a slot decodes the file
client-side and draws a waveform. **No server analysis, no pipeline.**

`Treatment` is one collapsed row reading `all automatic`. Expanded, it
shows the intents from 1.1 as compact selects. Most users never open it.

One button runs one pipeline. It is disabled until both slots are
filled, and its label says exactly what it will do.

### 4.3 During and after the run

A single progress rail with the real stages, streamed from the server:
`intake → plan → transform → mix → master → check`. After intake, **the
plan appears** — the same table from 1.3, in plain language:

> Your vocal is already finished and was recorded to this beat.
> Tuning off · timing kept · reverb kept · beat not looped.

The result is a player, the plan, the critic's gates, and a
`Change something and run again` control that re-opens Treatment with
the detected values pre-filled — Nectar's Relearn, not a blank form.

### 4.4 Recording

The existing `record.js` (mic permission, latency calibration, room
check) is good and is kept. It gains:

- **Record over the beat.** If the beat slot is filled, recording the
  vocal plays the beat through monitoring, with the calibrated latency
  compensated on the captured take. This makes `relationship: locked`
  true by construction, and the UI says so.
- **Live guidance, not live correction.** An AudioWorklet runs pitch
  detection on the input and shows, in real time, the note being sung
  against the beat's key — flat/sharp and in/out of scale. Correction
  is applied in the render.

  This is deliberate. Real-time correction in the monitor path is
  latency-bound and, more importantly, would hide pitch problems from
  the performer while recording them into the take. Every reference
  tool separates guidance from correction. The UI states plainly that
  guidance is a monitor aid and the take is captured unprocessed.

### 4.5 Visual design

Dark, with purple as the single accent. The existing `tokens.css`
discipline is kept and its reasoning still holds: one accent, semantic
colours reserved (green in-range, amber approaching, red clipped), 4 px
spacing scale, tabular numerals, 2 px radius.

Changes: `--accent` becomes purple with a gradient
(`--accent-grad`) used in exactly three places — the primary action, the
progress rail, and the active slot border. Nowhere else.

Explicitly avoided, because they are what makes an interface look
generated rather than designed: glassmorphism, blurred gradient blobs,
oversized border radii, gradient text, emoji as iconography, centred
hero sections, and decorative motion. This is a tool. Density, clear
hierarchy, and restraint are the aesthetic.

Accessibility: contrast ratios verified against WCAG AA, full keyboard
path through the flow, `prefers-reduced-motion` already honoured,
semantic HTML with the slots as real form controls.

### 4.6 API

One new endpoint: `POST /api/render` taking both files and the intents,
returning a job id, with progress over the existing job mechanism. The
per-upload analysis endpoints remain for the CLI and for backward
compatibility but the dashboard no longer calls them.

---

## Risks

- **Intake misclassification.** Mitigated by conservative defaults
  (ambiguity resolves toward "finished", i.e. toward doing less),
  explicit user override, and reporting evidence for every call.
- **Backend swaps regress quality.** Each is behind capabilities with
  fallback, and the fidelity regression plus the existing suite must
  stay green across the swap.
- **Phase 3 is the least specified.** It depends on Phase 1's plan
  existing and will be re-examined before it starts; if it needs its own
  design pass, it gets one.
- **Live pitch guidance is browser-dependent.** It degrades to a level
  meter where AudioWorklet or the pitch backend is unavailable; the
  recording path never depends on it.

## Definition of done

1. `voc.mp3` + `beat.mp3` renders faithfully in under two minutes and
   the fidelity regression passes.
2. `108.mp3` + `preview (20).mp3` renders without looping the intro and
   without retuning a tuned vocal.
3. A genuinely raw take over an arbitrary beat still gets full
   production, verified by ear and by the critic.
4. One dashboard, one run, and the plan is legible to someone who has
   never read this document.
