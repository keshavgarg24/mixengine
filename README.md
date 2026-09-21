# mixengine

Takes a vocal take and an instrumental and produces a mixed, mastered
stereo master — matching the vocal to a compatible beat, aligning it to that
beat's groove, correcting pitch against the chords underneath it, arranging
it into a song with a shape, mixing, mastering to a genre-appropriate
target, and grading the result against technical and musical gates.

**Current state: honest summary.** The engine renders end to end and the
output is technically correct — correct format, true peak held at −1.0 dBTP,
no clipping, mono-safe — and now has an arrangement: hooks are found by
repetition, layers follow an energy contour, and the mix moves across the
song rather than sitting at one setting. What it still lacks is listed under
[Known limits](#known-limits), which is a shorter list than it was and is
kept accurate deliberately.

---

## Install

```bash
python3 -m venv .venv && . .venv/bin/activate     # Python 3.9 to 3.13
pip install -e ".[quality,stretch,web]"
```

Nothing beyond numpy, scipy, librosa and soundfile is required. Every other
component is probed at runtime, and `doctor` reports what each missing one
costs rather than the engine refusing to start.

```bash
mixengine doctor
```

Optional, in rough order of impact:

| Component | Install | What it buys |
|---|---|---|
| Rubber Band | `brew install rubberband` | Formant-preserving stretch and pitch shift |
| Demucs | `make install-models` | Stems → drum-safe pitch shift, ducking without pumping |
| torchcrepe | `make install-models` | Accurate f0, which improves key detection and tuning; the full model on an Apple or NVIDIA GPU, the tiny one on a CPU |
| madmom | `make install-models` | Downbeat tracking for bar-accurate placement |
| pyloudnorm | included in `[quality]` | Real ITU-R BS.1770 loudness rather than an RMS proxy |

`make install-models` installs torch, torchcrepe and demucs, madmom from a
pinned commit of its repository (the 2018 PyPI release does not import
under numpy 2), and a headless Chromium for the browser DSP tests. All of it
needs Python 3.9 to 3.13 -- torch and numba have no wheels beyond -- and
`make venv` checks before creating anything.

Analyses are cached by content and schema version, and every document also
records the backends it was made with. Installing a better one -- madmom,
torchcrepe, demucs -- makes the next lookup re-analyse instead of serving
the older result, and a document made with better backends elsewhere is
kept rather than downgraded.

---

## Use

**Interface** — the main way in:

```bash
mixengine serve
```

Four views: import and inspect beats, **record a take with live coaching**,
upload a take and see its matches, and audition renders with the critic's
verdict on each. Every number shown is measured; anything the engine could
not measure shows an em dash rather than a plausible-looking zero.

There is no build step. The interface is native ES modules and plain CSS,
served straight from the package — `pip install mixengine` then `mixengine
serve` is the whole setup, with no Node toolchain anywhere in the path.

**Command line**, for batch work:

```bash
mixengine analyze-beats --beats data/beats --out data/dna/beats
mixengine analyze-vocal --vocal take.wav --bpm 140 --key f#_minor
mixengine match  --vocal take.wav -n 5
mixengine render --vocal take.wav --beats 3 --variants 1
```

Telling the engine a known BPM or key is the highest-value optional input
there is. A rubato or dry rap take is genuinely hard to measure, and a wrong
tempo propagates into every stage after it.

---

## How it works

```
BEAT (once, cached)          VOCAL (per take)
  load + quality probe         de-bleed against a known beat, if given
  beats / downbeats            classify → separate only if needed
  groove template   ◄──┐       restore: dereverb, denoise, declip
  key + chords         │       f0 → note events → key
  structure, pocket    │       phrases, onsets, delivery type
  genre, space         │                │
  stems                │                ▼
         └─────────────┼──────►  MATCH  (7 weighted dimensions,
                       │                  6-rung relaxation ladder)
                       │                │
                       │                ▼
                       └──────►  RENDER
                                   pitch → tempo (verified) → tuning
                                   → variable-rate alignment → arrangement
                                   → layers → space → mix → master
                                                │
                                                ▼
                                             CRITIC → repair loop
```

### The parts that are load-bearing

**Groove, not a grid.** Each beat's microtiming signature is measured once
and cached; vocals are aligned to *those* positions rather than to a
mathematically exact grid. Landing on them is what "in the pocket" means.
The engine never injects random jitter to "humanise" — scaling expert
performers' microtiming shows groove ratings falling when deviations are
exaggerated, and fully quantised versions rating as highly as the human
originals. A template that does not repeat reliably is discarded rather
than applied.

**Variable-rate alignment, not a single ratio.** The take's onsets are
matched to grid slots by a monotonic dynamic program, so no two syllables
collapse onto one slot and none crosses another — failures that only appear
over a whole take and that per-onset nearest-neighbour cannot avoid. Tempo
drift is measured and removed first, because an assignment computed before
that faithfully reproduces the drift. The warp is then applied segment by
segment at each segment's own ratio.

**Chord-aware tuning, not scale-snapping.** A note is judged against the
chord sounding underneath it at that instant. Only genuinely harsh
intervals are corrected; deliberate tensions are left alone, gestures
(scoops, slides, melisma) are skipped entirely, and the blues degrees are
protected in the genres built on them. Correction strength scales with how
exposed the note is.

**An arrangement, not a loop.** The hook is found by repetition — the
oldest and most reliable structural signal in popular music — and the
section labels that follow from it drive an energy contour. That one
contour then decides both halves of the problem: which layers exist where
(doubles on the hook, harmony at the peak, ad-libs in the gaps, nothing in
the first verse) and how the mix moves (level and brightness riding across
the song). Because they share a target, the arrangement and the mix push
the same way instead of each guessing separately.

**Doubles that are performances, not copies.** A delayed copy of a signal
is a comb filter, not a double. Generated layers are warped along a slow
bounded random walk and detuned, which is what a second take actually
differs by, and the two halves of a wide pair are generated independently
so they survive a mono fold.

**The vocal in the beat's room.** The beat's RT60 and direct-to-reverberant
ratio are measured from the decay after its own transients, and the vocal is
moved *toward* that space — never all the way, because a lead is
conventionally closer than the track around it. This is the main reason an
automatic mix reads as pasted on top.

**Subtraction, not separation, when the beat is known.** A vocal recorded
over speakers carries the beat back into the microphone. Running a source
separator on that is solving an underdetermined problem when a determined
one is available: the beat is *known*, so its contribution can be estimated
and subtracted. Measured on synthetic bleed at −12 to −30 dB, this removes
about 15 dB while leaving the voice 34 dB below the error floor.

**Balance over vocal-active regions.** The vocal and the beat are measured
over the *same* phrase windows. Integrated loudness across a whole file is
dragged down by the silence between phrases, so matching to it makes a
vocal too loud while it is actually singing.

**A critic that re-measures the output.** Gates run on the rendered audio,
not on the render parameters, so drift accumulated during processing is
caught. Each failing gate carries a deterministic repair, overrides
accumulate across attempts, and the best-scoring attempt is kept.

**Honest degradation.** Unknown is distinguished from zero everywhere. A
low-confidence key will not trigger a pitch shift; an unreachable loudness
target is reported rather than reached by crushing the dynamics; a take with
no silence in it reports an unmeasurable noise floor instead of a wrong one;
a genre classifier that cannot separate two families uses the neutral
profile rather than picking one.

---

## Layout

```
src/mixengine/
  core/       types, Musical IR, keys, audio I/O, capability probing
  musical/    theory, groove, salience, energy  ← pure logic, no audio deps
  analysis/   MIR, beat DNA, vocal DNA, genre, matching
  align/      monotonic onset↔grid assignment, DTW, variable-rate warping
  arrange/    structure, song plan, generated layers, mix automation
  audio/      dsp, transform, tuning, timing, debleed, space, mixer,
              master, critic, pipeline
  capture/    real-time frame analysis, pitch tracking, coaching
  api/        service layer and HTTP endpoints
  web/        the interface
    worklets/   the audio-thread DSP: capture, metering, pitch
    js/audio/   ring buffer, latency calibration, WAV, live coach
    js/core/    api client, state, DOM helpers
    js/ui/      one module per view, plus the canvas waveform
    css/        tokens, base, components, views
```

[`ARCHITECTURE.md`](ARCHITECTURE.md) covers the dependency rules, what
crosses a process boundary, and how to add a stage.

`musical/` is deliberately free of audio dependencies. Every musical
judgement the engine makes can be verified in milliseconds on any machine,
independently of whether separation models or stretch libraries are
installed — and a disagreement about what the engine *should* do is settled
by reading one function rather than by rendering a file and listening.

---

## Recording

The capture layer follows a finding that runs against the obvious design:
**concurrent visual feedback measurably degrades the take in progress.**
Studies of singers training with live visual pitch feedback report
performance decrement from the added cognitive load, with results worsening
at the moment feedback is delivered. The benefit is to learning across
sessions, not to the take being recorded now.

So the engine splits guidance three ways:

- **Before** — key, tempo, bar count, mic technique, room verdict, and a
  range check against the singer's measured tessitura. This last one is the
  most valuable item in the product: no downstream processing fixes a melody
  sitting outside someone's range.
- **During** — only faults that destroy the recording and cannot be repaired:
  clipping, dead channel, level, proximity, plosives. Nothing about pitch or
  timing. An attention budget rate-limits everything except clipping.
- **After** — full detail, with pitch judged against measured human norms
  (professionals average ~25 cents of deviation, non-professionals ~34.5),
  not against perfection.

### The browser side

The live half runs in the browser, because a cue that arrives 300 ms after
the fault it describes is worse than no cue. Its thresholds mirror
`capture/coach.py` and say so; Python remains the source of truth for the
post-take report.

**An AudioWorklet, not a ScriptProcessor.** The metering and capture run on
the audio render thread at real-time priority, allocating nothing. A
ScriptProcessor runs its callback on the main thread, so a layout pass
drops audio — and a take cannot be re-recorded after the fact.

**A lock-free ring buffer in shared memory.** The worklet writes, the main
thread reads, neither blocks. This needs cross-origin isolation, so the
server sends COOP and COEP; where they are unavailable the engine falls
back to copying blocks through `postMessage` and says which path it took.

**Measurement that matches the engine's.** True peak is 4× oversampled, so
it sees the inter-sample peaks that make a −0.5 dBFS signal clip a consumer
D/A. Loudness is K-weighted per ITU-R BS.1770 — the same measure the
mastering stage targets — with coefficients re-derived by bilinear
transform so it stays correct away from 48 kHz. Pitch is McLeod's method,
the same algorithm as `capture/realtime.py`.

**Round-trip latency measured, not guessed.** A short sweep is played and
found again in the microphone feed, which gives the real output-to-input
delay including the air gap. `AudioContext.outputLatency` knows nothing
about the input path or the room. Someone recording over speakers is
uniformly late by that round trip — 30–80 ms on a laptop, a third of a
sixteenth at 140 BPM — and no later stage can tell that apart from a singer
who was simply behind the beat.

**Raw PCM at 24 bits, not MediaRecorder's Opus**, and every browser
"enhancement" — echo cancellation, noise suppression, automatic gain — is
explicitly disabled. All three are tuned for speech intelligibility on a
call: echo cancellation ducks against the beat, noise suppression treats
sustained tones as noise, and AGC moves the gain during the performance so
every later level decision is fighting one already applied invisibly.

---

## Tests

```bash
make check          # lint, types, and the Python suite
make test           # the Python suite alone
```

**282 Python tests**, no audio dependencies required for the majority.
Tests assert musical claims rather than implementation details, and many
lock down specific bugs found by running the engine — each of those names
the failure it prevents, because a test whose reason has been forgotten is
the first one someone deletes.

**32 browser audio tests** at `/static/test.html`, or headless:

```bash
python scripts/run_browser_tests.py
```

The worklet DSP cannot be checked by the Python suite, so it is asserted
against signals whose correct answer is known analytically: a 0 dBFS 1 kHz
sine must read −3.01 LKFS, a waveform built to overshoot by 3.01 dB must
measure 3.01 dB, a probe planted at a known sample must be found there.
Four of those checks failed the first time they ran against code that
looked entirely reasonable.

---

## Known limits

Things that are genuinely not done, stated plainly, and separated from the
things that are design decisions.

**Not done:**

- **Analysis remains the binding constraint.** Vocal tempo and key are
  estimated, and estimates carry error into every later stage. Two things
  narrow that: tempo now comes from *two* independent estimators whose
  agreement sets its confidence, and when a take was recorded over a known
  beat the tempo is taken from the beat rather than estimated at all.
  Neither is the same as a better pitch tracker; RMVPE-class f0 and Beat
  This!-class beat tracking would be the next real gain, and the
  capability probing exists so they can be dropped in.
- **Structure needs repetition to find anything.** A through-composed take
  with no repeated phrase gets a flat arrangement, correctly labelled as
  such. The hook detector has no opinion about lyrics.
- **Live monitoring latency cannot be removed from a browser.** Direct
  monitoring happens in an audio interface's hardware, before the signal
  reaches the computer. The recorder measures the delay, states it, and
  says when it is past the point where singers start dragging — but it
  cannot make it shorter.

**Design decisions, not defects:**

- **Genre classification is nine explicit templates, not a trained model.**
  A template can be read and argued with when the output is "this is drill,
  so duck 3 dB harder"; it needs no training data and no torch; and it
  abstains across families rather than guessing. A producer's own tag
  always wins. A model would be more accurate on a wider label set and is a
  reasonable future addition, but the current behaviour is intended.
- **A "drum fill" transition is delivered as a reverse swell of the beat's
  own last beat.** A real fill needs drum samples or a drum stem to write
  into, and neither is guaranteed to exist. The report names what was
  delivered.
- **The harmony layer is a single voice, a third above.** It is chord-aware
  per note and folds into the singer's range, but it is one part. Stacked
  harmonies are a production choice this engine does not make on its own.
