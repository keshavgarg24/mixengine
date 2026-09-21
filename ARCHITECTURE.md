# Architecture

How the pieces fit, and why the boundaries are where they are. The README
covers what the engine does; this covers how it is built and what a change
in one place costs elsewhere.

## The shape

```
                       ┌──────────────────────┐
  browser  ──────────► │  api/app.py          │  HTTP, validation, paths
                       │  api/service.py      │  caching, jobs, workspace
                       └──────────┬───────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        ▼                         ▼                         ▼
  ┌───────────┐           ┌─────────────┐           ┌─────────────┐
  │ analysis/ │           │  audio/     │           │  arrange/   │
  │ what is   │  ───────► │  what to do │ ◄──────── │  song shape │
  │ this?     │           │  to it      │           │             │
  └─────┬─────┘           └──────┬──────┘           └──────┬──────┘
        │                        │                         │
        └────────────┬───────────┴─────────────────────────┘
                     ▼
              ┌─────────────┐        ┌──────────┐
              │  musical/   │        │  align/  │
              │  theory,    │        │  onset ↔ │
              │  groove,    │        │  grid    │
              │  salience,  │        └──────────┘
              │  energy     │
              └─────────────┘
                     │
              ┌──────▼──────┐
              │   core/     │  types, IR, keys, audio I/O, capabilities
              └─────────────┘
```

Dependencies point downward. `core/` imports nothing else in the package;
`musical/` imports only `core/`; nothing below `audio/` imports from it.

## The one rule that shapes everything

**`musical/` has no audio dependencies.** Not librosa, not soundfile, not
numpy arrays of samples — only numbers and dataclasses.

This is not tidiness. Every musical judgement the engine makes — which note
to correct, how hard to pull an onset, where the energy should sit — is
decided there, so all of it can be tested in milliseconds on any machine
regardless of whether Rubber Band or Demucs is installed. It also means a
disagreement about what the engine *should* do is settled by reading one
function, not by rendering a file and listening to it.

The cost is real: `audio/` has to marshal its arrays into note events and
onset times before asking a question. That marshalling is the price of
being able to answer the question at all.

## Understand → decide → execute

Each render stage is split the same way.

| | Understands | Decides | Executes |
|---|---|---|---|
| Tuning | `analysis.track_pitch` | `musical/theory` | `audio/tuning` |
| Timing | `analysis.detect_onsets` | `musical/groove`, `salience` | `align/` |
| Arrangement | `arrange/structure` | `arrange/plan` | `arrange/layers` |
| Mix | `analysis/*` DNA | `musical/energy` | `audio/mixer` |
| Master | `audio/master` metering | `config.GenreProfile` | `audio/master` |

The middle column never touches audio and the right column never makes a
musical choice. When a render sounds wrong, that split says which of the
two to look at.

## What crosses a process boundary

**The engine ↔ the browser.** JSON over HTTP, files on disk. The browser
never computes anything the engine will later rely on — except one thing.

**The one exception: the recorder's live meters.** A cue that arrives 300 ms
after the fault it describes is worse than no cue, so level, clipping,
proximity, loudness and pitch are computed in an AudioWorklet on the
browser's audio thread. Those thresholds are a deliberate *mirror* of
`capture/coach.py`, and both files say so. Python remains the source of
truth: the post-take report, which is what a take is actually judged on, is
computed there from the recorded audio. The live copy exists so that what is
said during a take agrees with what is said after it.

Both sides are tested against the same kind of fixture — a signal whose
correct answer is known analytically. `web/test.html` does it for the
browser; `tests/` does it for Python.

## Caching

Beat analysis is expensive (separation, beat tracking, chord estimation,
groove extraction) and completely deterministic for a given file and
analyser version. It is cached by **content hash**, salted with
`DNA_SCHEMA_VERSION`. Each document also records the analysis backends
that were available when it was made (`analysis_backends`: rhythm,
pitch, separation), and a lookup re-analyses when this machine now has
a better one (`capabilities.improvement_over`), while keeping a document
made with better backends elsewhere.

Content-addressed rather than by filename, for two reasons: renaming or
re-uploading the same audio costs nothing, and editing a file in place
actually re-analyses it. The CLI and the service once used different
schemes, and the same beat imported both ways appeared twice in every
catalog listing and twice in every match. There is now one implementation,
in `core/audio_io.content_key`.

## Failure policy

**The pipeline never raises for a musical reason.** A low-confidence result
ships with an honest label rather than failing, because a labelled B-grade
render is more useful than an error message.

Every stage that can decline does so explicitly and says why in its report:

```json
{"applied": false,
 "note": "decay estimates disagree too much to act on (spread 1.16 across 85 measurements)"}
```

Unknown is distinguished from zero everywhere, all the way to the interface,
which renders an em dash rather than a plausible-looking `0.0`.

## Jobs

Renders take tens of seconds to minutes. They run on a worker thread with
stage-level progress, and the client polls with a backoff. The job record
outlives the connection that started it.

There is no queue and no persistence: this is a local tool, and a job that
does not survive a restart is an acceptable trade for not requiring Redis.
Introducing a second concurrent user is the point at which that stops being
true.

## The critic loop

Gates run on the **rendered audio**, not on the parameters that produced it,
so drift accumulated during processing is caught. Each failing gate carries
a deterministic repair; overrides accumulate across attempts; the
best-scoring attempt is kept.

Measuring the output rather than the settings is what caught the sync gate
measuring against a grid the beat itself did not follow — the gate scored a
beat against *itself* at 36 ms, and a vocal that passed it would have been
off the beat.

## Adding a stage

1. Put the measurement in `analysis/`, returning a value **and a confidence**.
2. Put the decision in `musical/` or `arrange/`, as pure logic.
3. Put the processing in `audio/`, taking the decision as an argument.
4. Have it report what it did, including when it declined and why.
5. Add a gate to `audio/critic.py` only if the failure is detectable in the
   finished audio.
6. Write the test against a fixture whose correct answer is known in
   advance, not against the output the code currently produces.

Step 6 is the one that matters. Most of the defects found in this codebase
were found by a test that knew the right answer independently — a beat
measured against its own grid, an impulse response with a stated RT60, a
waveform built to overshoot by exactly 3.01 dB.
