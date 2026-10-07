# The voice path: what is built, what is not, and what has to be decided

The aim: take a person's voice and words, in English, Hindi or Punjabi, and
produce a vocal that is already studio-ready for the beat it will sit on, so
that mixing and mastering are the easy part.

## The chain

```
take / lyrics ─► understand ─► PLAN ─► SCORE ─► 1 carrier ─► 2 voice transfer ─► mix ─► master
                 (built)       (built)  (built)  (not built)   (not built)       (built) (built)
```

1. **Carrier**: a model that sings or speaks the score in *any* licensed voice,
   with correct timing and pitch.
2. **Voice transfer**: converts the carrier's voice to the person's. It keeps
   pitch and timing, so everything depends on stage 1 being right.

No single released model does both. They are chained so that each one's
weakness is irrelevant: the converter cannot fix timing, and the carrier
cannot sing in a stranger's voice.

## Built and verified in this repository

| Piece | Where | Verified by |
| --- | --- | --- |
| Per-note velocity and vibrato (rate, depth) | `analysis.annotate_expression` | synthetic notes with known vibrato; 3 real takes |
| Language: declared, detected or defaulted, never forced | `analysis/lyrics.py` | Hindi speech, 2 English raps, stubbed model tests |
| Language question and UI selector | `core/questions.py`, `web/` | real HTTP calls: declared → no question, auto Hindi → asked |
| Transcript timing verdict, gating every use of word times | `lyrics.timing_check` | see "what the measurements said" |
| Words placed on notes, only when timing is trusted | `perform/lyric_align.py` | 21 tests, mutation-checked |
| Score (JSON + MIDI, UTF-8 so Hindi and Punjabi survive) | `perform/score.py`, `mixengine score` | real take end to end; round trip of Devanagari and Gurmukhi |

## Not built, and why

Nothing below can be built honestly without a decision that is not mine to
make, or a model that needs its own environment (every voice model pins its
own torch; the engine runs 2.7.1 and a trial install of chatterbox wanted 2.6.0).

### Language × delivery: what exists

| | Sung | Rapped / spoken |
| --- | --- | --- |
| **English** | Synthesizer V Studio 2 Pro (commercial use included; scripting API, headless render not documented) or a voicebank commissioned and owned | speech carrier + word-level placement |
| **Hindi** | **nothing** open or commercial | Indic Parler-TTS (Apache-2.0, Hindi yes) or IndicF5 |
| **Punjabi** | **nothing** | **IndicF5 only**; Indic Parler-TTS does not list Punjabi |

- Synthesizer V sings English, Japanese, Korean, Mandarin, Cantonese and
  Spanish. Its cross-lingual feature does not extend to Hindi or Punjabi.
- IndicF5 covers 11 Indian languages including both, and clones a voice from a
  reference clip. Its terms forbid cloning a voice without permission, so a
  consent step is part of the product, not an option. It is fine-tuned from F5-TTS, whose
  weights are CC-BY-NC; whether IndicF5's MIT label holds up for a paid product
  is a question for whoever signs off licences. **Not verified.**
- No Hindi or Punjabi singing dataset was found. A voicebank means commissioning one.

**Consequence:** Hindi and Punjabi can reach rap and spoken word. Sung Hindi
or Punjabi has no carrier today and would need a commissioned voicebank
(DiffSinger or NNSVS, MIT) with a Hindi/Punjabi phoneme front end.

### Rap cannot be driven note by note

A pitch tracker segments rap coarsely: on the test take one note spans 1.5 s
and carries three words. Rap needs **word-level timing**, not a note list, which
is exactly what the transcript is least able to give (below). A rap carrier
should be handed words and target times from the grid. The engine's own
syllable-onset detector is the best source available for those times (its
alignment lands within about 12 ms of the grid), but word-to-onset matching on
rap has not been measured and is the first thing Phase 5 must prove.

## What the measurements said (and corrected)

These are in the record because several earlier conclusions were wrong.

1. **Whisper's output on rap is not stable.** The same 36 s take gave 95, 9,
   43, 6 and 12 words on five runs. Seeding does not make it repeat. Cause:
   rap trips the compression and confidence checks, which trigger random
   re-decoding. The decode is now fixed at temperature 0.
2. **More words is not better.** Temperature 0 returns 229 words where the
   default returned 50–70, but 50–79% of them have zero duration. Word count
   was never accuracy.
3. **Word times agree with the audio only sometimes.** The share of word starts
   landing near an acoustic onset, against what random times would score:
   0.6× to 2.0× across takes and settings. Rap onsets are dense (about 6/s) so
   chance is already 32–53%. No window size or decode setting fixed it. So every
   use of word times (line starts for bar phase, phrase attribution for hook
   detection, words on notes) is gated on `timing_check`.
4. **"Restoration destroys transcription" was noise.** The earlier 80 → 6 words
   figure came from single runs of a random decoder. Measured repeatably,
   restored and raw give the same (24 vs 24 and 208 vs 218 words). The
   pre-restoration capture in `vocal_dna` is harmless but its stated reason is
   withdrawn.
5. **The spectral gap was overstated.** The phone take's 125 Hz to 4 kHz slope
   exceeds the studio stem's by about 18 dB, not 24, and the engine's own chain
   already closes the top end (4 kHz: -25.4 → -14.4 dB against -13.2 for the
   studio take). What remains is about +7.6 dB of proximity bass at 125 Hz and a
   5 dB hole at 1–2 kHz. That is a tonal-balance fix, not a reason to re-synthesise.
6. **Whisper's `chunk_length` leaks state:** after one call with it, later calls
   without it inherit it. Anyone testing window sizes in one process gets
   contaminated numbers.
7. **mido encodes MIDI lyrics as Latin-1.** Hindi and Punjabi text is accepted at
   construction and fails at save. `MidiFile(..., charset="utf-8")` is the fix;
   the outer `meta_charset()` context does not work because `save` overrides it.

## Language handling, as built

- A declared language wins and is never second-guessed.
- A confident detection (≥ 0.7) is used and marked unconfirmed.
- An unsure detection falls back to English and says so. Forcing English onto
  Hindi does not fail, it **translates**, and the word times then belong to
  nothing. Scoring candidate languages on an excerpt was tried and is worse:
  forced to Hindi, an English rap produced 70 plausible words and outscored
  the correct English decode.
- Hindi and Punjabi are always asked about, even when detected confidently,
  because they share most of their sound and the detector cannot separate
  them (summed word confidence 22.5 vs 23.3 on Hindi audio). Everything
  downstream that needs the right one (the transcript, later the voice) should
  have the person's answer, not a guess.
- Indic languages transcribe with `small`, which writes Devanagari; `base`
  writes Urdu script and mishears more.

**Unverified:** no Punjabi audio was available. The Punjabi path was tested only
by declaring it on Hindi speech. Whisper's Punjabi script (Gurmukhi vs
Shahmukhi) on real Punjabi is not confirmed.

## Phases, with the gate on each

| Phase | Work | Gate before moving on |
| --- | --- | --- |
| 0 | Done: expression in the plan, language, timing gate, score export | — |
| 1 | **By hand:** one song through carrier then voice transfer, outside the engine | Does it beat the current render when a person listens? If not, stop. |
| 2 | Decide carriers per language (below) | Licence sign-off; a Hindi/Punjabi rap sample judged by a native speaker |
| 3 | Carrier worker(s) in their own container, JSON score in, audio out | Reproduces the Phase 1 result unattended |
| 4 | Voice transfer worker (RVC v2, MIT; 3–5 min of restored reference per user) | Similarity and artefacts judged on real takes |
| 5 | Rap from text: placement of words on the grid | Onsets within the grid tolerance the aligner already enforces (≤ 12 ms) |
| 6 | Sung topline from lyrics | Genuinely open research; skip unless Phase 1–4 succeed |

### Decisions that are yours

- Whether to commission a voicebank (the only route to sung Hindi/Punjabi, and
  to a headless, owned English carrier).
- Whether IndicF5's licence chain is acceptable for a paid product.
- The consent flow for voice cloning (required by IndicF5's terms and sensible regardless).
- The supported-language list. English, Hindi and Punjabi are wired through
  intents, the CLI, the API and the UI.

## Sources

- [Synthesizer V languages](https://dreamtonics.zendesk.com/hc/en-us/articles/44278346439833-What-languages-are-supported-by-Synthesizer-V-voices)
  and [commercial use](https://dreamtonics.com/synthesizerv/)
- [Synthesizer V scripting manual](https://resource.dreamtonics.com/scripting/)
- [Indic Parler-TTS (21 languages, Apache-2.0)](https://model.aibase.com/models/details/1915693256136089601)
- [IndicF5 (11 languages including Punjabi)](https://www.aimodels.fyi/models/huggingFace/indicf5-ai4bharat)
- [Open singing datasets and their licences](https://arxiv.org/html/2409.13832v1)
- [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
