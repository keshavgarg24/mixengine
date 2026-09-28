"""
What was said, and when it was said.

Three stages of the engine were guessing at something a transcript knows
outright.

**Which phrases are the hook.** Repetition is the signal, and structure.py
finds it with chroma and pitch contour. Those describe how a phrase sounds,
which is the right feature for singing and a weak one for rap: two verses
of a rap share a key, a voice and a register, and differ only in their
words. Two phrases carrying the same words are the same material, and
nothing else has to be inferred.

**Where the bars fall.** Every other anchor the engine has -- onsets,
phrase starts, energy -- moves when the processing chain changes, which is
why one performance placed differently across its own encodings. Word
starts do not: they come from the attention alignment inside the model,
not from a level threshold, so the same performance yields the same word
times whether it arrived clipped, as an m4a, or clean.

**Whether the words survived the mix.** The critic measures whether the
vocal is loud enough. Loud and unintelligible is a different failure, and
until now nothing looked for it.

The transcript does not have to be correct to do any of this. A rap take
mis-heard as "manookantaparava" is mis-heard the same way every time it
repeats, so the repetition still reads; and the word's start time is
right even where the word is wrong. Accuracy matters for the third use
and is reported so the critic can discount a poor one.

Language detection is forced, not trusted. Left to itself the model read a
dry English rap take as Punjabi at 0.63 confidence and returned four
segments containing nothing but commas -- a silent, total failure. Sung
and rapped vowels are long and pitched and do not look like the speech the
detector was trained on, so a low-confidence guess is discarded in favour
of the configured language.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from ..audio import dsp
from ..core.capabilities import CAPS

log = logging.getLogger("mixengine.lyrics")

# The model runs on CPU through CTranslate2 (no Metal backend), so size is
# bought with time. `base` transcribes at 5-14x real time on an M-series
# CPU and `small` at 2x, for a transcript that is better in its wording and
# no better in its word times -- and the word times are what two of the
# three uses depend on. `small` is a config change, not a code change.
MODEL_SIZE = "base"
COMPUTE_TYPE = "int8"

# Below this, the detected language is discarded. See the module note.
LANGUAGE_MIN_PROB = 0.7
DEFAULT_LANGUAGE = "en"

# Whisper transcribes at 16 kHz mono.
ASR_SR = 16000

# A take quieter than this reads badly for want of level alone; anything
# above it is left exactly as it is. The target is where real takes that
# transcribe well actually sit, not the engine's mixing level.
ASR_QUIET_FLOOR_DB = -35.0
ASR_QUIET_TARGET_DB = -25.0
ASR_MAX_GAIN_DB = 40.0

# A word the model is this unsure of is kept in the transcript but not
# counted toward intelligibility, and not used as a timing anchor.
WORD_MIN_PROB = 0.4

# Two lines above this token similarity are the same material.
SAME_LINE_SIMILARITY = 0.75

_MODEL: Optional[Any] = None
_MODEL_SIZE_LOADED = ""

_PUNCT = re.compile(r"[^\w\s']+", re.UNICODE)


@dataclass
class Word:
    text: str
    start: float
    end: float
    probability: float = 0.0

    def to_dict(self) -> dict:
        return {"text": self.text, "start": round(self.start, 3),
                "end": round(self.end, 3),
                "probability": round(self.probability, 3)}


@dataclass
class Line:
    text: str
    start: float
    end: float
    words: List[Word] = field(default_factory=list)

    @property
    def tokens(self) -> List[str]:
        return normalise(self.text)

    def to_dict(self) -> dict:
        return {"text": self.text, "start": round(self.start, 3),
                "end": round(self.end, 3),
                "words": [w.to_dict() for w in self.words]}


def normalise(text: str) -> List[str]:
    """Lowercased word tokens with punctuation removed.

    The model punctuates its own guesses and the punctuation is not
    evidence of anything, so two renderings of the same line that differ
    only in a comma must compare as identical.
    """
    return _PUNCT.sub(" ", (text or "").lower()).split()


def prepare(y: np.ndarray, sr: int) -> np.ndarray:
    """The signal to transcribe, at the rate the model wants it.

    Taken before restoration, and this is the whole reason the function
    exists. De-reverb and the high-pass are tuned to make a take sit in a
    mix, and they cost the transcriber almost everything: on a real take
    the same audio yielded 80 words before `condition_vocal` and 6 after
    it. Every stage between the file and here -- bleed cancellation,
    separation, level -- shifts no sample in time, so a transcript taken
    from this signal has word times that remain true of the restored take
    the render is built from.

    Returned as mono at 16 kHz, which is both what the model consumes and
    a twelfth of the memory of keeping a second copy of the take.

    The level is left alone unless the take is far too quiet to read. The
    engine's own working level is the wrong target here: the model's
    features keep an absolute offset rather than normalising it away, so
    it is level-sensitive, and raising a take by the 3.7 dB that staging
    wanted cost two thirds of its words. Only a take quiet enough to be
    unreadable is touched, and then only up to `ASR_QUIET_TARGET_DB`.
    """
    import librosa
    mono = dsp.to_mono(y).astype(np.float32)
    if mono.size == 0:
        return mono
    if sr != ASR_SR:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=ASR_SR)
    rms = float(np.sqrt(np.mean(np.square(mono.astype(np.float64)))))
    if rms > 0.0:
        level_db = 20.0 * np.log10(rms)
        if level_db < ASR_QUIET_FLOOR_DB:
            gain = min(ASR_QUIET_TARGET_DB - level_db, ASR_MAX_GAIN_DB)
            mono = (mono * (10.0 ** (gain / 20.0))).astype(np.float32)
            peak = float(np.abs(mono).max())
            if peak > 1.0:
                mono = (mono / peak).astype(np.float32)
            log.info("  lyrics: take was %.0f dBFS; raised %+.0f dB to be "
                     "readable", level_db, gain)
    return mono


def _model(size: str = MODEL_SIZE):
    global _MODEL, _MODEL_SIZE_LOADED
    if _MODEL is None or _MODEL_SIZE_LOADED != size:
        from faster_whisper import WhisperModel
        _MODEL = WhisperModel(size, device="cpu", compute_type=COMPUTE_TYPE)
        _MODEL_SIZE_LOADED = size
    return _MODEL


def transcribe(y: np.ndarray, sr: int, *,
               language: Optional[str] = None,
               model_size: str = MODEL_SIZE) -> Optional[dict]:
    """Transcribe a take into lines and word timings.

    `language` forces a language; None means detect it and fall back to
    `DEFAULT_LANGUAGE` when the detection is not confident. Returns None
    when no transcriber is installed, so every caller keeps the behaviour
    it had before this module existed.
    """
    if not CAPS.whisper:
        return None
    try:
        import librosa
        mono = dsp.to_mono(y).astype(np.float32)
        if mono.size == 0:
            return None
        audio = (librosa.resample(mono, orig_sr=sr, target_sr=ASR_SR)
                 if sr != ASR_SR else mono).astype(np.float32)
        model = _model(model_size)
        segments, info = model.transcribe(
            audio, language=language, word_timestamps=True,
            vad_filter=False, beam_size=5,
            condition_on_previous_text=False)

        detected = getattr(info, "language", None)
        prob = float(getattr(info, "language_probability", 0.0) or 0.0)
        forced = None
        if language is None and prob < LANGUAGE_MIN_PROB:
            # Redo it in the fallback language. A sung vowel does not look
            # like the speech the detector was trained on, and its wrong
            # guess costs the whole transcript rather than a few words.
            forced = DEFAULT_LANGUAGE
            log.info("  lyrics: language read as %s at %.2f; "
                     "transcribing as %s instead", detected, prob, forced)
            segments, info = model.transcribe(
                audio, language=forced, word_timestamps=True,
                vad_filter=False, beam_size=5,
                condition_on_previous_text=False)

        lines: List[Line] = []
        for s in segments:
            words = [Word(text=w.word.strip(), start=float(w.start),
                          end=float(w.end),
                          probability=float(getattr(w, "probability", 0.0) or 0.0))
                     for w in (getattr(s, "words", None) or [])
                     if w.word and w.word.strip()]
            text = (s.text or "").strip()
            if not text and not words:
                continue
            lines.append(Line(text=text, start=float(s.start),
                              end=float(s.end), words=words))

        all_words = [w for ln in lines for w in ln.words]
        confident = [w for w in all_words if w.probability >= WORD_MIN_PROB]
        doc = {
            "model": model_size,
            "language": forced or detected,
            "language_probability": round(prob, 3),
            "language_forced": forced is not None,
            "text": " ".join(ln.text for ln in lines).strip(),
            "lines": [ln.to_dict() for ln in lines],
            "n_words": len(all_words),
            "n_confident_words": len(confident),
            "mean_word_probability": round(
                float(np.mean([w.probability for w in all_words])), 3)
            if all_words else 0.0,
        }
        log.info("  lyrics: %d words in %d lines (%s, mean confidence %.2f)",
                 doc["n_words"], len(lines), doc["language"],
                 doc["mean_word_probability"])
        return doc
    except Exception as e:                                   # noqa: BLE001
        log.warning("transcription unavailable (%s)", e)
        return None


def shift_times(doc: Optional[dict], scale: float = 1.0,
                offset_s: float = 0.0) -> Optional[dict]:
    """The same transcript, with its times moved to where the audio now is.

    The transcript is taken from the uploaded take. By the time the
    arrangement is planned the vocal has been stretched to the beat's
    tempo and moved onto a bar line, so a word's recorded time no longer
    names the sample it is at. Both of those are affine -- a scale and a
    shift -- and applying them keeps the words attached to the audio.

    The timing stages after those move individual onsets by a few tens of
    milliseconds, bounded by the audibility threshold they work to. That
    is far below the length of the phrase a word gets assigned to, so it
    is not modelled.
    """
    if not doc:
        return None
    scale = float(scale) if np.isfinite(scale) and scale > 0 else 1.0
    offset_s = float(offset_s) if np.isfinite(offset_s) else 0.0
    if abs(scale - 1.0) < 1e-6 and abs(offset_s) < 1e-6:
        return doc

    def at(t: Any) -> float:
        return round(float(t or 0.0) * scale + offset_s, 3)

    out = dict(doc)
    out["lines"] = [
        {**ln, "start": at(ln.get("start")), "end": at(ln.get("end")),
         "words": [{**w, "start": at(w.get("start")), "end": at(w.get("end"))}
                   for w in (ln.get("words") or [])]}
        for ln in (doc.get("lines") or [])]
    out["time_map"] = {"scale": round(scale, 6), "offset_s": round(offset_s, 4)}
    return out


def words_of(doc: Optional[dict], min_probability: float = WORD_MIN_PROB
             ) -> List[Word]:
    """Every word the model was reasonably sure of, in time order."""
    if not doc:
        return []
    out = [Word(text=w.get("text", ""), start=float(w.get("start") or 0.0),
                end=float(w.get("end") or 0.0),
                probability=float(w.get("probability") or 0.0))
           for ln in (doc.get("lines") or [])
           for w in (ln.get("words") or [])]
    out = [w for w in out if w.probability >= min_probability and w.text]
    out.sort(key=lambda w: w.start)
    return out


def word_onsets(doc: Optional[dict],
                min_probability: float = WORD_MIN_PROB) -> np.ndarray:
    """Word start times, in seconds.

    These are the timing anchors that do not move with the processing
    chain: they come from the model's attention alignment rather than from
    any level threshold, so a clipped take and a clean one give the same
    times for the same performance.
    """
    return np.asarray([w.start for w in words_of(doc, min_probability)],
                      dtype=np.float64)


def line_starts(doc: Optional[dict],
                min_probability: float = WORD_MIN_PROB) -> np.ndarray:
    """When each sung or rapped line begins, in seconds.

    The first confident word of each transcribed segment. This is the
    level-independent counterpart of a phrase start, and it exists
    because the phrase start is not stable: it comes from an energy
    threshold, so a take that arrived clipped, quiet or as an m4a yields
    different ones for the same performance, and every decision resting on
    them moved with the encoding. A word start comes from the model's
    attention alignment, which does not depend on level at all.

    Lines, not words: a start per word is dense enough that every bar
    phase looks equally good, and it is lines that begin on bar lines.
    """
    if not doc:
        return np.zeros(0, dtype=np.float64)
    out: List[float] = []
    for ln in (doc.get("lines") or []):
        confident = [w for w in (ln.get("words") or [])
                     if float(w.get("probability") or 0.0) >= min_probability
                     and (w.get("text") or "").strip()]
        if confident:
            out.append(float(confident[0].get("start") or 0.0))
    return np.asarray(sorted(out), dtype=np.float64)


def line_similarity(a: Sequence[str], b: Sequence[str]) -> float:
    """How far two token sequences are the same material, in [0, 1].

    Sequence ratio rather than a bag of words: "my brain my brain all the
    other things" and "all the other things my brain my brain" share every
    token and are not the same line. Short sequences are compared as
    written, because a two-word ad-lib matching another two-word ad-lib is
    exactly the repetition being looked for.
    """
    if not a or not b:
        return 0.0
    return float(SequenceMatcher(None, list(a), list(b)).ratio())


def tokens_between(doc: Optional[dict], start_s: float, end_s: float,
                   min_probability: float = WORD_MIN_PROB) -> List[str]:
    """The words spoken inside a time window, normalised.

    Phrase boundaries come from the audio and line boundaries come from
    the model, and the two do not coincide. A phrase is described by
    whatever words fall inside it.
    """
    out: List[str] = []
    for w in words_of(doc, min_probability):
        if w.start >= start_s and w.start < end_s:
            out.extend(normalise(w.text))
    return out


def phrase_texts(doc: Optional[dict],
                 phrases: Sequence[Tuple[int, int]], sr: int,
                 min_probability: float = WORD_MIN_PROB) -> List[List[str]]:
    """One token list per phrase, aligned to the phrase list given."""
    if not doc or not phrases:
        return [[] for _ in phrases]
    return [tokens_between(doc, s / float(sr), e / float(sr), min_probability)
            for s, e in phrases]


def phrase_lyrics(doc: Optional[dict],
                  phrases: Sequence[Tuple[int, int]], sr: int,
                  min_probability: float = WORD_MIN_PROB
                  ) -> List[Tuple[List[str], float]]:
    """Per phrase: its tokens, and how sure the model was of them.

    The confidence travels with the words because a caller comparing two
    phrases should weigh the comparison by it. A transcript of a take the
    model could barely read produces tokens that repeat for reasons of
    its own rather than the singer's, and a consumer that cannot tell the
    difference will hear a hook in the noise.
    """
    if not doc or not phrases:
        return [([], 0.0) for _ in phrases]
    words = words_of(doc, min_probability)
    out: List[Tuple[List[str], float]] = []
    for s, e in phrases:
        a, b = s / float(sr), e / float(sr)
        inside = [w for w in words if a <= w.start < b]
        tokens = [t for w in inside for t in normalise(w.text)]
        conf = (float(np.mean([w.probability for w in inside]))
                if inside else 0.0)
        out.append((tokens, conf))
    return out


def intelligibility(doc: Optional[dict]) -> Optional[dict]:
    """How clearly the words came through, for the critic.

    The model's own per-word probability is the measure. It is not a
    perfect proxy for whether a listener would catch the line, but it
    moves the right way for the thing being guarded against: a vocal
    buried under a beat, smeared by reverb, or crushed by the limiter
    transcribes worse than the same vocal sitting properly in the mix.
    """
    if not doc:
        return None
    n = int(doc.get("n_words") or 0)
    if n == 0:
        return {"verdict": "no_words", "mean_probability": 0.0,
                "confident_fraction": 0.0, "n_words": 0}
    confident = int(doc.get("n_confident_words") or 0)
    frac = confident / float(n)
    mean = float(doc.get("mean_word_probability") or 0.0)
    verdict = ("clear" if frac >= 0.8 else
               "muddy" if frac >= 0.5 else "unclear")
    return {"verdict": verdict, "mean_probability": round(mean, 3),
            "confident_fraction": round(frac, 3), "n_words": n}
