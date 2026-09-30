"""
Finding the song inside a vocal take.

A take arrives as continuous audio with phrase boundaries and nothing else.
Everything an arranger needs -- which phrases are the hook, which are
verses, where the bridge is, where the energy should climb -- has to be
recovered from the audio itself.

The method is repetition, which is the oldest and most reliable signal in
structural analysis of popular music: the hook is the material that comes
back. Chroma averaged over each phrase and compared phrase to phrase,
following Bartsch & Wakefield's chroma-similarity approach to refrain
finding, with pitch-contour and duration features added because two
phrases of a rap verse have near-identical chroma and are told apart by
their melodic shape rather than their harmony.

Bartsch & Wakefield aggregate chroma *per beat* rather than over a fixed
hop, which normalises for tempo so that two repeats line up frame by
frame. This does not, and `_features` still takes the `beats` it would
need: the phrase-mean chroma here is tempo-invariant for a different
reason -- it collapses the whole phrase to one vector, so nothing has to
line up -- and the melodic contour, which is the feature that actually
separates two rap verses, is already resampled to a fixed sixteen points.
Beat-synchronous frames would buy a finer comparison than either, and
changing it moves every similarity value against a threshold tuned to the
current ones, so it is a measured change rather than a free one.

The classifier is deliberately conservative. Labelling a verse as a hook
puts doubles, harmonies and a lift in the wrong place, which is a worse
outcome than treating a real hook as a verse and simply doing less.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.types import FloatSeq

from ..core.types import Section

log = logging.getLogger("mixengine.arrange.structure")

# Two phrases count as the same material above this cosine similarity.
# Set from the gap between "the same hook sung twice" and "two verses of
# the same song", which is narrower than it sounds -- verses share a key,
# a tempo and a voice, and differ mainly in melodic contour.
SIMILARITY_THRESHOLD = 0.82

# A hook must come back. One occurrence is a verse that happened to be
# catchy, and treating it as a hook puts the whole arrangement's lift in a
# place the listener never returns to.
MIN_HOOK_REPEATS = 2

# Phrases shorter than this are ad-libs or breaths, not structure.
MIN_PHRASE_S = 0.8
ADLIB_MAX_S = 1.4

# A phrase needs this many transcribed words before its words are compared
# at all. One word matching one word is a coincidence, not a repeat.
MIN_LYRIC_TOKENS = 2

# What a perfectly-transcribed lyric match is worth against chroma (0.5)
# and melodic contour (0.3). The largest single weight, because identical
# words are the one piece of evidence that is not circumstantial -- but it
# is scaled by the transcriber's own confidence before it is used, so this
# is a ceiling reached only on a take the model read cleanly.
LYRIC_WEIGHT = 0.6


@dataclass
class PhraseFeature:
    index: int
    start: float
    end: float
    chroma: np.ndarray = field(default_factory=lambda: np.zeros(12))
    contour: np.ndarray = field(default_factory=lambda: np.zeros(0))
    rms_db: float = -60.0
    median_midi: float = 0.0
    onset_rate: float = 0.0
    tokens: List[str] = field(default_factory=list)
    token_confidence: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class StructureResult:
    phrases: List[PhraseFeature] = field(default_factory=list)
    groups: List[int] = field(default_factory=list)      # group id per phrase
    labels: List[str] = field(default_factory=list)
    hook_group: int = -1
    similarity: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    method: str = "repetition"
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "n_phrases": len(self.phrases),
            "hook_group": self.hook_group,
            "labels": list(self.labels),
            "groups": [int(g) for g in self.groups],
            "hook_phrases": [i for i, l in enumerate(self.labels) if l == "hook"],
            "note": self.note,
        }


def analyze(y: np.ndarray, sr: int, phrases: Sequence[Tuple[int, int]],
            *, beats: Optional[FloatSeq] = None,
            performance_type: str = "sung",
            lyrics_doc: Optional[dict] = None) -> StructureResult:
    """Label each phrase of a vocal take. Never raises; degrades to labels.

    `lyrics_doc` is the take's transcript when one was made. Words settle
    what chroma and contour can only suggest, and they settle it best
    exactly where those two are weakest: two verses of a rap share a key,
    a register and a voice, and differ in nothing but what is said.
    """
    res = StructureResult()
    feats = _features(y, sr, phrases, beats)
    _attach_lyrics(feats, lyrics_doc, phrases, sr)
    res.phrases = feats
    if len(feats) < 2:
        res.labels = ["verse"] * len(feats)
        res.groups = list(range(len(feats)))
        res.note = "too few phrases to find structure"
        return res

    res.similarity = _similarity(feats)
    res.groups = _cluster(res.similarity)
    res.hook_group = _pick_hook(feats, res.groups, performance_type)
    res.labels = _label(feats, res.groups, res.hook_group)
    if any(f.tokens for f in feats):
        res.method = "repetition+lyrics"
    if res.hook_group < 0:
        res.note = ("no phrase repeats often enough to be a hook; "
                    "arranged as a continuous verse")
    return res


def _attach_lyrics(feats: List[PhraseFeature], doc: Optional[dict],
                   phrases: Sequence[Tuple[int, int]], sr: int) -> None:
    """Hang each phrase's words on its feature record.

    `_features` drops phrases too short to measure, so the features are
    not one-to-one with the phrase list and are matched by their own
    recorded index rather than by position.
    """
    if not doc:
        return
    try:
        from ..analysis import lyrics as lyrics_mod
        per = lyrics_mod.phrase_lyrics(doc, phrases, sr)
    except Exception as exc:                                # pragma: no cover
        log.debug("lyrics unavailable for structure: %s", exc)
        return
    for f in feats:
        if 0 <= f.index < len(per):
            f.tokens, f.token_confidence = per[f.index]


# ─────────────────────────────────────────────────────────────────────────────

def _features(y: np.ndarray, sr: int, phrases: Sequence[Tuple[int, int]],
              beats: Optional[FloatSeq]) -> List[PhraseFeature]:
    from ..audio import dsp
    mono = dsp.to_mono(dsp.as_2d(y))
    out: List[PhraseFeature] = []
    try:
        import librosa
    except ImportError:
        librosa = None  # type: ignore[assignment]

    for i, (s, e) in enumerate(phrases):
        s, e = int(max(0, s)), int(min(len(mono), e))
        if e - s < int(sr * 0.2):
            continue
        seg = mono[s:e]
        f = PhraseFeature(index=i, start=s / sr, end=e / sr)
        f.rms_db = float(dsp.rms_db(seg))
        if librosa is not None:
            try:
                c = librosa.feature.chroma_cqt(y=seg, sr=sr, hop_length=2048)
                v = c.mean(axis=1)
                n = float(np.linalg.norm(v))
                f.chroma = v / n if n > 1e-9 else v
                on = librosa.onset.onset_detect(y=seg, sr=sr, units="time")
                f.onset_rate = len(on) / max(f.duration, 1e-3)
                f0 = librosa.yin(seg, fmin=65, fmax=1000, sr=sr,
                                 frame_length=2048)
                good = np.isfinite(f0) & (f0 > 60)
                if good.any():
                    midi = 69 + 12 * np.log2(f0[good] / 440.0)
                    f.median_midi = float(np.median(midi))
                    # The melodic shape, as a fixed-length contour of pitch
                    # relative to the phrase's own centre. This replaced a
                    # spectral-centroid contour, which looked reasonable and
                    # failed in practice: every phrase of one voice has
                    # nearly the same centroid shape, and what little
                    # distinguished them was smeared by the alignment warp
                    # -- run through the full pipeline, hook detection
                    # dropped from 7/8 phrases correct to 4/7. The melody
                    # *is* the thing that repeats; compare the melody.
                    if midi.size >= 4:
                        pos = np.flatnonzero(good) / max(f0.size - 1, 1)
                        f.contour = np.interp(np.linspace(0, 1, 16),
                                              pos, midi) - f.median_midi
            except Exception as exc:                        # pragma: no cover
                log.debug("phrase features failed on %d: %s", i, exc)
        out.append(f)
    return out


def _contour_distance(a: np.ndarray, b: np.ndarray, max_shift: int = 1) -> float:
    """Mean absolute difference between two contours, the best of a few
    small slides.

    A hook sung again is the same melody with slightly different timing,
    and so is a hook after the alignment warp has moved its onsets: on
    this project's own fixture the warp shifted the first hook's contour
    by under a sixteenth of the phrase and its pointwise similarity to the
    other two repeats fell from 0.87 to 0.81, across the 0.82 threshold,
    and the song lost a hook. Letting the contours slide by one point in
    either direction -- a sixteenth of the phrase, well over any
    displacement the warp makes -- recovered both links with margin and
    moved no unrelated pair at all. The slide is never larger than that;
    beyond it the comparison would start to match different melodies.
    """
    n = int(a.size)
    best = float(np.mean(np.abs(a - b)))
    for s in range(1, max_shift + 1):
        if s >= n:
            break
        best = min(best,
                   float(np.mean(np.abs(a[s:] - b[:n - s]))),
                   float(np.mean(np.abs(a[:n - s] - b[s:]))))
    return best


def _similarity(feats: Sequence[PhraseFeature]) -> np.ndarray:
    n = len(feats)
    sim = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = feats[i], feats[j]
            parts: List[Tuple[float, float]] = []
            if a.chroma.size == 12 and b.chroma.size == 12:
                parts.append((float(np.dot(a.chroma, b.chroma)), 0.5))
            if a.contour.size and a.contour.size == b.contour.size:
                # Mean absolute difference in semitones, mapped so that the
                # same melody re-sung (under a semitone of average
                # deviation) still scores high while a genuinely different
                # line (several semitones apart most of the way) scores
                # near zero. A raw Euclidean norm was far too steep here:
                # sixteen points of one-semitone jitter -- normal for a
                # repeat that went through the warp -- gave e^-4.
                mad = _contour_distance(a.contour, b.contour)
                parts.append((float(np.exp(-mad / 1.5)), 0.3))
            # Duration and density: a hook repeat is close to the same
            # length and the same busyness as the first time it appeared.
            dur = min(a.duration, b.duration) / max(a.duration, b.duration, 1e-6)
            parts.append((float(dur), 0.1))
            rate = min(a.onset_rate, b.onset_rate) / max(
                a.onset_rate, b.onset_rate, 1e-6)
            parts.append((float(np.clip(rate, 0.0, 1.0)), 0.1))
            # What was actually said. Weighted by how sure the transcriber
            # was of these two phrases, which keeps a transcript the model
            # could barely read from inventing a hook: its tokens then
            # repeat for reasons of its own rather than the singer's, and
            # at low confidence they are worth almost nothing against the
            # chroma and contour that carry the rest of the decision.
            if (len(a.tokens) >= MIN_LYRIC_TOKENS
                    and len(b.tokens) >= MIN_LYRIC_TOKENS):
                from ..analysis.lyrics import line_similarity
                conf = min(a.token_confidence, b.token_confidence)
                w = LYRIC_WEIGHT * float(np.clip(conf, 0.0, 1.0))
                if w > 0.01:
                    parts.append((line_similarity(a.tokens, b.tokens), w))

            total_w = sum(w for _, w in parts)
            s = sum(v * w for v, w in parts) / total_w if total_w > 0 else 0.0
            sim[i, j] = sim[j, i] = float(np.clip(s, 0.0, 1.0))
    return sim


def _cluster(sim: np.ndarray, threshold: float = SIMILARITY_THRESHOLD
             ) -> List[int]:
    """Single-link clustering over the similarity matrix.

    Single-link rather than average-link on purpose: a hook that drifts
    across a song -- the last chorus sung harder and higher than the first
    -- stays one group as long as each repeat resembles the one before it.
    Average-link splits exactly that case in two.
    """
    n = sim.shape[0]
    group = list(range(n))

    def find(x):
        while group[x] != x:
            group[x] = group[group[x]]
            x = group[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                a, b = find(i), find(j)
                if a != b:
                    group[max(a, b)] = min(a, b)

    roots = [find(i) for i in range(n)]
    remap: Dict[int, int] = {}
    out = []
    for r in roots:
        if r not in remap:
            remap[r] = len(remap)
        out.append(remap[r])
    return out


def _pick_hook(feats: Sequence[PhraseFeature], groups: Sequence[int],
               performance_type: str) -> int:
    """Which repetition group is the hook.

    Repetition is necessary but not sufficient: the most repeated material
    in a rap track is often the least interesting, because a flat delivery
    makes consecutive bars look alike. So repetition is scored alongside
    the two things that actually distinguish a hook from a verse -- it sits
    higher in the voice, and it is louder.
    """
    counts: Dict[int, List[int]] = {}
    for i, g in enumerate(groups):
        counts.setdefault(g, []).append(i)

    best, best_score = -1, 0.0
    all_midi = [f.median_midi for f in feats if f.median_midi > 0]
    ref_midi = float(np.median(all_midi)) if all_midi else 0.0
    ref_db = float(np.median([f.rms_db for f in feats])) if feats else -60.0

    for g, members in counts.items():
        if len(members) < MIN_HOOK_REPEATS:
            continue
        long_enough = [i for i in members
                       if feats[i].duration >= MIN_PHRASE_S]
        if len(long_enough) < MIN_HOOK_REPEATS:
            continue

        repeats = min(len(long_enough) / 4.0, 1.0)
        midi = [feats[i].median_midi for i in long_enough
                if feats[i].median_midi > 0]
        pitch_lift = 0.0
        if midi and ref_midi > 0:
            pitch_lift = float(np.clip(
                (float(np.mean(midi)) - ref_midi) / 5.0, -1.0, 1.0))
        loud = float(np.clip(
            (float(np.mean([feats[i].rms_db for i in long_enough]))
             - ref_db) / 4.0, -1.0, 1.0))

        w_pitch = 0.20 if performance_type in ("rap",) else 0.30
        score = 0.55 * repeats + w_pitch * pitch_lift + 0.15 * loud
        if score > best_score:
            best, best_score = g, score
    return best


def _label(feats: Sequence[PhraseFeature], groups: Sequence[int],
           hook_group: int) -> List[str]:
    n = len(feats)
    labels = ["verse"] * n
    if n == 0:
        return labels

    counts: Dict[int, int] = {}
    for g in groups:
        counts[g] = counts.get(g, 0) + 1
    median_dur = float(np.median([f.duration for f in feats])) or 1.0

    for i in range(n):
        if groups[i] == hook_group and hook_group >= 0:
            labels[i] = "hook"

    # An ad-lib is short *relative to the take*, unique, and tucked against
    # a real phrase. Testing duration alone mislabels every phrase of a song
    # built from short lines -- on a take of 1.2-second phrases it turned
    # the verses and the bridge into ad-libs while the hooks, found
    # correctly by repetition, survived only because that test ran first.
    for i, f in enumerate(feats):
        if labels[i] == "hook":
            continue
        if f.duration > min(ADLIB_MAX_S, median_dur * 0.55):
            continue
        if counts.get(groups[i], 0) > 1:
            continue
        near = ((i > 0 and f.start - feats[i - 1].end < 1.0)
                or (i + 1 < n and feats[i + 1].start - f.end < 1.0))
        if near:
            labels[i] = "adlib"

    # The first phrase, if it is short and sits before any hook, is an intro
    # rather than a verse: giving it verse-level presence makes a song start
    # at full height.
    if labels[0] not in ("hook",) and feats[0].duration < median_dur * 0.7:
        labels[0] = "intro"

    # A phrase directly before a hook is the approach to it.
    for i in range(n - 1):
        if labels[i + 1] == "hook" and labels[i] == "verse":
            labels[i] = "prehook"

    # A late, unique phrase between two hooks is a bridge. The test is
    # structural -- unlike anything else and placed where a bridge goes --
    # rather than acoustic, because a bridge has no reliable sound.
    singles = [i for i in range(n)
               if sum(1 for g in groups if g == groups[i]) == 1]
    for i in singles:
        if labels[i] not in ("verse", "prehook"):
            continue
        if i < n * 0.45 or i >= n - 1:
            continue
        if "hook" in labels[:i] and "hook" in labels[i + 1:]:
            labels[i] = "bridge"
            break
    return labels


def sections_from_labels(feats: Sequence[PhraseFeature],
                         labels: Sequence[str],
                         total_s: float,
                         downbeats: Optional[FloatSeq] = None
                         ) -> List[Section]:
    """Turn per-phrase labels into contiguous sections covering the song.

    Consecutive phrases with the same label merge, gaps are absorbed by the
    section before them, and boundaries snap to the nearest downbeat when
    one is available. A section boundary that lands mid-bar makes every
    arrangement move built on it -- a lift, a drop, a filter sweep -- sound
    like a mistake rather than a decision.
    """
    if not feats:
        return []
    db = np.asarray(downbeats if downbeats is not None else [], dtype=np.float64)

    def snap(t: float) -> float:
        if db.size == 0:
            return t
        j = int(np.argmin(np.abs(db - t)))
        # Only snap when the downbeat is close enough that moving there does
        # not swallow a phrase.
        return float(db[j]) if abs(db[j] - t) < 0.6 else t

    out: List[Section] = []
    cur_label = labels[0]
    cur_start = 0.0
    for i in range(1, len(feats)):
        if labels[i] == cur_label:
            continue
        boundary = snap((feats[i - 1].end + feats[i].start) / 2.0)
        if boundary > cur_start + 0.3:
            out.append(Section(start=cur_start, end=boundary, label=cur_label))
            cur_start = boundary
        cur_label = labels[i]
    out.append(Section(start=cur_start, end=max(total_s, cur_start + 0.5),
                       label=cur_label))
    return out
