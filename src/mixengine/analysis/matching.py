"""
Beat matching -- the retrieval layer.

Two-stage, exactly as an information-retrieval system would be built:

  Stage 1  cheap metadata filter, catalog -> ~500 candidates
  Stage 2  mashability re-ranking on precomputed features -> top N

The point of this layer is not merely convenience. By selecting pairings
that need almost no transformation, it removes the large pitch shifts and
time stretches that the renderer handles worst. The engine's hardest failure
modes never occur, because they are filtered out before rendering. A hard
signal-processing problem becomes an easy search problem.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import CFG, GENRE_NEIGHBOURS
from ..core.keys import Key, key_compatibility

log = logging.getLogger("mixengine.matching")


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Match:
    beat_id: str
    title: Optional[str] = None
    score: float = 0.0
    semitone_shift: int = 0
    tempo_ratio: float = 1.0
    tempo_interpretation: str = "direct"     # direct | halftime | doubletime
    target_bpm: float = 0.0

    sub_scores: Dict[str, float] = field(default_factory=dict)
    penalties: Dict[str, float] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    relaxation_level: int = 0
    beat_dna: Optional[dict] = None

    @property
    def is_viable(self) -> bool:
        return self.score >= CFG.match.min_viable_score

    @property
    def transform_summary(self) -> str:
        bits = []
        if self.semitone_shift:
            bits.append(f"{self.semitone_shift:+d} semitone"
                        f"{'s' if abs(self.semitone_shift) != 1 else ''}")
        pct = (self.tempo_ratio - 1.0) * 100
        if abs(pct) > 0.5:
            bits.append(f"{pct:+.1f}% tempo")
        if self.tempo_interpretation != "direct":
            bits.append(self.tempo_interpretation)
        return ", ".join(bits) if bits else "no transformation needed"

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("beat_dna", None)
        d["transform_summary"] = self.transform_summary
        d["is_viable"] = self.is_viable
        d["score_pct"] = round(self.score * 100)
        return d


@dataclass
class MatchReport:
    matches: List[Match] = field(default_factory=list)
    catalog_size: int = 0
    candidates_considered: int = 0
    relaxation_level: int = 0
    relaxation_reason: str = ""
    any_viable: bool = False
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "matches": [m.to_dict() for m in self.matches],
            "catalog_size": self.catalog_size,
            "candidates_considered": self.candidates_considered,
            "relaxation_level": self.relaxation_level,
            "relaxation_reason": self.relaxation_reason,
            "any_viable": self.any_viable,
            "message": self.message,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 -- candidate filtering
# ─────────────────────────────────────────────────────────────────────────────

def _tempo_options(vocal_bpm: float, beat_bpm: float,
                   window: float) -> List[Tuple[float, str, float]]:
    """Every metrically valid way to reconcile two tempos.

    Returns `(ratio, interpretation, error)` where `ratio` is the stretch
    factor to apply to the vocal. Half- and double-time readings are first
    class options, not fallbacks: a 140 BPM vocal sits perfectly over a
    70 BPM beat, and refusing that reading throws away good matches.
    """
    if vocal_bpm <= 0 or beat_bpm <= 0:
        return [(1.0, "unknown", 0.0)]

    out: List[Tuple[float, str, float]] = []
    for mult, name in ((1.0, "direct"), (2.0, "halftime"), (0.5, "doubletime")):
        effective = beat_bpm * mult
        ratio = effective / vocal_bpm
        err = abs(ratio - 1.0)
        if err <= window * 2.5:
            out.append((ratio, name, err))
    if not out:
        ratio = beat_bpm / vocal_bpm
        out.append((ratio, "direct", abs(ratio - 1.0)))
    out.sort(key=lambda x: x[2])
    return out


def _passes_filter(beat: dict, req: dict, level: int,
                   vocal_key: Optional[Key], vocal_bpm: float) -> bool:
    """Stage-1 metadata filter at a given relaxation level."""
    if not beat.get("active", True):
        return False
    if beat.get("sale_status") not in (None, "sale", "free"):
        return False
    if beat.get("status") != "ok":
        return False

    # -- Tempo -------------------------------------------------------------
    window = CFG.match.tempo_windows[min(level, len(CFG.match.tempo_windows) - 1)]
    if level >= 2:
        window = CFG.match.tempo_windows[-1]
    beat_bpm = float(beat.get("bpm") or 0.0)
    if vocal_bpm > 0 and beat_bpm > 0:
        opts = _tempo_options(vocal_bpm, beat_bpm, window)
        if not opts or opts[0][2] > window * 2.0:
            return False

    # -- Key ---------------------------------------------------------------
    if not beat.get("is_atonal") and vocal_key is not None:
        beat_key = Key.from_dict(beat.get("key"))
        if beat_key is not None:
            max_shift = 0 if level < 2 else (1 if level < 4 else CFG.match.max_semitone_shift)
            _, quality = key_compatibility(vocal_key, beat_key, max_shift=max_shift)
            threshold = 0.80 if level < 2 else (0.62 if level < 4 else 0.40)
            if quality < threshold:
                return False

    # -- Genre -------------------------------------------------------------
    if level < 5:
        wanted = set(req.get("genres") or [])
        if wanted:
            g = (beat.get("genre") or "").lower().replace(" ", "_")
            if level >= 1:
                for base in list(wanted):
                    wanted.update(GENRE_NEIGHBOURS.get(base, []))
            if g and g not in wanted:
                return False

    # -- Pocket ------------------------------------------------------------
    min_pocket = float(req.get("min_pocket_score") or 0.0)
    if level >= 3:
        min_pocket = min(min_pocket, 0.45)
    if level >= 4:
        min_pocket = 0.0
    if float(beat.get("pocket_score") or 0.0) < min_pocket:
        return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 -- mashability scoring
# ─────────────────────────────────────────────────────────────────────────────

def score_pair(vocal: dict, beat: dict, level: int = 0) -> Match:
    """Score one (vocal, beat) pairing and choose its best transform."""
    m = CFG.match
    vocal_key = Key.from_dict(vocal.get("key"))
    beat_key = Key.from_dict(beat.get("key"))
    vocal_bpm = float(vocal.get("bpm") or 0.0)
    beat_bpm = float(beat.get("bpm") or 0.0)
    atonal = bool(beat.get("is_atonal"))

    match = Match(beat_id=beat.get("beat_id", "?"),
                  title=beat.get("title"),
                  relaxation_level=level,
                  beat_dna=beat)

    # ── Harmonic ──────────────────────────────────────────────────────────
    if atonal:
        shift, harmonic = 0, 0.85          # drum-only fits anything
        match.reasons.append("drum-based beat - works in any key")
    elif vocal_key is None or beat_key is None:
        shift, harmonic = 0, 0.5
        match.warnings.append("key could not be determined on one side")
    else:
        shift, harmonic = key_compatibility(vocal_key, beat_key,
                                            max_shift=m.max_semitone_shift)

        # A pitch shift is an irreversible, audible transformation, and here
        # it is being justified entirely by two key estimates. When either
        # estimate is weak the shift is as likely to move the vocal away from
        # the beat as toward it -- and unlike a wrong tempo, a wrong
        # transposition cannot be recovered from later in the chain.
        #
        # This was not hypothetical: on a real render the engine detected the
        # vocal's key at 0.49 confidence and applied a 2-semitone shift on the
        # strength of it. Nothing anywhere in the scoring or planning path
        # consulted confidence at all.
        #
        # So the shift is suppressed below a confidence floor, and the
        # harmonic score is blended toward neutral in proportion to how
        # uncertain the evidence is. An uncertain key should let the other
        # scoring dimensions decide, not drive the most destructive
        # transformation in the engine.
        key_conf = min(float(vocal.get("key_confidence") or 0.0),
                       float(beat.get("key_confidence") or 0.0))
        if shift != 0 and key_conf < m.min_key_confidence_for_shift:
            match.warnings.append(
                f"key estimate is uncertain ({key_conf:.2f}); rendering without "
                f"the {shift:+d} semitone shift rather than risk transposing "
                f"the wrong way")
            shift = 0
            _, harmonic = key_compatibility(vocal_key, beat_key, max_shift=0)

        trust = float(np.clip(key_conf / max(m.min_key_confidence_for_shift, 1e-6),
                              0.0, 1.0))
        harmonic = harmonic * trust + 0.5 * (1.0 - trust)

        if shift == 0 and harmonic > 0.95:
            match.reasons.append(
                f"same key family ({beat_key.name} / {vocal_key.name})")
        elif shift == 0 and key_conf >= m.min_key_confidence_for_shift:
            match.reasons.append(f"harmonically compatible ({beat_key.name})")
        elif shift != 0:
            match.reasons.append(
                f"needs {shift:+d} semitone shift to reach {vocal_key.name}")
    match.semitone_shift = int(shift)

    # ── Tempo ─────────────────────────────────────────────────────────────
    if vocal_bpm > 0 and beat_bpm > 0:
        window = CFG.match.tempo_windows[min(level, len(CFG.match.tempo_windows) - 1)]
        opts = _tempo_options(vocal_bpm, beat_bpm, window)
        ratio, interp, err = opts[0]
        match.tempo_ratio = float(ratio)
        match.tempo_interpretation = interp
        match.target_bpm = beat_bpm * (2.0 if interp == "halftime"
                                       else 0.5 if interp == "doubletime" else 1.0)
        if err < 0.02:
            match.reasons.append(f"tempo aligns almost exactly ({beat_bpm:.0f} BPM)")
        elif err < 0.06:
            match.reasons.append(f"minor tempo adjustment ({(ratio-1)*100:+.1f}%)")
        if interp != "direct":
            match.reasons.append(f"works as {interp}")
    else:
        match.tempo_ratio = 1.0
        match.target_bpm = beat_bpm
        match.tempo_interpretation = "unknown"
        match.warnings.append("vocal tempo unknown - placement will follow phrases")

    # ── Transform cost ────────────────────────────────────────────────────
    # Deliberately weighted high: a beat needing zero transformation almost
    # always beats a theoretically better match needing 2 semitones and 8%
    # stretch, because the artifacts cost more than the theoretical gain.
    shift_cost = abs(match.semitone_shift) / max(m.max_semitone_shift, 1)
    stretch_amount = abs(match.tempo_ratio - 1.0)
    stretch_limit = m.max_stretch_ratio - 1.0          # e.g. 0.12
    stretch_cost = min(stretch_amount / stretch_limit, 1.0)
    transform = float(np.clip(1.0 - (0.55 * shift_cost + 0.45 * stretch_cost), 0, 1))

    # Past the stretch limit, artifact quality does not degrade linearly --
    # it falls off a cliff. Time-stretching a vocal by 25% is audibly
    # damaged no matter how good the algorithm, so the score must collapse
    # rather than taper, or these pairings out-rank genuinely good ones.
    if stretch_amount > stretch_limit:
        excess = (stretch_amount - stretch_limit) / max(stretch_limit, 1e-6)
        transform *= float(np.clip(1.0 - excess * 0.85, 0.05, 1.0))
        match.warnings.append(
            f"requires a {stretch_amount*100:.0f}% time stretch - "
            f"audible artifacts likely")

    if match.semitone_shift == 0 and stretch_amount < 0.02:
        match.reasons.append("no transformation needed - highest quality path")

    # ── Rhythmic ──────────────────────────────────────────────────────────
    rhythmic = _rhythmic_fit(vocal, beat, match)

    # ── Pocket / spectral ─────────────────────────────────────────────────
    pocket = float(beat.get("pocket_score") or 0.5)
    rate = float(vocal.get("syllable_rate") or 0.0)
    need = 0.62 if rate > 3.5 else 0.45
    pocket_fit = float(np.clip(pocket / max(need, 1e-6), 0.0, 1.2)) / 1.2
    if pocket > 0.7:
        match.reasons.append("plenty of space in the mids for the vocal")
    elif pocket < 0.4:
        match.warnings.append("busy midrange - vocal will need heavy carving")

    # ── Vibe / metadata ───────────────────────────────────────────────────
    vibe = _vibe_similarity(vocal, beat)
    metadata = _metadata_fit(vocal, beat, match)
    popularity = _popularity(beat)

    # ── Penalties ─────────────────────────────────────────────────────────
    penalties: Dict[str, float] = {}
    # Atonal (drum-only) beats have no harmonic content to clash with, so
    # any chord estimate on them is noise and must not be penalised.
    diss = 0.0 if atonal else _dissonance_penalty(vocal, beat, match.semitone_shift)
    if diss > 0.02:
        penalties["dissonance"] = round(diss * m.p_dissonance, 4)
        if diss > 0.25:
            match.warnings.append("sustained vocal notes clash with the chords")

    if beat.get("has_vocal_content"):
        ratio = float(beat.get("vocal_content_ratio") or 0.0)
        penalties["vocal_collision"] = round(min(ratio, 1.0) * m.p_vocal_collision, 4)
        match.warnings.append("beat contains its own vocals - they will be ducked")

    # ── Combine ───────────────────────────────────────────────────────────
    subs = {
        "harmonic": round(float(harmonic), 4),
        "transform": round(float(transform), 4),
        "rhythmic": round(float(rhythmic), 4),
        "pocket": round(float(pocket_fit), 4),
        "vibe": round(float(vibe), 4),
        "metadata": round(float(metadata), 4),
        "popularity": round(float(popularity), 4),
    }
    total = (m.w_harmonic * subs["harmonic"]
             + m.w_transform * subs["transform"]
             + m.w_rhythmic * subs["rhythmic"]
             + m.w_pocket * subs["pocket"]
             + m.w_vibe * subs["vibe"]
             + m.w_metadata * subs["metadata"]
             + m.w_popularity * subs["popularity"])
    total -= sum(penalties.values())

    match.sub_scores = subs
    match.penalties = penalties
    match.score = float(np.clip(total, 0.0, 1.0))
    return match


def _rhythmic_fit(vocal: dict, beat: dict, match: Match) -> float:
    """Syllable density against beat density, plus grid-stability needs."""
    rate = float(vocal.get("syllable_rate") or 0.0)
    bpm = float(beat.get("bpm") or 0.0)
    if rate <= 0 or bpm <= 0:
        return 0.55

    # Syllables per beat -- roughly 0.8-2.5 is comfortable across genres.
    per_beat = rate / (bpm / 60.0)
    if per_beat < 0.4 or per_beat > 4.5:
        fit = 0.35
    elif 0.8 <= per_beat <= 2.5:
        fit = 1.0
    else:
        fit = 0.7

    # Rapped vocals need a tight grid to quantise against.
    stability = float(beat.get("grid_stability") or 0.5)
    if vocal.get("performance_type") in ("rap", "melodic_rap"):
        fit *= float(np.clip(0.65 + stability * 0.35, 0.6, 1.0))
        if stability > 0.9:
            match.reasons.append("tight programmed grid suits the rap delivery")
    return float(np.clip(fit, 0.0, 1.0))


def _vibe_similarity(vocal: dict, beat: dict) -> float:
    """Cosine similarity of vibe embeddings, if both sides have them.

    Returns a neutral 0.5 when embeddings are absent so the term neither
    rewards nor punishes. Populate `embedding_vibe` (CLAP or MuQ-MuLan) to
    activate it -- this is the term that catches "right key, right tempo,
    completely wrong feel".
    """
    a = vocal.get("embedding_vibe")
    b = beat.get("embedding_vibe")
    if not a or not b:
        return 0.5
    va, vb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if va.shape != vb.shape or va.size == 0:
        return 0.5
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom <= 0:
        return 0.5
    return float(np.clip((np.dot(va, vb) / denom + 1.0) / 2.0, 0.0, 1.0))


def _metadata_fit(vocal: dict, beat: dict, match: Match) -> float:
    req = vocal.get("beat_requirements", {}) or {}
    score, n = 0.0, 0

    wanted = req.get("genres") or []
    g = (beat.get("genre") or "").lower().replace(" ", "_")
    if wanted:
        n += 1
        if g in wanted[:4]:
            score += 1.0
            match.reasons.append(f"{g} suits this delivery")
        elif g in wanted:
            score += 0.7

    vm = {x.lower() for x in (vocal.get("mood") or [])}
    bm = {x.lower() for x in (beat.get("mood") or [])}
    if vm and bm:
        n += 1
        score += 1.0 if vm & bm else 0.3

    return score / n if n else 0.6


def _popularity(beat: dict) -> float:
    """Mild logarithmic popularity prior.

    Logarithmic, not linear: a linear boost would mean the same fifty beats
    are recommended forever and the long tail never surfaces.
    """
    plays = float(beat.get("play_count") or 0.0)
    playlists = float(beat.get("playlist_count") or 0.0)
    p = math.log1p(plays) / math.log1p(5000.0)
    c = min(playlists / 4.0, 1.0)
    return float(np.clip(0.6 * p + 0.4 * c, 0.0, 1.0))


def _dissonance_penalty(vocal: dict, beat: dict, shift: int) -> float:
    """Fraction of sustained vocal note time landing on non-chord tones.

    This is the term that catches genuinely bad pairings that a global key
    comparison misses. A vocal holding a note a semitone off the underlying
    chord for two bars is exactly what a listener hears as "wrong", even
    when both tracks nominally share a key.
    """
    notes = vocal.get("notes") or []
    chords = beat.get("chords") or []
    if not notes or not chords:
        return 0.0

    chord_pcs = []
    for c in chords[:64]:
        root = (int(c.get("root", 0)) + shift) % 12
        third = 4 if c.get("quality") == "maj" else 3
        chord_pcs.append({root, (root + third) % 12, (root + 7) % 12})
    if not chord_pcs:
        return 0.0

    # Union of chord tones across the progression -- a note that is a chord
    # tone somewhere in the progression is unlikely to sound wrong.
    union: set = set()
    for s in chord_pcs:
        union |= s

    bad_time, total_time = 0.0, 0.0
    for n in notes:
        dur = float(n.get("duration", 0.0))
        if dur < 0.22:                 # short passing notes are fine anywhere
            continue
        total_time += dur
        if int(n.get("pc", 0)) % 12 not in union:
            bad_time += dur
    if total_time <= 0:
        return 0.0
    return float(np.clip(bad_time / total_time, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Top-level retrieval
# ─────────────────────────────────────────────────────────────────────────────

def find_matches(vocal_dna: dict, catalog: Sequence[dict],
                 n: Optional[int] = None,
                 enforce_diversity: bool = True) -> MatchReport:
    """Rank the catalog for a vocal, relaxing filters until enough survive.

    The relaxation ladder is ordered least-damaging-first: widen tempo,
    then adjacent genres, then allow pitch shifting, then accept busier
    beats. The level reached is reported, because consistently deep
    relaxation for a given genre or key is a catalog gap worth acting on
    commercially.
    """
    n = n or CFG.match.n_results
    req = vocal_dna.get("beat_requirements", {}) or {}
    vocal_key = Key.from_dict(vocal_dna.get("key"))
    vocal_bpm = float(vocal_dna.get("bpm") or 0.0)

    usable = [b for b in catalog if b.get("status") == "ok"]
    report = MatchReport(catalog_size=len(usable))

    if not usable:
        report.message = "No analysed beats in the catalog. Run beat DNA extraction first."
        return report

    ladder = [
        (0, "exact match"),
        (1, "widened tempo window and adjacent genres"),
        (2, "allowed 1-semitone pitch shift"),
        (3, "accepted busier beats"),
        (4, "allowed 2-semitone shift"),
        (5, "dropped genre filter"),
    ]

    candidates: List[dict] = []
    level_used, reason_used = 0, "exact match"
    for level, reason in ladder:
        candidates = [b for b in usable
                      if _passes_filter(b, req, level, vocal_key, vocal_bpm)]
        level_used, reason_used = level, reason
        if len(candidates) >= max(n * 3, 10):
            break
        if len(candidates) >= n and level >= 2:
            break

    if not candidates:
        candidates = usable
        level_used, reason_used = 6, "no filter survived - scoring whole catalog"

    report.candidates_considered = len(candidates)
    report.relaxation_level = level_used
    report.relaxation_reason = reason_used
    if level_used > 0:
        log.info("relaxation level %d (%s): %d candidates",
                 level_used, reason_used, len(candidates))

    scored = [score_pair(vocal_dna, b, level=level_used) for b in candidates]
    scored.sort(key=lambda m: m.score, reverse=True)

    final = _diversify(scored, n) if enforce_diversity else scored[:n]
    report.matches = final
    report.any_viable = any(m.is_viable for m in final)

    if not final:
        report.message = "No compatible beats found."
    elif not report.any_viable:
        report.message = (
            "No strong matches in the catalog for this vocal. The closest "
            "options are shown, but they need significant transformation - "
            "results will be rough. Consider a different vocal take or "
            "expanding the catalog in this key/tempo range.")
    else:
        best = final[0]
        n_viable = sum(1 for m in final if m.is_viable)
        report.message = (
            f"Found {n_viable} strong match{'es' if n_viable != 1 else ''}. "
            f"Best: {best.title or best.beat_id} "
            f"({best.score*100:.0f}% - {best.transform_summary}).")
        # Be honest when results only survived by dropping filters -- the
        # scores are relative to a compromised candidate pool.
        if level_used >= 4:
            report.message += (
                f" Note: filters were relaxed ({reason_used}) to find these, "
                f"so they are the best available rather than a close fit. "
                f"A larger catalog in this key and tempo range would help.")
    return report


def _diversify(scored: List[Match], n: int) -> List[Match]:
    """Enforce spread across producers and genres.

    Five near-identical beats is a worse result set than three good ones
    plus two genuinely different options, even if the five scored higher.
    """
    out: List[Match] = []
    per_producer: Dict[str, int] = {}
    per_genre: Dict[str, int] = {}
    max_genre = max(2, n // 2)

    for m in scored:
        if len(out) >= n:
            break
        dna = m.beat_dna or {}
        owner = dna.get("owner") or "unknown"
        genre = (dna.get("genre") or "unknown").lower()
        if per_producer.get(owner, 0) >= CFG.match.max_per_producer:
            continue
        if per_genre.get(genre, 0) >= max_genre:
            continue
        out.append(m)
        per_producer[owner] = per_producer.get(owner, 0) + 1
        per_genre[genre] = per_genre.get(genre, 0) + 1

    # Backfill if diversity constraints left us short.
    if len(out) < n:
        chosen = {id(m) for m in out}
        for m in scored:
            if len(out) >= n:
                break
            if id(m) not in chosen:
                out.append(m)
    return out[:n]


def explain(match: Match) -> str:
    """Human-readable justification, shown next to each recommendation.

    Showing *why* a beat matched is what makes the recommendation
    trustworthy, and it teaches users what to look for.
    """
    lines = [f"{match.score*100:.0f}% match - {match.transform_summary}"]
    for r in match.reasons[:4]:
        lines.append(f"  + {r}")
    for w in match.warnings[:3]:
        lines.append(f"  ! {w}")
    return "\n".join(lines)
