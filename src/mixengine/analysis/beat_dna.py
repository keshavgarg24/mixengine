"""
Beat DNA extraction -- the offline catalog analysis.

Runs once per beat, at producer-upload time, and is cached forever. The
output document mirrors the `audio_dna` sub-document in the beats
collection so it can be written straight into Mongo.

Producer metadata (`tempo`, `key`, `genre`) is used as a *prior*, never as
ground truth: tags are frequently half/double time or refer to the key of a
sample rather than the beat. Both the tagged and detected values are
stored, plus a `verified` flag, so tagging accuracy becomes measurable.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, Optional

import numpy as np

from . import analysis, genre
from ..core import audio_io
from ..core.capabilities import CAPS, improvement_over
from ..audio import dsp, separation, space
from ..config import SR, DNA_SCHEMA_VERSION
from ..core.keys import parse_key

log = logging.getLogger("mixengine.beat_dna")


def extract(path: str,
            metadata: Optional[dict] = None,
            stems_dir: Optional[str] = None,
            do_separation: bool = True,
            beat_id: Optional[str] = None) -> dict:
    """Analyse one beat. Returns the full DNA document.

    `metadata` is the Mongo document (or any dict with `tempo`, `key`,
    `genre`, `mood`, `_id`, ...). Missing metadata is fine -- everything is
    detected from audio.
    """
    t_start = time.time()
    metadata = metadata or {}
    beat_id = beat_id or _derive_id(path, metadata)

    log.info("analysing beat: %s", beat_id)

    y, sr, quality = audio_io.load(path, sr=SR)
    if quality.is_silent:
        return _failed(beat_id, path, "audio is silent", metadata)

    # -- Priors from producer metadata -------------------------------------
    bpm_tag = _num(metadata.get("tempo"))
    key_tag = parse_key(metadata.get("key"))
    genre_tag = metadata.get("genre")

    # -- Rhythm ------------------------------------------------------------
    rhythm = analysis.analyze_rhythm(y, sr, bpm_hint=bpm_tag)

    # -- Harmony -----------------------------------------------------------
    atonal = analysis.is_atonal(y, sr)
    if atonal:
        key_res = analysis.KeyResult(method="atonal")
        chords = []
    else:
        key_res = analysis.detect_key_audio(y, sr, hint=key_tag)
        chords = analysis.estimate_chords(y, sr, rhythm.downbeats, key_res.key)

    # -- Groove ------------------------------------------------------------
    # Measured once per beat and cached with the DNA, because it is the
    # target every vocal placed on this beat will be aligned to. A vocal
    # landing on these positions is in the pocket; one landing on a
    # mathematically exact grid is merely on time, which is not the same
    # thing and is audibly stiffer.
    from ..audio.timing import extract_beat_groove
    beat_onsets = analysis.detect_onsets(y, sr)
    groove = extract_beat_groove(beat_onsets, rhythm.beats, rhythm.downbeats,
                                 bpm=rhythm.bpm,
                                 beats_per_bar=rhythm.beats_per_bar)

    # -- Structure ---------------------------------------------------------
    sections = analysis.analyze_structure(y, sr, rhythm.beats, rhythm.downbeats)

    # -- Mix character -----------------------------------------------------
    pocket = analysis.pocket_score(y, sr)
    pocket_sections = analysis.pocket_by_section(y, sr, sections)
    spectral = analysis.spectral_profile(y, sr)
    has_vocals, vocal_ratio = analysis.detect_vocal_content(y, sr)
    lufs = audio_io.integrated_lufs(y, sr)

    # -- Acoustic space ----------------------------------------------------
    # Measured once and cached, because it is the target every vocal placed
    # on this beat gets matched into. A vocal recorded in a different room
    # from the one the beat was produced in reads as pasted on top however
    # well the levels are set.
    space_profile = space.estimate(y, sr, onsets=beat_onsets)

    # -- Genre -------------------------------------------------------------
    # The producer's tag wins when there is one; otherwise this is what
    # makes the per-genre mix profile engage at all. Without it an untagged
    # beat fell through to the default profile and every genre-specific
    # decision downstream -- loudness target, vocal ratio, ducking depth,
    # how hard to tune -- was unreachable.
    genre_result = genre.detect(
        y, sr, bpm=rhythm.bpm, spectral=spectral,
        dynamic_range_db=float(quality.peak_db - quality.rms_db),
        swing_ratio=float(groove.swing_ratio), onsets=beat_onsets,
        beats=rhythm.beats, tagged=genre_tag)
    effective_genre = genre_result.genre if genre_result.usable else None
    if effective_genre and not genre_tag:
        log.info("  genre: %s (%.2f confidence) -- %s", effective_genre,
                 genre_result.confidence,
                 "; ".join(genre_result.evidence[:2]) or "no strong evidence")
    elif genre_result.note:
        log.info("  genre: %s", genre_result.note)

    # -- Stems -------------------------------------------------------------
    stems: Dict[str, str] = {}
    if do_separation and stems_dir:
        target = os.path.join(stems_dir, beat_id)
        stems = separation.separate_beat_stems(path, target)

    # -- Reference curve for mastering -------------------------------------
    f, mag = dsp.long_term_spectrum(y, sr)
    step = max(1, len(f) // 96)
    ref_curve = {"freqs": [round(float(v), 1) for v in f[::step]],
                 "db": [round(float(v), 2) for v in mag[::step]]}

    # -- Verification of producer tags -------------------------------------
    bpm_verified = bool(
        bpm_tag and rhythm.bpm > 0
        and abs(rhythm.bpm - bpm_tag) / max(bpm_tag, 1e-9) < 0.04)
    key_verified = bool(key_tag and key_res.key and key_tag == key_res.key)

    # Which value the matcher should actually use. A verified tag is the
    # safest source; otherwise trust the detector when it is confident, and
    # fall back to the tag when it is not.
    if bpm_verified:
        bpm_use, bpm_src = float(bpm_tag or 0.0), "verified_tag"
    elif rhythm.confidence >= 0.55 or not bpm_tag:
        bpm_use, bpm_src = float(rhythm.bpm), "detected"
    else:
        bpm_use, bpm_src = float(bpm_tag), "tag_fallback"

    if key_verified:
        key_use, key_src = key_res.key, "verified_tag"
    elif key_res.key and key_res.confidence >= 0.5:
        key_use, key_src = key_res.key, "detected"
    elif key_tag:
        key_use, key_src = key_tag, "tag_fallback"
    else:
        key_use, key_src = key_res.key, "detected_low_confidence"

    duration = len(y) / sr
    bars = (int(duration / rhythm.bar_duration_s)
            if rhythm.bar_duration_s > 0 else 0)

    dna = {
        "beat_id": beat_id,
        "serial_number": metadata.get("serial_number"),
        "mongo_id": _mongo_id(metadata),
        "title": metadata.get("title"),
        "source_path": path,
        "version": DNA_SCHEMA_VERSION,
        "analyzed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "analysis_seconds": round(time.time() - t_start, 1),
        "status": "ok",
        # What made this document. The record is of what was *available*,
        # not what happened to run: a backend that was installed but failed
        # would otherwise leave a document that every lookup tries to
        # improve on, fails the same way, and rewrites -- forever.
        "analysis_backends": CAPS.analysis_backends(
            separated=bool(do_separation and stems_dir and CAPS.can_separate)),

        # -- Rhythm --------------------------------------------------------
        "bpm": round(bpm_use, 2),
        "bpm_source": bpm_src,
        "bpm_detected": round(float(rhythm.bpm), 2),
        "bpm_tagged": bpm_tag,
        "bpm_confidence": round(float(rhythm.confidence), 3),
        "bpm_verified": bpm_verified,
        "bpm_alternates": rhythm.alternates,
        "beats_per_bar": rhythm.beats_per_bar,
        "grid_stability": round(float(rhythm.grid_stability), 3),
        "downbeat_offset_s": (round(float(rhythm.downbeats[0]), 4)
                              if len(rhythm.downbeats) else 0.0),
        "beats": [round(float(b), 4) for b in rhythm.beats],
        "downbeats": [round(float(b), 4) for b in rhythm.downbeats],
        "rhythm_method": rhythm.method,
        "groove": groove.to_dict(),
        "swing_ratio": round(float(groove.swing_ratio), 4),
        "groove_consistency": round(float(groove.consistency), 4),

        # -- Harmony -------------------------------------------------------
        "key": key_use.to_dict() if key_use else None,
        "key_source": key_src,
        "key_detected": key_res.key.to_dict() if key_res.key else None,
        "key_tagged": key_tag.to_dict() if key_tag else None,
        "key_confidence": round(float(key_res.confidence), 3),
        "key_verified": key_verified,
        "key_candidates": key_res.candidates[:5],
        "camelot": key_use.camelot if key_use else None,
        "is_atonal": bool(atonal),
        "chords": chords[:128],

        # -- Structure -----------------------------------------------------
        "duration_s": round(duration, 2),
        "duration_bars": bars,
        "sections": sections,
        "loop_points": _loop_points(sections),

        # -- Mix character -------------------------------------------------
        "pocket_score": round(float(pocket), 3),
        "pocket_by_section": pocket_sections,
        "spectral": spectral,
        "lufs_integrated": (round(float(lufs), 2) if np.isfinite(lufs) else None),
        "dynamic_range_db": round(float(quality.peak_db - quality.rms_db), 2),
        "true_peak_db": round(float(quality.true_peak_db), 2),
        "has_vocal_content": bool(has_vocals),
        "vocal_content_ratio": vocal_ratio,
        "reference_curve": ref_curve,
        "space": space_profile.to_dict(),

        # -- Stems ---------------------------------------------------------
        "stems": stems,
        "has_stems": bool(stems),

        # -- Metadata passthrough (used by the matcher's filters) ----------
        "genre": effective_genre,
        "genre_tagged": genre_tag,
        "genre_detection": genre_result.to_dict(),
        "mood": metadata.get("mood", []),
        "instrument": metadata.get("instrument", []),
        "play_count": metadata.get("play_count", 0),
        "owner": _mongo_id({"_id": metadata.get("owner")}),
        "playlist_count": (metadata.get("playlist_membership", {}) or {}).get("count", 0),
        "price": metadata.get("price", {}),
        "active": metadata.get("active", True),
        "sale_status": metadata.get("status"),
        "preview_url": metadata.get("preview"),

        "quality": quality.to_dict(),
    }

    log.info("  %s: %.1f BPM (%s) | %s | pocket %.2f | %d sections | %.1fs",
             beat_id, dna["bpm"], bpm_src,
             key_use.name if key_use else "atonal",
             pocket, len(sections), dna["analysis_seconds"])

    if bpm_tag and not bpm_verified:
        log.warning("  %s: tagged tempo %.0f disagrees with detected %.1f",
                    beat_id, bpm_tag, rhythm.bpm)
    if key_tag and key_res.key and not key_verified:
        log.warning("  %s: tagged key %s disagrees with detected %s",
                    beat_id, key_tag.name, key_res.key.name)
    return dna


# ─────────────────────────────────────────────────────────────────────────────
# Batch
# ─────────────────────────────────────────────────────────────────────────────

def extract_catalog(beats_dir: str, out_dir: str,
                    metadata_by_file: Optional[Dict[str, dict]] = None,
                    stems_dir: Optional[str] = None,
                    do_separation: bool = True,
                    skip_existing: bool = True) -> list:
    """Analyse every audio file in `beats_dir`, writing one JSON per beat.

    Checkpointed by output existence, so an interrupted run (a Kaggle
    session timing out) resumes where it stopped rather than restarting.

    Output files are named by *content hash*, matching what the service
    layer does. They used to be named by the derived beat id, and the two
    schemes disagreed: importing one file through both the CLI and the web
    interface produced `beat.json` and `beat-f60a...json`, two entries for
    one beat in every catalog listing and two candidates in every match.
    Content addressing also means a file edited in place is re-analysed
    rather than served stale from a name that did not change.
    """
    os.makedirs(out_dir, exist_ok=True)
    files = audio_io.list_audio_files(beats_dir)
    if not files:
        log.warning("no audio files found in %s", beats_dir)
        return []

    metadata_by_file = metadata_by_file or {}
    results = []
    log.info("found %d beat files in %s", len(files), beats_dir)

    for i, path in enumerate(files, 1):
        stem = os.path.splitext(os.path.basename(path))[0]
        meta = (metadata_by_file.get(os.path.basename(path))
                or metadata_by_file.get(stem) or {})
        beat_id = _derive_id(path, meta)
        out_path = os.path.join(out_dir, f"{cache_key(path)}.json")

        if skip_existing and os.path.exists(out_path):
            existing = audio_io.read_json(out_path)
            why = None
            if existing is not None and is_current(existing):
                why = can_improve(existing,
                                  want_stems=bool(do_separation and stems_dir))
                if why is None:
                    log.info("[%d/%d] %s: cached, skipping", i, len(files), beat_id)
                    results.append(existing)
                    continue
            if why:
                log.info("[%d/%d] %s: re-analysing, %s", i, len(files), beat_id, why)

        log.info("[%d/%d] processing %s", i, len(files), os.path.basename(path))
        try:
            dna = extract(path, metadata=meta, stems_dir=stems_dir,
                          do_separation=do_separation, beat_id=beat_id)
        except Exception as e:
            log.exception("failed on %s: %s", path, e)
            dna = _failed(beat_id, path, str(e), meta)

        audio_io.write_json(out_path, dna)
        results.append(dna)

    ok = sum(1 for r in results if r.get("status") == "ok")
    log.info("catalog analysis complete: %d/%d succeeded", ok, len(results))
    return results


def is_current(doc: Optional[dict]) -> bool:
    """Whether a DNA document was produced by this analyser version.

    Bumping `DNA_SCHEMA_VERSION` is the documented way to invalidate every
    cached analysis, and it does invalidate them -- new analyses are written
    under new content keys. It did not, until this check, *hide* the old
    ones: both loaders accepted any document with `status == "ok"`, so a
    schema bump left the previous file loading alongside the new one and
    the same beat appeared twice in every catalog listing.
    """
    return bool(doc and doc.get("status") == "ok"
                and doc.get("version") == DNA_SCHEMA_VERSION)


def can_improve(doc: Optional[dict], want_stems: bool = False,
                now: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Why a cached beat analysis is worth redoing on this machine, or None.

    A document that failed is not a cache entry, and is left to the caller.
    Stems only count when they were asked for, because they are only ever
    produced on request.
    """
    if not doc or doc.get("status") != "ok":
        return None
    return improvement_over(doc.get("analysis_backends"),
                            want_separation=want_stems, now=now)


def load_catalog(dna_dir: str) -> list:
    """Load every current beat DNA document from a directory."""
    if not os.path.isdir(dna_dir):
        return []
    out = []
    stale = 0
    for f in sorted(os.listdir(dna_dir)):
        if not f.endswith(".json"):
            continue
        d = audio_io.read_json(os.path.join(dna_dir, f))
        if is_current(d):
            out.append(d)
        elif d and d.get("status") == "ok":
            stale += 1
    if stale:
        log.info("catalog: ignored %d analysis file(s) from an older schema; "
                 "re-run analyze-beats to refresh them", stale)
    return out


def to_mongo_update(dna: dict) -> dict:
    """Shape a DNA document as a `$set` payload for the beats collection.

    Large arrays (full beat grids, reference curves) are dropped -- they
    belong in object storage, not in every document the API returns.
    """
    slim = {k: v for k, v in dna.items()
            if k not in ("beats", "downbeats", "reference_curve",
                         "key_candidates", "quality", "source_path")}
    slim["grid_summary"] = {
        "n_beats": len(dna.get("beats", [])),
        "n_downbeats": len(dna.get("downbeats", [])),
        "first_downbeat_s": dna.get("downbeat_offset_s", 0.0),
    }
    return {"$set": {"audio_dna": slim}}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def cache_key(path: str) -> str:
    """Content hash of a beat file. The same key the service layer uses."""
    return audio_io.content_key(path, DNA_SCHEMA_VERSION)


def _derive_id(path: str, metadata: dict) -> str:
    if metadata.get("serial_number"):
        return f"beat-{metadata['serial_number']}"
    mid = _mongo_id(metadata)
    if mid:
        return f"beat-{mid}"
    return os.path.splitext(os.path.basename(path))[0]


def _mongo_id(metadata: dict) -> Optional[str]:
    v = (metadata or {}).get("_id")
    if isinstance(v, dict):
        return v.get("$oid")
    return str(v) if v else None


def _num(v) -> Optional[float]:
    try:
        f = float(v)
        return f if f > 0 else None
    except (TypeError, ValueError):
        return None


def _loop_points(sections: list) -> list:
    return [[s.get("start_bar", 0), s.get("end_bar", 0)]
            for s in sections
            if s.get("label") in ("verse", "chorus")
            and s.get("end_bar", 0) > s.get("start_bar", 0)]


def _failed(beat_id: str, path: str, reason: str, metadata: dict) -> dict:
    return {
        "beat_id": beat_id,
        "source_path": path,
        "version": DNA_SCHEMA_VERSION,
        "status": "failed",
        "error": reason,
        "genre": metadata.get("genre"),
        "title": metadata.get("title"),
    }
