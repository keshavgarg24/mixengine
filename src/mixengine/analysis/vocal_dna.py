"""
Vocal DNA extraction.

Converts a raw upload into a symbolic representation -- notes, phrases,
timings, key -- plus a *beat requirements profile* describing what kind of
instrumental would suit it.

The symbolic conversion is the pivot of the whole design. Once the vocal is
notes-and-timings rather than a blob of samples, key detection becomes
reliable, tuning becomes surgical, and every musical decision happens in the
domain where music theory actually applies.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import analysis, lyrics
from ..core import audio_io
from ..core.capabilities import CAPS, improvement_over
from ..audio import dsp, separation, debleed
from ..config import CFG, SR, DNA_SCHEMA_VERSION, GENRE_NEIGHBOURS
from ..core.keys import Key, compatible_camelot_set

log = logging.getLogger("mixengine.vocal_dna")

# Performance type -> plausible genres, used to seed retrieval.
_GENRE_BY_PERFORMANCE: Dict[str, List[str]] = {
    "rap": ["trap", "drill", "hip_hop", "boom_bap"],
    "melodic_rap": ["trap", "melodic_trap", "hip_hop", "rnb", "drill"],
    "sung": ["rnb", "pop", "afrobeats", "soul"],
    "spoken": ["hip_hop", "lofi", "ambient"],
}


def extract(path: str,
            conditioned_out: Optional[str] = None,
            do_separation: bool = True,
            user_bpm: Optional[float] = None,
            user_key: Optional[str] = None,
            vocal_id: Optional[str] = None,
            reference_beat_path: Optional[str] = None,
            reference_beat_dna: Optional[dict] = None,
            reference_stated: bool = True) -> dict:
    """Analyse an uploaded vocal. Returns the DNA document.

    `user_bpm` / `user_key` are optional hints from the upload form. Asking
    the user one question is far cheaper and more reliable than trying to
    infer tempo from a rubato a cappella, and it is the single highest-value
    piece of optional input the product can collect.

    `reference_beat_path` is the beat the singer was recording over, when
    that is known. Supplying it turns bleed removal from an underdetermined
    separation problem into a determined subtraction, and is worth far more
    than any other optional input for a take recorded on speakers.
    `reference_beat_dna` is that beat's analysis; with it, the vocal's tempo
    is taken from the beat rather than estimated, because the singer sang to
    it. `reference_stated` is whether anyone said so: the two-step flow
    passes the beat it was given without that claim, so bleed is looked
    for -- a take recorded over speakers carries the beat, and the beat is
    known, so it is cancelled rather than guessed at by a separator -- but
    the beat's tempo is trusted as the take's only when the bleed proves
    the take was performed to it, or the person said it was.
    """
    t_start = time.time()
    vocal_id = vocal_id or os.path.splitext(os.path.basename(path))[0]
    log.info("analysing vocal: %s", vocal_id)

    y, sr, quality = audio_io.load(path, sr=SR)
    if quality.is_silent:
        return {"vocal_id": vocal_id, "status": "failed",
                "error": "audio is silent", "version": DNA_SCHEMA_VERSION}

    # ── Stage 0: de-bleed against a known beat ────────────────────────────
    # Runs before classification, because removing the beat is what makes
    # the classifier see a clean vocal rather than a mixture -- and the
    # classifier's answer decides whether a separator runs at all. When the
    # beat is known this is strictly better than separation: separation
    # guesses what the instrumental was, this subtracts the one we have.
    bleed_report = None
    if reference_beat_path and os.path.exists(reference_beat_path):
        ref, _, _ = audio_io.load(reference_beat_path, sr=sr)
        y, bleed_report = debleed.cancel(y, ref, sr)
        if not bleed_report.get("applied"):
            log.info("  debleed: %s", bleed_report.get("note", "not applied"))

    # ── Stage 1: classify, then separate only if needed ───────────────────
    kind = separation.classify_vocal_input(y, sr)
    log.info("  input type: %s (%s)", kind.kind, kind.reason)

    separated = False
    if kind.needs_separation and do_separation and separation.CAPS.can_separate:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            stems = separation.separate(path, tmp)
            vpath = stems.get("vocals")
            if vpath and os.path.exists(vpath):
                y, sr, _ = audio_io.load(vpath, sr=SR)
                separated = True
                log.info("  separated vocals from input")
            else:
                log.warning("  separation produced no vocal stem; using input as-is")
    elif kind.needs_separation:
        log.warning("  input needs separation but no backend available; "
                    "results will be degraded")

    # ── Stage 2: restoration ──────────────────────────────────────────────
    quality_post = audio_io.probe_quality(y, sr, path)
    noise_in = analysis.noise_verdict(y, sr, analysis.detect_phrases(y, sr))
    # Kept before restoration, because restoration is what the transcriber
    # cannot survive -- see `lyrics.prepare`. Held as mono at 16 kHz, so
    # the second copy costs a twelfth of the take.
    asr_audio = lyrics.prepare(y, sr) if CAPS.whisper else None
    y, restore_report = separation.condition_vocal(y, sr, quality_post,
                                                   separated=separated)
    log.info("  restoration: %s", {k: v for k, v in restore_report.items() if v})
    # Working level first, so the floor below and every threshold after
    # it are measured on the take the mixer will receive.
    y, staged_db = analysis.stage_level(y, sr)
    if staged_db:
        restore_report["gain_db"] = staged_db
        log.info("  level: %+.1f dB to the working level", staged_db)
    # The mixer's expander opens above this floor. It must be the floor of
    # the take the mixer receives -- the restored one -- not the input's:
    # a threshold set 6 dB over a noise floor that restoration has since
    # lowered by 12 sits inside the quiet syllables and chews them.
    noise_floor_db = float(dsp.noise_floor_db(y, sr))

    # What is left under the voice after restoration is the verdict the
    # person hears about: a take the chain could make clean needs no
    # question, one it could not is theirs to re-record or accept.
    phrases = analysis.detect_phrases(y, sr)
    voice = analysis.voice_presence(y, sr, phrases)
    if voice:
        log.info("  voice: %s (speech in %.0f%% of the phrases)",
                 voice["verdict"], voice["speech_in_phrases"] * 100)
    # What was said, and when. Skipped when the voice model heard no
    # voice: there are no words in an instrumental, and the transcriber
    # would spend a minute inventing some.
    words = None
    if asr_audio is not None and (not voice
                                  or voice.get("verdict") != "no_voice"):
        words = lyrics.transcribe(asr_audio, lyrics.ASR_SR)
    noise_out = analysis.noise_verdict(y, sr, phrases)
    noise = {"verdict": noise_out["verdict"],
             "snr_db": noise_out["snr_db"],
             "input_snr_db": noise_in["snr_db"],
             "input_verdict": noise_in["verdict"],
             "removed_db": round(noise_out["snr_db"] - noise_in["snr_db"], 1),
             "gap_fraction": noise_out["gap_fraction"]}
    log.info("  noise: %s (%.1f dB over the floor; was %s, %.1f dB)",
             noise["verdict"], noise["snr_db"], noise["input_verdict"],
             noise["input_snr_db"])
    if phrases and noise_out["snr_db"] < separation.GAP_GATE_SNR_DB:
        # The floor that restoration could not remove is silenced where
        # nothing was performed. Measured floor above stays pre-gate, so
        # the mixer's expander still works the short gaps inside phrases.
        y = separation.mute_between_phrases(y, sr, phrases)
        restore_report["gap_gate"] = True
        noise["gaps_muted"] = True

    if conditioned_out:
        audio_io.save(conditioned_out, y, sr)

    duration = len(y) / sr

    # ── Stage 3: symbolic analysis ────────────────────────────────────────
    pitch = analysis.track_pitch(y, sr)
    onsets = analysis.detect_onsets(y, sr)
    # Syllables are counted where the voice is, over the time it sounds.
    # Onsets the detector fires in the gaps of a noisy take are not
    # syllables, and a take that is a third silence is not a slow one.
    active_s = sum(e - s for s, e in phrases) / sr if phrases else duration
    if phrases and onsets.size:
        starts = np.array([s / sr for s, _ in phrases])
        ends = np.array([e / sr for _, e in phrases])
        inside = ((onsets[:, None] >= starts[None, :])
                  & (onsets[:, None] <= ends[None, :])).any(axis=1)
        onsets = onsets[inside]
    harmonicity = analysis.phrase_harmonicity(y, sr, phrases)
    span = analysis.performance_span(phrases, sr, duration, harmonicity)
    if span["lead_in_s"] or span["tail_s"]:
        log.info("  performance %.2f-%.2fs; lead-in %.1fs, tail %.1fs -> %s (%s)",
                 span["start_s"], span["end_s"], span["lead_in_s"],
                 span["tail_s"], span["default"], span["reason"])

    notes = pitch.notes or []
    midis = [n["midi"] for n in notes]
    durations = [n["duration"] for n in notes]

    # Key from the note histogram -- not from chroma. See analysis.py.
    key_res = analysis.detect_key_from_notes(midis, durations)
    user_key_parsed = None
    if user_key:
        from ..core.keys import parse_key
        user_key_parsed = parse_key(user_key)
        if user_key_parsed:
            key_res.key = user_key_parsed
            key_res.confidence = 0.95
            key_res.method = "user_supplied"

    # Tempo. Precedence: the beat that was actually playing while the take
    # was recorded, then a tempo the user typed in, then detection.
    tempo = analysis.estimate_vocal_tempo_detailed(onsets)
    bpm_det, bpm_conf, bpm_alts = tempo["bpm"], tempo["confidence"], tempo["alternates"]
    tempo_verification: Optional[dict] = None
    extra_warnings: List[str] = []
    if noise["verdict"] == "severe":
        extra_warnings.append(
            "background noise is nearly as loud as the voice (%.0f dB under "
            "it after restoration); it will be heard in the render -- a "
            "cleaner take is the fix" % noise["snr_db"])
    elif noise["verdict"] == "heavy":
        extra_warnings.append(
            "heavy background noise: restoration took it down but some "
            "remains under the words (%.0f dB under the voice)"
            % noise["snr_db"])
    if span["default"] == "trim":
        extra_warnings.append("lead-in cut by default: %s" % span["reason"])
    if voice and voice["verdict"] == "no_voice":
        extra_warnings.append(
            "no voice was found in this file (the voice detector hears "
            "speech in %.0f%% of its phrases); it sounds like an "
            "instrumental or a tone, not a take"
            % (voice["speech_in_phrases"] * 100))

    ref_bpm = float((reference_beat_dna or {}).get("bpm") or 0.0)
    performed_to_it = reference_stated or bool((bleed_report or {}).get("applied"))
    if ref_bpm > 0 and performed_to_it:
        assert reference_beat_dna is not None      # ref_bpm came from it
        # The singer heard this beat and sang to it. That is not an estimate
        # of the vocal's tempo, it is the vocal's tempo, and it outranks
        # anything measured from the onsets alone. The onsets are still
        # checked against the beat's own grid: if they do not fit it, the
        # take was not really performed to it, and that is worth saying.
        ref_beats = np.asarray(reference_beat_dna.get("beats") or [], dtype=np.float64)
        grid = analysis.subdivide(ref_beats, 4) if ref_beats.size >= 4 else np.zeros(0)
        bpm, bpm_conf, bpm_src = ref_bpm, 0.95, "reference_beat"
        bpm_alts = [round(ref_bpm * f, 2) for f in (0.5, 2.0)]
        if grid.size >= 4 and bpm_det > 0:
            v_bpm, v_src, v_err = analysis.verify_tempo(onsets, bpm_det, grid)
            tempo_verification = {"detected_bpm": round(bpm_det, 2),
                                  "verified_bpm": round(v_bpm, 2),
                                  "source": v_src,
                                  "grid_fit_ms": round(v_err * 1000, 1)
                                  if np.isfinite(v_err) else None}
            if np.isfinite(v_err) and v_err > 0.09:
                extra_warnings.append(
                    f"the take does not sit on the reference beat's grid "
                    f"(mean onset error {v_err * 1000:.0f} ms); it may not "
                    f"have been performed to that beat")
        if user_bpm and user_bpm > 0 and not analysis._octave_related(user_bpm, ref_bpm):
            extra_warnings.append(
                f"you entered {user_bpm:.0f} BPM but the reference beat is "
                f"{ref_bpm:.1f}; using the beat, since it is what was playing")
    elif user_bpm and user_bpm > 0:
        bpm, bpm_conf, bpm_src = float(user_bpm), 0.95, "user_supplied"
        bpm_alts = [round(bpm * f, 2) for f in (0.5, 2.0)]
    elif bpm_conf >= 0.25:
        bpm, bpm_src = bpm_det, "detected"
    else:
        bpm, bpm_src = 0.0, "unknown"
        log.info("  no stable tempo detected -- phrase-anchored placement "
                 "will be used instead of grid-locked stretching")

    performance, performance_conf, performance_reason = \
        analysis.classify_performance_ex(pitch, onsets, active_s)

    # ── Stage 4: descriptive features ─────────────────────────────────────
    voiced_f0 = pitch.f0[pitch.voiced] if pitch.voiced.size else np.array([])
    if voiced_f0.size:
        f0_lo = float(np.percentile(voiced_f0, 5))
        f0_hi = float(np.percentile(voiced_f0, 95))
        f0_med = float(np.median(voiced_f0))
    else:
        f0_lo = f0_hi = f0_med = 0.0

    phrase_levels = [dsp.rms_db(y[s:e]) for s, e in phrases] if phrases else []
    phrase_levels = [p for p in phrase_levels if np.isfinite(p)]

    active_lufs = audio_io.loudness_region_lufs(y, sr, phrases)
    sibilance = _sibilance_ratio(y, sr)
    resonances = dsp.find_resonances(y, sr, n=3)
    syllable_rate = len(onsets) / active_s if active_s > 0 else 0.0

    tuning_dev = ([abs(n["cents_dev"]) for n in notes] if notes else [])
    in_key_frac = 0.0
    if key_res.key and notes:
        from ..core.keys import notes_in_key
        in_key_frac = notes_in_key([n["pc"] for n in notes], key_res.key)

    dna = {
        "vocal_id": vocal_id,
        "source_path": path,
        "conditioned_path": conditioned_out,
        "version": DNA_SCHEMA_VERSION,
        "analyzed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "analysis_seconds": round(time.time() - t_start, 1),
        "status": "ok",
        # What was available to make this document; see beat_dna for why
        # it is that and not what happened to run.
        "analysis_backends": CAPS.analysis_backends(
            separated=bool(kind.needs_separation and do_separation
                           and CAPS.can_separate)),

        # -- Input handling ------------------------------------------------
        "input_type": kind.kind,
        "input_type_confidence": round(float(kind.confidence), 3),
        "was_separated": bool(kind.needs_separation and do_separation),
        "needs_separation": bool(kind.needs_separation),
        "restoration": restore_report,
        "debleed": bleed_report,

        # -- Musical ------------------------------------------------------
        "key": key_res.key.to_dict() if key_res.key else None,
        "key_confidence": round(float(key_res.confidence), 3),
        "key_candidates": key_res.candidates[:5],
        "key_method": key_res.method,
        "in_key_fraction": round(float(in_key_frac), 3),

        "bpm": round(float(bpm), 2),
        "bpm_source": bpm_src,
        "bpm_detected": bpm_det,
        "bpm_confidence": round(float(bpm_conf), 3),
        "bpm_alternates": bpm_alts,
        "bpm_estimators": {"histogram": tempo.get("histogram_bpm"),
                           "autocorrelation": tempo.get("autocorr_bpm"),
                           "agreement": tempo.get("agreement")},
        "bpm_verification": tempo_verification,

        "performance_type": performance,
        "performance_confidence": round(float(performance_conf), 2),
        "performance_reason": performance_reason,
        "syllable_rate": round(float(syllable_rate), 2),
        "performance_span": span,
        "noise": noise,
        "voice": voice,
        "lyrics": words,
        "intelligibility": lyrics.intelligibility(words),

        # -- Symbolic -----------------------------------------------------
        "duration_s": round(duration, 2),
        "n_notes": len(notes),
        "notes": notes[:600],
        "pitch_method": pitch.method,
        "phrases": [[int(s), int(e)] for s, e in phrases],
        "phrases_s": [[round(s / sr, 3), round(e / sr, 3)] for s, e in phrases],
        "n_phrases": len(phrases),
        "onsets_s": [round(float(o), 4) for o in onsets[:800]],

        # -- Voice character ----------------------------------------------
        "f0_low_hz": round(f0_lo, 1),
        "f0_high_hz": round(f0_hi, 1),
        "f0_median_hz": round(f0_med, 1),
        "range_low_note": _hz_to_note(f0_lo),
        "range_high_note": _hz_to_note(f0_hi),
        "voice_type": _voice_type(f0_med),
        "vibrato_mean_cents": (round(float(np.mean([n["vibrato"] for n in notes])), 1)
                               if notes else 0.0),
        "tuning_deviation_cents": (round(float(np.median(tuning_dev)), 1)
                                   if tuning_dev else 0.0),

        # -- Mix-relevant measurements -------------------------------------
        "active_lufs": (round(float(active_lufs), 2)
                        if np.isfinite(active_lufs) else None),
        "phrase_level_spread_db": (round(float(np.std(phrase_levels)), 2)
                                   if len(phrase_levels) > 1 else 0.0),
        "sibilance_ratio": round(float(sibilance), 4),
        "resonances": [[round(f, 1), round(e, 2)] for f, e in resonances],
        "noise_floor_db": round(noise_floor_db, 2),
        "quality": quality_post.to_dict(),
        "repairs": list(quality.repairs),
        "warnings": list(quality_post.warnings) + extra_warnings,
    }

    dna["beat_requirements"] = build_requirements(dna)
    dna["summary"] = human_summary(dna)

    log.info("  %s | %s | %.0f BPM (%s) | %d notes | %d phrases | %.1fs "
             "(analysed in %.0fs)",
             vocal_id, dna["key"]["name"] if dna["key"] else "key unknown",
             bpm, bpm_src, len(notes), len(phrases), dna["duration_s"],
             dna["analysis_seconds"])
    return dna


# ─────────────────────────────────────────────────────────────────────────────
# Beat requirements profile
# ─────────────────────────────────────────────────────────────────────────────

def build_requirements(dna: dict) -> dict:
    """Translate the vocal's properties into a beat specification.

    This is the object the retrieval layer queries with, and it is also
    what gets shown to the user before anything is rendered -- the moment
    the product demonstrates it understood the upload.
    """
    from ..core.keys import Key as K

    key = K.from_dict(dna.get("key"))
    bpm = float(dna.get("bpm") or 0.0)
    performance = dna.get("performance_type", "sung")

    # Tempo windows. If tempo is unknown we cannot filter on it at all, and
    # say so rather than inventing a range.
    tempo_range: Optional[List[float]]
    half: Optional[List[float]]
    double: Optional[List[float]]
    if bpm > 0:
        w = CFG.match.tempo_windows[0]
        tempo_range = [round(bpm * (1 - w), 1), round(bpm * (1 + w), 1)]
        half = [round(bpm * 0.5 * (1 - w), 1), round(bpm * 0.5 * (1 + w), 1)]
        double = [round(bpm * 2 * (1 - w), 1), round(bpm * 2 * (1 + w), 1)]
    else:
        tempo_range = half = double = None

    compatible = compatible_camelot_set(key, max_shift=0) if key else []
    compatible_shifted = (compatible_camelot_set(key, max_shift=1) if key else [])

    genres = _GENRE_BY_PERFORMANCE.get(performance, ["pop", "rnb"])
    expanded: List[str] = list(genres)
    for g in genres:
        expanded.extend(GENRE_NEIGHBOURS.get(g, []))
    expanded = list(dict.fromkeys(expanded))   # ordered de-duplication

    # Denser vocals need more space in the midrange.
    rate = float(dna.get("syllable_rate") or 0.0)
    min_pocket = 0.62 if rate > 3.5 else (0.52 if rate > 2.0 else 0.42)

    return {
        "tempo_range": tempo_range,
        "tempo_range_halftime": half,
        "tempo_range_doubletime": double,
        "tempo_known": bpm > 0,
        "key": key.name if key else None,
        "camelot": key.camelot if key else None,
        "compatible_camelot": compatible,
        "compatible_camelot_with_shift": compatible_shifted,
        "compatible_key_names": ([Key(*_camelot_to_key(c)).name
                                  for c in compatible] if key else []),
        "genres": expanded[:8],
        "min_pocket_score": min_pocket,
        "needs_pocket_hz": [220, 4000],
        "avoid": (["dense_vocal_chops"] if rate > 3.0 else []),
        "prefer_grid_stability": 0.8 if performance in ("rap", "melodic_rap") else 0.0,
    }


def human_summary(dna: dict) -> str:
    """One-paragraph, user-facing description of the analysis."""
    key = dna.get("key")
    req = dna.get("beat_requirements", {}) or {}
    parts = []

    parts.append(key["name"] if key else "key undetermined")
    if dna.get("bpm"):
        src = dna.get("bpm_source")
        suffix = " (you told us)" if src == "user_supplied" else ""
        parts.append(f"~{dna['bpm']:.0f} BPM{suffix}")
    else:
        parts.append("no stable tempo")
    parts.append(dna.get("performance_type", "vocal").replace("_", " "))
    parts.append(dna.get("voice_type", ""))

    head = " · ".join(p for p in parts if p)
    genres = ", ".join(req.get("genres", [])[:3])
    tr = req.get("tempo_range")
    tempo_txt = f" at {tr[0]:.0f}-{tr[1]:.0f} BPM" if tr else ""
    keys = ", ".join(req.get("compatible_key_names", [])[:3])

    body = f"Best fit: {genres}{tempo_txt}"
    if keys:
        body += f" in {keys}"
    return f"{head}. {body}."


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sibilance_ratio(y: np.ndarray, sr: int) -> float:
    """Energy in the 5-9.5 kHz sibilance band relative to total.

    Sets the de-esser threshold. A bright female vocal and a dark male
    vocal need very different settings, and this is how the chain finds
    out which it has without being told.
    """
    f, mag = dsp.long_term_spectrum(y, sr)
    if f.size == 0:
        return 0.0
    lin = 10.0 ** (mag / 20.0)
    total = float(np.sum(lin)) + 1e-12
    band = (f >= 5000) & (f <= 9500)
    return float(np.sum(lin[band]) / total)


def _hz_to_note(hz: float) -> str:
    if hz <= 0:
        return ""
    from ..config import NOTE_NAMES
    midi = 69 + 12 * np.log2(hz / 440.0)
    return f"{NOTE_NAMES[int(round(midi)) % 12]}{int(round(midi)) // 12 - 1}"


def _voice_type(f0_median: float) -> str:
    if f0_median <= 0:
        return ""
    if f0_median < 130:
        return "male_bass"
    if f0_median < 175:
        return "male_baritone"
    if f0_median < 220:
        return "male_tenor"
    if f0_median < 280:
        return "female_alto"
    return "female_soprano"


def _camelot_to_key(code: str) -> Tuple[int, str]:
    from ..core.keys import _CAMELOT_REVERSE
    k = _CAMELOT_REVERSE.get(code)
    return (k.pc, k.mode) if k else (0, "major")


def save(dna: dict, out_dir: str) -> str:
    path = os.path.join(out_dir, f"{dna['vocal_id']}.json")
    return audio_io.write_json(path, dna)


def can_improve(doc: Optional[dict],
                now: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Why a cached vocal analysis is worth redoing on this machine, or None.

    Separation counts only for a take that needed it: a clean a cappella
    was never going to be separated, so a newly installed separator is no
    reason to analyse it again.
    """
    if not doc or doc.get("status") != "ok":
        return None
    # The render reads the restored take from this path. An analysis made
    # before it was written renders the noisy original.
    cond = doc.get("conditioned_path")
    if not cond or not os.path.exists(cond):
        return "restored take is not on disk"
    # The noise verdict and the performance span decide what the person
    # is asked before a render; a document without them cannot ask.
    if "noise" not in doc or "performance_span" not in doc:
        return "analysed before the take was judged for noise and lead-in"
    if CAPS.silero_vad and "voice" not in doc:
        return "analysed before the take was checked for a voice"
    # A take with a voice in it and no transcript was analysed before a
    # transcriber was installed. The hook, the bar phase and the
    # intelligibility check all read the transcript.
    if CAPS.whisper and "lyrics" not in doc \
            and (doc.get("voice") or {}).get("verdict") != "no_voice":
        return "analysed before the words were transcribed"
    return improvement_over(doc.get("analysis_backends"),
                            want_separation=bool(doc.get("needs_separation")),
                            now=now)
