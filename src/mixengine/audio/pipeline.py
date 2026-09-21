"""
End-to-end orchestration.

    upload vocal -> Vocal DNA -> match catalog -> render variants
                 -> critic -> ranked deliverables

Every stage has a fallback and the pipeline never raises for musical
reasons: a low-confidence result ships with an honest label rather than
failing. A labelled B-grade result is always better than an error message.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import align, arrange
from ..analysis import analysis, matching, vocal_dna
from ..arrange import automation
from ..core import audio_io
from . import (critic, dsp, master, mixer, separation, space, timing,
               transform, tuning)
from ..core.capabilities import CAPS, require_render
from ..core.types import Phrase, Section
from ..musical import groove as groove_mod
from ..config import CFG, SR, VARIANTS, VariantSpec

log = logging.getLogger("mixengine.pipeline")


@dataclass
class RenderResult:
    variant: str = ""
    label: str = ""
    description: str = ""
    path: str = ""
    beat_id: str = ""
    beat_title: str = ""
    score: float = 0.0
    match_score: float = 0.0
    duration_s: float = 0.0
    render_seconds: float = 0.0
    transform: Dict = field(default_factory=dict)
    mix_report: Dict = field(default_factory=dict)
    master_report: Dict = field(default_factory=dict)
    critic_report: Dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "variant": self.variant, "label": self.label,
            "description": self.description, "path": self.path,
            "beat_id": self.beat_id, "beat_title": self.beat_title,
            "score": round(self.score, 4), "score_pct": round(self.score * 100),
            "match_score": round(self.match_score, 4),
            "duration_s": round(self.duration_s, 2),
            "render_seconds": round(self.render_seconds, 1),
            "transform": self.transform, "mix": self.mix_report,
            "master": self.master_report, "critic": self.critic_report,
            "warnings": self.warnings,
        }


# ═════════════════════════════════════════════════════════════════════════════
# Single render
# ═════════════════════════════════════════════════════════════════════════════

def render_variant(vocal_audio: np.ndarray, sr: int, vdna: dict,
                   bdna: dict, match: matching.Match,
                   variant: VariantSpec, out_path: str,
                   beat_audio: Optional[np.ndarray] = None,
                   stems: Optional[Dict[str, np.ndarray]] = None,
                   extra_overrides: Optional[dict] = None) -> RenderResult:
    """Render one (vocal, beat, variant) combination."""
    t0 = time.time()
    o = dict(variant.overrides)
    if extra_overrides:
        for k, v in extra_overrides.items():
            o[k] = o.get(k, 0.0) + v

    profile = CFG.genre(bdna.get("genre"))
    result = RenderResult(variant=variant.key, label=variant.label,
                          description=variant.description,
                          beat_id=bdna.get("beat_id", "?"),
                          beat_title=bdna.get("title") or bdna.get("beat_id", "?"),
                          match_score=match.score)
    tinfo: Dict = {}

    # ── Load the beat if not provided ─────────────────────────────────────
    if beat_audio is None:
        beat_audio, _, _ = audio_io.load(bdna["source_path"], sr=sr)
    if stems is None:
        stems = separation.load_stems(bdna.get("stems") or {}, sr=sr)

    beats_s = np.asarray(bdna.get("beats") or [], dtype=np.float64)
    downbeats_s = np.asarray(bdna.get("downbeats") or [], dtype=np.float64)

    # ── 1. Pitch ──────────────────────────────────────────────────────────
    plan = transform.plan_pitch_shift(match.semitone_shift, bool(stems))
    tinfo["pitch"] = plan
    if plan["beat_shift"]:
        if stems:
            stems = transform.shift_beat_stems(stems, sr, plan["beat_shift"])
            beat_audio = _sum_stems(stems, len(beat_audio))
        else:
            beat_audio = transform.pitch_shift(beat_audio, sr, plan["beat_shift"])
    if plan["quality"] == "incompatible":
        result.warnings.append(plan["note"])

    v = dsp.as_2d(vocal_audio)

    # ── 2. Tempo ──────────────────────────────────────────────────────────
    # The matcher's ratio comes from two tempo *estimates*, and the vocal's
    # is the less reliable of the two. Before stretching, check the ratio
    # against the vocal's actual onsets: if leaving it alone fits the beat's
    # grid better than stretching does, the detected tempo was wrong and the
    # stretch would introduce the very misalignment it claims to fix.
    ratio = float(match.tempo_ratio)
    onsets_pre = analysis.detect_onsets(v, sr)
    if abs(ratio - 1.0) > 0.002 and len(onsets_pre) >= 6:
        # Score against the beat's *measured* subdivision grid, not a grid
        # generated from its tempo. A fraction of a BPM of tracker error
        # accumulates enough phase over half a minute to invert this
        # comparison entirely.
        ref_grid = analysis.subdivide(beats_s, 4) if len(beats_s) >= 4 else np.zeros(0)
        if ref_grid.size >= 4:
            err_stretched = analysis.grid_fit_error(
                np.asarray(onsets_pre) * (1.0 / ratio), ref_grid)
            err_as_is = analysis.grid_fit_error(onsets_pre, ref_grid)
            if err_as_is < err_stretched * 0.85:
                log.info("  tempo: leaving the vocal unstretched -- it already "
                         "fits the grid better (%.0f ms vs %.0f ms stretched)",
                         err_as_is * 1000, err_stretched * 1000)
                result.warnings.append(
                    "detected vocal tempo looked wrong; rendered without a "
                    "time stretch because the onsets already fit the beat")
                ratio = 1.0
                tinfo["tempo_override"] = {
                    "reason": "onsets_fit_better_unstretched",
                    "error_as_is_ms": round(err_as_is * 1000, 1),
                    "error_stretched_ms": round(err_stretched * 1000, 1)}

    if abs(ratio - 1.0) > 0.002:
        v = transform.time_stretch(v, sr, 1.0 / ratio, preserve_formants=True)
        tinfo["time_stretch"] = {"ratio": round(1.0 / ratio, 4),
                                 "method": "tempo_ratio",
                                 "interpretation": match.tempo_interpretation}
    else:
        tinfo["time_stretch"] = {"ratio": 1.0, "method": "none"}

    # Phrase boundaries must be recomputed after stretching.
    phrases = analysis.detect_phrases(v, sr)

    # ── 3. Vocal pitch shift, if the plan calls for it ────────────────────
    if plan["vocal_shift"]:
        v = transform.pitch_shift(v, sr, plan["vocal_shift"], preserve_formants=True)

    # ── 3b. Coarse placement ──────────────────────────────────────────────
    # The whole take is shifted so its first phrase starts on a bar line
    # *before* any fine timing work. This used to run after the grid
    # alignment, which is the wrong order in a way that compounds: the
    # aligner places every onset on its absolute grid slot, and a global
    # shift of up to four seconds applied afterwards moves all of them off
    # again by exactly that amount. Coarse first, fine last -- the stage
    # with the most precise information must have the final say.
    v, align_info = transform.align_to_downbeat(v, sr, phrases, downbeats_s, beats_s)
    tinfo["alignment"] = align_info
    phrases = analysis.detect_phrases(v, sr)

    # ── 4. Tuning ─────────────────────────────────────────────────────────
    # Chord-aware rather than scale-aware. A note is judged against what is
    # actually sounding underneath it, gestures are left alone, and the
    # blues degrees are protected in the genres that depend on them.
    tune_strength = max(0.0, profile.tune_strength + float(o.get("tune_strength", 0.0)))
    perf = vdna.get("performance_type", "sung")
    if perf == "rap":
        tune_strength = 0.0            # tuning a rap vocal sounds wrong
    if tune_strength > 0.02:
        voice = vdna.get("voice") or {}
        harm_ctx = tuning.HarmonicContext.from_beat_dna(
            bdna, semitone_shift=plan["beat_shift"],
            genre=bdna.get("genre"),
            tessitura_high_midi=float(voice.get("tessitura_high_midi") or 0.0))
        notes = tuning.notes_from_dna(
            [n.to_dict() if hasattr(n, "to_dict") else n
             for n in analysis.track_pitch(v, sr).notes])
        v, tune_report = tuning.tune_musical(
            v, sr, notes, harm_ctx, base_strength=tune_strength)
        tinfo["tuning"] = tune_report
        phrases = analysis.detect_phrases(v, sr)

    # ── 5. Timing quantisation ────────────────────────────────────────────
    # The target is the beat's own measured groove, not an exact grid.
    # Correction is weighted by musical salience, so a downbeat is pulled
    # firmly and an off-beat sixteenth is barely touched -- and anything
    # below the audibility threshold is left alone entirely.
    time_ctx = timing.TimingContext.from_beat_dna(bdna)
    time_ctx.phrases = [Phrase(start=s / sr, end=e / sr,
                               start_sample=int(s), end_sample=int(e))
                        for s, e in phrases]
    q_strength = float(o.get("quantize_strength", 0.0))
    genre_strength = groove_mod.quantize_strength(
        bdna.get("genre"), perf,
        grid_consistency=float(bdna.get("grid_stability") or 0.0))
    q_strength = max(q_strength, genre_strength)
    aligned = False
    if q_strength > 0.02 and len(beats_s) > 2:
        onsets = analysis.detect_onsets(v, sr)
        # Variable-rate first. A monotonic onset-to-slot assignment plus one
        # warp handles tempo drift and keeps syllables in order, neither of
        # which independent per-onset nudges can do. It declines -- and says
        # why -- when there are too few onsets or no usable grid, and the
        # per-onset quantiser then runs as before.
        v, a_report = align.align_to_grid(
            v, sr, onsets, time_ctx, strength=q_strength, subdivision=16,
            phrases=time_ctx.phrases)
        tinfo["align"] = a_report
        aligned = bool(a_report.get("enabled"))
        if not aligned:
            v, q_report = timing.quantize_musical(
                v, sr, onsets, time_ctx, strength=q_strength, subdivision=16)
            tinfo["quantize"] = q_report
        phrases = analysis.detect_phrases(v, sr)

    # ── 6. Phrase-level fallback ──────────────────────────────────────────
    # Phrase warping is a coarser version of what step 5 just did. Running
    # both means the second one re-warps material already placed on the
    # grid, against phrase boundaries that are themselves less reliable
    # than the onsets -- so it only runs when variable-rate alignment did
    # not.
    stability = float(bdna.get("grid_stability") or 0.0)
    v, warp_info = transform.warp_phrases_to_grid(
        v, sr, phrases, downbeats_s,
        enabled=(not aligned and stability > 0.6 and len(downbeats_s) > 3))
    tinfo["warp"] = warp_info
    phrases = analysis.detect_phrases(v, sr)

    # ── 7. Fit the beat to the vocal ──────────────────────────────────────
    target_len = len(v) + int(sr * 1.5)
    beat_audio, fit_info = transform.fit_beat_to_vocal(
        beat_audio, sr, target_len, downbeats_s, bdna.get("sections"))
    tinfo["beat_fit"] = fit_info
    if stems:
        stems = {k: dsp.pad_to(s, len(beat_audio)) for k, s in stems.items()}

    n = max(len(v), len(beat_audio))
    v = dsp.pad_to(v, n)
    beat_audio = dsp.pad_to(beat_audio, n)

    # ── 8. Arrangement ────────────────────────────────────────────────────
    # Built here, after every timing and tuning stage has settled, because
    # the plan is keyed to phrase positions and those move until alignment
    # is done. Everything downstream -- the vocal chain, the layers, the
    # balance -- serves the one contour this produces.
    song_plan = arrange.build_plan(
        v, sr, phrases, genre=bdna.get("genre"), performance_type=perf,
        downbeats=downbeats_s, beats=beats_s,
        beat_sections=[Section.from_dict(s) for s in (bdna.get("sections") or [])],
        allow_layers=profile.stereo_double)
    tinfo["arrangement"] = song_plan.to_dict()
    if song_plan.structure is not None:
        log.info("  arrangement: %s", _describe_plan(song_plan))

    # ── 9. Vocal chain ────────────────────────────────────────────────────
    v_proc, vocal_chain = mixer.process_vocal(v, sr, vdna, profile, phrases, o)
    v_proc, auto_report = automation.apply_vocal(v_proc, sr, song_plan)
    vocal_chain["automation"] = auto_report

    # Place the vocal in the beat's acoustic space. Before the sends, so the
    # reverb and delay that follow are applied to a vocal that is already in
    # the right room rather than being asked to create one.
    beat_space = space.SpaceProfile(
        rt60_by_band={k: float(vv) for k, vv in
                      (bdna.get("space", {}).get("rt60_by_band") or {}).items()},
        rt60_mean=float((bdna.get("space") or {}).get("rt60_mean") or 0.0),
        drr_db=float((bdna.get("space") or {}).get("drr_db") or 0.0),
        n_decays=int((bdna.get("space") or {}).get("n_decays") or 0),
        confidence=float((bdna.get("space") or {}).get("confidence") or 0.0),
        note=str((bdna.get("space") or {}).get("note") or ""))
    v_proc, space_report = space.match(v_proc, sr, beat_space)
    vocal_chain["space"] = space_report

    # ── 10. Sends and layers ──────────────────────────────────────────────
    bpm = float(match.target_bpm or bdna.get("bpm") or 0.0)
    sends, send_info = mixer.build_sends(v_proc, sr, bpm, profile, o)

    hook = song_plan.hook_regions or _hook_regions(phrases)
    voice = vdna.get("voice") or {}
    tess = (float(voice.get("tessitura_low_midi") or 0.0),
            float(voice.get("tessitura_high_midi") or 0.0))

    # A harmony part needs the note events of the *final* vocal -- the ones
    # measured for tuning are stale, because alignment has moved everything
    # since -- and the chords underneath them. One extra pitch pass, taken
    # only when a harmony was actually planned. Rap gets no harmony at all:
    # a third above a rapped line is not a production choice anyone makes.
    wanted_layers = dict(song_plan.layer_regions)
    harm_notes, harm_chord_at, harm_key_at = None, None, None
    if "harmony" in wanted_layers:
        if perf == "rap":
            wanted_layers.pop("harmony")
            tinfo["harmony_skipped"] = "rap delivery"
        else:
            harm_ctx2 = tuning.HarmonicContext.from_beat_dna(
                bdna, semitone_shift=plan["beat_shift"], genre=bdna.get("genre"),
                tessitura_high_midi=tess[1])
            harm_notes = tuning.notes_from_dna(
                [nt.to_dict() if hasattr(nt, "to_dict") else nt
                 for nt in analysis.track_pitch(v, sr).notes])
            harm_chord_at, harm_key_at = harm_ctx2.chord_at, harm_ctx2.key_at

    built, layer_report = arrange.build_layers(
        v_proc, sr, wanted=wanted_layers, tessitura=tess,
        seed=_stable_seed(result.beat_id, variant.key),
        notes=harm_notes, chord_at=harm_chord_at, key_at=harm_key_at)
    tinfo["layers"] = layer_report
    if built:
        n_v = len(dsp.as_2d(v_proc))
        doubles = arrange.sum_layers(built, n_v)
        doubles = doubles * dsp.db_to_lin(float(o.get("double_gain_db", 0.0)))
        # Layers come up into a hook and fall away from it. Held at one
        # level through a section change they announce themselves as an
        # overdub rather than reading as part of the arrangement.
        ride = automation.layer_gain_curve(song_plan, "double", n_v, sr)
        if ride is not None:
            doubles = doubles * (10.0 ** (ride / 20.0))[:, None]
            tinfo["layers"]["ride_db"] = [round(float(ride.min()), 2),
                                          round(float(ride.max()), 2)]
    else:
        # No plan-driven layers -- either the take has no hook or the genre
        # does not use them. The flat doubler is still better than nothing
        # on a hook that was found by phrase position alone.
        doubles = mixer.build_doubles(v_proc, sr, profile, hook, o)

    # ── 10b. Transitions ──────────────────────────────────────────────────
    # The plan said where the energy steps and which devices the step
    # earns. Built from the render's own material where it can be -- the
    # vocal's last phrase reversed into the boundary, the beat's own last
    # beat swelled -- and synthesised where it cannot.
    fx_bus, beat_gain, fx_report = arrange.build_transitions(
        v_proc, beat_audio, sr, song_plan.transitions,
        bar_s=float(time_ctx.bar_duration_s or 0.0),
        beat_s=(60.0 / bpm) if bpm > 0 else 0.0,
        seed=_stable_seed(result.beat_id, variant.key, "fx"))
    tinfo["transitions"] = fx_report

    # ── 11. Beat processing ───────────────────────────────────────────────
    beat_proc, beat_chain = mixer.process_beat(stems or {}, beat_audio, v_proc,
                                               sr, profile, o)
    if fx_report.get("beat_muted_s", 0.0) > 0:
        beat_proc = beat_proc * dsp.pad_to(beat_gain[:, None], len(beat_proc))

    # ── 12. Balance ───────────────────────────────────────────────────────
    mix, balance = mixer.balance_and_sum(v_proc, beat_proc, sr, phrases,
                                         profile, sends, doubles, o, fx=fx_bus)

    # ── 13. Master ────────────────────────────────────────────────────────
    mastered, master_report = master.master(mix, sr, profile,
                                            bdna.get("reference_curve"), o)

    audio_io.save(out_path, mastered, sr)

    result.path = out_path
    result.duration_s = len(mastered) / sr
    result.transform = tinfo
    result.mix_report = {"vocal_chain": vocal_chain, "beat_chain": beat_chain,
                         "balance": balance, "sends": send_info}
    result.master_report = master_report
    result.render_seconds = time.time() - t0

    # ── 14. Critic ────────────────────────────────────────────────────────
    # The critic must see the vocal as it actually sits in the mix, which
    # means after the balance stage's gain and after mastering's makeup.
    # Handing it `v_proc` -- the vocal before `balance_and_sum` applied the
    # VIR gain -- made the `vocal_presence` gate wrong by exactly that gain,
    # which is clipped to +/-24 dB. The gate could then fail a perfectly
    # audible vocal, and its repair (`vir_db: +1.5`) would push a mix that
    # was already correct further out of balance.
    vocal_in_mix = v_proc * dsp.db_to_lin(float(balance.get("vocal_gain_db", 0.0)))
    master_makeup = _master_gain_db(master_report)
    if master_makeup:
        vocal_in_mix = vocal_in_mix * dsp.db_to_lin(master_makeup)

    cr = critic.evaluate(mastered, sr, variant.key, profile, vdna, bdna,
                         rendered_vocal=vocal_in_mix,
                         master_report=master_report,
                         semitone_shift=plan["beat_shift"])
    result.critic_report = cr.to_dict()
    result.score = cr.score
    result.warnings.extend(g.message for g in cr.errors)

    log.info("  [%s] %s -> %.0f%% (%.1fs)", variant.key, result.beat_title,
             cr.score * 100, result.render_seconds)
    return result


def _stable_seed(*parts: str) -> int:
    """A seed that survives restarts.

    `hash()` on a string is salted per process, so using it here would make
    two renders of the same take produce different doubles -- and a mix
    that cannot be reproduced cannot be debugged when someone reports that
    one of them sounded wrong.
    """
    return zlib.crc32("|".join(parts).encode("utf-8")) % 100000


def _describe_plan(plan) -> str:
    """One line describing the arrangement, for the render log."""
    st = plan.structure
    if st is None or not st.labels:
        return plan.note or "no structure"
    counts: Dict[str, int] = {}
    for lab in st.labels:
        counts[lab] = counts.get(lab, 0) + 1
    shape = ", ".join(f"{n}x {lab}" for lab, n in sorted(counts.items()))
    bits = [shape]
    if plan.contour is not None:
        bits.append(f"energy {plan.contour.contrast:.2f} contrast")
    if plan.layer_regions:
        bits.append("layers: " + ", ".join(sorted(plan.layer_regions)))
    elif plan.note:
        bits.append(plan.note)
    return " | ".join(bits)


def _master_gain_db(master_report: Optional[dict]) -> float:
    """Total gain the master chain applied, so the critic can follow the vocal.

    Read from `total_gain_db`, which the master chain reports directly.

    This function previously summed a list of individual stage keys. When
    the loudness stage was rewritten those keys were renamed, and because
    the sum simply skipped anything it could not find, it silently returned
    0.0 dB while the chain was actually applying 12 dB. The critic then
    measured the vocal 12 dB below where it really sat and could fail a
    perfectly audible mix.

    Reading one authoritative value avoids that whole failure mode: if the
    key is ever missing, the fallback is visible in the report rather than
    disguised as a plausible number.
    """
    if not master_report:
        return 0.0
    total = master_report.get("total_gain_db")
    if isinstance(total, (int, float)) and np.isfinite(total):
        return float(total)
    log.warning("master report has no total_gain_db; critic will measure the "
                "vocal without master makeup applied")
    return 0.0


def _sum_stems(stems: Dict[str, np.ndarray], n: int) -> np.ndarray:
    parts = [dsp.match_channels(dsp.pad_to(s, n), 2) for s in stems.values() if s is not None]
    if not parts:
        return np.zeros((n, 2), dtype=np.float32)
    ch = max(p.shape[1] for p in parts)
    return np.sum([dsp.match_channels(p, ch) for p in parts], axis=0).astype(np.float32)


def _is_better(candidate: RenderResult, incumbent: RenderResult) -> bool:
    """Rank two renders of the same variant.

    Clearing every hard gate always wins, regardless of score: a defective
    master is not a stylistic choice, so a passing render at 70% beats a
    failing one at 85%.
    """
    c_pass = bool(candidate.critic_report.get("passed"))
    i_pass = bool(incumbent.critic_report.get("passed"))
    if c_pass != i_pass:
        return c_pass
    return candidate.score > incumbent.score


def _hook_regions(phrases: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Guess which phrases are the hook, for doubling.

    Heuristic: the back half of the take. With Whisper transcription
    available, repeated lyric lines identify the hook properly -- that is
    the upgrade path here.
    """
    if len(phrases) < 4:
        return list(phrases)
    return list(phrases[len(phrases) // 2:])


# ═════════════════════════════════════════════════════════════════════════════
# Full pipeline
# ═════════════════════════════════════════════════════════════════════════════

def run(vocal_path: str, catalog: Sequence[dict], out_dir: str,
        n_beats: int = 3, variants_per_beat: int = 1,
        user_bpm: Optional[float] = None, user_key: Optional[str] = None,
        vdna: Optional[dict] = None,
        beat_ids: Optional[Sequence[str]] = None) -> dict:
    """Upload to finished songs.

    `n_beats` x `variants_per_beat` renders are produced. The default (3
    beats, 1 variant each) is the breadth-first configuration: it maximises
    beat discovery, which is what a marketplace wants. Set `n_beats=1,
    variants_per_beat=5` for depth-first when the user has already chosen.
    """
    # The one thing that cannot be degraded around. Everything else optional
    # is probed and worked around; without librosa and soundfile there is no
    # audio to work on, and failing here with the install line beats failing
    # forty seconds in with an AttributeError.
    require_render()
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    log.info("=" * 68)
    log.info("PIPELINE: %s", os.path.basename(vocal_path))
    log.info("=" * 68)

    # ── Stage 1: Vocal DNA ────────────────────────────────────────────────
    if vdna is None:
        conditioned = os.path.join(out_dir, "_conditioned_vocal.wav")
        vdna = vocal_dna.extract(vocal_path, conditioned_out=conditioned,
                                 user_bpm=user_bpm, user_key=user_key)
    if vdna.get("status") != "ok":
        return {"status": "failed", "error": vdna.get("error", "vocal analysis failed"),
                "vocal_dna": vdna}

    log.info("VOCAL: %s", vdna.get("summary", ""))

    # ── Stage 2: Match ────────────────────────────────────────────────────
    if beat_ids:
        chosen = [b for b in catalog if b.get("beat_id") in set(beat_ids)]
        report = matching.MatchReport(
            matches=[matching.score_pair(vdna, b) for b in chosen],
            catalog_size=len(catalog), candidates_considered=len(chosen),
            message=f"Using {len(chosen)} user-selected beat(s).")
        report.matches.sort(key=lambda m: m.score, reverse=True)
        report.any_viable = any(m.is_viable for m in report.matches)
    else:
        report = matching.find_matches(vdna, catalog, n=max(n_beats, 5))

    log.info("MATCHING: %s", report.message)
    for m in report.matches[:n_beats]:
        log.info("  %-28s %3.0f%%  %s", (m.title or m.beat_id)[:28],
                 m.score * 100, m.transform_summary)

    if not report.matches:
        return {"status": "no_matches", "vocal_dna": vdna,
                "match_report": report.to_dict(),
                "message": report.message, "renders": []}

    # ── Stage 3: Render ───────────────────────────────────────────────────
    conditioned_path = vdna.get("conditioned_path")
    src = conditioned_path if (conditioned_path and os.path.exists(conditioned_path)) \
        else vocal_path
    vocal_audio, sr, _ = audio_io.load(src, sr=SR)

    variants = VARIANTS[:variants_per_beat]
    renders: List[RenderResult] = []
    failures: List[dict] = []

    for m in report.matches[:n_beats]:
        bdna = m.beat_dna or {}
        src_path = bdna.get("source_path")
        if not src_path or not os.path.exists(src_path):
            # By far the most likely explanation for a run that matches
            # successfully and then renders nothing: the DNA was built on
            # one machine and the audio is not where it says it is. Recorded
            # explicitly so that outcome is diagnosable from result.json.
            log.warning("beat audio missing for %s (%s); skipping",
                        m.beat_id, src_path or "no source_path in DNA")
            failures.append({"beat_id": m.beat_id, "stage": "load",
                             "error": "beat audio not found",
                             "source_path": src_path})
            continue

        beat_audio, _, _ = audio_io.load(src_path, sr=SR)
        stems = separation.load_stems(bdna.get("stems") or {}, sr=SR)
        if stems:
            log.info("  using %d stems for %s", len(stems), m.beat_id)

        for variant in variants:
            name = f"{vdna['vocal_id']}__{m.beat_id}__{variant.key}.wav"
            out_path = os.path.join(out_dir, name)
            attempt_path = os.path.join(out_dir, f".repair__{name}")
            try:
                r = render_variant(vocal_audio, sr, vdna, bdna, m, variant,
                                   out_path, beat_audio=beat_audio, stems=stems)
            except Exception as e:
                log.exception("render failed for %s/%s: %s", m.beat_id, variant.key, e)
                # Record the failure instead of dropping it. The engine's
                # only recorded real run reported `"renders": []` with no
                # indication of why, because every exception was swallowed
                # here and left no trace in result.json.
                failures.append({"beat_id": m.beat_id, "variant": variant.key,
                                 "stage": "render", "error": repr(e)})
                continue

            # Repair loop: a failed gate maps to a specific parameter change.
            #
            # Two things this has to get right. Overrides **accumulate**
            # across attempts -- attempt two starts from where attempt one
            # left off, rather than discarding it and re-deriving a fix
            # from scratch. And the best render seen so far is **kept**: a
            # repair can easily make things worse, and the previous version
            # overwrote the same output path every time and never compared,
            # so a worse attempt silently replaced a better one.
            best = r
            cumulative: Dict[str, float] = {}
            attempt = 0
            while critic.should_repair(r.critic_report, attempt):
                attempt += 1
                for k, v in r.critic_report["repairs"].items():
                    cumulative[k] = cumulative.get(k, 0.0) + float(v)
                log.info("  repair attempt %d for %s: %s",
                         attempt, variant.key, cumulative)
                try:
                    r = render_variant(vocal_audio, sr, vdna, bdna, m, variant,
                                       attempt_path, beat_audio=beat_audio,
                                       stems=stems, extra_overrides=cumulative)
                except Exception as e:
                    log.warning("repair render failed: %s", e)
                    break
                if _is_better(r, best):
                    best = r

            if best.path != out_path:
                # The winner was a repair attempt: promote its file.
                try:
                    shutil.move(best.path, out_path)
                    best.path = out_path
                except OSError as e:
                    log.warning("could not promote repaired render: %s", e)
            if os.path.exists(attempt_path):
                os.remove(attempt_path)
            if attempt and best is not r:
                log.info("  keeping attempt with the best score (%.0f%%)",
                         best.score * 100)
            renders.append(best)

    renders.sort(key=lambda r: (r.critic_report.get("passed", False), r.score),
                 reverse=True)

    elapsed = time.time() - t0
    out = {
        "status": "ok" if renders else "no_renders",
        "failures": failures,
        "vocal_id": vdna["vocal_id"],
        "vocal_summary": vdna.get("summary"),
        "vocal_dna": {k: v for k, v in vdna.items()
                      if k not in ("notes", "onsets_s", "phrases", "quality")},
        "match_report": report.to_dict(),
        "renders": [r.to_dict() for r in renders],
        "total_seconds": round(elapsed, 1),
        "capabilities_tier": CAPS.tier,
    }
    audio_io.write_json(os.path.join(out_dir, "result.json"), out)

    log.info("=" * 68)
    log.info("DONE in %.0fs -- %d renders, best %.0f%%",
             elapsed, len(renders), renders[0].score * 100 if renders else 0)
    log.info("=" * 68)
    return out


def analyze_only(vocal_path: str, catalog: Sequence[dict],
                 n: int = 5, user_bpm: Optional[float] = None,
                 user_key: Optional[str] = None) -> dict:
    """Analyse and match without rendering.

    This is the fast path the product should show first: it completes in
    roughly a quarter of the time of a full render and is what makes the
    system feel intelligent before any audio is produced.
    """
    vdna = vocal_dna.extract(vocal_path, user_bpm=user_bpm, user_key=user_key)
    if vdna.get("status") != "ok":
        return {"status": "failed", "error": vdna.get("error")}
    report = matching.find_matches(vdna, catalog, n=n)
    return {
        "status": "ok",
        "vocal_summary": vdna.get("summary"),
        "requirements": vdna.get("beat_requirements"),
        "vocal_dna": {k: v for k, v in vdna.items()
                      if k not in ("notes", "onsets_s", "phrases", "quality")},
        "match_report": report.to_dict(),
    }
