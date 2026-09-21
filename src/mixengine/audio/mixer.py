"""
The mix stage.

Every parameter here is derived from a measurement of the actual audio.
There are no fixed thresholds: the high-pass corner comes from the singer's
measured fundamental, the gate from the measured noise floor, the
compressor threshold from a search for a target gain reduction, the
de-esser from the measured sibilance.

Balance is set by the vocal-to-instrumental ratio over **vocal-active
regions only**. Normalising the vocal and the beat each to -14 LUFS
integrated and summing them -- as the original engine did -- is wrong,
because a vocal is silent between phrases and its integrated loudness is
dragged down by those gaps. Matching it to the beat therefore makes the
vocal too loud while it is actually singing.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ..core import audio_io
from . import dsp
from ..core.capabilities import CAPS
from ..config import CFG, GenreProfile
from ..core.types import Phrase
from ..musical import salience

log = logging.getLogger("mixengine.mixer")


# ═════════════════════════════════════════════════════════════════════════════
# Vocal chain
# ═════════════════════════════════════════════════════════════════════════════

def process_vocal(vocal: np.ndarray, sr: int, vocal_dna: dict,
                  profile: GenreProfile, phrases: Sequence[Tuple[int, int]],
                  overrides: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
    """The adaptive vocal chain.

    Order matters and follows studio practice: clean up, then control
    dynamics, then shape tone, then add space. Level riding happens *before*
    compression so the compressor sees already-consistent material and can
    stay gentle -- this is the difference between a natural vocal and a
    squashed one.
    """
    o = overrides or {}
    v = dsp.as_2d(vocal)
    report: Dict = {}

    air_db = profile.air_db + float(o.get("air_db", 0.0))
    comp_target = max(1.0, profile.comp_target_gr_db + float(o.get("comp_target_gr_db", 0.0)))

    # ── 1. High-pass from the measured fundamental ────────────────────────
    f0_low = float(vocal_dna.get("f0_low_hz") or 0.0)
    if f0_low > 40:
        hpf = float(np.clip(f0_low * profile.hpf_ratio, 55.0, 160.0))
    else:
        hpf = 90.0
    v = dsp.highpass(v, sr, hpf, order=4)
    report["highpass_hz"] = round(hpf, 1)

    # ── 2. Expander from the measured noise floor ─────────────────────────
    floor = float(vocal_dna.get("noise_floor_db")
                  or dsp.noise_floor_db(v, sr))
    threshold = floor + CFG.mix.gate_above_floor_db
    v = dsp.gate(v, sr, threshold, ratio=CFG.mix.gate_ratio,
                 attack_ms=CFG.mix.gate_attack_ms,
                 release_ms=CFG.mix.gate_release_ms)
    report["gate_threshold_db"] = round(threshold, 1)
    report["measured_noise_floor_db"] = round(floor, 1)

    # ── 3. Notch the measured resonances ──────────────────────────────────
    resonances = vocal_dna.get("resonances") or dsp.find_resonances(v, sr, n=3)
    notches = []
    for item in resonances[:CFG.mix.n_resonance_notches]:
        freq, excess = (item[0], item[1]) if isinstance(item, (list, tuple)) else (item, 3.0)
        if excess < 2.0:
            continue
        cut = -float(np.clip(excess * 0.45, 0.8, CFG.mix.resonance_max_cut_db))
        v = dsp.peaking_eq(v, sr, float(freq), cut, q=CFG.mix.resonance_q)
        notches.append([round(float(freq), 1), round(cut, 2)])
    report["resonance_notches"] = notches

    # ── 4. Level rides before compression ─────────────────────────────────
    spread = float(vocal_dna.get("phrase_level_spread_db") or 0.0)
    if phrases and spread > 1.5:
        # Each phrase is weighted by how much level consistency matters
        # there. Short phrases are ad-libs and asides whose level is their
        # point; they are pulled less than the lines that carry the song.
        ph_objs = [Phrase(start=s / sr, end=e / sr, start_sample=int(s),
                          end_sample=int(e)) for s, e in phrases]
        weights = [salience.level_salience((p.start + p.end) / 2.0, phrase=p)
                   * (0.6 if p.duration < 0.8 else 1.0) for p in ph_objs]
        v = dsp.level_ride(v, sr, phrases,
                           max_gain_db=CFG.mix.ride_max_gain_db,
                           smoothing_s=CFG.mix.ride_smoothing_s,
                           weights=weights)
        report["level_ride"] = {"applied": True,
                                "phrase_spread_db": round(spread, 2),
                                "mean_weight": round(float(np.mean(weights)), 3)}
    else:
        report["level_ride"] = {"applied": False}

    # ── 5. De-ess from the measured sibilance ─────────────────────────────
    sib = float(vocal_dna.get("sibilance_ratio") or 0.0)
    sensitivity = float(np.clip(sib / 0.035, 0.5, 2.0))
    v, applied = dsp.deesser(v, sr, CFG.mix.deess_low_hz, CFG.mix.deess_high_hz,
                             max_gr_db=CFG.mix.deess_max_gr_db,
                             sensitivity=sensitivity)
    report["deesser"] = {"sensitivity": round(sensitivity, 2),
                         "mean_reduction_db": round(applied, 2),
                         "sibilance_ratio": round(sib, 4)}

    # ── 6. Compression toward a target reduction ──────────────────────────
    v, comp_info = dsp.auto_compressor(
        v, sr, target_gr_db=comp_target, ratio=CFG.mix.comp_ratio,
        attack_ms=CFG.mix.comp_attack_ms, release_ms=CFG.mix.comp_release_ms,
        knee_db=CFG.mix.comp_knee_db)
    comp_info["target_gr_db"] = round(comp_target, 2)
    report["compressor"] = comp_info

    # ── 7. Tone shaping ───────────────────────────────────────────────────
    # Low-mid cleanup scaled by how much energy is actually there.
    f, mag = dsp.long_term_spectrum(v, sr)
    lin = 10.0 ** (mag / 20.0)
    total = float(np.sum(lin)) + 1e-12
    mud = float(np.sum(lin[(f >= 200) & (f <= 420)])) / total
    mud_cut = -float(np.clip((mud - 0.13) * 22.0, 0.0, 3.5))
    if mud_cut < -0.3:
        v = dsp.peaking_eq(v, sr, 300.0, mud_cut, q=1.1)
    report["lowmid_cut_db"] = round(mud_cut, 2)

    # Presence and air.
    v = dsp.peaking_eq(v, sr, 3000.0, min(air_db * 0.45, 2.5), q=0.8)
    v = dsp.shelf_eq(v, sr, 8500.0, air_db, kind="high")
    report["presence_db"] = round(min(air_db * 0.45, 2.5), 2)
    report["air_db"] = round(air_db, 2)

    return dsp.as_2d(v).astype(np.float32), report


# ═════════════════════════════════════════════════════════════════════════════
# Sends
# ═════════════════════════════════════════════════════════════════════════════

def build_sends(vocal: np.ndarray, sr: int, bpm: float,
                profile: GenreProfile,
                overrides: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
    """Reverb and delay, both timed from the actual BPM.

    Delay time is a real note value (`60000/BPM` ms for a quarter note), and
    reverb predelay is derived the same way. Untimed effects are what make
    an automated mix sound smeared rather than rhythmic.
    """
    o = overrides or {}
    wet_level = max(0.0, profile.reverb_wet + float(o.get("reverb_wet", 0.0)))
    delay_wet = max(0.0, CFG.mix.delay_wet + float(o.get("delay_wet", 0.0)))
    info: Dict = {}

    v = dsp.as_2d(vocal)
    send = np.zeros((len(v), 2), dtype=np.float32)

    beat_ms = 60000.0 / bpm if bpm > 0 else 500.0
    delay_ms = beat_ms * (profile.delay_note * 4.0)
    predelay_ms = beat_ms * (CFG.mix.reverb_predelay_note * 4.0)
    info["delay_ms"] = round(delay_ms, 1)
    info["predelay_ms"] = round(predelay_ms, 1)
    info["reverb_wet"] = round(wet_level, 3)
    info["delay_wet"] = round(delay_wet, 3)

    # -- Tempo-synced delay -------------------------------------------------
    if delay_wet > 0.001:
        mono = dsp.to_mono(v)
        d = int(delay_ms * 0.001 * sr)
        if 0 < d < len(mono):
            tap = np.zeros_like(mono)
            fb = CFG.mix.delay_feedback
            gain = 1.0
            pos = d
            for _ in range(4):
                if pos >= len(mono):
                    break
                tap[pos:] += mono[:len(mono) - pos] * gain
                gain *= fb
                pos += d
            # Delays sit behind the vocal, so band-limit them.
            tap2 = dsp.highpass(tap[:, None], sr, 320.0, order=2)
            tap2 = dsp.lowpass(tap2, sr, 5200.0, order=2)
            send += np.repeat(tap2, 2, axis=1) * delay_wet

    # -- Reverb -------------------------------------------------------------
    if wet_level > 0.001:
        rev = _reverb(v, sr, predelay_ms, profile, wet_level)
        if rev is not None:
            send += rev
    return send.astype(np.float32), info


def _reverb(v: np.ndarray, sr: int, predelay_ms: float,
            profile: GenreProfile, wet: float) -> Optional[np.ndarray]:
    """Pedalboard reverb when available, synthetic plate otherwise."""
    pre = int(predelay_ms * 0.001 * sr)
    mono = dsp.to_mono(v)
    src = np.concatenate([np.zeros(pre, dtype=np.float32), mono])[:len(mono)]
    # High-pass the reverb send so low end never gets washy.
    src = dsp.highpass(src[:, None], sr, 300.0, order=2)

    if CAPS.pedalboard:
        try:
            from pedalboard import Pedalboard, Reverb
            board = Pedalboard([Reverb(room_size=float(np.clip(wet * 5.0, 0.15, 0.75)),
                                       damping=0.55, wet_level=1.0, dry_level=0.0,
                                       width=0.9)])
            stereo = np.repeat(src, 2, axis=1).T.astype(np.float32)
            out = board(stereo, sr).T
            return dsp.pad_to(out, len(v)) * wet
        except Exception as e:
            log.debug("pedalboard reverb unavailable (%s); using synthetic", e)

    return _synthetic_plate(src, sr, len(v)) * wet


def _synthetic_plate(src: np.ndarray, sr: int, n: int) -> np.ndarray:
    """Cheap Schroeder-style plate: comb filters into allpass, decorrelated.

    Not a replacement for a real algorithm, but it prevents the fallback
    path from having no reverb at all, which sounds noticeably unfinished.
    """
    from scipy import signal as sps
    mono = dsp.to_mono(src)
    out = np.zeros((n, 2), dtype=np.float32)

    combs = [(0.0297, 0.78), (0.0371, 0.76), (0.0411, 0.74), (0.0437, 0.72)]
    for ch, offset in enumerate((0.0, 0.0011)):
        acc = np.zeros(n, dtype=np.float64)
        for delay_s, fb in combs:
            d = int((delay_s + offset) * sr)
            if d < 1 or d >= n:
                continue
            y = sps.lfilter([1.0], np.concatenate([[1.0], np.zeros(d - 1), [-fb]]),
                            dsp.pad_to(mono[:, None], n)[:, 0])
            acc += y
        acc /= max(len(combs), 1)
        # Two allpass stages to smear the comb resonances.
        for delay_s, g in ((0.005, 0.7), (0.0017, 0.7)):
            d = int(delay_s * sr)
            if d < 1 or d >= n:
                continue
            b = np.concatenate([[-g], np.zeros(d - 1), [1.0]])
            a = np.concatenate([[1.0], np.zeros(d - 1), [-g]])
            acc = sps.lfilter(b, a, acc)
        out[:, ch] = acc
    out = dsp.lowpass(out, sr, 7000.0, order=2)
    return (out * 0.28).astype(np.float32)


def build_doubles(vocal: np.ndarray, sr: int, profile: GenreProfile,
                  regions: Optional[Sequence[Tuple[int, int]]] = None,
                  overrides: Optional[dict] = None) -> np.ndarray:
    """Delayed, panned copies of the vocal.

    Cheap to produce and one of the most recognisable "produced" signals in
    modern pop and rap. Restricted to the hook when a region list is given,
    because doubling everything flattens the arrangement.
    """
    o = overrides or {}
    if not profile.stereo_double:
        return np.zeros((len(dsp.as_2d(vocal)), 2), dtype=np.float32)

    gain = CFG.mix.double_gain_db + float(o.get("double_gain_db", 0.0))
    v = dsp.as_2d(vocal)
    src = v
    if regions:
        src = np.zeros_like(v)
        for s, e in regions:
            s, e = int(max(0, s)), int(min(len(v), e))
            if e > s:
                src[s:e] = v[s:e]

    left = dsp.haas_double(src, sr, CFG.mix.double_offset_ms[0], gain,
                           -CFG.mix.double_pan)
    right = dsp.haas_double(src, sr, CFG.mix.double_offset_ms[1], gain,
                            CFG.mix.double_pan)
    out = left + right
    # Keep doubles out of the lead's fundamental range so they add width
    # rather than mud.
    return dsp.highpass(out, sr, 220.0, order=2).astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# Beat processing
# ═════════════════════════════════════════════════════════════════════════════

def process_beat(stems: Dict[str, np.ndarray], beat_full: np.ndarray,
                 vocal: np.ndarray, sr: int, profile: GenreProfile,
                 overrides: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
    """Duck and unmask the beat's tonal content, leaving drums untouched.

    With stems, ducking and spectral carving apply only to bass and melody,
    so the groove keeps its punch while the midrange opens up for the
    vocal. Without stems, the same processing is applied to the full beat
    but gently -- pumping the drums is worse than a slightly crowded mix.
    """
    o = overrides or {}
    duck_db = max(0.0, profile.duck_depth_db + float(o.get("duck_depth_db", 0.0)))
    mask_strength = float(np.clip(profile.mask_strength + float(o.get("mask_strength", 0.0)),
                                  0.0, 0.85))
    report: Dict = {"duck_depth_db": round(duck_db, 2),
                    "mask_strength": round(mask_strength, 2)}

    has_stems = bool(stems) and any(k in stems for k in ("bass", "other"))
    n = len(dsp.as_2d(beat_full))

    if has_stems:
        report["mode"] = "stem_aware"
        drums = dsp.pad_to(stems["drums"], n) if "drums" in stems else None
        n_ch = dsp.as_2d(beat_full).shape[1]
        tonal_parts = [dsp.match_channels(dsp.pad_to(stems[k], n), n_ch) for k in ("bass", "other") if k in stems]
        extra = [dsp.pad_to(stems[k], n) for k in ("vocals",) if k in stems]

        tonal = np.sum(tonal_parts, axis=0) if tonal_parts else np.zeros((n, 1), np.float32)

        # The beat's own vocal chops compete directly with the user's vocal,
        # so they get ducked considerably harder than the instruments.
        if extra:
            chops = np.sum(extra, axis=0)
            chops = dsp.sidechain_duck(chops, vocal, sr, depth_db=duck_db + 6.0,
                                       attack_ms=CFG.mix.duck_attack_ms,
                                       release_ms=CFG.mix.duck_release_ms)
            tonal = tonal + chops
            report["ducked_beat_vocals_db"] = round(duck_db + 6.0, 2)

        tonal = dsp.sidechain_duck(tonal, vocal, sr, depth_db=duck_db,
                                   attack_ms=CFG.mix.duck_attack_ms,
                                   release_ms=CFG.mix.duck_release_ms)
        if mask_strength > 0.02:
            tonal = dsp.spectral_unmask(tonal, vocal, sr, strength=mask_strength,
                                        low_hz=CFG.mix.mask_low_hz,
                                        high_hz=CFG.mix.mask_high_hz)
        out = tonal if drums is None else (dsp.match_channels(tonal, max(tonal.shape[1], drums.shape[1]))
                                           + dsp.match_channels(drums, max(tonal.shape[1], drums.shape[1])))
        report["drums_untouched"] = drums is not None
    else:
        report["mode"] = "full_beat"
        report["note"] = ("no stems available - ducking applied to the whole "
                          "beat at reduced depth to avoid pumping the drums")
        out = dsp.sidechain_duck(beat_full, vocal, sr, depth_db=duck_db * 0.6,
                                 attack_ms=CFG.mix.duck_attack_ms,
                                 release_ms=CFG.mix.duck_release_ms)
        if mask_strength > 0.02:
            out = dsp.spectral_unmask(out, vocal, sr, strength=mask_strength * 0.8,
                                      low_hz=CFG.mix.mask_low_hz,
                                      high_hz=CFG.mix.mask_high_hz)
    return dsp.as_2d(out).astype(np.float32), report


# ═════════════════════════════════════════════════════════════════════════════
# Balance and summing
# ═════════════════════════════════════════════════════════════════════════════

def balance_and_sum(vocal: np.ndarray, beat: np.ndarray, sr: int,
                    phrases: Sequence[Tuple[int, int]],
                    profile: GenreProfile,
                    sends: Optional[np.ndarray] = None,
                    doubles: Optional[np.ndarray] = None,
                    overrides: Optional[dict] = None,
                    fx: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
    """Set the vocal level by target VIR over active regions, then sum.

    The key measurement: the vocal's loudness and the beat's loudness are
    both measured over the *same* vocal-active windows. Their difference is
    the vocal-to-instrumental ratio, and the vocal is gained so that ratio
    hits the genre target.

    `sends` and `doubles` are vocal-derived and follow the vocal's gain.
    `fx` -- transition effects -- is summed at unity: a riser or an impact
    was already rendered at its intended absolute level and has no business
    moving with the vocal fader.
    """
    o = overrides or {}
    target_vir = profile.vir_db + float(o.get("vir_db", 0.0))

    n = max(len(dsp.as_2d(vocal)), len(dsp.as_2d(beat)))
    n_ch = max(dsp.as_2d(vocal).shape[1], dsp.as_2d(beat).shape[1], 2)

    v = dsp.match_channels(dsp.pad_to(vocal, n), n_ch)
    b = dsp.match_channels(dsp.pad_to(beat, n), n_ch)

    regions = [(int(s), int(e)) for s, e in phrases if e > s] if phrases else []
    if not regions:
        regions = [(0, n)]
    regions = [(max(0, s), min(n, e)) for s, e in regions if min(n, e) > max(0, s)]

    v_lufs = audio_io.loudness_region_lufs(v, sr, regions)
    b_lufs = audio_io.loudness_region_lufs(b, sr, regions)

    if np.isfinite(v_lufs) and np.isfinite(b_lufs):
        current_vir = v_lufs - b_lufs
        gain_db = float(np.clip(target_vir - current_vir, -24.0, 24.0))
    else:
        current_vir, gain_db = 0.0, 0.0

    v = v * dsp.db_to_lin(gain_db)

    mix = v + b
    if sends is not None and len(sends):
        mix = mix + dsp.match_channels(dsp.pad_to(sends, n), n_ch) * dsp.db_to_lin(gain_db)
    if doubles is not None and len(doubles):
        mix = mix + dsp.match_channels(dsp.pad_to(doubles, n), n_ch) * dsp.db_to_lin(gain_db)
    if fx is not None and len(fx):
        mix = mix + dsp.match_channels(dsp.pad_to(fx, n), n_ch)

    report = {
        "target_vir_db": round(target_vir, 2),
        "measured_vir_db": round(float(current_vir), 2),
        "vocal_gain_db": round(gain_db, 2),
        "vocal_active_lufs": (round(float(v_lufs), 2) if np.isfinite(v_lufs) else None),
        "beat_under_vocal_lufs": (round(float(b_lufs), 2) if np.isfinite(b_lufs) else None),
        "n_active_regions": len(regions),
        "method": "vocal_to_instrumental_ratio_over_active_regions",
    }
    log.info("  balance: VIR %.1f -> %.1f dB (vocal %+.1f dB)",
             current_vir, target_vir, gain_db)
    return dsp.as_2d(mix).astype(np.float32), report
