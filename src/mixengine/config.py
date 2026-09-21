"""
Central configuration for the mixing engine.

Every tunable number in the system lives here. Nothing downstream should
hard-code a threshold, a target level, or a weight -- if you find yourself
writing a magic number in another module, it belongs in this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Global audio constants
# ─────────────────────────────────────────────────────────────────────────────

SR = 44100                      # working sample rate for all rendering
ANALYSIS_SR = 22050             # analysis runs at half rate -- 2-4x faster, no
                                # meaningful accuracy loss for rhythm/harmony
HOP = 512                       # analysis hop length at ANALYSIS_SR
N_FFT = 2048

DNA_SCHEMA_VERSION = "2.1.0"    # bump to force catalog re-analysis

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Kessler key profiles (Krumhansl & Kessler 1982).
# Used for key detection by correlating against a pitch-class distribution.
KK_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
KK_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


# ─────────────────────────────────────────────────────────────────────────────
# Analysis thresholds
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AnalysisConfig:
    # -- Rhythm ------------------------------------------------------------
    tempo_min: float = 50.0
    tempo_max: float = 210.0
    # A beat grid is "stable" (quantized/programmed) above this score.
    grid_stability_quantized: float = 0.85

    # -- Key ---------------------------------------------------------------
    # Correlation gap between best and second-best key required for the
    # detection to count as confident.
    key_confidence_gap: float = 0.06
    # Producer tag is trusted unless the detector disagrees this strongly.
    key_tag_override_confidence: float = 0.75

    # -- Vocal pitch -------------------------------------------------------
    f0_min: float = 65.0            # C2 -- below this is almost never sung
    f0_max: float = 1200.0          # D6
    # torchcrepe periodicity below this is treated as unvoiced.
    voicing_threshold: float = 0.40
    # Minimum duration for an f0 run to count as a note event.
    min_note_dur_s: float = 0.06

    # -- Phrasing ----------------------------------------------------------
    # Silence longer than this splits one phrase from the next.
    phrase_gap_s: float = 0.35
    min_phrase_dur_s: float = 0.25
    # A phrase is a breath-to-breath unit -- a handful of bars. Anything
    # longer is a detection failure, not a long phrase, and it silently
    # breaks every stage that treats a phrase as its unit of work, so
    # overlong regions are split at their quietest interior point.
    max_phrase_dur_s: float = 12.0
    # Fallback only: used when the level histogram has no usable valley to
    # split on. Voice activity is normally derived adaptively from the
    # measured distribution rather than from a fixed offset below the peak.
    silence_rel_db: float = 38.0

    # -- Structure ---------------------------------------------------------
    min_section_bars: int = 4
    target_n_sections: int = 8

    # -- Quality gates on the incoming vocal -------------------------------
    min_snr_db: float = 12.0
    max_clipping_pct: float = 0.5
    # Estimated RT60 above this means the recording is too roomy to fix.
    max_reverb_rt60_s: float = 0.55
    # A lowpass cliff below this suggests a lossy/low-bandwidth source.
    min_bandwidth_hz: float = 13000.0


# ─────────────────────────────────────────────────────────────────────────────
# Matching / retrieval
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MatchConfig:
    # Scoring weights -- must sum to 1.0 (validated at import).
    w_harmonic: float = 0.30
    w_transform: float = 0.20      # deliberately high: artifacts cost more
                                   # than theoretical harmonic gain
    w_rhythmic: float = 0.15
    w_pocket: float = 0.15
    w_vibe: float = 0.10
    w_metadata: float = 0.05
    w_popularity: float = 0.05

    # Penalties (subtracted after the weighted sum)
    p_dissonance: float = 0.25
    p_vocal_collision: float = 0.15

    # -- Hard retrieval limits --------------------------------------------
    # Beyond these the transformation artifacts dominate; reject instead.
    max_semitone_shift: int = 2
    max_stretch_ratio: float = 1.12     # and 1/1.12 on the other side
    # Tempo windows, as a fraction, applied in the relaxation ladder.
    tempo_windows: Tuple[float, ...] = (0.06, 0.10, 0.15)

    # A pairing scoring below this is reported as incompatible.
    min_viable_score: float = 0.45

    # Minimum key-detection confidence (on the weaker of the two sides)
    # before a pitch shift may be applied on harmonic grounds. Transposition
    # is irreversible and audible; below this the harmonic evidence is too
    # weak to justify it, and the pairing renders untransposed instead.
    min_key_confidence_for_shift: float = 0.55

    # Result-set diversity
    max_per_producer: int = 1
    n_results: int = 5


# ─────────────────────────────────────────────────────────────────────────────
# Mixing / mastering
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GenreProfile:
    """Per-genre mix and master targets.

    `vir_db` is the vocal-to-instrumental ratio measured *over vocal-active
    regions only*. This is the single most important balance number in the
    system -- integrated LUFS over the whole file is the wrong metric because
    the gaps between phrases drag the vocal's average down.
    """
    name: str
    lufs_target: float          # final master integrated LUFS
    vir_db: float               # vocal minus instrumental, active regions
    duck_depth_db: float        # how far the tonal stems duck under the vocal
    mask_strength: float        # 0-1, spectral carve depth
    hpf_ratio: float            # HPF corner = fundamental * this
    air_db: float               # high-shelf boost at 8 kHz
    comp_target_gr_db: float    # target average gain reduction on the vocal
    reverb_wet: float
    delay_note: float           # 0.25 = quarter note, 0.125 = eighth
    tune_strength: float        # 0 = off, 1 = hard tune
    stereo_double: bool         # generate doubled/panned hook layers


GENRE_PROFILES: Dict[str, GenreProfile] = {
    "trap": GenreProfile("trap", -8.5, -1.0, 2.5, 0.55, 0.85, 3.5, 5.0, 0.06, 0.25, 0.55, True),
    "drill": GenreProfile("drill", -8.0, -0.5, 3.0, 0.60, 0.85, 4.0, 5.5, 0.05, 0.25, 0.60, True),
    "hip_hop": GenreProfile("hip_hop", -9.0, -1.5, 2.0, 0.50, 0.85, 3.0, 4.5, 0.07, 0.25, 0.35, True),
    "rnb": GenreProfile("rnb", -10.5, -2.5, 1.5, 0.45, 0.90, 3.0, 4.0, 0.12, 0.375, 0.30, True),
    "pop": GenreProfile("pop", -9.5, -2.0, 2.0, 0.50, 0.90, 3.5, 4.5, 0.10, 0.25, 0.45, True),
    "afrobeats": GenreProfile("afrobeats", -9.0, -2.0, 2.0, 0.50, 0.90, 3.0, 4.0, 0.09, 0.25, 0.35, True),
    "drum_and_bass": GenreProfile("drum_and_bass", -8.5, -1.5, 2.5, 0.55, 0.85, 3.5, 4.5, 0.08, 0.125, 0.30, False),
    "lofi": GenreProfile("lofi", -12.0, -3.0, 1.0, 0.35, 0.90, 1.5, 3.0, 0.15, 0.375, 0.10, False),
    "default": GenreProfile("default", -10.0, -2.0, 2.0, 0.50, 0.88, 3.0, 4.5, 0.08, 0.25, 0.35, True),
}

# Genre adjacency, used by the relaxation ladder when candidates are thin.
GENRE_NEIGHBOURS: Dict[str, List[str]] = {
    "trap": ["drill", "hip_hop", "rnb"],
    "drill": ["trap", "hip_hop"],
    "hip_hop": ["trap", "boom_bap", "rnb"],
    "rnb": ["pop", "hip_hop", "afrobeats"],
    "pop": ["rnb", "afrobeats"],
    "afrobeats": ["pop", "rnb", "dancehall"],
    "lofi": ["hip_hop", "jazz"],
}


@dataclass
class MixConfig:
    # -- Vocal chain -------------------------------------------------------
    gate_above_floor_db: float = 6.0     # gate opens this far above the
                                         # *measured* noise floor
    gate_ratio: float = 4.0
    gate_attack_ms: float = 2.0
    gate_release_ms: float = 120.0

    # De-esser
    deess_low_hz: float = 5000.0
    deess_high_hz: float = 9500.0
    deess_max_gr_db: float = 8.0

    # Compressor
    comp_ratio: float = 3.0
    comp_attack_ms: float = 8.0
    comp_release_ms: float = 90.0
    comp_knee_db: float = 6.0

    # Vocal rides: normalise each phrase toward the median phrase level
    # before compression. Capped so quiet phrases stay expressive.
    ride_max_gain_db: float = 6.0
    ride_smoothing_s: float = 0.15

    # Resonance suppression: notch the N worst peaks in the long-term spectrum
    n_resonance_notches: int = 3
    resonance_max_cut_db: float = 4.5
    resonance_q: float = 6.0

    # -- Beat processing ---------------------------------------------------
    duck_attack_ms: float = 12.0
    duck_release_ms: float = 220.0
    # Spectral masking is only applied inside this band.
    mask_low_hz: float = 220.0
    mask_high_hz: float = 5200.0

    # -- Sends -------------------------------------------------------------
    reverb_predelay_note: float = 0.0625   # 1/16 note
    delay_feedback: float = 0.28
    delay_wet: float = 0.10
    double_offset_ms: Tuple[float, float] = (11.0, 17.0)
    double_gain_db: float = -7.0
    double_pan: float = 0.72

    # -- Master ------------------------------------------------------------
    glue_ratio: float = 1.6
    glue_target_gr_db: float = 1.5
    limiter_ceiling_db: float = -1.0
    true_peak_db: float = -1.0
    # How much gain may be pushed *into* the limiter to reach the loudness
    # target, measured beyond the point where the limiter starts working.
    # Loudness is always reachable given enough limiting, but past a few dB
    # the cost is transient snap -- and the platform normalises the level
    # back down afterwards while the squashed dynamics stay squashed. When
    # the target needs more than this, the engine delivers quieter and says
    # so rather than crushing the master.
    max_limiting_db: float = 6.0
    # Spectral match toward the beat's own tonal balance, 0-1.
    tonal_match_strength: float = 0.35


# ─────────────────────────────────────────────────────────────────────────────
# Critic / QC gates
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CriticConfig:
    max_true_peak_db: float = -0.9
    lufs_tolerance_db: float = 1.5
    # How much further under target a master may sit, with the true peak
    # already on the ceiling, before the shortfall is a warning rather
    # than a note. Past the tolerance the limiter was the binding
    # constraint (`MasterConfig.max_limiting_db`) and the master declined
    # to crush the dynamics for the last decibels.
    lufs_ceiling_grace_db: float = 2.0
    lufs_ceiling_window_db: float = 0.6      # "on the ceiling": this close to the true-peak limit
    max_mono_loss_db: float = 3.0
    max_sync_error_ms: float = 35.0
    max_tuning_error_cents: float = 45.0
    min_vocal_presence_db: float = -30.0     # vocal must be audible
    max_silence_gap_s: float = 3.0
    max_repair_attempts: int = 2

    # Ranking weights over the critic's own sub-scores.
    w_gates: float = 0.40
    w_harmonic: float = 0.25
    w_clarity: float = 0.20
    w_loudness: float = 0.15


# ─────────────────────────────────────────────────────────────────────────────
# Render variants -- the 5 deliverables, spanning the subjective space
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VariantSpec:
    key: str
    label: str
    description: str
    overrides: Dict[str, float] = field(default_factory=dict)


VARIANTS: List[VariantSpec] = [
    VariantSpec(
        "clean", "Clean / Radio",
        "Balanced, natural tuning, genre-standard loudness.",
        {},
    ),
    VariantSpec(
        "forward", "Forward / Aggressive",
        "Vocal pushed forward, harder compression, deeper duck, brighter.",
        {"vir_db": 1.5, "duck_depth_db": 1.5, "air_db": 1.5,
         "comp_target_gr_db": 1.5, "lufs_target": 0.7},
    ),
    VariantSpec(
        "wide", "Wide / Atmospheric",
        "Doubles, longer reverb and delay throws, vocal sits back.",
        {"vir_db": -1.5, "reverb_wet": 0.06, "delay_wet": 0.06,
         "double_gain_db": 2.0},
    ),
    VariantSpec(
        "tuned", "Tuned / Modern",
        "Stronger pitch correction and tighter timing quantisation.",
        {"tune_strength": 0.35, "quantize_strength": 0.5, "air_db": 1.0},
    ),
    VariantSpec(
        "warm", "Warm / Intimate",
        "Softer top end, gentler compression, more dynamic range.",
        {"air_db": -1.5, "comp_target_gr_db": -1.5, "lufs_target": -2.0,
         "duck_depth_db": -0.5},
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Paths:
    """All I/O locations. Override root for Kaggle vs local."""
    root: str = "./data"

    @property
    def beats(self) -> str: return f"{self.root}/beats"

    @property
    def vocals(self) -> str: return f"{self.root}/vocals"

    @property
    def dna(self) -> str: return f"{self.root}/dna"

    @property
    def beat_dna(self) -> str: return f"{self.dna}/beats"

    @property
    def vocal_dna(self) -> str: return f"{self.dna}/vocals"

    @property
    def stems(self) -> str: return f"{self.root}/stems"

    @property
    def cache(self) -> str: return f"{self.root}/cache"

    @property
    def outputs(self) -> str: return f"{self.root}/outputs"


# ─────────────────────────────────────────────────────────────────────────────
# Assembled config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    match: MatchConfig = field(default_factory=MatchConfig)
    mix: MixConfig = field(default_factory=MixConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    paths: Paths = field(default_factory=Paths)
    seed: int = 1337

    def genre(self, name: str | None) -> GenreProfile:
        """Look up a genre profile, falling back to `default`."""
        if not name:
            return GENRE_PROFILES["default"]
        key = str(name).lower().strip().replace(" ", "_").replace("&", "and")
        return GENRE_PROFILES.get(key, GENRE_PROFILES["default"])

    def to_dict(self) -> dict:
        return asdict(self)


CFG = Config()


def _validate() -> None:
    m = CFG.match
    total = (m.w_harmonic + m.w_transform + m.w_rhythmic + m.w_pocket
             + m.w_vibe + m.w_metadata + m.w_popularity)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Match weights must sum to 1.0, got {total:.6f}")

    c = CFG.critic
    ctotal = c.w_gates + c.w_harmonic + c.w_clarity + c.w_loudness
    if abs(ctotal - 1.0) > 1e-6:
        raise ValueError(f"Critic weights must sum to 1.0, got {ctotal:.6f}")


_validate()
