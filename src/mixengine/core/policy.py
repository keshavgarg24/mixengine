"""
What the engine may do to this audio.

One pure function, no audio, no I/O. The decisions it makes used to be
scattered across six modules, each with its own idea of what the input
was, which is how a finished vocal on its own beat ended up retuned,
requantised, de-reverbed and laid over a looped intro -- with every
stage individually behaving as designed.

Having them in one place makes the engine's behaviour a table you can
read, test without audio, and show to the user before it runs.
"""

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict

from ..analysis.intake import KeyDecision, Relationship, VocalState
from .intents import Intents

log = logging.getLogger("mixengine.policy")

# A "full mix" call this confident is worth two minutes of separation.
# The reference render separated on 0.56 -- a coin flip.
SEPARATION_CONFIDENCE = 0.75

# Reverb longer than this on a raw take is a room worth removing.
DEREVERB_RT60_S = 0.8

# Flex-Tune semantics: leave notes already close alone. Antares calls
# 50-100 cents "significantly off"; this sits inside that.
TUNING_DEAD_ZONE_CENTS = 35.0

_STAGES = ("separation", "dereverb", "tuning", "alignment", "arrangement",
           "beat_fit", "vocal_chain", "space", "ducking")


@dataclass(frozen=True)
class StageDecision:
    """What one stage may do, how much, and why."""

    enabled: bool
    strength: float = 0.0
    method: str = "none"
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RenderPlan:
    """Every stage decision for one render."""

    separation: StageDecision
    dereverb: StageDecision
    tuning: StageDecision
    alignment: StageDecision
    arrangement: StageDecision
    beat_fit: StageDecision
    vocal_chain: StageDecision
    space: StageDecision
    ducking: StageDecision

    key: KeyDecision
    offset_s: float
    is_locked: bool
    vocal_state: str

    def summary(self) -> str:
        """The plan in the language a musician would use."""
        origin = ("was recorded to this beat" if self.is_locked
                  else "is being matched to this beat")
        doing = [
            "tuning %s" % ("on" if self.tuning.enabled else "off"),
            ("timing kept" if self.alignment.method == "single_offset"
             else "timing aligned"),
            ("reverb kept" if self.space.method == "keep"
             else "reverb matched to the beat"),
            ("beat not looped" if self.beat_fit.method == "pad"
             else "beat %s" % self.beat_fit.method),
        ]
        return ("Your vocal is %s and %s. %s."
                % (self.vocal_state, origin, " · ".join(doing)))

    def to_dict(self) -> Dict[str, Any]:
        return {"stages": {n: getattr(self, n).to_dict() for n in _STAGES},
                "key": self.key.to_dict(),
                "offset_s": round(float(self.offset_s), 4),
                "is_locked": self.is_locked,
                "vocal_state": self.vocal_state,
                "summary": self.summary()}


def plan(vdna: dict, bdna: dict, vocal_state: VocalState,
         relationship: Relationship, key_decision: KeyDecision,
         intents: Intents = Intents.AUTO) -> RenderPlan:
    """Decide what every stage may do.

    Ambiguity resolves toward doing less. A flat render can be asked for
    more; a destroyed one cannot be recovered.
    """
    locked = relationship.is_locked
    finished = vocal_state.is_mixed
    tuned = vocal_state.is_tuned

    # ── Separation ───────────────────────────────────────────────────
    sep_conf = float(vdna.get("input_type_confidence") or 0.0)
    is_full_mix = vdna.get("input_type") == "full_mix"
    if intents.separate == "never":
        separation = StageDecision(False, reason="you asked us not to separate")
    elif intents.separate == "always":
        separation = StageDecision(True, 1.0, "demucs",
                                   "you asked us to separate")
    elif locked:
        separation = StageDecision(
            False, reason="the vocal was recorded to this beat, so the "
                          "bleed is the beat itself")
    elif is_full_mix and sep_conf >= SEPARATION_CONFIDENCE:
        separation = StageDecision(
            True, 1.0, "demucs",
            "instrumental content detected at %.2f confidence" % sep_conf)
    else:
        separation = StageDecision(
            False, reason="no confident instrumental content (%.2f); "
                          "separating would cost minutes and risk the take"
                          % sep_conf)

    # ── Dereverb ─────────────────────────────────────────────────────
    if vocal_state.reverb_is_intentional:
        dereverb = StageDecision(
            False, reason="the %.2fs tail is part of the vocal's sound, "
                          "not a room" % vocal_state.rt60_s)
    elif vocal_state.rt60_s > DEREVERB_RT60_S:
        dereverb = StageDecision(
            True, 0.7, "spectral",
            "a %.2fs room on an otherwise raw take" % vocal_state.rt60_s)
    else:
        dereverb = StageDecision(False, reason="no problematic room")

    # ── Tuning ───────────────────────────────────────────────────────
    if intents.tune is not None:
        tuning = StageDecision(
            enabled=intents.tune > 0.0, strength=intents.tune,
            method="flex" if intents.tune > 0.0 else "none",
            reason=("you asked for tuning at %.2f" % intents.tune
                    if intents.tune > 0 else "you asked for tuning off"))
    elif tuned:
        tuning = StageDecision(
            False, reason="%.0f%% of the note time is already on the grid"
                          % (vocal_state.tuned_fraction * 100))
    else:
        tuning = StageDecision(
            True, 0.5, "flex",
            "only %.0f%% of the note time is on the grid; correcting notes "
            "more than %.0f cents off"
            % (vocal_state.tuned_fraction * 100, TUNING_DEAD_ZONE_CENTS))

    # ── Alignment ────────────────────────────────────────────────────
    if locked:
        alignment = StageDecision(
            True, 0.0, "single_offset",
            "recorded to this beat; applying the measured %+.2fs lag and "
            "nothing else" % relationship.offset_s)
    elif finished:
        alignment = StageDecision(
            True, 0.2, "phrase_anchor",
            "a finished vocal's timing is a performance; anchoring phrases "
            "to downbeats without quantising inside them")
    else:
        alignment = StageDecision(
            True, 0.45, "grid_nudge",
            "a raw take over a new beat; nudging onsets toward the grid")
    if intents.timing is not None:
        alignment = StageDecision(
            enabled=intents.timing > 0.0, strength=intents.timing,
            method=("single_offset" if locked and intents.timing == 0.0
                    else alignment.method),
            reason="you asked for timing at %.2f" % intents.timing)

    # ── Arrangement ──────────────────────────────────────────────────
    if locked:
        arrangement = StageDecision(
            False, reason="the arrangement is the one it was recorded to")
    else:
        arrangement = StageDecision(
            True, 1.0, "structure_aware",
            "matching a new beat, so the sections are ours to choose")

    # ── Beat fit ─────────────────────────────────────────────────────
    v_duration = float(vdna.get("duration_s") or 0.0)
    b_duration = float(bdna.get("duration_s") or 0.0)
    if locked or b_duration >= v_duration - 0.5:
        beat_fit = StageDecision(
            True, 0.0, "pad",
            "the beat (%.0fs) already covers the vocal (%.0fs)"
            % (b_duration, v_duration))
    else:
        beat_fit = StageDecision(
            True, 1.0, "loop_last_section",
            "the beat (%.0fs) is shorter than the vocal (%.0fs); looping "
            "its last full section" % (b_duration, v_duration))

    # ── Vocal chain ──────────────────────────────────────────────────
    if finished:
        vocal_chain = StageDecision(
            True, 0.3, "finish",
            "the vocal is already mixed; levelling and seating it only")
    else:
        vocal_chain = StageDecision(
            True, 1.0, "produce", "a raw take needs the full chain")

    # ── Space ────────────────────────────────────────────────────────
    if intents.space is not None:
        space = StageDecision(True, 1.0, intents.space,
                              "you asked to %s the space" % intents.space)
    elif vocal_state.reverb_is_intentional:
        space = StageDecision(True, 0.0, "keep",
                              "the vocal brought its own space")
    else:
        space = StageDecision(True, 1.0, "match",
                              "placing the vocal in the beat's room")

    # ── Ducking ──────────────────────────────────────────────────────
    if bool(bdna.get("has_stems") or bdna.get("stems")):
        ducking = StageDecision(True, 1.0, "stems",
                                "ducking the tonal stems under the vocal")
    else:
        ducking = StageDecision(
            True, 0.4, "band_limited",
            "no stems, so ducking is limited to the vocal band to avoid "
            "pumping the drums")

    p = RenderPlan(separation=separation, dereverb=dereverb, tuning=tuning,
                   alignment=alignment, arrangement=arrangement,
                   beat_fit=beat_fit, vocal_chain=vocal_chain, space=space,
                   ducking=ducking, key=key_decision,
                   offset_s=relationship.offset_s, is_locked=locked,
                   vocal_state=vocal_state.state)
    _log_plan(p)
    return p


def _log_plan(p: RenderPlan) -> None:
    log.info("── plan ───────────────────")
    log.info("  %s", p.summary())
    for name in _STAGES:
        d = getattr(p, name)
        log.info("  %-12s %s  %s", name, "on " if d.enabled else "off",
                 d.reason)
    log.info("  %-12s     %s", "key", p.key.evidence)
