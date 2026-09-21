"""
Vocal coaching: what to tell a singer, when, and when to say nothing.

The design here is shaped by one uncomfortable research finding, and it
runs against the obvious product instinct.

Real-time visual pitch feedback **makes the take worse while it is
happening.** Studies of singers training with concurrent visual feedback
report a significant performance decrement attributed to the extra
information-processing load, and a repeated pattern of results worsening at
the precise moment feedback is delivered. One study of an augmented
instrument found visual feedback had no effect at all and warned that
learning could be harmed in some cases. The benefit that does exist is to
*learning over sessions*, not to the take being recorded right now.

So a live pitch meter on screen during a take is not a feature. It is a
tax on the performance, paid in the currency the engine most needs: a good
source recording.

What follows from that:

  * **Before the take** is where guidance belongs. Attention is free, and
    knowing the key, the range and where the phrases land measurably
    changes what the singer does.
  * **During the take**, interrupt only for faults that destroy the
    recording and cannot be repaired afterwards -- clipping, the singer
    walking off mic, a dead channel. Pitch and timing are explicitly *not*
    in that category, because both are fixable downstream and neither is
    worth breaking concentration for.
  * **After the take** is where detail belongs, paired with knowledge of
    results rather than a raw trace. That is the condition under which the
    retention benefits in the literature actually appeared.

Tolerances come from measurement, not intuition. Mean pitch inaccuracy is
around 25 cents for professional singers and 34.5 for non-professionals,
so any threshold tighter than that would flag expert singing as wrong. The
engine's own tuner corrects far below these thresholds silently; the coach
only speaks when a human would.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.types import FloatSeq

from .realtime import FrameStats

# ─────────────────────────────────────────────────────────────────────────────
# Cue taxonomy
# ─────────────────────────────────────────────────────────────────────────────

# Shown before recording starts. Attention is free here.
PREROLL = "preroll"
# Interrupts a take in progress. Reserved for unrecoverable faults.
LIVE = "live"
# Delivered after the take, with the result attached.
POST = "post"

# Severity within a level.
FATAL = "fatal"        # the take is unusable; stop and redo
WARN = "warn"
INFO = "info"
GOOD = "good"

# Measured reference points -- see module docstring.
PRO_PITCH_ERROR_CENTS = 25.0
AMATEUR_PITCH_ERROR_CENTS = 34.5

# Gain staging. Peaks at -6 dBFS with RMS near -18 dBFS leaves headroom for
# transients while keeping the signal far above the converter's noise.
TARGET_PEAK_DB = -6.0
TARGET_RMS_DB = -18.0


@dataclass
class Cue:
    """One thing to tell the singer."""
    id: str
    level: str                 # preroll | live | post
    severity: str              # fatal | warn | info | good
    message: str               # short, imperative, glanceable
    detail: str = ""           # the why, shown on demand
    at_s: Optional[float] = None
    value: Optional[float] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


# ═════════════════════════════════════════════════════════════════════════════
# Before the take
# ═════════════════════════════════════════════════════════════════════════════

def preroll_guidance(*, beat_key_name: Optional[str] = None,
                     beat_bpm: float = 0.0,
                     beat_bars: int = 0,
                     scale_midi: FloatSeq = (),
                     voice_low_midi: float = 0.0,
                     voice_high_midi: float = 0.0,
                     performance_type: str = "",
                     noise_floor_db: float = -70.0,
                     rt60_s: float = 0.0,
                     has_beat: bool = True) -> List[Cue]:
    """Everything worth saying before the singer starts.

    Two categories. Musical guidance is only possible when a beat is
    supplied -- key, tempo, where bars fall. Technical guidance applies
    either way and is what determines whether the recording is salvageable
    at all.

    The range check is the single most valuable item here: if the beat's
    key puts the melody outside the singer's comfortable range, no amount
    of downstream processing fixes it, and the answer is to transpose the
    beat before recording rather than to fight it afterwards.
    """
    cues: List[Cue] = []

    if has_beat and beat_key_name:
        cues.append(Cue("key", PREROLL, INFO,
                        "Key: %s" % beat_key_name,
                        "Aim for notes in this key. The engine corrects small "
                        "deviations, but it cannot fix a phrase sung in the "
                        "wrong key."))
    if has_beat and beat_bpm > 0:
        bar_s = 240.0 / beat_bpm
        cues.append(Cue("tempo", PREROLL, INFO,
                        "%.0f BPM - one bar is %.1f s" % (beat_bpm, bar_s),
                        "You get a two-bar count-in. Start your first word on "
                        "the downbeat after it, or just before if the line "
                        "has a pickup."))
    if has_beat and beat_bars:
        cues.append(Cue("length", PREROLL, INFO,
                        "%d bars available" % beat_bars))

    # Range check against the singer's measured tessitura.
    if voice_low_midi > 0 and voice_high_midi > 0 and len(scale_midi):
        notes = np.asarray(scale_midi, dtype=np.float64)
        below = notes[notes < voice_low_midi - 1.0]
        above = notes[notes > voice_high_midi + 1.0]
        if above.size and above.size >= notes.size * 0.25:
            cues.append(Cue(
                "range_high", PREROLL, WARN,
                "This key sits high for your range",
                "About %d%% of the melody is above where you sing "
                "comfortably. Transposing the beat down 2-3 semitones before "
                "recording will sound better than straining and correcting "
                "it afterwards." % round(100.0 * above.size / notes.size),
                value=float(above.size) / notes.size))
        elif below.size and below.size >= notes.size * 0.25:
            cues.append(Cue(
                "range_low", PREROLL, WARN,
                "This key sits low for your range",
                "Consider transposing the beat up. Notes at the bottom of "
                "your range lose projection and pick up more room.",
                value=float(below.size) / notes.size))
        else:
            cues.append(Cue("range_ok", PREROLL, GOOD,
                            "This key suits your range"))

    # Environment.
    if rt60_s > 0.6:
        cues.append(Cue(
            "room_bad", PREROLL, FATAL,
            "This room is too live to record in",
            "Estimated reverb tail %.2f s. Room reverb is baked into the "
            "recording and cannot be removed cleanly; everything downstream "
            "inherits it. Move somewhere soft -- a wardrobe, a curtained "
            "corner -- or hang blankets behind the mic." % rt60_s,
            value=rt60_s))
    elif rt60_s > 0.35:
        cues.append(Cue("room_live", PREROLL, WARN,
                        "Room is a little live (%.2f s tail)" % rt60_s,
                        "Usable, but soft furnishings behind you will "
                        "noticeably tighten the result.", value=rt60_s))

    if noise_floor_db > -45.0:
        cues.append(Cue(
            "noisy", PREROLL, WARN,
            "Background noise is high",
            "Noise floor %.0f dB. Turn off fans, air conditioning and "
            "anything with a motor. Noise removal costs vocal detail, so "
            "not recording it is always better." % noise_floor_db,
            value=noise_floor_db))

    if performance_type == "rap":
        cues.append(Cue("rap_technique", PREROLL, INFO,
                        "Stay close and consistent on the mic",
                        "Consonants carry rap. Keep a steady distance so the "
                        "level does not jump between bars."))
    elif performance_type == "sung":
        cues.append(Cue("sung_technique", PREROLL, INFO,
                        "Back off slightly on loud notes",
                        "Moving a few inches back on the big notes keeps them "
                        "from overloading and preserves their tone."))

    cues.append(Cue("mic_distance", PREROLL, INFO,
                    "Sit a hand-span from the mic, slightly off-axis",
                    "About 15-20 cm, angled 15-30 degrees off your mouth. "
                    "That angle keeps sibilance and plosives off the capsule "
                    "while keeping the voice present. Closer than 10 cm adds "
                    "10-15 dB of boomy low end that has to be EQ'd back out."))
    return cues


# ═════════════════════════════════════════════════════════════════════════════
# During the take
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class LiveCoachConfig:
    """Thresholds and, more importantly, the attention budget."""
    # A cue costs concentration. These limits exist so the coach cannot
    # chatter its way into degrading the performance it is supervising.
    min_gap_s: float = 6.0
    max_cues_per_minute: int = 4
    # Sustained conditions must persist before they are worth mentioning;
    # a single frame is noise.
    sustain_s: float = 0.8

    clip_frames_before_alert: int = 3
    quiet_db: float = -38.0
    loud_rms_db: float = -9.0
    silence_s: float = 4.0

    # Proximity is judged against the singer's own baseline, not against an
    # absolute number. The low/mid ratio depends on the voice -- a baritone
    # carries more energy below 250 Hz than a soprano at any distance -- so
    # a fixed threshold is wrong in principle, not merely uncalibrated. The
    # first seconds of voice establish where they started; a cue fires when
    # they move a factor of two from it. The hard bounds cover the extreme
    # cases before a baseline exists. The browser coach mirrors these.
    proximity_baseline_s: float = 4.0
    proximity_close_factor: float = 2.0
    proximity_far_factor: float = 0.5
    proximity_hard_high: float = 2.8
    proximity_hard_low: float = 0.2


class LiveCoach:
    """Emits live cues, sparingly.

    Consumes `FrameStats` and returns cues only for conditions that make a
    recording unusable. Deliberately absent: pitch, timing, vibrato,
    expression. Those are all either fixable downstream or actively harmed
    by interrupting to mention them.

    The rate limiter is not politeness. It is the mechanism that keeps this
    from reproducing the exact effect the research warns about.
    """

    def __init__(self, config: Optional[LiveCoachConfig] = None,
                 sr: int = 48000):
        self.cfg = config or LiveCoachConfig()
        self.sr = sr
        self._last_cue_at: float = -1e9
        self._recent: List[float] = []
        self._emitted: Dict[str, float] = {}
        self._clip_run = 0
        self._state_since: Dict[str, float] = {}
        self._last_voice_at: Optional[float] = None
        self._started_at: Optional[float] = None
        self._prox_samples: List[float] = []
        self._prox_voiced_s: float = 0.0
        self._prox_last_t: Optional[float] = None
        self.proximity_baseline: Optional[float] = None

    # -- proximity baseline ------------------------------------------------

    def _track_baseline(self, f: FrameStats, t: float) -> None:
        """Accumulate the singer's own low/mid ratio until a baseline exists."""
        if self.proximity_baseline is not None:
            return
        dt = 0.0 if self._prox_last_t is None else max(0.0, t - self._prox_last_t)
        self._prox_last_t = t
        if not f.is_voice or f.low_mid_ratio <= 0:
            return
        self._prox_voiced_s += dt
        self._prox_samples.append(float(f.low_mid_ratio))
        if (self._prox_voiced_s >= self.cfg.proximity_baseline_s
                and len(self._prox_samples) >= 20):
            self.proximity_baseline = float(np.median(self._prox_samples))

    def _proximity_state(self, f: FrameStats) -> Tuple[bool, bool]:
        """`(too_close, too_far)` for this frame, relative to the baseline."""
        c = self.cfg
        r = float(f.low_mid_ratio)
        b = self.proximity_baseline
        if b is not None and b > 0:
            close = r > b * c.proximity_close_factor
            far = f.is_voice and r < b * c.proximity_far_factor
        else:
            close = r > c.proximity_hard_high
            far = f.is_voice and 0 < r < c.proximity_hard_low
        return close, far

    # -- budget ------------------------------------------------------------

    def _may_speak(self, t: float, cue_id: str, severity: str) -> bool:
        if severity == FATAL:
            # Clipping destroys the recording. It overrides the budget,
            # because staying quiet to protect concentration is pointless
            # if the take is being ruined while we stay quiet.
            return (t - self._emitted.get(cue_id, -1e9)) > 3.0
        if (t - self._last_cue_at) < self.cfg.min_gap_s:
            return False
        self._recent = [x for x in self._recent if t - x < 60.0]
        if len(self._recent) >= self.cfg.max_cues_per_minute:
            return False
        if (t - self._emitted.get(cue_id, -1e9)) < 20.0:
            return False
        return True

    def _emit(self, t: float, cue: Cue) -> Optional[Cue]:
        if not self._may_speak(t, cue.id, cue.severity):
            return None
        self._last_cue_at = t
        self._recent.append(t)
        self._emitted[cue.id] = t
        cue.at_s = round(t, 2)
        return cue

    def _sustained(self, key: str, t: float, active: bool) -> bool:
        """True once a condition has held for `sustain_s`."""
        if not active:
            self._state_since.pop(key, None)
            return False
        since = self._state_since.setdefault(key, t)
        return (t - since) >= self.cfg.sustain_s

    # -- main --------------------------------------------------------------

    def update(self, frames: Sequence[FrameStats]) -> List[Cue]:
        out: List[Cue] = []
        for f in frames:
            t = f.time_s
            if self._started_at is None:
                self._started_at = t
            self._track_baseline(f, t)
            cue = self._check(f, t)
            if cue is not None:
                out.append(cue)
        return out

    def _check(self, f: FrameStats, t: float) -> Optional[Cue]:
        c = self.cfg

        # 1. Clipping -- unrecoverable, so it outranks everything.
        self._clip_run = self._clip_run + 1 if f.clipped else 0
        if self._clip_run >= c.clip_frames_before_alert:
            return self._emit(t, Cue(
                "clipping", LIVE, FATAL, "Clipping - turn the gain down",
                "The waveform is hitting the ceiling and the distortion is "
                "recorded into the file. It cannot be removed afterwards.",
                value=f.peak_db))

        if f.is_voice:
            self._last_voice_at = t

        # 2. Dead channel or the singer has stopped.
        if self._last_voice_at is not None and (t - self._last_voice_at) > c.silence_s:
            if self._sustained("silent", t, True):
                return self._emit(t, Cue(
                    "no_signal", LIVE, WARN, "No signal",
                    "Nothing is reaching the mic. Check it is selected, "
                    "unmuted and pointed at you.", value=f.rms_db))

        if not f.is_voice:
            return None

        # 3. Level -- too low costs signal-to-noise permanently.
        if self._sustained("quiet", t, f.rms_db < c.quiet_db):
            return self._emit(t, Cue(
                "too_quiet", LIVE, WARN, "Move closer or raise the gain",
                "You are %.0f dB below where you should be, which means more "
                "room and hiss come up with the voice later."
                % (TARGET_RMS_DB - f.rms_db), value=f.rms_db))

        if self._sustained("hot", t, f.rms_db > c.loud_rms_db and not f.clipped):
            return self._emit(t, Cue(
                "too_hot", LIVE, WARN, "Running hot - ease the gain down",
                "Not clipping yet, but there is no headroom left for a "
                "louder line.", value=f.rms_db))

        # 4. Proximity -- measurable, and fixable in the moment, which is
        #    what makes it worth the interruption.
        too_close, too_far = self._proximity_state(f)
        if self._sustained("close", t, too_close):
            return self._emit(t, Cue(
                "too_close", LIVE, WARN, "Closer than you started - back off a little",
                "The low end is building up from proximity effect -- at under "
                "10 cm that is 10-15 dB of extra bass that has to be EQ'd "
                "back out, taking body with it. Judged against where you "
                "were standing at the start of the take.",
                value=f.low_mid_ratio))

        if self._sustained("far", t, too_far):
            return self._emit(t, Cue(
                "too_far", LIVE, INFO, "Come in closer",
                "The voice is thin and the room is coming up with it.",
                value=f.low_mid_ratio))

        # 5. Plosives -- one head-turn fixes it permanently.
        if f.is_plosive:
            return self._emit(t, Cue(
                "plosive", LIVE, INFO, "Angle slightly off the mic",
                "A p/b/t blast hit the capsule. Turning your head about 30 "
                "degrees sends the air past the mic instead of into it."))
        return None


# ═════════════════════════════════════════════════════════════════════════════
# After the take
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class TakeReport:
    """What the take was actually like, and whether to keep it."""
    duration_s: float = 0.0
    voiced_s: float = 0.0
    peak_db: float = -120.0
    rms_db: float = -120.0
    noise_floor_db: float = -120.0
    snr_db: float = 0.0
    clipped_pct: float = 0.0
    median_pitch_error_cents: float = 0.0
    pitch_stability_cents: float = 0.0
    level_spread_db: float = 0.0
    proximity_drift: float = 0.0
    plosive_count: int = 0
    sibilance_ratio: float = 0.0
    usable: bool = True
    grade: str = "ok"                 # excellent | good | ok | rough | unusable
    cues: List[Cue] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["cues"] = [c.to_dict() for c in self.cues]
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 3) if np.isfinite(v) else None
        return d


def _nearest_scale_cents(midi: float, scale_midi: FloatSeq) -> float:
    """Distance in cents to the nearest allowed note."""
    if not len(scale_midi):
        return abs(midi - round(midi)) * 100.0
    arr = np.asarray(scale_midi, dtype=np.float64)
    pcs = np.unique(np.round(arr).astype(int) % 12)
    base = int(round(midi))
    best = min((abs(midi - cand) for cand in range(base - 2, base + 3)
                if cand % 12 in pcs), default=abs(midi - round(midi)))
    return float(best) * 100.0


def take_report(frames: Sequence[FrameStats], *,
                scale_midi: FloatSeq = (),
                performance_type: str = "sung",
                noise_floor_db: Optional[float] = None) -> TakeReport:
    """Summarise a finished take and say whether it is worth keeping.

    This is where detail belongs. The literature's retention benefits came
    from feedback paired with knowledge of results rather than from a live
    trace, and a singer reading this between takes has attention available
    in a way they did not while performing.

    Pitch is judged against measured human norms, not against perfection.
    Professionals average around 25 cents of deviation and non-professionals
    around 34.5, so a take is only called out for intonation once it is
    meaningfully worse than an ordinary human performance -- which is also
    the point beyond which the engine's own correction starts to sound like
    correction.
    """
    rep = TakeReport()
    if not frames:
        rep.usable, rep.grade = False, "unusable"
        rep.cues.append(Cue("empty", POST, FATAL, "Nothing was recorded"))
        return rep

    voiced = [f for f in frames if f.is_voice]
    rep.duration_s = float(frames[-1].time_s)
    rep.voiced_s = float(len(voiced)) * (frames[1].time_s - frames[0].time_s) \
        if len(frames) > 1 else 0.0
    rep.peak_db = max(f.peak_db for f in frames)
    rep.clipped_pct = 100.0 * sum(1 for f in frames if f.clipped) / len(frames)
    rep.plosive_count = sum(1 for f in frames if f.is_plosive)

    # The noise floor must be measured from frames that are *not* voice.
    # Taking a low percentile of every frame looks equivalent and is not: a
    # continuous take with no gaps between phrases has no quiet frames, so
    # the percentile lands on the voice itself, SNR computes as ~0 dB, and a
    # perfectly good performance is declared unusable. When there is no
    # silence to measure, the honest answer is that the floor is unknown --
    # not a number that happens to be wrong.
    quiet = np.array([f.rms_db for f in frames if not f.is_voice])
    floor_known = True
    if noise_floor_db is not None:
        rep.noise_floor_db = float(noise_floor_db)
    elif quiet.size >= 10:
        rep.noise_floor_db = float(np.percentile(quiet, 50))
    else:
        floor_known = False
        rep.noise_floor_db = float("-inf")

    if not voiced:
        rep.usable, rep.grade = False, "unusable"
        rep.cues.append(Cue("no_voice", POST, FATAL, "No voice detected",
                            "The recording contains only background noise."))
        return rep

    v_rms = np.array([f.rms_db for f in voiced])
    rep.rms_db = float(np.median(v_rms))
    rep.snr_db = (rep.rms_db - rep.noise_floor_db) if floor_known else float("nan")
    rep.level_spread_db = float(np.percentile(v_rms, 90) - np.percentile(v_rms, 10))
    rep.sibilance_ratio = float(np.median([f.sibilance_ratio for f in voiced]))
    prox = np.array([f.low_mid_ratio for f in voiced])
    rep.proximity_drift = float(np.percentile(prox, 90) - np.percentile(prox, 10))

    pitched = [f for f in voiced if f.pitch.voiced and f.pitch.midi > 0]
    if pitched:
        errs = [_nearest_scale_cents(f.pitch.midi, scale_midi) for f in pitched]
        rep.median_pitch_error_cents = float(np.median(errs))
        midis = np.array([f.pitch.midi for f in pitched])
        # Frame-to-frame wobble on sustained notes. Distinct from being flat
        # or sharp: this is unsteadiness rather than mis-aim.
        rep.pitch_stability_cents = float(np.median(np.abs(np.diff(midis))) * 100.0) \
            if midis.size > 1 else 0.0

    # ── Verdict ───────────────────────────────────────────────────────────
    cues: List[Cue] = []
    fatal = False

    if rep.clipped_pct > 0.5:
        fatal = True
        cues.append(Cue("clipped", POST, FATAL,
                        "%.1f%% of the take is clipped" % rep.clipped_pct,
                        "Distortion is baked in and cannot be removed. Lower "
                        "the gain and record it again.", value=rep.clipped_pct))
    elif rep.clipped_pct > 0.02:
        cues.append(Cue("clipped_light", POST, WARN,
                        "Occasional clipping (%.2f%%)" % rep.clipped_pct,
                        "Mostly repairable, but the loudest words will have "
                        "lost some detail.", value=rep.clipped_pct))

    # Only judge signal-to-noise when the floor was actually measurable.
    # Declaring a take unusable on the strength of an unmeasured quantity
    # is worse than saying nothing about it.
    if floor_known and np.isfinite(rep.snr_db) and rep.snr_db < 12.0:
        fatal = fatal or rep.snr_db < 6.0
        cues.append(Cue("snr", POST, FATAL if rep.snr_db < 6.0 else WARN,
                        "Signal-to-noise is low (%.0f dB)" % rep.snr_db,
                        "Background noise will be audible once the vocal is "
                        "compressed. Removing it costs breath and air.",
                        value=rep.snr_db))

    if rep.peak_db < -20.0:
        cues.append(Cue("quiet_take", POST, WARN,
                        "Recorded very quietly (peak %.0f dB)" % rep.peak_db,
                        "Usable, but you are throwing away resolution. Aim "
                        "for peaks near %.0f dB." % TARGET_PEAK_DB,
                        value=rep.peak_db))

    if rep.level_spread_db > 14.0:
        cues.append(Cue("level_spread", POST, INFO,
                        "Level varies a lot across the take (%.0f dB)"
                        % rep.level_spread_db,
                        "The engine rides this automatically, but a steadier "
                        "distance gives a more natural result than heavy "
                        "correction does.", value=rep.level_spread_db))

    if rep.proximity_drift > 1.2:
        cues.append(Cue("proximity_drift", POST, INFO,
                        "You moved around the mic during the take",
                        "The tone shifts between phrases as a result. Marking "
                        "a spot on the floor helps more than it sounds like "
                        "it should.", value=rep.proximity_drift))

    if rep.plosive_count > 4:
        cues.append(Cue("plosives", POST, INFO,
                        "%d plosive hits" % rep.plosive_count,
                        "Angle about 30 degrees off the mic, or add a pop "
                        "filter.", value=float(rep.plosive_count)))

    if rep.sibilance_ratio > 0.14:
        cues.append(Cue("sibilance", POST, INFO,
                        "Bright sibilance",
                        "The de-esser will handle it, but angling slightly "
                        "off-axis keeps more of the top end intact.",
                        value=rep.sibilance_ratio))

    # Pitch, judged against human norms rather than against the grid.
    pe = rep.median_pitch_error_cents
    if pe > 0:
        if pe <= PRO_PITCH_ERROR_CENTS:
            cues.append(Cue("pitch_good", POST, GOOD,
                            "Intonation is tight (%.0f cents)" % pe,
                            "That is inside the range measured for "
                            "professional singers.", value=pe))
        elif pe <= AMATEUR_PITCH_ERROR_CENTS + 8.0:
            cues.append(Cue("pitch_ok", POST, INFO,
                            "Intonation is normal (%.0f cents)" % pe,
                            "Typical for a good take. Correction will be "
                            "light and should stay inaudible.", value=pe))
        else:
            cues.append(Cue("pitch_loose", POST, WARN,
                            "Intonation is loose (%.0f cents off)" % pe,
                            "Correcting this much starts to become audible. "
                            "Another take against a pitch reference will "
                            "sound more natural than heavy tuning.", value=pe))

    if rep.pitch_stability_cents > 45.0 and performance_type == "sung":
        cues.append(Cue("wobble", POST, INFO,
                        "Sustained notes are unsteady",
                        "More breath support will hold long notes still.",
                        value=rep.pitch_stability_cents))

    rep.cues = cues
    rep.usable = not fatal
    if fatal:
        rep.grade = "unusable"
    elif any(c.severity == WARN for c in cues):
        rep.grade = "rough" if len([c for c in cues if c.severity == WARN]) > 1 else "ok"
    elif pe and pe <= PRO_PITCH_ERROR_CENTS and (
            not floor_known or rep.snr_db > 24.0):
        rep.grade = "excellent"
    else:
        rep.grade = "good"
    return rep


def rank_takes(reports: Sequence[TakeReport]) -> List[int]:
    """Order take indices best-first.

    Usability dominates: a clipped take with perfect intonation is still
    unusable, and ranking it above a merely-good one would be wrong.
    """
    order = {"excellent": 4, "good": 3, "ok": 2, "rough": 1, "unusable": 0}

    def score(i_r: Tuple[int, TakeReport]) -> Tuple:
        _, r = i_r
        return (r.usable, order.get(r.grade, 0), r.snr_db,
                -r.median_pitch_error_cents, -r.clipped_pct)

    return [i for i, _ in sorted(enumerate(reports), key=score, reverse=True)]
