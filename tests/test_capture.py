"""
Tests for vocal capture: frame analysis, pitch tracking, and coaching.

The pitch tests check accuracy against synthesised tones of known
frequency, so a regression shows up as a number rather than as a feeling.

The coaching tests are the more important ones, because they encode a
design constraint that is easy to lose: concurrent feedback during a take
measurably degrades the performance, so the live coach must stay quiet
about anything that is either fixable downstream or not worth breaking
concentration for. A test suite that only checked "does it produce cues"
would happily pass a system that chatters the take into ruin.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mixengine.capture import realtime                          # noqa: E402
from mixengine.capture.coach import (                                  # noqa: E402
    FATAL, POST, PREROLL, LiveCoach,
    LiveCoachConfig, preroll_guidance, rank_takes, take_report,
)
from mixengine.capture.realtime import (                               # noqa: E402
    FrameAnalyzer, FrameStats, PitchEstimate, PitchTracker,
    midi_to_hz, nsdf,
)

SR = 48000


def tone(hz, seconds=0.5, sr=SR, amp=0.3, harmonics=(1, 2, 3, 4),
         gains=(1.0, 0.5, 0.25, 0.12), seed=0, noise=0.0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    y = np.zeros(t.size)
    for h, g in zip(harmonics, gains):
        y += g * np.sin(2 * np.pi * hz * h * t)
    y = y / max(np.max(np.abs(y)), 1e-9) * amp
    if noise > 0:
        y += rng.normal(0, noise, t.size)
    return y.astype(np.float32)


def frames_from(signal, sr=SR):
    fa = FrameAnalyzer(sr=sr)
    return fa.push(signal), fa


def synth_frames(n, *, rms_db=-18.0, peak_db=-6.0, voiced=True, midi=60.0,
                 clarity=0.95, low_mid=1.0, sib=0.05, clipped=False,
                 plosive=False, hop_s=0.00533, t0=0.0):
    out = []
    for i in range(n):
        out.append(FrameStats(
            time_s=t0 + i * hop_s, rms_db=rms_db, peak_db=peak_db,
            clipped=clipped, near_clip=peak_db > -1.0,
            pitch=PitchEstimate(hz=midi_to_hz(midi), midi=midi,
                                clarity=clarity, voiced=voiced),
            low_mid_ratio=low_mid, sibilance_ratio=sib,
            is_voice=voiced, is_plosive=plosive))
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Pitch
# ═════════════════════════════════════════════════════════════════════════════

class TestNSDF(unittest.TestCase):

    def test_is_bounded(self):
        """The bound is the reason MPM was chosen over YIN: it makes the
        clarity value comparable across frames and material."""
        for hz in (110.0, 220.0, 440.0):
            n = nsdf(tone(hz, 0.05).astype(np.float64))
            self.assertLessEqual(float(np.max(n)), 1.0 + 1e-9)
            self.assertGreaterEqual(float(np.min(n)), -1.0 - 1e-9)

    def test_peaks_at_the_period(self):
        sr, hz = SR, 200.0
        x = tone(hz, 0.05, sr).astype(np.float64)
        x = x - x.mean()
        n = nsdf(x)
        lag = int(np.argmax(n[40:400])) + 40
        self.assertAlmostEqual(sr / lag, hz, delta=hz * 0.03)

    def test_empty_input_is_safe(self):
        self.assertEqual(nsdf(np.zeros(2)).size, 0)


class TestPitchTracker(unittest.TestCase):

    def track(self, hz, **kw):
        tr = PitchTracker(SR)
        sig = tone(hz, 0.4, **kw)
        ests = [tr(sig[i:i + 2048]) for i in range(0, sig.size - 2048, 256)]
        return [e for e in ests if e.voiced]

    def test_accuracy_across_the_vocal_range(self):
        """Within 10 cents across the range a singer actually uses."""
        for hz in (98.0, 146.8, 220.0, 329.6, 523.3):
            got = self.track(hz)
            self.assertTrue(got, "no voiced frames at %.1f Hz" % hz)
            med = float(np.median([e.hz for e in got]))
            cents = abs(realtime.cents_between(med, hz))
            self.assertLess(cents, 10.0,
                            "%.1f Hz -> %.1f Hz (%.1f cents)" % (hz, med, cents))

    def test_parabolic_interpolation_beats_integer_lags(self):
        """Without sub-sample refinement the estimate is quantised to
        integer lags, which is worse than 10 cents in the upper register."""
        hz = 440.0
        got = self.track(hz)
        med = float(np.median([e.hz for e in got]))
        raw_lag = round(SR / hz)
        quantised = SR / raw_lag
        self.assertLessEqual(abs(realtime.cents_between(med, hz)),
                             abs(realtime.cents_between(quantised, hz)) + 6.0)

    def test_clarity_is_high_on_a_clean_tone(self):
        self.assertGreater(float(np.median([e.clarity for e in self.track(220.0)])),
                           0.85)

    def test_noise_is_reported_as_unvoiced(self):
        tr = PitchTracker(SR)
        rng = np.random.default_rng(0)
        noise = rng.normal(0, 0.2, SR // 2).astype(np.float32)
        voiced = [tr(noise[i:i + 2048]).voiced
                  for i in range(0, noise.size - 2048, 256)]
        self.assertLess(sum(voiced) / max(len(voiced), 1), 0.5)

    def test_silence_is_unvoiced(self):
        tr = PitchTracker(SR)
        self.assertFalse(tr(np.zeros(2048, dtype=np.float32)).voiced)

    def test_dc_offset_does_not_break_detection(self):
        """MPM is not DC-invariant -- an offset drives the NSDF toward 1 at
        every lag and hides the zero crossings. The tracker must remove it."""
        tr = PitchTracker(SR)
        sig = tone(220.0, 0.4) + 0.5            # large constant offset
        got = [tr(sig[i:i + 2048]) for i in range(0, sig.size - 2048, 256)]
        voiced = [e for e in got if e.voiced]
        self.assertTrue(voiced)
        self.assertAlmostEqual(float(np.median([e.hz for e in voiced])), 220.0,
                               delta=8.0)

    def test_octave_jumps_are_repaired(self):
        """No voice moves an octave between adjacent 5 ms frames."""
        tr = PitchTracker(SR)
        tr._prev_midi, tr._prev_voiced = 57.0, True    # A3
        est = tr(tone(midi_to_hz(69.0), 0.05))         # detector says A4
        if est.voiced:
            self.assertLess(abs(est.midi - 57.0), 6.0)

    def test_survives_moderate_noise(self):
        got = self.track(220.0, noise=0.02)
        self.assertTrue(got)
        self.assertAlmostEqual(float(np.median([e.hz for e in got])), 220.0,
                               delta=10.0)


class TestFrameAnalyzer(unittest.TestCase):

    def test_hop_rate_and_buffering(self):
        """Block size is decoupled from hop size, because a device delivers
        whatever it likes and the caller should not have to care."""
        fa = FrameAnalyzer(sr=SR)
        sig = tone(220.0, 1.0)
        got = []
        for i in range(0, sig.size, 777):           # deliberately odd blocks
            got.extend(fa.push(sig[i:i + 777]))
        expected = (sig.size - 2048) // 256
        self.assertAlmostEqual(len(got), expected, delta=3)

    def test_detects_voice_and_level(self):
        frames, _ = frames_from(tone(220.0, 0.5, amp=0.3))
        voiced = [f for f in frames if f.is_voice]
        self.assertTrue(voiced)
        self.assertAlmostEqual(float(np.median([f.rms_db for f in voiced])),
                               -16.0, delta=8.0)

    def test_detects_clipping(self):
        sig = np.clip(tone(220.0, 0.3, amp=2.0), -1.0, 1.0)
        frames, _ = frames_from(sig)
        self.assertTrue(any(f.clipped for f in frames))

    def test_proximity_ratio_tracks_low_end(self):
        """Proximity effect is a measurable low/mid imbalance, which is what
        makes 'back off the mic' an objective instruction."""
        close = tone(180.0, 0.4, harmonics=(1, 2), gains=(1.0, 0.15))
        far = tone(180.0, 0.4, harmonics=(1, 2, 3, 4), gains=(0.3, 1.0, 0.8, 0.6))
        fc = [f for f in frames_from(close)[0] if f.is_voice]
        ff = [f for f in frames_from(far)[0] if f.is_voice]
        self.assertGreater(float(np.median([f.low_mid_ratio for f in fc])),
                           float(np.median([f.low_mid_ratio for f in ff])))

    def test_silence_yields_no_voice(self):
        frames, _ = frames_from(np.zeros(SR // 2, dtype=np.float32))
        self.assertFalse(any(f.is_voice for f in frames))

    def test_noise_floor_is_learned(self):
        fa = FrameAnalyzer(sr=SR)
        rng = np.random.default_rng(0)
        fa.push(rng.normal(0, 10 ** (-50 / 20.0), SR).astype(np.float32))
        self.assertLess(fa.noise_floor_db, -35.0)


# ═════════════════════════════════════════════════════════════════════════════
# Coaching -- the design constraints
# ═════════════════════════════════════════════════════════════════════════════

class TestLiveCoachStaysQuiet(unittest.TestCase):
    """The core constraint. Concurrent feedback degrades the take, so the
    live coach must not mention anything recoverable."""

    def test_says_nothing_about_pitch_ever(self):
        c = LiveCoach()
        # A minute of confidently, badly out-of-tune singing.
        frames = synth_frames(11000, midi=60.4, clarity=0.95)
        ids = {cue.id for cue in c.update(frames)}
        for banned in ("pitch", "flat", "sharp", "tuning", "intonation"):
            self.assertFalse(any(banned in i for i in ids),
                             "live coach mentioned pitch: %s" % ids)

    def test_says_nothing_about_timing(self):
        c = LiveCoach()
        ids = {cue.id for cue in c.update(synth_frames(11000))}
        for banned in ("timing", "rush", "drag", "late", "early", "grid"):
            self.assertFalse(any(banned in i for i in ids))

    def test_a_good_take_is_never_interrupted(self):
        c = LiveCoach()
        cues = c.update(synth_frames(11000, rms_db=-18.0, peak_db=-6.0,
                                     low_mid=1.0))
        self.assertEqual(cues, [], "interrupted a clean take: %s" % cues)


class TestLiveCoachSpeaksWhenItMatters(unittest.TestCase):
    """Only for faults that destroy the recording."""

    def test_clipping_is_reported_immediately(self):
        c = LiveCoach()
        cues = c.update(synth_frames(40, clipped=True, peak_db=0.0))
        self.assertTrue(any(cue.id == "clipping" for cue in cues))
        self.assertEqual([cue.severity for cue in cues if cue.id == "clipping"][0],
                         FATAL)

    def test_clipping_overrides_the_attention_budget(self):
        """Staying quiet to protect concentration is pointless if the take
        is being destroyed while we stay quiet."""
        c = LiveCoach()
        c.update(synth_frames(300, low_mid=3.0))        # spend the budget
        cues = c.update(synth_frames(40, clipped=True, peak_db=0.0, t0=2.0))
        self.assertTrue(any(cue.id == "clipping" for cue in cues))

    def test_too_close_is_reported(self):
        c = LiveCoach()
        cues = c.update(synth_frames(400, low_mid=3.0))
        self.assertTrue(any(cue.id == "too_close" for cue in cues))

    def test_too_quiet_is_reported(self):
        c = LiveCoach()
        cues = c.update(synth_frames(400, rms_db=-46.0))
        self.assertTrue(any(cue.id == "too_quiet" for cue in cues))

    def test_transient_conditions_are_ignored(self):
        """A single frame is noise, not a condition."""
        c = LiveCoach()
        cues = c.update(synth_frames(3, low_mid=3.0) + synth_frames(400))
        self.assertFalse(any(cue.id == "too_close" for cue in cues))


class TestAttentionBudget(unittest.TestCase):

    def test_rate_limit_is_enforced(self):
        """The limiter is not politeness -- it is what stops the coach
        reproducing the effect the research warns about."""
        cfg = LiveCoachConfig(max_cues_per_minute=3, min_gap_s=6.0)
        c = LiveCoach(cfg)
        frames = []
        for k in range(12):
            frames += synth_frames(200, low_mid=3.0, t0=k * 2.0)
            frames += synth_frames(200, rms_db=-46.0, t0=k * 2.0 + 1.0)
        in_first_minute = [q for q in c.update(frames) if (q.at_s or 0) < 60.0
                           and q.severity != FATAL]
        self.assertLessEqual(len(in_first_minute), 3)

    def test_same_cue_is_not_repeated_immediately(self):
        c = LiveCoach()
        cues = c.update(synth_frames(3000, low_mid=3.0))
        times = [q.at_s for q in cues if q.id == "too_close"]
        for a, b in zip(times, times[1:]):
            self.assertGreaterEqual(b - a, 15.0)


class TestPreroll(unittest.TestCase):

    def test_musical_guidance_requires_a_beat(self):
        with_beat = preroll_guidance(beat_key_name="A Minor", beat_bpm=140,
                                     beat_bars=16, has_beat=True)
        bare = preroll_guidance(has_beat=False)
        self.assertTrue(any(c.id == "key" for c in with_beat))
        self.assertTrue(any(c.id == "tempo" for c in with_beat))
        self.assertFalse(any(c.id == "key" for c in bare))
        self.assertFalse(any(c.id == "tempo" for c in bare))

    def test_technical_guidance_applies_either_way(self):
        for has_beat in (True, False):
            cues = preroll_guidance(has_beat=has_beat)
            self.assertTrue(any(c.id == "mic_distance" for c in cues))

    def test_range_check_catches_a_key_that_is_too_high(self):
        """The most valuable pre-roll item: no downstream processing fixes
        a melody outside the singer's range."""
        cues = preroll_guidance(scale_midi=[76, 78, 79, 81, 83],
                                voice_low_midi=48, voice_high_midi=64)
        self.assertTrue(any(c.id == "range_high" for c in cues))

    def test_range_check_passes_a_suitable_key(self):
        cues = preroll_guidance(scale_midi=[55, 57, 59, 60, 62],
                                voice_low_midi=48, voice_high_midi=67)
        self.assertTrue(any(c.id == "range_ok" for c in cues))

    def test_unusable_room_is_flagged_as_fatal(self):
        cues = preroll_guidance(rt60_s=0.85)
        fatal = [c for c in cues if c.severity == FATAL]
        self.assertTrue(fatal)
        self.assertEqual(fatal[0].id, "room_bad")

    def test_all_cues_are_preroll_level(self):
        for c in preroll_guidance(beat_key_name="C Major", beat_bpm=120,
                                  rt60_s=0.7, noise_floor_db=-30):
            self.assertEqual(c.level, PREROLL)


class TestTakeReport(unittest.TestCase):

    def test_clean_take_grades_well(self):
        # Real takes have gaps between phrases; those quiet frames are what
        # the noise floor is measured from.
        frames = (synth_frames(600, rms_db=-18.0, peak_db=-6.0)
                  + synth_frames(120, rms_db=-58.0, voiced=False, t0=3.2)
                  + synth_frames(600, rms_db=-18.0, peak_db=-6.0, t0=3.9))
        r = take_report(frames, scale_midi=[60, 62, 64, 65, 67, 69, 71])
        self.assertTrue(r.usable)
        self.assertIn(r.grade, ("excellent", "good", "ok"))

    def test_clipped_take_is_unusable(self):
        r = take_report(synth_frames(2000, clipped=True, peak_db=0.0))
        self.assertFalse(r.usable)
        self.assertEqual(r.grade, "unusable")
        self.assertTrue(any(c.severity == FATAL for c in r.cues))

    def test_empty_take_is_handled(self):
        r = take_report([])
        self.assertFalse(r.usable)
        self.assertEqual(r.grade, "unusable")

    def test_silence_only_take_is_unusable(self):
        r = take_report(synth_frames(500, voiced=False, rms_db=-70.0))
        self.assertFalse(r.usable)

    def test_pitch_is_judged_against_human_norms(self):
        """Professionals average 25 cents of error. A threshold tighter than
        that would flag expert singing as wrong."""
        tight = take_report(synth_frames(600, midi=60.05),
                            scale_midi=[60, 62, 64, 65, 67, 69, 71])
        loose = take_report(synth_frames(600, midi=60.55),
                            scale_midi=[60, 62, 64, 65, 67, 69, 71])
        self.assertTrue(any(c.id == "pitch_good" for c in tight.cues))
        self.assertTrue(any(c.id == "pitch_loose" for c in loose.cues))

    def test_normal_human_intonation_is_not_criticised(self):
        r = take_report(synth_frames(600, midi=60.30),   # 30 cents -- normal
                        scale_midi=[60, 62, 64, 65, 67, 69, 71])
        self.assertFalse(any(c.id == "pitch_loose" for c in r.cues))

    def test_low_snr_is_reported(self):
        frames = (synth_frames(300, rms_db=-40.0, voiced=True)
                  + synth_frames(300, rms_db=-44.0, voiced=False))
        r = take_report(frames)
        self.assertTrue(any(c.id == "snr" for c in r.cues))

    def test_no_silent_frames_does_not_fake_an_snr_verdict(self):
        """A continuous take has no gaps to measure the floor from. Saying
        nothing beats declaring a good performance unusable."""
        r = take_report(synth_frames(900, rms_db=-18.0, peak_db=-6.0))
        self.assertTrue(r.usable)
        self.assertFalse(any(c.id == "snr" for c in r.cues))

    def test_plosives_are_counted(self):
        frames = synth_frames(200) + synth_frames(8, plosive=True) + synth_frames(200)
        self.assertEqual(take_report(frames).plosive_count, 8)

    def test_all_cues_are_post_level(self):
        r = take_report(synth_frames(600, clipped=True))
        for c in r.cues:
            self.assertEqual(c.level, POST)

    def test_report_is_json_safe(self):
        import json
        json.dumps(take_report(synth_frames(600)).to_dict())


class TestRankTakes(unittest.TestCase):

    def test_usable_always_beats_unusable(self):
        good = take_report(synth_frames(600, rms_db=-18.0))
        bad = take_report(synth_frames(600, clipped=True, peak_db=0.0))
        self.assertEqual(rank_takes([bad, good])[0], 1)

    def test_empty_list(self):
        self.assertEqual(rank_takes([]), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestProximityBaseline(unittest.TestCase):
    """Proximity is relative to where the singer started, because the
    absolute low/mid ratio is a property of the voice, not the distance."""

    def test_a_singer_who_starts_close_and_stays_close_is_not_nagged(self):
        c = LiveCoach()
        # Ratio 2.2 would have tripped the old absolute threshold of 1.9.
        cues = c.update(synth_frames(1600, low_mid=2.2))
        self.assertFalse(any(cue.id == "too_close" for cue in cues))
        self.assertIsNotNone(c.proximity_baseline)

    def test_moving_closer_than_the_baseline_is_reported(self):
        c = LiveCoach()
        frames = synth_frames(1000, low_mid=1.0)
        frames += synth_frames(400, low_mid=2.6, t0=frames[-1].time_s + 0.005)
        cues = c.update(frames)
        self.assertTrue(any(cue.id == "too_close" for cue in cues))

    def test_moving_away_from_the_baseline_is_reported(self):
        c = LiveCoach()
        frames = synth_frames(1000, low_mid=1.0)
        frames += synth_frames(400, low_mid=0.35, t0=frames[-1].time_s + 0.005)
        cues = c.update(frames)
        self.assertTrue(any(cue.id == "too_far" for cue in cues))

    def test_extreme_values_still_fire_before_a_baseline_exists(self):
        c = LiveCoach()
        cues = c.update(synth_frames(400, low_mid=3.0))
        self.assertTrue(any(cue.id == "too_close" for cue in cues))
        self.assertIsNone(c.proximity_baseline)   # under 4 s of voice
