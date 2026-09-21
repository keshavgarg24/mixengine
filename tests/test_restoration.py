"""
Tests for genre detection, bleed cancellation and acoustic space matching.

The bleed and space tests build their own ground truth: a known impulse
response convolved onto a known signal, so the measurement can be checked
against the number it was given rather than against an opinion. Several of
these lock down failures that produced confident, plausible, wrong output --
which is the only kind worth writing a test for.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mixengine.analysis import genre                                     # noqa: E402
from mixengine.audio import debleed, dsp, space                          # noqa: E402

SR = 22050


def clicks(duration=10.0, every=0.55, sr=SR, freq=1200.0):
    y = np.zeros(int(duration * sr), dtype=np.float32)
    n = int(sr * 0.004)
    w = (np.hanning(n) * np.sin(2 * np.pi * freq * np.arange(n) / sr)
         ).astype(np.float32)
    for t in np.arange(0.2, duration - 0.5, every):
        i = int(t * sr)
        y[i:i + n] += 0.8 * w
    return y


def convolve(x, ir):
    from scipy.signal import fftconvolve
    return fftconvolve(x, ir)[:len(x)].astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════

class TestGenre(unittest.TestCase):

    def _detect(self, **kw):
        base = {"bpm": 140.0,
                "spectral": {"centroid_hz": 3400.0, "flatness": 0.02,
                             "band_energy": {"sub": 0.17, "air": 0.5}},
                "dynamic_range_db": 13.0, "swing_ratio": 0.5}
        base.update(kw)
        return genre.detect(None, SR, **base)

    def test_a_producer_tag_wins_outright(self):
        r = self._detect(tagged="lofi")
        self.assertEqual(r.genre, "lofi")
        self.assertEqual(r.source, "tagged")
        self.assertTrue(r.usable)

    def test_an_unknown_tag_falls_through_to_detection(self):
        r = self._detect(tagged="phonk")
        self.assertEqual(r.source, "detected")
        self.assertIn("phonk", r.note)

    def test_no_tempo_declines(self):
        r = self._detect(bpm=0.0)
        self.assertIsNone(r.genre)
        self.assertFalse(r.usable)

    def test_a_close_call_inside_one_family_is_still_usable(self):
        """Trap and drill land within a few hundredths of each other because
        they *are* close, and their mix profiles differ by half a dB.
        Abstaining to the neutral default there gives up more than picking
        either one could cost."""
        r = self._detect()
        self.assertIn(r.genre, ("trap", "drill"))
        self.assertEqual(r.family, "rap_808")
        self.assertTrue(r.usable)

    def test_a_close_call_across_families_abstains(self):
        forced = genre.GenreResult(
            genre="trap", confidence=0.8, margin=0.02, runner_up="rnb")
        self.assertFalse(forced.usable)
        self.assertEqual(forced.family, "rap_808")

    def test_halftime_tempo_still_reads_as_trap(self):
        fast = self._detect(bpm=140.0)
        slow = self._detect(bpm=70.0)
        self.assertEqual(genre.FAMILIES.get(fast.genre),
                         genre.FAMILIES.get(slow.genre))

    def test_lofi_is_distinguished_by_its_spectrum(self):
        r = self._detect(bpm=82.0, swing_ratio=0.60,
                         dynamic_range_db=18.0,
                         spectral={"centroid_hz": 1200.0, "flatness": 0.12,
                                   "band_energy": {"sub": 0.08, "air": 0.10}})
        self.assertEqual(r.genre, "lofi")

    def test_drum_and_bass_is_distinguished_by_tempo(self):
        r = self._detect(bpm=174.0, onsets=list(np.arange(0, 30, 0.12)),
                         beats=list(np.arange(0, 30, 60 / 174.0)))
        self.assertEqual(r.genre, "drum_and_bass")

    def test_evidence_is_present_for_a_confident_answer(self):
        r = self._detect()
        self.assertTrue(r.evidence)

    def test_missing_rhythm_features_do_not_penalise_every_genre(self):
        """Scoring an unmeasured zero as 'outside the range' would punish
        every genre that expects hi-hats whenever the audio was absent."""
        with_audio = self._detect()
        self.assertGreater(with_audio.confidence, 0.4)


class TestDebleed(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(3)
        self.beat = np.zeros(int(SR * 12), dtype=np.float32)
        for t in np.arange(0.1, 11.5, 0.31):
            i = int(t * SR)
            n = int(SR * 0.05)
            env = np.exp(-np.arange(n) / (SR * 0.02))
            f = 80.0 if int(t / 0.31) % 4 == 0 else 3500.0
            self.beat[i:i + n] += (0.6 * env
                                   * np.sin(2 * np.pi * f * np.arange(n) / SR))

        self.voice = np.zeros_like(self.beat)
        self.gaps = np.ones(len(self.beat), dtype=bool)
        for k in range(6):
            s = int(SR * (0.4 + k * 1.8))
            n = int(SR * 1.0)
            if s + n > len(self.voice):
                break
            t = np.arange(n) / SR
            f = 220.0 * 2 ** (rng.integers(0, 8) / 12.0)
            self.voice[s:s + n] = ((0.35 * np.sin(2 * np.pi * f * t)
                                    + 0.1 * np.sin(2 * np.pi * 2 * f * t))
                                   * np.hanning(n))
            self.gaps[s:s + n] = False

        ir = np.zeros(int(SR * 0.09), dtype=np.float32)
        ir[0] = 1.0
        ir[int(SR * 0.01)] = 0.45
        ir[int(SR * 0.025)] = 0.28
        self.offset = int(SR * 0.83)
        bleed = convolve(self.beat, ir)
        self.bleed = np.zeros_like(bleed)
        self.bleed[self.offset:] = bleed[:len(bleed) - self.offset]

    def _mic(self, level_db):
        return (self.voice + self.bleed * dsp.db_to_lin(level_db))[:, None]

    def test_removes_the_bleed_and_keeps_the_voice(self):
        mic = self._mic(-14.0)
        out, rep = debleed.cancel(mic, self.beat[:, None], SR)
        self.assertTrue(rep["applied"], rep.get("note"))
        clean = dsp.to_mono(dsp.as_2d(out))

        def err(x):
            e = x[:len(self.voice)] - self.voice
            return 10 * np.log10(np.sum(e ** 2) / np.sum(self.voice ** 2))

        self.assertLess(err(clean), err(dsp.to_mono(mic)) - 6.0)

    def test_never_makes_the_recording_louder(self):
        """Subtracting a wrong-phase estimate adds energy. Before the output
        was clamped to the input magnitude, this stage made the gaps between
        phrases 15.6 dB louder than it found them, and one sample reached
        22.0 from an input peaking at 0.47."""
        mic = self._mic(-14.0)
        out, _ = debleed.cancel(mic, self.beat[:, None], SR)
        self.assertLessEqual(float(np.abs(out).max()),
                             float(np.abs(mic).max()) * 1.05)

    def test_the_gaps_get_quieter_not_louder(self):
        mic = self._mic(-14.0)
        out, rep = debleed.cancel(mic, self.beat[:, None], SR)
        if not rep["applied"]:
            self.skipTest("nothing was cancelled on this fixture")
        a = float(np.sum(dsp.to_mono(mic)[self.gaps] ** 2))
        b = float(np.sum(dsp.to_mono(dsp.as_2d(out))[self.gaps] ** 2))
        self.assertLess(b, a)

    def test_a_clean_take_passes_through_untouched(self):
        out, rep = debleed.cancel(self.voice[:, None], self.beat[:, None], SR)
        self.assertFalse(rep["applied"])
        np.testing.assert_allclose(out, self.voice[:, None], atol=1e-6)

    def test_the_true_offset_is_among_the_candidates(self):
        """A looping beat's correlation peaks once per bar, so the tallest
        peak is regularly the wrong one -- which is why the caller tries
        several and keeps whichever cancels."""
        offs, _ = debleed.offset_candidates(self._mic(-14.0),
                                            self.beat[:, None], SR,
                                            k=debleed.N_CANDIDATES)
        truth = self.offset / SR
        self.assertTrue(np.any(np.abs(offs - truth) < 0.05),
                        f"true offset {truth:.3f} not in {offs[:8]}")

    def test_istft_does_not_amplify_the_edges(self):
        """The first and last half-window are covered by only part of the
        window sum. A round-trip test that skipped those regions passed
        while the real path produced a sample of 22.0 from a 0.47 peak."""
        win = np.hanning(debleed.N_FFT).astype(np.float32)
        x = np.sin(2 * np.pi * 300 * np.arange(SR * 3) / SR).astype(np.float32)
        X = debleed._stft(x, win, debleed.HOP)
        X = X * 0.5                     # any modification breaks the cancelling
        y = debleed._istft(X, win, debleed.HOP, len(x))
        self.assertLessEqual(float(np.abs(y).max()),
                             float(np.abs(x).max()) * 1.05)

    def test_empty_input_is_safe(self):
        out, rep = debleed.cancel(np.zeros((0, 1)), self.beat[:, None], SR)
        self.assertFalse(rep["applied"])


class TestSpace(unittest.TestCase):

    def test_recovers_a_known_rt60(self):
        for truth in (0.25, 0.55, 1.10):
            with self.subTest(rt60=truth):
                ir = space.synth_rir(SR, truth)
                p = space.estimate(convolve(clicks(12.0), ir), SR)
                self.assertTrue(p.measured, p.note)
                self.assertLess(abs(p.rt60_mean - truth) / truth, 0.20)

    def test_a_dry_signal_reports_no_room(self):
        p = space.estimate(clicks(10.0), SR)
        self.assertFalse(p.measured)
        self.assertTrue(p.note)

    def test_disagreeing_estimates_are_not_acted_on(self):
        """A median over measurements that disagree is not describing one
        room, and convolving an invented space onto the vocal is worse than
        leaving the mismatch."""
        p = space.SpaceProfile(rt60_mean=0.5, n_decays=40, confidence=0.05)
        self.assertFalse(p.measured)

    def test_match_only_ever_adds_reverberation(self):
        big = space.estimate(convolve(clicks(12.0), space.synth_rir(SR, 1.2)), SR)
        small = space.estimate(convolve(clicks(12.0), space.synth_rir(SR, 0.3)), SR)
        wet_vocal = convolve(clicks(12.0), space.synth_rir(SR, 1.2))[:, None]
        _, rep = space.match(wet_vocal, SR, small, current=big)
        self.assertFalse(rep["applied"])
        self.assertIn("dereverb", rep["note"])

    def test_a_dry_vocal_gets_a_real_wet_level(self):
        """Treating an unmeasurable vocal as a zero-decibel DRR gap floored
        the wet level at 1%, so a dry take going into a 0.9-second room came
        out dry."""
        target = space.estimate(
            convolve(clicks(12.0), space.synth_rir(SR, 0.9)), SR)
        _, rep = space.match(clicks(8.0)[:, None], SR, target)
        self.assertTrue(rep["applied"], rep.get("note"))
        self.assertGreater(rep["wet"], 0.05)

    def test_wet_level_is_capped(self):
        target = space.estimate(
            convolve(clicks(12.0), space.synth_rir(SR, 2.0)), SR)
        if not target.measured:
            self.skipTest("target room not measurable on this fixture")
        _, rep = space.match(clicks(8.0)[:, None], SR, target, max_wet=0.2)
        self.assertLessEqual(rep["wet"], 0.2 + 1e-9)

    def test_unmeasured_target_declines(self):
        _, rep = space.match(clicks(8.0)[:, None], SR, space.SpaceProfile())
        self.assertFalse(rep["applied"])

    def test_synth_rir_respects_per_band_decays(self):
        """A single broadband decay leaves the top end ringing long after a
        real surface would have absorbed it."""
        ir = space.synth_rir(SR, 1.0, {"500_2000": 1.0, "2000_6000": 0.15,
                                       "6000_12000": 0.12})
        low = dsp.to_mono(dsp.bandpass(ir[:, None], SR, 500.0, 2000.0))
        high = dsp.to_mono(dsp.bandpass(ir[:, None], SR, 2000.0, 6000.0))
        tail = slice(int(SR * 0.4), None)
        self.assertGreater(float(np.sqrt(np.mean(low[tail] ** 2))),
                           float(np.sqrt(np.mean(high[tail] ** 2))) * 2)

    def test_reverb_fraction_maps_drr_sensibly(self):
        self.assertAlmostEqual(space._reverb_fraction(0.0), 0.5, places=6)
        self.assertGreater(space._reverb_fraction(-10.0), 0.9)
        self.assertLess(space._reverb_fraction(10.0), 0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestSpacePerBand(unittest.TestCase):
    """A pad holding a note is internally consistent and reads as a long
    decay. It must be recognised as contamination from the room prior, not
    allowed to veto the bands that are measuring the room."""

    def _room_with_pad(self, rt60=0.30):
        y = convolve(clicks(12.0, freq=6000.0), space.synth_rir(SR, rt60))
        # A sustained 700 Hz pad for the whole take: a "decay" of ~forever
        # in the 500-2000 band.
        t = np.arange(y.size) / SR
        y = y + (0.12 * np.sin(2 * np.pi * 700 * t)).astype(np.float32)
        return y

    def test_a_sustained_mid_band_does_not_veto_the_measurement(self):
        p = space.estimate(self._room_with_pad(0.30), SR)
        self.assertTrue(p.measured, p.note)
        self.assertLess(abs(p.rt60_mean - 0.30) / 0.30, 0.35)

    def test_pad_band_is_excluded_from_the_room(self):
        """A held note yields no decay at all, or a wildly long one. Either
        way it must not reach `rt60_by_band`, which is what synthesis uses."""
        p = space.estimate(self._room_with_pad(0.30), SR)
        self.assertNotIn("500_2000", p.rt60_by_band, p.band_detail)
        self.assertTrue(p.measured, p.note)

    def test_missing_bands_are_extended_by_the_room_prior(self):
        filled = space.fill_bands({"6000_12000": 0.30}, 0.30)
        self.assertGreater(filled["120_500"], filled["500_2000"])
        self.assertGreater(filled["500_2000"], filled["2000_6000"])
        self.assertGreater(filled["2000_6000"], 0.30)
        self.assertAlmostEqual(filled["6000_12000"], 0.30)

    def test_tail_decay_is_used_on_dense_material(self):
        """Continuous material offers no inter-onset decays; the fade into
        silence at the end is the one measurement it still provides."""
        rt = 0.40
        ir = space.synth_rir(SR, rt)
        dense = convolve(clicks(10.0, every=0.06, freq=5000.0), ir)
        # Cut the last click train short so the final decay is clean.
        dense[-int(SR * 2.5):] = convolve(clicks(2.5, every=5.0, freq=5000.0), ir)
        p = space.estimate(dense, SR)
        self.assertTrue(p.measured, p.note)
        self.assertLess(abs(p.rt60_mean - rt) / rt, 0.4)
