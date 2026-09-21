/**
 * Live coaching, rate limited.
 *
 * A mirror of `capture/coach.py`'s `LiveCoachConfig`. Python remains the
 * source of truth: the post-take report, which is what a take is actually
 * judged on, is computed there from the recorded audio. These thresholds
 * exist so that what is said *during* a take agrees with what is said after
 * it. If one changes, change both.
 *
 * The rate limiter is not politeness. Concurrent visual feedback measurably
 * degrades the performance it supervises — studies of singers training with
 * live pitch displays report the decrement appearing at the moment feedback
 * is delivered, with the benefit accruing to later sessions rather than the
 * take in progress. So this reports only faults that destroy a recording and
 * cannot be repaired afterwards, and it reports them sparingly.
 *
 * Nothing here mentions pitch, timing, vibrato or expression. Every one of
 * those is either fixable downstream or actively harmed by interrupting to
 * mention it.
 */

export const CONFIG = {
  minGapS: 6.0,
  maxCuesPerMinute: 4,
  sustainS: 0.8,
  clipFramesBeforeAlert: 3,
  quietDb: -38.0,
  loudRmsDb: -9.0,
  deadPeakDb: -70.0,
  /* Proximity is judged against the singer's own baseline, not an absolute
     number. The low/mid ratio depends on the voice -- a baritone carries
     more energy below 250 Hz than a soprano at any distance -- so a fixed
     threshold is wrong in principle rather than merely uncalibrated. The
     first seconds of voice establish what "where they started" sounds
     like; a cue fires when they move a factor of two from it. The hard
     bounds remain for the extreme cases before a baseline exists. */
  proximityBaselineS: 4.0,
  proximityCloseFactor: 2.0,
  proximityFarFactor: 0.5,
  proximityHardHigh: 2.8,
  proximityHardLow: 0.2,
};

export const TARGET = { peakDb: -6.0, rmsDb: -18.0 };

export class LiveCoach {
  constructor(config = {}) {
    this.cfg = { ...CONFIG, ...config };
    this.reset();
  }

  reset() {
    this.cues = [];
    this.lastCueAt = -Infinity;
    this.cueTimes = [];
    this.sustain = new Map();
    this.clipRun = 0;
    this.proxSamples = [];
    this.proxBaseline = null;
    this.voicedS = 0;
    this.lastT = null;
  }

  /** Accumulate the singer's own low/mid ratio until a baseline exists. */
  trackBaseline(ratio, voiced, now) {
    if (this.proxBaseline != null) return;
    const dt = this.lastT == null ? 0 : Math.max(0, now - this.lastT);
    this.lastT = now;
    if (!voiced || ratio <= 0) return;
    this.voicedS += dt;
    this.proxSamples.push(ratio);
    if (this.voicedS >= this.cfg.proximityBaselineS && this.proxSamples.length >= 20) {
      const s = [...this.proxSamples].sort((a, b) => a - b);
      this.proxBaseline = s[Math.floor(s.length / 2)];
    }
  }

  /** True once a condition has held for `sustainS`. One frame is noise. */
  held(key, active, now) {
    if (!active) { this.sustain.delete(key); return false; }
    if (!this.sustain.has(key)) this.sustain.set(key, now);
    return now - this.sustain.get(key) >= this.cfg.sustainS;
  }

  emit(id, severity, message, now, { urgent = false } = {}) {
    if (!urgent) {
      if (now - this.lastCueAt < this.cfg.minGapS) return null;
      this.cueTimes = this.cueTimes.filter(t => now - t < 60);
      if (this.cueTimes.length >= this.cfg.maxCuesPerMinute) return null;
    }
    const last = this.cues[0];
    if (last && last.id === id && now - last.at < this.cfg.minGapS) return null;

    this.lastCueAt = now;
    this.cueTimes.push(now);
    const cue = { id, severity, message, at: now };
    this.cues.unshift(cue);
    this.cues = this.cues.slice(0, 6);
    return cue;
  }

  /**
   * Feed one metrics frame. Returns a new cue, or null.
   *
   * @param {object} m   from CaptureEngine.readMetrics()
   * @param {number} now seconds since the take started
   */
  push(m, now) {
    const c = this.cfg;

    // Clipping bypasses the budget entirely. It is the one fault that
    // destroys a take outright and the one the singer can fix instantly.
    if (m.clipped > 0) {
      if (++this.clipRun >= c.clipFramesBeforeAlert) {
        this.clipRun = 0;
        return this.emit('clip', 'fatal', 'Clipping — turn the gain down', now,
                         { urgent: true });
      }
    } else {
      this.clipRun = 0;
    }

    if (!m.alive) {
      return this.emit('dead_thread', 'fatal',
                       'The audio engine stopped responding', now, { urgent: true });
    }
    if (m.dropped > 0) {
      return this.emit('dropped', 'fatal',
                       'Samples are being dropped — close other tabs', now,
                       { urgent: true });
    }
    if (this.held('dead', m.peakDb < c.deadPeakDb, now)) {
      return this.emit('dead', 'fatal', 'No signal — check the microphone', now,
                       { urgent: true });
    }

    if (this.held('quiet', m.rmsDb < c.quietDb && m.peakDb > c.deadPeakDb, now)) {
      return this.emit('quiet', 'warn', 'Too quiet — move closer or raise the gain', now);
    }
    if (this.held('loud', m.rmsDb > c.loudRmsDb, now)) {
      return this.emit('loud', 'warn', 'Very hot — back off a little', now);
    }

    const proximity = m.mid > 1e-9 ? m.low / m.mid : 0;
    const voiced = m.rmsDb > c.quietDb;
    this.trackBaseline(proximity, voiced, now);
    const b = this.proxBaseline;
    const near = b != null ? proximity > b * c.proximityCloseFactor
                           : proximity > c.proximityHardHigh;
    const far = voiced && (b != null ? proximity < b * c.proximityFarFactor
                                     : proximity < c.proximityHardLow);
    if (this.held('near', near, now)) {
      return this.emit('near', 'warn', 'Closer than you started — the low end is building up', now);
    }
    if (this.held('far', far, now)) {
      return this.emit('far', 'warn', 'Further than you started — the room is taking over', now);
    }

    return null;
  }
}

/** Level classification for the meter's colour, shared with the trace. */
export function levelTone(m) {
  if (!m) return '';
  if (m.truePeakDb > -1 || m.clipped > 0) return 'clip';
  if (m.rmsDb > CONFIG.loudRmsDb) return 'hot';
  if (m.rmsDb < CONFIG.quietDb) return 'low';
  return 'good';
}

export function noteName(hz) {
  if (!hz || hz <= 0) return null;
  const midi = 69 + 12 * Math.log2(hz / 440);
  const rounded = Math.round(midi);
  const names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
  const cents = Math.round((midi - rounded) * 100);
  return {
    name: names[((rounded % 12) + 12) % 12] + (Math.floor(rounded / 12) - 1),
    cents,
    midi,
  };
}
