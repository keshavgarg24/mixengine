/**
 * Capture and metering on the audio render thread.
 *
 * Everything here runs inside `process()`, which has a hard deadline of one
 * render quantum — 2.67 ms at 48 kHz — and must never allocate. Every buffer
 * is sized once in the constructor, every filter keeps its state in a plain
 * number, and no closure is created per call. A garbage collection pause on
 * this thread is a dropout in the middle of a take, and a take cannot be
 * re-recorded after the fact.
 *
 * What it computes, and why each is here rather than on the main thread:
 *
 *   **Sample peak, RMS, clip count.** Trivial, but must see *every* sample.
 *   An AnalyserNode on the main thread samples whatever happens to be in its
 *   buffer when requestAnimationFrame fires, so between two frames at 60 Hz
 *   it misses roughly three quarters of the audio — and a clip is a handful
 *   of samples. A meter that can miss clipping is worse than no meter,
 *   because it is trusted.
 *
 *   **True peak, 4x oversampled.** Inter-sample peaks routinely exceed the
 *   sample peak by 1-3 dB, so a signal metering -0.5 dBFS can convert to
 *   +1 dBTP and distort in every consumer D/A and lossy encoder. The engine
 *   masters to -1.0 dBTP; the recorder has no business reporting a quieter
 *   number than the thing it feeds.
 *
 *   **K-weighted loudness (ITU-R BS.1770).** The same measure the mastering
 *   stage targets. An RMS number and a LUFS number for the same signal can
 *   differ by several dB depending on spectral content, and a singer setting
 *   their level against the wrong one arrives at the wrong level.
 *
 *   **Band energies for proximity.** Bands match `capture/realtime.py`
 *   exactly (80-250 Hz over 400-2000 Hz), because the live hint and the
 *   post-take report must not disagree about how close someone was standing.
 *
 *   **Pitch, by McLeod's method.** Computed here but *not* shown during a
 *   take: concurrent pitch feedback measurably degrades the performance it
 *   is supervising. It drives the tuner and the range check, which happen
 *   before anyone is performing.
 */

const RQ = 128;                      // render quantum

// ── Pitch: MPM parameters, mirroring capture/realtime.py ──────────────────
const MPM_CUTOFF = 0.93;
const MIN_CLARITY = 0.55;
const PITCH_DECIMATION = 4;          // analyse at sr/4
const PITCH_WINDOW = 512;            // 2048 at full rate
const PITCH_INTERVAL = 1024;         // ~21 ms at 48 kHz
const F0_MIN = 65;
const F0_MAX = 1050;

// ── Loudness gate windows ────────────────────────────────────────────────
const MOMENTARY_S = 0.4;
const SHORT_S = 3.0;

const METRIC = {
  SEQ: 0, PEAK: 1, RMS: 2, CLIPPED: 3, LOW: 4, MID: 5, HIGH: 6,
  TRUE_PEAK: 7, FRAMES: 8, DROPPED: 9, PITCH_HZ: 10, CLARITY: 11,
  LUFS_M: 12, LUFS_S: 13, COUNT: 14,
};

const HEADER_SLOTS = 4;
const IDX_WRITE = 0;
const IDX_READ = 1;

/** Transposed direct form II biquad. One instance, state in two numbers. */
class Biquad {
  constructor() { this.reset(); this.setPassthrough(); }
  reset() { this.z1 = 0; this.z2 = 0; }
  setPassthrough() { this.b0 = 1; this.b1 = 0; this.b2 = 0; this.a1 = 0; this.a2 = 0; }

  setCoefficients(b0, b1, b2, a0, a1, a2) {
    this.b0 = b0 / a0; this.b1 = b1 / a0; this.b2 = b2 / a0;
    this.a1 = a1 / a0; this.a2 = a2 / a0;
  }

  /** RBJ bandpass, constant skirt gain. */
  setBandpass(sr, f0, q) {
    const w0 = 2 * Math.PI * Math.min(f0, sr * 0.49) / sr;
    const alpha = Math.sin(w0) / (2 * q);
    this.setCoefficients(alpha, 0, -alpha, 1 + alpha, -2 * Math.cos(w0), 1 - alpha);
  }

  /**
   * BS.1770 stage 1: a high-frequency shelf standing in for the acoustic
   * effect of a head in a diffuse field. The standard tabulates coefficients
   * at 48 kHz only; these are re-derived from the underlying f0/Q/gain by
   * bilinear transform so the meter is still correct at 44.1 or 96 kHz
   * rather than quietly mis-weighted.
   */
  setK1(sr) {
    const f0 = 1681.974450955533;
    const G = 3.999843853973347;
    const Q = 0.7071752369554196;
    const K = Math.tan(Math.PI * f0 / sr);
    const Vh = Math.pow(10, G / 20);
    const Vb = Math.pow(Vh, 0.4996667741545416);
    const den = 1 + K / Q + K * K;
    this.setCoefficients(
      (Vh + Vb * K / Q + K * K) / den,
      2 * (K * K - Vh) / den,
      (Vh - Vb * K / Q + K * K) / den,
      1,
      2 * (K * K - 1) / den,
      (1 - K / Q + K * K) / den);
  }

  /** BS.1770 stage 2: the RLB high-pass. */
  setK2(sr) {
    const f0 = 38.13547087602444;
    const Q = 0.5003270373238773;
    const K = Math.tan(Math.PI * f0 / sr);
    this.setCoefficients(
      1, -2, 1,
      1,
      2 * (K * K - 1) / (1 + K / Q + K * K),
      (1 - K / Q + K * K) / (1 + K / Q + K * K));
  }

  process(x) {
    const y = this.b0 * x + this.z1;
    this.z1 = this.b1 * x - this.a1 * y + this.z2;
    this.z2 = this.b2 * x - this.a2 * y;
    return y;
  }
}

/**
 * Windowed-sinc polyphase interpolator for true-peak detection.
 *
 * Two details decide whether this measures anything at all, and both were
 * found by testing it against a signal with a known overshoot rather than
 * by reading the code.
 *
 * *The window must be centred on the sinc, not on the tap array.* Centring
 * it on the array leaves a half-sample offset that weights one side of the
 * sinc more than the other. The filter still looks like an interpolator and
 * still passes DC, so nothing obviously breaks -- it just stops resolving
 * anything near Nyquist.
 *
 * *Twelve taps is not enough.* The inter-sample peaks that matter live at
 * high frequencies, which is exactly where a short windowed sinc has
 * already rolled off. With 12 taps a signal built to overshoot by 3 dB
 * measured an overshoot of 0.00 dB -- the meter reported the sample peak
 * and called it true peak. Thirty-two taps per phase puts the passband edge
 * comfortably past 20 kHz.
 */
function buildPolyphase(phases, taps) {
  const bank = [];
  const centre = (taps - 1) / 2;
  for (let p = 0; p < phases; p++) {
    const h = new Float32Array(taps);
    const frac = p / phases;
    let sum = 0;
    for (let i = 0; i < taps; i++) {
      const x = i - centre - frac;
      const sinc = Math.abs(x) < 1e-9 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x);
      // Blackman, centred on the sinc's own centre so the taper is
      // symmetric about the point being interpolated.
      const t = (x + centre + 1) / (taps + 1);
      const w = 0.42 - 0.5 * Math.cos(2 * Math.PI * t) + 0.08 * Math.cos(4 * Math.PI * t);
      h[i] = sinc * w;
      sum += h[i];
    }
    // Unity DC gain, so a constant input is reproduced exactly and the
    // meter cannot invent level that is not there.
    if (sum !== 0) for (let i = 0; i < taps; i++) h[i] /= sum;
    bank.push(h);
  }
  return bank;
}

class CaptureProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() { return []; }

  constructor(options) {
    super();
    const opt = options.processorOptions || {};
    this.sr = sampleRate;
    this.armed = false;
    this.frames = 0;
    this.dropped = 0;
    this.seq = 0;

    // Shared memory, when the page is cross-origin isolated.
    this.ring = null;
    this.ringHeader = null;
    this.ringData = null;
    this.ringCapacity = 0;
    if (opt.ringBuffer) {
      this.ringHeader = new Int32Array(opt.ringBuffer, 0, HEADER_SLOTS);
      this.ringData = new Float32Array(opt.ringBuffer, HEADER_SLOTS * 4);
      this.ringCapacity = this.ringData.length;
    }
    this.metrics = opt.metrics ? new Float32Array(opt.metrics) : null;

    // Fallback path: a pre-allocated staging block, transferred when full.
    this.stageSize = 4096;
    this.stage = new Float32Array(this.stageSize);
    this.stageFill = 0;

    // ── filters ──────────────────────────────────────────────────────────
    this.k1 = new Biquad(); this.k1.setK1(this.sr);
    this.k2 = new Biquad(); this.k2.setK2(this.sr);
    this.bpLow = new Biquad(); this.bpLow.setBandpass(this.sr, 141, 1.2);   // 80-250
    this.bpMid = new Biquad(); this.bpMid.setBandpass(this.sr, 894, 1.0);   // 400-2000
    this.bpHigh = new Biquad(); this.bpHigh.setBandpass(this.sr, 6890, 2.0); // 5k-9.5k

    // ── loudness sliding sums ────────────────────────────────────────────
    this.mLen = Math.max(1, Math.round(MOMENTARY_S * this.sr));
    this.sLen = Math.max(1, Math.round(SHORT_S * this.sr));
    this.mBuf = new Float32Array(this.mLen);
    this.sBuf = new Float32Array(this.sLen);
    this.mPos = 0; this.sPos = 0; this.mSum = 0; this.sSum = 0;

    // ── true peak ────────────────────────────────────────────────────────
    this.tpPhases = 4;
    this.tpTaps = 32;
    this.tpBank = buildPolyphase(this.tpPhases, this.tpTaps);
    this.tpHist = new Float32Array(this.tpTaps);
    this.tpPos = 0;

    // ── band envelopes ───────────────────────────────────────────────────
    this.envLow = 0; this.envMid = 0; this.envHigh = 0;
    this.envCoef = Math.exp(-1 / (0.05 * this.sr));

    // ── pitch ────────────────────────────────────────────────────────────
    this.pDecim = PITCH_DECIMATION;
    this.pSr = this.sr / this.pDecim;
    this.pBuf = new Float32Array(PITCH_WINDOW);
    this.pFill = 0;
    this.pCounter = 0;
    this.pDecimCount = 0;
    this.pAcc = 0;
    this.nsdf = new Float32Array(PITCH_WINDOW);
    this.maxima = new Int32Array(64);
    this.pitchHz = 0;
    this.clarity = 0;
    this.minLag = Math.max(2, Math.floor(this.pSr / F0_MAX));
    this.maxLag = Math.min(PITCH_WINDOW - 2, Math.ceil(this.pSr / F0_MIN));
    // Anti-alias before decimating, or everything above pSr/2 folds down
    // onto the very range we are trying to measure.
    this.pLp = new Biquad();
    this.pLp.setBandpass(this.sr, 900, 0.5);

    this.port.onmessage = ev => this.onCommand(ev.data);
  }

  onCommand(msg) {
    switch (msg.type) {
      case 'arm':
        this.armed = true;
        this.frames = 0;
        this.dropped = 0;
        this.stageFill = 0;
        break;
      case 'disarm':
        this.armed = false;
        this.flushStage();
        this.port.postMessage({ type: 'stopped', frames: this.frames,
                                dropped: this.dropped });
        break;
      case 'reset':
        this.k1.reset(); this.k2.reset();
        this.mSum = 0; this.sSum = 0;
        this.mBuf.fill(0); this.sBuf.fill(0);
        break;
    }
  }

  pushShared(block) {
    const w = Atomics.load(this.ringHeader, IDX_WRITE);
    const r = Atomics.load(this.ringHeader, IDX_READ);
    const used = w >= r ? w - r : this.ringCapacity - r + w;
    const free = this.ringCapacity - used - 1;
    const n = block.length;
    if (free < n) { this.dropped += n - Math.max(free, 0); }
    const take = Math.min(n, free);
    if (take <= 0) return;

    const first = Math.min(take, this.ringCapacity - w);
    this.ringData.set(block.subarray(0, first), w);
    if (take > first) this.ringData.set(block.subarray(first, take), 0);
    Atomics.store(this.ringHeader, IDX_WRITE, (w + take) % this.ringCapacity);
  }

  flushStage() {
    if (this.stageFill === 0) return;
    // One allocation per 4096 samples (85 ms), only on the fallback path.
    const out = this.stage.slice(0, this.stageFill);
    this.port.postMessage({ type: 'audio', samples: out }, [out.buffer]);
    this.stageFill = 0;
  }

  pushStaged(block) {
    for (let i = 0; i < block.length; i++) {
      this.stage[this.stageFill++] = block[i];
      if (this.stageFill >= this.stageSize) this.flushStage();
    }
  }

  truePeak(x) {
    this.tpHist[this.tpPos] = x;
    this.tpPos = (this.tpPos + 1) % this.tpTaps;
    let peak = Math.abs(x);
    for (let p = 0; p < this.tpPhases; p++) {
      const h = this.tpBank[p];
      let acc = 0;
      let idx = this.tpPos;
      for (let i = 0; i < this.tpTaps; i++) {
        acc += h[i] * this.tpHist[idx];
        idx = (idx + 1) % this.tpTaps;
      }
      const a = Math.abs(acc);
      if (a > peak) peak = a;
    }
    return peak;
  }

  detectPitch() {
    const n = PITCH_WINDOW;
    const buf = this.pBuf;
    const nsdf = this.nsdf;

    // Normalised square difference, McLeod & Wyvill. Bounded [-1, 1], which
    // is what makes a fixed clarity threshold meaningful across levels --
    // plain autocorrelation is not, and a threshold on it tracks loudness
    // instead of periodicity.
    let m0 = 0;
    for (let i = 0; i < n; i++) m0 += buf[i] * buf[i];
    if (m0 < 1e-7) { this.clarity = 0; return; }

    let best = 0;
    for (let lag = this.minLag; lag <= this.maxLag; lag++) {
      let ac = 0, m = 0;
      const lim = n - lag;
      for (let i = 0; i < lim; i++) {
        const a = buf[i], b = buf[i + lag];
        ac += a * b;
        m += a * a + b * b;
      }
      nsdf[lag] = m > 1e-12 ? (2 * ac) / m : 0;
      if (nsdf[lag] > best) best = nsdf[lag];
    }

    if (best < MIN_CLARITY) { this.clarity = best; this.pitchHz = 0; return; }

    // Key maxima: the first peak above cutoff * global max, not the global
    // max itself. Taking the tallest peak chooses whichever octave happens
    // to correlate best and flips between them on sustained notes.
    const threshold = MPM_CUTOFF * best;
    let chosen = -1;
    for (let lag = this.minLag + 1; lag < this.maxLag; lag++) {
      if (nsdf[lag] > nsdf[lag - 1] && nsdf[lag] >= nsdf[lag + 1] &&
          nsdf[lag] >= threshold) { chosen = lag; break; }
    }
    if (chosen < 0) { this.clarity = best; this.pitchHz = 0; return; }

    // Parabolic interpolation: the true period rarely lands on a sample.
    const y0 = nsdf[chosen - 1], y1 = nsdf[chosen], y2 = nsdf[chosen + 1];
    const denom = 2 * (2 * y1 - y0 - y2);
    const shift = Math.abs(denom) > 1e-12 ? (y2 - y0) / denom : 0;
    const period = chosen + shift;

    this.pitchHz = period > 0 ? this.pSr / period : 0;
    this.clarity = y1;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const ch = input[0];
    if (!ch) return true;
    const n = ch.length;

    let peak = 0, sumsq = 0, clipped = 0, tp = 0;

    for (let i = 0; i < n; i++) {
      const x = ch[i];
      const a = x < 0 ? -x : x;
      if (a > peak) peak = a;
      sumsq += x * x;
      if (a >= 0.999) clipped++;

      const t = this.truePeak(x);
      if (t > tp) tp = t;

      // K-weighted energy for loudness.
      const k = this.k2.process(this.k1.process(x));
      const kk = k * k;
      this.mSum += kk - this.mBuf[this.mPos];
      this.mBuf[this.mPos] = kk;
      this.mPos = this.mPos + 1 >= this.mLen ? 0 : this.mPos + 1;
      this.sSum += kk - this.sBuf[this.sPos];
      this.sBuf[this.sPos] = kk;
      this.sPos = this.sPos + 1 >= this.sLen ? 0 : this.sPos + 1;

      // Band envelopes for proximity.
      const l = Math.abs(this.bpLow.process(x));
      const m = Math.abs(this.bpMid.process(x));
      const h = Math.abs(this.bpHigh.process(x));
      this.envLow = l + this.envCoef * (this.envLow - l);
      this.envMid = m + this.envCoef * (this.envMid - m);
      this.envHigh = h + this.envCoef * (this.envHigh - h);

      // Decimate for pitch: low-pass, then take every Nth sample.
      this.pAcc = this.pLp.process(x);
      if (++this.pDecimCount >= this.pDecim) {
        this.pDecimCount = 0;
        this.pBuf.copyWithin(0, 1);
        this.pBuf[PITCH_WINDOW - 1] = this.pAcc;
        if (this.pFill < PITCH_WINDOW) this.pFill++;
      }
    }

    if (this.armed) {
      if (this.ringData) this.pushShared(ch);
      else this.pushStaged(ch);
      this.frames += n;
    }

    this.pCounter += n;
    if (this.pCounter >= PITCH_INTERVAL && this.pFill >= PITCH_WINDOW) {
      this.pCounter = 0;
      this.detectPitch();
    }

    if (this.metrics) {
      const v = this.metrics;
      const db = x => (x > 1e-7 ? 20 * Math.log10(x) : -120);
      v[METRIC.PEAK] = db(peak);
      v[METRIC.RMS] = db(Math.sqrt(sumsq / n));
      v[METRIC.TRUE_PEAK] = db(tp);
      v[METRIC.CLIPPED] = clipped;
      v[METRIC.LOW] = this.envLow;
      v[METRIC.MID] = this.envMid;
      v[METRIC.HIGH] = this.envHigh;
      v[METRIC.FRAMES] = this.frames;
      v[METRIC.DROPPED] = this.dropped;
      v[METRIC.PITCH_HZ] = this.pitchHz;
      v[METRIC.CLARITY] = this.clarity;
      // -0.691 is the BS.1770 calibration offset that puts a 1 kHz sine at
      // -3 dBFS on a single channel at -3.0 LUFS.
      v[METRIC.LUFS_M] = this.mSum > 1e-12
        ? -0.691 + 10 * Math.log10(this.mSum / this.mLen) : -120;
      v[METRIC.LUFS_S] = this.sSum > 1e-12
        ? -0.691 + 10 * Math.log10(this.sSum / this.sLen) : -120;
      v[METRIC.SEQ] = ++this.seq;
    }

    return true;
  }
}

registerProcessor('capture-processor', CaptureProcessor);
