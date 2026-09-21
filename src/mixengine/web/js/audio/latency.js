/**
 * Round-trip latency measurement by acoustic loopback.
 *
 * When someone records over playback, everything they hear is already late:
 * the browser schedules a sample, it queues through the output device,
 * leaves the speaker, crosses the room, and queues again through the input
 * device. The singer performs in time with what reached their ears, so the
 * recording is uniformly late by that whole round trip — commonly 30-80 ms
 * on a laptop, which at 140 BPM is a third of a sixteenth note.
 *
 * No later stage can fix this properly. To the alignment stage a fixed
 * offset looks exactly like a singer who is behind the beat, and correcting
 * it there means warping a performance that was actually in time. Measured
 * here it is a single subtraction, and it is exact.
 *
 * `AudioContext.outputLatency` reports only what the output device admits
 * to. It omits the input path entirely and knows nothing about the air gap,
 * so on the machines where it matters most it is the least complete. A
 * loopback measurement asks the actual question: a known signal is played,
 * the microphone records, and the lag between them is measured directly.
 *
 * The probe is a short linear sweep rather than a click. A click has its
 * energy concentrated in one instant, so any room reflection produces a
 * correlation peak as tall as the direct sound and the measurement picks
 * whichever happened to be louder. A sweep correlates against its own
 * frequency progression, which the reflection cannot imitate, and
 * compresses to a sharp unambiguous peak.
 */

const PROBE_MS = 12;
const PROBE_F0 = 900;
const PROBE_F1 = 7500;
const LISTEN_MS = 600;
const REPEATS = 5;

// How far above the background the chosen peak must stand before the
// measurement is believed. Reporting a number that is not the probe would
// be worse than reporting nothing: the offset gets applied to every take
// recorded afterwards.
const MIN_CONFIDENCE = 0.25;

// A peak at least this fraction of the tallest counts as a real arrival of
// the probe rather than background structure. The first such peak is the
// direct sound.
const DIRECT_CUTOFF = 0.6;

export function buildProbe(sampleRate) {
  const n = Math.round(sampleRate * PROBE_MS / 1000);
  const out = new Float32Array(n);
  const k = (PROBE_F1 - PROBE_F0) / (n / sampleRate);
  for (let i = 0; i < n; i++) {
    const t = i / sampleRate;
    const phase = 2 * Math.PI * (PROBE_F0 * t + 0.5 * k * t * t);
    // Hann taper: an abrupt start would splatter energy across the spectrum
    // and blunt the very peak we are trying to locate.
    const w = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (n - 1));
    out[i] = Math.sin(phase) * w * 0.55;
  }
  return out;
}

/**
 * Lag, in samples, of `probe` inside `signal`, with a normalised score.
 *
 * Direct correlation rather than FFT: the probe is a few hundred samples and
 * the search window under thirty thousand, so this is a few million
 * multiply-adds — well under a frame — and it avoids shipping an FFT that
 * exists for nothing else.
 */
export function findProbe(signal, probe) {
  const n = signal.length, m = probe.length;
  if (n <= m) return { lag: -1, score: 0 };

  let probeEnergy = 0;
  for (let i = 0; i < m; i++) probeEnergy += probe[i] * probe[i];
  if (probeEnergy < 1e-12) return { lag: -1, score: 0 };

  const limit = n - m;
  const scores = new Float32Array(limit);
  let bestLag = -1, bestScore = -Infinity;
  for (let lag = 0; lag < limit; lag++) {
    let dot = 0, energy = 0;
    for (let i = 0; i < m; i++) {
      const s = signal[lag + i];
      dot += s * probe[i];
      energy += s * s;
    }
    const score = energy > 1e-12 ? dot / Math.sqrt(energy * probeEnergy) : 0;
    scores[lag] = score;
    if (score > bestScore) { bestScore = score; bestLag = lag; }
  }

  // The answer is the *earliest* strong peak, not the tallest one.
  //
  // A room reflection is the same probe arriving later, so it correlates
  // just as well as the direct sound and is sometimes louder -- a hard
  // surface close to the microphone will beat the direct path. Taking the
  // maximum therefore measures the reflection, and over-reports the latency
  // by however far the sound travelled. Physics settles it: a reflection
  // took a longer path, so it can never arrive first.
  //
  // This is the same reasoning as key-maxima picking in the pitch tracker,
  // where taking the tallest peak chooses an arbitrary octave.
  const threshold = DIRECT_CUTOFF * bestScore;
  let chosen = bestLag;
  for (let lag = 1; lag < limit - 1; lag++) {
    if (scores[lag] >= threshold &&
        scores[lag] > scores[lag - 1] && scores[lag] >= scores[lag + 1]) {
      chosen = lag;
      break;
    }
  }

  // Confidence is the peak against the *background*, not against other
  // peaks. Measuring it against the runner-up looked right and rejected the
  // one case that always happens: with a single reflection 15 ms behind the
  // direct sound, a detection that was correct to the sample scored 0.997
  // and produced a margin of 0.005.
  let sum = 0;
  for (let lag = 0; lag < limit; lag++) sum += Math.abs(scores[lag]);
  const background = sum / limit;
  const margin = scores[chosen] - background;
  return { lag: chosen, score: scores[chosen], margin, peakScore: bestScore };
}

/**
 * Play the probe, record it, and return the measured round trip.
 *
 * @param {CaptureEngine} engine  an open capture engine
 * @returns {{seconds:number, samples:number, confidence:number,
 *            reported:number, measurements:number[], note:string}}
 */
export async function measure(engine, { repeats = REPEATS, onProgress } = {}) {
  if (!engine.ctx) throw new Error('The microphone is not open.');
  const ctx = engine.ctx;
  const sr = ctx.sampleRate;
  const probe = buildProbe(sr);

  const probeBuffer = ctx.createBuffer(1, probe.length, sr);
  probeBuffer.copyToChannel(probe, 0);

  const results = [];
  for (let attempt = 0; attempt < repeats; attempt++) {
    onProgress?.(attempt / repeats);

    engine.chunks = [];
    engine.recordedFrames = 0;
    if (engine.ring) engine.ring.clear();
    engine.node.port.postMessage({ type: 'arm' });

    // A short lead-in so the capture path is certainly running before the
    // probe sounds; without it the first attempt measures a delay that
    // includes the arming itself.
    await new Promise(r => setTimeout(r, 80));

    const src = ctx.createBufferSource();
    src.buffer = probeBuffer;
    src.connect(ctx.destination);
    const framesBeforeProbe = engine.recordedFrames + drainNow(engine);
    src.start();

    await new Promise(r => setTimeout(r, LISTEN_MS));
    drainNow(engine);
    engine.node.port.postMessage({ type: 'disarm' });
    await new Promise(r => setTimeout(r, 40));
    drainNow(engine);

    const captured = concat(engine.chunks);
    engine.chunks = [];
    if (captured.length <= probe.length) continue;

    const search = captured.subarray(Math.max(0, framesBeforeProbe));
    const { lag, score, margin } = findProbe(search, probe);
    if (lag >= 0 && margin >= MIN_CONFIDENCE) {
      results.push({ samples: lag, score, margin });
    }
  }

  engine.chunks = [];
  engine.recordedFrames = 0;
  onProgress?.(1);

  if (!results.length) {
    return {
      seconds: 0, samples: 0, confidence: 0,
      reported: engine.reportedLatency, measurements: [],
      note: 'The probe was never heard. Turn the speakers up, or leave the ' +
            'headphones off for this measurement — it has to travel through ' +
            'the air to be measured.',
    };
  }

  const sorted = results.map(r => r.samples).sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)];

  // Spread across attempts is the honest confidence signal. A stable path
  // returns the same lag every time; one that wanders was measuring
  // something other than the probe.
  const spread = sorted[sorted.length - 1] - sorted[0];
  const stable = spread < sr * 0.01;
  const confidence = Math.min(1, results.length / repeats) * (stable ? 1 : 0.4);

  return {
    seconds: median / sr,
    samples: median,
    confidence,
    reported: engine.reportedLatency,
    measurements: sorted.map(s => +(s / sr * 1000).toFixed(1)),
    note: stable
      ? ''
      : `The measurements disagree by ${(spread / sr * 1000).toFixed(0)} ms. ` +
        'Something is moving — check nothing is adjusting the volume.',
  };
}

function drainNow(engine) {
  const before = engine.recordedFrames;
  engine.drain();
  return engine.recordedFrames - before;
}

function concat(chunks) {
  const total = chunks.reduce((n, c) => n + c.length, 0);
  const out = new Float32Array(total);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.length; }
  return out;
}
