/**
 * Browser-side audio tests.
 *
 * These exist because the DSP in the worklet cannot be checked by reading
 * it. Every assertion below is against a signal whose correct answer is
 * known analytically — a sine at a stated level, a waveform constructed to
 * overshoot by exactly 3.01 dB, a probe planted at a known sample — and
 * four of them failed the first time they were run against code that looked
 * entirely reasonable:
 *
 *   - the true-peak meter reported an overshoot of 0.00 dB on a signal built
 *     to overshoot by 3.01, because its interpolation window was misaligned
 *     by half a sample and twelve taps had already rolled off;
 *   - the latency probe rejected a detection that was correct to the sample,
 *     because its confidence was measured against its own peak shoulders;
 *   - the same probe located a *reflection* rather than the direct sound
 *     whenever the reflection was louder;
 *   - the ring buffer's capacity contract was one frame short of what its
 *     own caller assumed.
 *
 * `OfflineAudioContext` rather than a live one: it renders deterministically,
 * runs faster than real time, and is not subject to the autoplay policy, so
 * this page needs no click to produce a result.
 */

import { buildProbe, findProbe } from './audio/latency.js';
import { MetricBlock, RingBuffer, sharedMemoryAvailable } from './audio/ring-buffer.js';
import { peaks, summarise } from './audio/wav.js';
import { LiveCoach, noteName } from './audio/coach.js';

const out = document.querySelector('#out');
let passed = 0, failed = 0;

function group(name) {
  const h = document.createElement('h2');
  h.textContent = name;
  out.appendChild(h);
}

function check(name, ok, detail = '') {
  const row = document.createElement('div');
  row.className = `t ${ok ? 'pass' : 'fail'}`;
  row.innerHTML = '';
  const mark = document.createElement('span');
  mark.className = 'mark';
  mark.textContent = ok ? 'ok' : 'FAIL';
  const label = document.createElement('span');
  label.textContent = name;
  const det = document.createElement('span');
  det.className = 'detail';
  det.textContent = detail;
  row.append(mark, label, det);
  out.appendChild(row);
  ok ? passed++ : failed++;
}

function near(a, b, tol) { return Math.abs(a - b) <= tol; }

const SR = 48000;

/** Render a signal through the capture worklet and read its meters back. */
async function meter(build, seconds = 4.0) {
  const ctx = new OfflineAudioContext(1, Math.round(SR * seconds), SR);
  await ctx.audioWorklet.addModule('/static/worklets/capture-processor.js');
  const metrics = MetricBlock.alloc();
  const node = new AudioWorkletNode(ctx, 'capture-processor', {
    numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1],
    processorOptions: { metrics: metrics.view.buffer },
  });
  build(ctx, node);
  node.connect(ctx.destination);
  await ctx.startRendering();
  return metrics.read();
}

function sineSource(ctx, node, hz, dbfs) {
  const o = ctx.createOscillator(), g = ctx.createGain();
  o.frequency.value = hz;
  g.gain.value = Math.pow(10, dbfs / 20);
  o.connect(g).connect(node);
  o.start();
}

function bufferSource(ctx, node, fill, seconds = 4.0) {
  const n = Math.round(SR * seconds);
  const buf = ctx.createBuffer(1, n, SR);
  fill(buf.getChannelData(0), n);
  const s = ctx.createBufferSource();
  s.buffer = buf;
  s.connect(node);
  s.start();
}

async function run() {
  /* ── environment ─────────────────────────────────────────────────────── */
  group('Environment');
  check('AudioWorklet available', typeof AudioWorkletNode !== 'undefined');
  check('cross-origin isolated', crossOriginIsolated === true,
        'required for SharedArrayBuffer');
  check('SharedArrayBuffer available', sharedMemoryAvailable(),
        sharedMemoryAvailable() ? 'lock-free ring buffer in use'
                                : 'falling back to postMessage transfer');

  /* ── level metering ──────────────────────────────────────────────────── */
  group('Level metering');
  const m6 = await meter((c, n) => sineSource(c, n, 440, -6));
  check('sample peak of a −6 dBFS sine', near(m6.peakDb, -6, 0.05),
        `${m6.peakDb.toFixed(2)} dB`);
  check('RMS of a sine is 3 dB below its peak', near(m6.rmsDb, -9, 0.4),
        `${m6.rmsDb.toFixed(2)} dB`);
  check('true peak is never below sample peak', m6.truePeakDb >= m6.peakDb - 0.01,
        `${m6.truePeakDb.toFixed(2)} vs ${m6.peakDb.toFixed(2)} dB`);

  /* ── ITU-R BS.1770 ───────────────────────────────────────────────────── */
  group('Loudness (ITU-R BS.1770)');
  const ref = await meter((c, n) => sineSource(c, n, 1000, 0));
  check('0 dBFS 1 kHz sine reads −3.01 LKFS', near(ref.lufsShort, -3.01, 0.1),
        `${ref.lufsShort.toFixed(2)} LUFS · standard says −3.01`);
  const q = await meter((c, n) => sineSource(c, n, 1000, -20));
  check('20 dB of attenuation moves loudness by 20 dB',
        near(q.lufsShort - ref.lufsShort, -20, 0.2),
        `${(q.lufsShort - ref.lufsShort).toFixed(2)} dB`);

  /* ── true peak ───────────────────────────────────────────────────────── */
  group('True peak (4× oversampled)');
  // A sine at sr/4 sampled 45° off its crest: samples reach A/√2 while the
  // continuous waveform reaches A, an overshoot of exactly 3.01 dB.
  const isp = await meter((c, n) => bufferSource(c, n, (d, len) => {
    for (let i = 0; i < len; i++) d[i] = 0.98 * Math.sin(2 * Math.PI * i / 4 + Math.PI / 4);
  }));
  const overshoot = isp.truePeakDb - isp.peakDb;
  check('inter-sample overshoot measured as 3.01 dB', near(overshoot, 3.01, 0.15),
        `${overshoot.toFixed(2)} dB · a sample-peak meter would report 0.00`);

  const dc = await meter((c, n) => bufferSource(c, n, (d, len) => d.fill(0.5)));
  check('constant signal invents no overshoot', near(dc.truePeakDb, dc.peakDb, 0.05),
        `${(dc.truePeakDb - dc.peakDb).toFixed(3)} dB`);

  /* ── pitch ───────────────────────────────────────────────────────────── */
  group('Pitch (McLeod)');
  for (const hz of [110, 220, 440, 880]) {
    const p = await meter((c, n) => sineSource(c, n, hz, -12));
    const cents = 1200 * Math.log2((p.pitchHz || 1) / hz);
    check(`${hz} Hz sine`, Math.abs(cents) < 10,
          `${p.pitchHz.toFixed(1)} Hz · ${cents.toFixed(1)} cents · clarity ${p.clarity.toFixed(3)}`);
  }
  const noisy = await meter((c, n) => bufferSource(c, n, (d, len) => {
    for (let i = 0; i < len; i++) d[i] = (Math.random() - 0.5) * 0.4;
  }));
  check('noise is reported as unpitched', noisy.clarity < 0.55 || noisy.pitchHz === 0,
        `clarity ${noisy.clarity.toFixed(3)}`);

  /* ── ring buffer ─────────────────────────────────────────────────────── */
  group('Ring buffer');
  const rb = RingBuffer.alloc(1000, SR, 1);
  const src = new Float32Array(600).map((_, i) => i / 600);
  rb.push(src);
  const back = rb.pull();
  check('round trip preserves every sample',
        back.length === 600 && back.every((v, i) => near(v, src[i], 1e-7)));

  rb.push(new Float32Array(700).fill(0.5));
  rb.pull(300);
  rb.push(new Float32Array(500).fill(0.25));
  const seam = rb.pull();
  check('reads correctly across the wrap point',
        seam.length === 900 && seam.slice(0, 400).every(v => v === 0.5)
                            && seam.slice(400).every(v => v === 0.25),
        `${seam.length} samples`);

  const small = RingBuffer.alloc(100, SR, 1);
  const wrote = small.push(new Float32Array(200).map((_, i) => i));
  const kept = small.pull();
  check('alloc(n) provides exactly n frames', wrote === 100 && kept.length === 100,
        `accepted ${wrote}`);
  check('overrun drops the newest, never the unread oldest',
        kept[0] === 0 && kept[99] === 99);

  /* ── latency probe ───────────────────────────────────────────────────── */
  group('Latency probe');
  const probe = buildProbe(SR);
  const KNOWN = 2137;
  const cases = [
    ['clean',            0.80, 0.005, []],
    ['quiet',            0.35, 0.020, []],
    ['buried in noise',  0.12, 0.030, []],
    ['one reflection',   0.50, 0.015, [[720, 0.30]]],
    ['reflection louder than direct', 0.40, 0.015, [[720, 0.65]]],
    ['live room',        0.45, 0.020, [[380, 0.35], [720, 0.28], [1500, 0.18]]],
  ];
  for (const [label, gain, noise, reflections] of cases) {
    const sig = new Float32Array(SR * 0.5);
    for (let i = 0; i < sig.length; i++) sig[i] = (Math.random() - 0.5) * noise * 2;
    for (let i = 0; i < probe.length; i++) {
      sig[KNOWN + i] += probe[i] * gain;
      for (const [d, g] of reflections) sig[KNOWN + d + i] += probe[i] * g;
    }
    const f = findProbe(sig, probe);
    const errMs = (f.lag - KNOWN) / SR * 1000;
    check(`finds the direct path · ${label}`,
          Math.abs(f.lag - KNOWN) <= 2 && f.margin >= 0.25,
          `${errMs.toFixed(2)} ms error · margin ${f.margin.toFixed(3)}`);
  }
  const empty = new Float32Array(SR * 0.5);
  for (let i = 0; i < empty.length; i++) empty[i] = (Math.random() - 0.5) * 0.06;
  const none = findProbe(empty, probe);
  check('reports nothing when the probe was never played', none.margin < 0.25,
        `margin ${none.margin.toFixed(3)}`);

  /* ── peaks and summary ───────────────────────────────────────────────── */
  group('Waveform reduction');
  const spike = new Float32Array(10000);
  spike[5000] = 1.0;
  const p = peaks(spike, 100);
  check('a single full-scale sample survives reduction',
        Math.max(...p) >= 0.999, 'naive decimation would lose it');
  const s = summarise(spike);
  check('clip count is exact', s.clippedSamples === 1);

  /* ── coach ───────────────────────────────────────────────────────────── */
  group('Live coach');
  const coach = new LiveCoach();
  const frame = { peakDb: -6, rmsDb: -18, truePeakDb: -5, clipped: 0,
                  low: 1, mid: 1, high: 1, dropped: 0, alive: true };
  check('says nothing when everything is fine',
        coach.push(frame, 1) === null);

  coach.reset();
  let clipCue = null;
  for (let i = 0; i < 5 && !clipCue; i++) {
    clipCue = coach.push({ ...frame, clipped: 4 }, i * 0.1);
  }
  check('clipping is raised immediately', clipCue?.severity === 'fatal',
        clipCue?.message || 'no cue');

  coach.reset();
  coach.push({ ...frame, clipped: 9 }, 0);
  coach.push({ ...frame, clipped: 9 }, 0.1);
  coach.push({ ...frame, clipped: 9 }, 0.2);
  let quiet = null;
  for (let t = 0.3; t < 4 && !quiet; t += 0.1) {
    quiet = coach.push({ ...frame, rmsDb: -50 }, t);
  }
  check('the attention budget suppresses a second cue',
        quiet === null, 'a non-urgent cue inside the gap must be withheld');

  // Proximity is relative to the singer's own baseline. A ratio that would
  // have tripped a fixed threshold is fine if that is where they started;
  // a move to double it is reported.
  const c2 = new LiveCoach();
  let nagged = null;
  for (let t = 0; t < 12 && !nagged; t += 0.02) {
    nagged = c2.push({ ...frame, low: 2.2, mid: 1 }, t);
  }
  check('a singer who starts close and stays close is not nagged', nagged === null,
        `baseline ${c2.proxBaseline?.toFixed(2) ?? 'none'}`);

  const c3 = new LiveCoach();
  let moved = null;
  for (let t = 0; t < 6; t += 0.02) c3.push({ ...frame, low: 1, mid: 1 }, t);
  for (let t = 6; t < 12 && !moved; t += 0.02) moved = c3.push({ ...frame, low: 2.6, mid: 1 }, t);
  check('moving to twice the baseline ratio is reported', moved?.id === 'near',
        moved?.message || 'no cue');

  const n = noteName(440);
  check('A440 is named correctly', n.name === 'A4' && Math.abs(n.cents) < 1,
        `${n.name} ${n.cents} cents`);

  const sum = document.querySelector('#summary');
  sum.textContent = failed === 0
    ? `${passed} checks passed.`
    : `${passed} passed, ${failed} FAILED.`;
  sum.className = failed === 0 ? 'good' : 'bad';
  document.title = failed === 0 ? `✓ ${passed} passed` : `✗ ${failed} failed`;
}

run().catch(err => {
  check('test runner completed', false, String(err));
  document.querySelector('#summary').textContent = 'The run did not finish.';
  document.querySelector('#summary').className = 'bad';
});
