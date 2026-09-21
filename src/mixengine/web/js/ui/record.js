/**
 * The recording view.
 *
 * Structured around the three-way split the capture research forces:
 * everything useful is said *before* the take, only unrecoverable faults are
 * raised *during* it, and the full assessment waits until *after*. The
 * layout follows that — guidance on the right where it can be read at
 * leisure, one meter in the middle that is legible at a glance, and the
 * report below only once there is a take to report on.
 *
 * The tuner is the one place pitch is shown, and it is disabled the moment
 * recording starts.
 */

import { api, postForm } from '../core/api.js';
import { $, clock, clockMs, DASH, delegate, esc, html, note, num, on, show }
  from '../core/dom.js';
import { prefs, store, subscribe, update } from '../core/store.js';
import { CAPTURE_STATE, CaptureEngine } from '../audio/capture.js';
import { LiveCoach, levelTone, noteName, TARGET } from '../audio/coach.js';
import { measure } from '../audio/latency.js';
import { encodeWav, summarise } from '../audio/wav.js';
import { LiveMeterTrace, Waveform } from './waveform.js';

const engine = new CaptureEngine();
const coach = new LiveCoach();

let trace = null;
let takeWave = null;
let beatAudio = null;
let beatInfo = null;
let rafId = 0;
let takeStart = 0;
let countingIn = false;

export function init() {
  trace = new LiveMeterTrace($('#rec-trace'));
  takeWave = new Waveform($('#take-wave'));

  on($('#rec-enable'), 'click', enableMic);
  on($('#rec-device'), 'change', ev => openDevice(ev.target.value));
  on($('#rec-beat'), 'change', () => selectBeat($('#rec-beat').value));
  on($('#rec-monitor'), 'change', ev => {
    engine.setMonitor(ev.target.checked);
    prefs.set('monitor', ev.target.checked);
    paintMonitorAdvice((engine.monitorLatency || 0) * 1000);
  });
  on($('#rec-go'), 'click', startTake);
  on($('#rec-stop'), 'click', stopTake);
  on($('#rec-room'), 'click', roomCheck);
  on($('#rec-calibrate'), 'click', calibrateLatency);

  delegate($('#rec-takes'), 'click', '[data-take-use]', (_e, el) => {
    useTake(Number(el.dataset.takeUse));
  });
  delegate($('#rec-takes'), 'click', '[data-take-play]', (_e, el) => {
    playTake(Number(el.dataset.takePlay));
  });
  delegate($('#rec-takes'), 'click', '[data-take-drop]', (_e, el) => {
    dropTake(Number(el.dataset.takeDrop));
  });

  engine.addEventListener('state', () => paintTransport());
  engine.addEventListener('opened', ev => paintDeviceInfo(ev.detail));

  subscribe('catalog', fillBeats);
  subscribe('takes', paintTakes);
  subscribe('latency', paintLatency);

  const saved = prefs.get('latencySeconds', null);
  if (saved != null) {
    engine.setLatency(saved);
    update('latency', { seconds: saved, confidence: prefs.get('latencyConfidence', 0),
                        restored: true });
  }

  window.addEventListener('beforeunload', () => engine.close());
  paintTransport();
  paintCues();
  // Subscriptions only fire on change, so the empty states have to be
  // painted once at boot or the panels come up blank.
  paintLatency(store.latency);
  paintTakes(store.takes);
  selectBeat('');
}

/* ── device ──────────────────────────────────────────────────────────────── */

async function enableMic() {
  const btn = $('#rec-enable');
  btn.disabled = true;
  btn.textContent = 'Opening…';
  try {
    await engine.open({ deviceId: prefs.get('deviceId', null) });
    // Labels are blank until permission is granted, so the device list is
    // only worth populating after the first successful open.
    const devices = await engine.listDevices();
    const sel = $('#rec-device');
    sel.innerHTML = devices.map(d =>
      `<option value="${esc(d.id)}">${esc(d.label)}</option>`).join('');
    sel.value = engine.deviceId || devices[0]?.id || '';
    show($('#rec-device-row'), devices.length > 1);

    $('#rec-monitor').checked = prefs.get('monitor', false);
    engine.setMonitor($('#rec-monitor').checked);

    btn.textContent = 'Microphone live';
    startMeterLoop();
    setStatus('');
  } catch (err) {
    btn.disabled = false;
    btn.textContent = 'Enable microphone';
    setStatus(err.message, 'bad');
  }
  paintTransport();
}

async function openDevice(id) {
  prefs.set('deviceId', id);
  stopMeterLoop();
  try {
    await engine.open({ deviceId: id });
    startMeterLoop();
    setStatus('');
  } catch (err) {
    setStatus(err.message, 'bad');
  }
  paintTransport();
}

const MONITOR_COMFORT_MS = 12;

function paintDeviceInfo(detail) {
  const monMs = (detail.monitorLatency || 0) * 1000;
  const bits = [
    `${detail.sampleRate} Hz`,
    detail.shared ? 'shared memory' : 'buffered transfer',
    `${(detail.latency * 1000).toFixed(0)} ms reported`,
    `monitoring ≈ ${monMs.toFixed(0)} ms`,
  ];
  $('#rec-devinfo').textContent = bits.join(' · ');
  paintMonitorAdvice(monMs);
}

function paintMonitorAdvice(monMs) {
  const host = $('#rec-monitor-note');
  if (!host) return;
  if (!$('#rec-monitor').checked) { host.innerHTML = ''; return; }
  if (monMs <= MONITOR_COMFORT_MS) {
    host.innerHTML = note(`Monitoring delay is about ${monMs.toFixed(0)} ms, ` +
                          'which reads as room rather than echo.', 'good');
    return;
  }
  host.innerHTML = note(
    `Monitoring through the browser adds about ${monMs.toFixed(0)} ms to your ` +
    'own voice. Past ~12 ms most singers find it distracting and start ' +
    'dragging behind the beat. If your interface has direct monitoring, use ' +
    'that and leave this off; otherwise sing with one ear uncovered.', 'warn');
}

/* ── metering ────────────────────────────────────────────────────────────── */

function startMeterLoop() {
  stopMeterLoop();
  const tick = () => {
    const m = engine.readMetrics();
    if (m) {
      paintMeter(m);
      if (engine.state === CAPTURE_STATE.RECORDING && !countingIn) {
        const now = engine.ctx.currentTime - takeStart;
        const cue = coach.push(m, now);
        if (cue) paintCues();
        paintClock(now);
      }
    }
    rafId = requestAnimationFrame(tick);
  };
  rafId = requestAnimationFrame(tick);
}

function stopMeterLoop() {
  cancelAnimationFrame(rafId);
  rafId = 0;
}

function paintMeter(m) {
  const tone = levelTone(m);
  const fill = $('#rec-level');
  const pct = Math.max(0, Math.min(1, (m.rmsDb + 60) / 60)) * 100;
  fill.style.width = pct.toFixed(1) + '%';
  fill.className = `level-fill is-${tone}`;

  const peak = $('#rec-peakbar');
  const ppct = Math.max(0, Math.min(1, (m.truePeakDb + 60) / 60)) * 100;
  peak.style.left = ppct.toFixed(1) + '%';
  peak.classList.toggle('is-over', m.truePeakDb > -1);

  $('#rec-peak').textContent = m.peakDb > -119 ? m.peakDb.toFixed(1) : DASH;
  $('#rec-tp').textContent = m.truePeakDb > -119 ? m.truePeakDb.toFixed(1) : DASH;
  $('#rec-rms').textContent = m.rmsDb > -119 ? m.rmsDb.toFixed(1) : DASH;
  $('#rec-lufs').textContent = m.lufsShort > -119 ? m.lufsShort.toFixed(1) : DASH;
  const proximity = m.mid > 1e-9 ? m.low / m.mid : 0;
  $('#rec-prox').textContent = proximity > 0 ? proximity.toFixed(2) : DASH;

  trace.push(m.rmsDb);

  // The tuner is silent during a take, deliberately. See coach.js.
  const tuning = $('#rec-tuner');
  if (engine.state === CAPTURE_STATE.RECORDING) {
    tuning.dataset.mode = 'muted';
    $('#rec-note').textContent = '·';
    $('#rec-cents').textContent = 'not shown while recording';
    $('#rec-needle').style.transform = 'translateX(0)';
    return;
  }
  tuning.dataset.mode = 'live';
  const n = m.clarity > 0.55 ? noteName(m.pitchHz) : null;
  if (n) {
    $('#rec-note').textContent = n.name;
    $('#rec-cents').textContent = `${n.cents > 0 ? '+' : ''}${n.cents} cents`;
    $('#rec-needle').style.transform =
      `translateX(${Math.max(-50, Math.min(50, n.cents))}px)`;
    tuning.classList.toggle('is-intune', Math.abs(n.cents) <= 10);
  } else {
    $('#rec-note').textContent = '·';
    $('#rec-cents').textContent = 'sing to tune';
    $('#rec-needle').style.transform = 'translateX(0)';
    tuning.classList.remove('is-intune');
  }
}

function paintClock(now) {
  $('#rec-clock').textContent = clockMs(now);
  if (!beatInfo?.bpm) { $('#rec-bar').textContent = DASH; return; }
  const bpb = beatInfo.beats_per_bar || 4;
  const beat = Math.floor(now / (60 / beatInfo.bpm));
  $('#rec-bar').textContent = `${Math.floor(beat / bpb) + 1}.${(beat % bpb) + 1}`;
}

/* ── beat selection and guidance ─────────────────────────────────────────── */

function fillBeats(catalog) {
  const sel = $('#rec-beat');
  const current = sel.value;
  sel.innerHTML = '<option value="">No beat — bare vocal</option>' +
    catalog.map(b => `<option value="${esc(b.beat_id)}">${esc(b.title || b.beat_id)}
      · ${num(b.bpm, 0)} BPM · ${esc(b.key_name || '')}</option>`).join('');
  if (current) sel.value = current;
}

async function selectBeat(id) {
  beatInfo = null;
  if (beatAudio) { beatAudio.pause(); beatAudio = null; }
  const host = $('#rec-guidance');
  try {
    const q = id ? `?beat_id=${encodeURIComponent(id)}` : '';
    const g = await api('/api/take/guidance' + q);
    beatInfo = g.beat;
    if (g.beat?.source_name) {
      beatAudio = new Audio(
        `/api/source/beats/${encodeURIComponent(g.beat.source_name)}`);
      beatAudio.preload = 'auto';
    }
    host.innerHTML = g.cues.length
      ? g.cues.map(cueMarkup).join('')
      : '<p class="quiet">Nothing specific to say before this take.</p>';
  } catch (err) {
    host.innerHTML = note(err.message, 'bad');
  }
  paintFacts();
  takeWave.setDownbeats(beatInfo?.downbeats || []);
}

function cueMarkup(c) {
  return `
    <div class="cue cue-${esc(c.severity)}">
      <p class="cue-msg">${esc(c.message)}</p>
      ${c.detail ? `<p class="cue-why">${esc(c.detail)}</p>` : ''}
    </div>`;
}

function paintFacts() {
  const b = beatInfo;
  $('#rec-facts').innerHTML = b ? `
    <div><dt>Key</dt><dd>${esc(b.key || DASH)}</dd></div>
    <div><dt>Tempo</dt><dd class="mono">${num(b.bpm, 1, ' BPM')}</dd></div>
    <div><dt>Bars</dt><dd class="mono">${esc(b.bars ?? DASH)}</dd></div>
    <div><dt>Length</dt><dd class="mono">${clock(b.duration_s)}</dd></div>`
    : `
    <div><dt>Mode</dt><dd>Free — no beat</dd></div>
    <div><dt>Guidance</dt><dd>Technical only</dd></div>`;
}

/* ── room and latency ────────────────────────────────────────────────────── */

async function roomCheck() {
  if (engine.state === CAPTURE_STATE.IDLE) return;
  const btn = $('#rec-room');
  btn.disabled = true;
  btn.textContent = 'Listening…';
  setStatus('Stay quiet for three seconds.');

  engine.start();
  await new Promise(r => setTimeout(r, 3000));
  const take = await engine.stop();

  btn.disabled = false;
  btn.textContent = 'Check the room';
  if (!take) { setStatus('Nothing was captured.', 'bad'); return; }

  try {
    const blob = encodeWav(take.samples, take.sampleRate);
    const r = await postForm('/api/take/preflight', {
      file: new File([blob], 'roomcheck.wav', { type: 'audio/wav' }),
      beat_id: $('#rec-beat').value || null,
    });
    $('#rec-room-result').innerHTML = `
      <dl class="facts">
        <div><dt>Noise floor</dt><dd class="mono">${num(r.quality.noise_floor_db, 1, ' dB')}</dd></div>
        <div><dt>Room decay</dt><dd class="mono">${num(r.quality.estimated_rt60_s, 2, ' s')}</dd></div>
      </dl>` + r.cues.map(cueMarkup).join('');
    setStatus(r.ok ? 'Room is usable.' : 'The room will limit how tight this can sound.',
              r.ok ? 'good' : 'warn');
  } catch (err) {
    setStatus(err.message, 'bad');
  }
}

async function calibrateLatency() {
  if (engine.state === CAPTURE_STATE.IDLE) return;
  const btn = $('#rec-calibrate');
  btn.disabled = true;
  setStatus('Measuring — a few short sweeps will play through the speakers.');

  try {
    const result = await measure(engine, {
      onProgress: p => { btn.textContent = `Measuring… ${Math.round(p * 100)}%`; },
    });
    if (result.confidence > 0) {
      engine.setLatency(result.seconds);
      prefs.set('latencySeconds', result.seconds);
      prefs.set('latencyConfidence', result.confidence);
      update('latency', result);
      setStatus(`Round trip is ${(result.seconds * 1000).toFixed(0)} ms. ` +
                'Takes recorded against playback will be corrected by that.',
                'good');
    } else {
      update('latency', result);
      setStatus(result.note, 'warn');
    }
  } catch (err) {
    setStatus(err.message, 'bad');
  }
  btn.disabled = false;
  btn.textContent = 'Calibrate latency';
}

function paintLatency(l) {
  const host = $('#rec-latency');
  if (!l) {
    host.innerHTML = '<p class="quiet">Not measured. Takes recorded over ' +
      'speaker playback will be late by the round trip.</p>';
    return;
  }
  if (!l.confidence) {
    host.innerHTML = note(l.note, 'warn');
    return;
  }
  host.innerHTML = `
    <dl class="facts">
      <div><dt>Round trip</dt><dd class="mono">${num(l.seconds * 1000, 0, ' ms')}</dd></div>
      <div><dt>Browser reported</dt><dd class="mono">${num((l.reported || 0) * 1000, 0, ' ms')}</dd></div>
      <div><dt>Confidence</dt><dd class="mono">${num(l.confidence * 100, 0, '%')}</dd></div>
    </dl>
    ${l.restored ? '<p class="quiet">Restored from the last session.</p>' : ''}
    ${l.note ? note(l.note, 'warn') : ''}`;
}

/* ── the take ────────────────────────────────────────────────────────────── */

async function startTake() {
  if (engine.state === CAPTURE_STATE.IDLE) return;
  coach.reset();
  trace.reset();
  paintCues();

  const bars = Number($('#rec-countin').value) || 0;
  if (bars > 0 && beatInfo?.bpm) await countIn(bars);

  takeStart = engine.ctx.currentTime;
  engine.start();
  if (beatAudio) { beatAudio.currentTime = 0; beatAudio.play().catch(() => {}); }
  $('#recorder').classList.add('is-live');
  paintTransport();
  setStatus('');
}

async function countIn(bars) {
  countingIn = true;
  const bpb = beatInfo.beats_per_bar || 4;
  const beatS = 60 / beatInfo.bpm;
  const el = $('#rec-count');
  show(el, true);
  for (let b = 0; b < bars * bpb; b++) {
    const accent = b % bpb === 0;
    el.textContent = String((b % bpb) + 1);
    el.classList.toggle('is-down', accent);
    click(accent);
    await new Promise(r => setTimeout(r, beatS * 1000));
  }
  show(el, false);
  countingIn = false;
}

function click(accent) {
  const ctx = engine.ctx;
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  osc.frequency.value = accent ? 1600 : 1000;
  gain.gain.setValueAtTime(0.18, ctx.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.05);
  osc.connect(gain).connect(ctx.destination);
  osc.start();
  osc.stop(ctx.currentTime + 0.06);
}

async function stopTake() {
  const result = await engine.stop();
  if (beatAudio) beatAudio.pause();
  $('#recorder').classList.remove('is-live');
  paintTransport();

  if (!result || result.seconds < 1.0) {
    setStatus('That take was under a second — nothing to review.', 'warn');
    return;
  }

  const stats = summarise(result.samples);
  const blob = encodeWav(result.samples, result.sampleRate);
  const index = store.takes.length;
  const name = `take-${index + 1}.wav`;

  const take = {
    name,
    seconds: result.seconds,
    samples: result.samples,
    sampleRate: result.sampleRate,
    url: URL.createObjectURL(blob),
    stats,
    latencyTrimmedMs: (result.latencySamples / result.sampleRate) * 1000,
    beat_id: $('#rec-beat').value || null,
    report: null,
    path: null,
  };
  update('takes', t => [take, ...t]);
  showTakeWave(take);
  setStatus('Reviewing the take…');

  try {
    const r = await postForm('/api/take/review', {
      file: new File([blob], name, { type: 'audio/wav' }),
      beat_id: take.beat_id,
    });
    take.report = r;
    take.path = r.path;
    update('takes', t => [...t]);
    setStatus('');
  } catch (err) {
    setStatus(err.message, 'bad');
  }
}

function paintTransport() {
  const open = engine.state !== CAPTURE_STATE.IDLE;
  const recording = engine.state === CAPTURE_STATE.RECORDING;
  $('#rec-enable').disabled = open;
  $('#rec-room').disabled = !open || recording;
  $('#rec-calibrate').disabled = !open || recording;
  $('#rec-go').disabled = !open || recording;
  show($('#rec-go'), !recording);
  show($('#rec-stop'), recording);
  $('#rec-monitor').disabled = !open;
}

function paintCues() {
  const host = $('#rec-cues');
  if (!coach.cues.length) {
    host.innerHTML = engine.state === CAPTURE_STATE.RECORDING
      ? '<p class="quiet">Nothing to fix. Keep going.</p>'
      : '<p class="quiet">Live cues appear here while you record.</p>';
    return;
  }
  host.innerHTML = coach.cues.map(c => `
    <div class="cue cue-${esc(c.severity)}">
      <span class="cue-time mono">${clock(c.at)}</span>
      <span class="cue-msg">${esc(c.message)}</span>
    </div>`).join('');
}

/* ── takes ───────────────────────────────────────────────────────────────── */

const GRADE_ORDER = ['excellent', 'good', 'ok', 'rough', 'unusable'];

function rank(take) {
  const r = take.report;
  if (!r) return [GRADE_ORDER.length + 1, 999];
  const g = GRADE_ORDER.indexOf(r.grade);
  return [g < 0 ? GRADE_ORDER.length : g, r.median_pitch_error_cents ?? 999];
}

function paintTakes(takes) {
  const host = $('#rec-takes');
  if (!takes.length) {
    host.innerHTML = '<p class="placeholder">Takes appear here once recorded.</p>';
    return;
  }
  const best = takes.reduce((a, b) => {
    const [ga, pa] = rank(a), [gb, pb] = rank(b);
    return (gb < ga || (gb === ga && pb < pa)) ? b : a;
  }, takes[0]);

  host.innerHTML = takes.map((t, i) => {
    const r = t.report;
    const cues = (r?.cues || []).slice(0, 3).map(c =>
      `<div class="cue cue-${esc(c.severity)}"><span class="cue-msg">${esc(c.message)}</span></div>`
    ).join('');
    return `
      <article class="take${t === best ? ' is-best' : ''}">
        <header>
          <span class="take-n mono">${takes.length - i}</span>
          <span class="take-name">${esc(t.name)}</span>
          ${t === best ? '<span class="pill amber">best</span>' : ''}
          ${r ? `<span class="pill">${esc(r.grade)}</span>` : '<span class="pill">reviewing…</span>'}
          <span class="take-len mono">${clock(t.seconds)}</span>
        </header>
        <dl class="facts compact">
          <div><dt>Peak</dt><dd class="mono">${num(t.stats.peakDb, 1, ' dB')}</dd></div>
          <div><dt>Level</dt><dd class="mono">${num(t.stats.rmsDb, 1, ' dB')}</dd></div>
          <div><dt>Clipped</dt><dd class="mono">${t.stats.clippedSamples || 0}</dd></div>
          <div><dt>Pitch error</dt><dd class="mono">${
            r?.median_pitch_error_cents != null
              ? num(r.median_pitch_error_cents, 0, ' ¢') : DASH}</dd></div>
          ${t.latencyTrimmedMs > 0.5 ? `<div><dt>Latency trimmed</dt>
            <dd class="mono">${num(t.latencyTrimmedMs, 0, ' ms')}</dd></div>` : ''}
        </dl>
        ${cues}
        <footer>
          <button class="btn sm" data-take-play="${i}">Show</button>
          <audio controls preload="none" src="${esc(t.url)}"></audio>
          <button class="btn sm" data-take-drop="${i}">Discard</button>
          <button class="btn sm primary" data-take-use="${i}"${r ? '' : ' disabled'}>Use this take</button>
        </footer>
      </article>`;
  }).join('');
}

function showTakeWave(take) {
  show($('#take-wave-panel'), true);
  takeWave.setAudio(take.samples, take.sampleRate);
  takeWave.setDownbeats(beatInfo?.downbeats || []);
  $('#take-wave-title').textContent = take.name;
}

function playTake(index) {
  const take = store.takes[index];
  if (take) showTakeWave(take);
}

function dropTake(index) {
  const take = store.takes[index];
  if (take?.url) URL.revokeObjectURL(take.url);
  update('takes', t => t.filter((_, i) => i !== index));
}

async function useTake(index) {
  const take = store.takes[index];
  if (!take?.path) return;
  update('activeTake', take);
  window.dispatchEvent(new CustomEvent('mixengine:use-take', { detail: take }));
}

function setStatus(message, tone = '') {
  $('#rec-status').innerHTML = note(message, tone);
}

export function teardown() {
  stopMeterLoop();
  engine.close();
}
