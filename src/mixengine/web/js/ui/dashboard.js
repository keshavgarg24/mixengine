/**
 * The dashboard: two slots, one run.
 *
 * Filling a slot decodes the file in the browser and draws it. That is
 * all it does. Nothing is uploaded, nothing is analysed, and no work
 * starts on the server until you press the button — the previous
 * interface fired a separate analysis on every upload, which meant the
 * engine ran three times to make one song and the user had to understand
 * its internal stages to operate it.
 */

import { api, pollJob } from '../core/api.js';
import { $, clock, esc, on } from '../core/dom.js';

const slots = {
  vocal: { file: null, buffer: null },
  beat: { file: null, buffer: null },
};

let audioCtx = null;
const ctx = () => (audioCtx ||= new (window.AudioContext || window.webkitAudioContext)());

/* ── Waveform ──────────────────────────────────────────────────────── */

/**
 * Draw peaks, not samples.
 *
 * A canvas is a few hundred pixels wide and a take is a few million
 * samples, so every column stands for thousands of them. Drawing the
 * extremes of each bucket keeps transients visible; drawing an average
 * would flatten exactly the detail that tells you where the words are.
 */
function drawWave(canvas, buffer) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const g = canvas.getContext('2d');
  g.scale(dpr, dpr);
  g.clearRect(0, 0, w, h);

  const data = buffer.getChannelData(0);
  const step = Math.max(1, Math.floor(data.length / w));
  const mid = h / 2;

  g.fillStyle = getComputedStyle(document.documentElement)
    .getPropertyValue('--wave').trim() || '#7A6F92';

  for (let x = 0; x < w; x++) {
    let min = 1, max = -1;
    const start = x * step;
    for (let i = 0; i < step; i++) {
      const v = data[start + i];
      if (v === undefined) break;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    if (min > max) continue;
    const y1 = mid + min * mid * 0.92;
    const y2 = mid + max * mid * 0.92;
    g.fillRect(x, y1, 1, Math.max(1, y2 - y1));
  }
}

/* ── Slots ─────────────────────────────────────────────────────────── */

async function fillSlot(kind, file) {
  const slot = slots[kind];
  slot.file = file;

  $(`#${kind}-name`).textContent = file.name;
  $(`#${kind}-empty`).hidden = true;
  $(`#${kind}-filled`).hidden = false;
  $(`#${kind}-meta`).textContent = 'reading…';

  try {
    const buf = await ctx().decodeAudioData(await file.arrayBuffer());
    slot.buffer = buf;
    $(`#${kind}-meta`).textContent = clock(buf.duration);
    drawWave($(`#${kind}-wave`), buf);
  } catch {
    // An undecodable file is still renderable — ffmpeg on the server
    // reads formats the browser will not. Say so rather than refusing.
    slot.buffer = null;
    $(`#${kind}-meta`).textContent = 'ready';
  }
  refreshGo();
}

function clearSlot(kind) {
  slots[kind] = { file: null, buffer: null };
  $(`#${kind}-empty`).hidden = false;
  $(`#${kind}-filled`).hidden = true;
  $(`#${kind}-meta`).textContent = '';
  refreshGo();
}

function refreshGo() {
  const go = $('#go');
  const label = $('#go-label');
  const haveV = !!slots.vocal.file, haveB = !!slots.beat.file;

  go.disabled = !(haveV && haveB);
  if (haveV && haveB) label.textContent = 'Make the render';
  else if (haveV) label.textContent = 'Add a beat';
  else if (haveB) label.textContent = 'Add a vocal';
  else label.textContent = 'Add a vocal and a beat';
}

/* ── Treatment ─────────────────────────────────────────────────────── */

const FIELDS = ['vocal_state', 'relationship', 'tune', 'timing', 'space', 'key'];

function readIntents() {
  const out = {};
  for (const f of FIELDS) {
    const el = $(`#i-${f}`);
    const v = (el?.value || '').trim();
    if (v && v !== 'auto') out[f] = v;
  }
  return out;
}

function refreshTreatState() {
  const n = Object.keys(readIntents()).length;
  $('#treat-state').textContent = n === 0
    ? 'everything automatic'
    : `${n} set by you, the rest automatic`;
}

/* ── The run ───────────────────────────────────────────────────────── */

const STAGES = ['intake', 'plan', 'transform', 'mix', 'master', 'check'];

function showStage(name, pct) {
  $('#rail-fill').style.width = `${Math.round(pct)}%`;
  $('#run-stage').textContent = name;
}

function paintPlan(plan) {
  if (!plan) return;
  $('#plan').hidden = false;
  $('#plan-summary').textContent = plan.summary || '';
  const stages = plan.stages || {};
  $('#plan-list').innerHTML = Object.entries(stages).map(([name, d]) => `
    <li class="${d.enabled ? '' : 'off'}">
      <b>${esc(name.replace(/_/g, ' '))}</b>
      <span>${esc(d.reason || (d.enabled ? 'on' : 'off'))}</span>
    </li>`).join('');
}

function paintResult(render) {
  $('#run').hidden = true;
  $('#result').hidden = false;
  $('#result-score').textContent = `${render.score_pct}% · ${clock(render.duration_s)}`;

  const url = render.download;
  $('#player').src = url;
  $('#download').href = url;
  $('#download').setAttribute('download', render.path.split('/').pop());

  const gates = (render.critic?.gates || []).filter(g => g.severity !== 'skipped');
  $('#gates').innerHTML = gates.map(g => {
    const tone = g.severity === 'info' ? 'info' : g.passed ? 'good'
      : g.severity === 'error' ? 'bad' : 'warn';
    return `<span class="chip ${tone}">${esc(g.name.replace(/_/g, ' '))}</span>`;
  }).join('');
}

async function run() {
  $('#go').disabled = true;
  $('#result').hidden = true;
  $('#run').hidden = false;
  $('#plan').hidden = true;
  showStage('Uploading…', 4);

  const body = new FormData();
  body.append('vocal', slots.vocal.file);
  body.append('beat', slots.beat.file);
  for (const [k, v] of Object.entries(readIntents())) body.append(k, v);

  try {
    const job = await api('/api/render', { method: 'POST', body });
    const result = await pollJob(job.job_id, {
      onProgress: j => {
        const i = STAGES.indexOf(j.stage);
        showStage(j.message || j.stage || 'Working…',
                  8 + (i < 0 ? 0 : (i + 1) / STAGES.length * 88));
      },
    });

    const render = result?.renders?.[0];
    if (!render) {
      throw new Error(result?.failures?.[0]?.error
        || 'The engine finished without producing a render.');
    }
    paintPlan(render.transform?.plan);
    showStage('Done', 100);
    paintResult(render);
  } catch (err) {
    showStage(err.message, 100);
    $('#rail-fill').style.background = 'var(--red)';
  } finally {
    $('#go').disabled = false;
  }
}

/* ── Wiring ────────────────────────────────────────────────────────── */

export function init() {
  for (const kind of ['vocal', 'beat']) {
    const slot = $(`#slot-${kind}`);
    const input = $(`#${kind}-file`);

    on(input, 'change', () => {
      if (input.files?.[0]) fillSlot(kind, input.files[0]);
    });

    // Drag and drop onto the slot it belongs to.
    on(slot, 'dragover', e => { e.preventDefault(); slot.classList.add('is-over'); });
    on(slot, 'dragleave', () => slot.classList.remove('is-over'));
    on(slot, 'drop', e => {
      e.preventDefault();
      slot.classList.remove('is-over');
      const f = e.dataTransfer?.files?.[0];
      if (f) fillSlot(kind, f);
    });
  }

  on(document, 'click', e => {
    const btn = e.target.closest('[data-act]');
    if (!btn) return;
    const { act, kind } = btn.dataset;
    if (act === 'upload') $(`#${kind}-file`).click();
    if (act === 'clear') clearSlot(kind);
    if (act === 'record') startRecording(kind);
  });

  for (const f of FIELDS) on($(`#i-${f}`), 'change', refreshTreatState);
  on($('#i-key'), 'input', refreshTreatState);
  on($('#go'), 'click', run);
  on($('#again'), 'click', () => {
    $('#result').hidden = true;
    $('#treatment').open = true;
    $('#treatment').scrollIntoView({ behavior: 'smooth', block: 'center' });
  });

  // Redraw on resize: the canvas is sized in device pixels, so a bare
  // CSS resize would leave it stretched.
  let t;
  window.addEventListener('resize', () => {
    clearTimeout(t);
    t = setTimeout(() => {
      for (const kind of ['vocal', 'beat']) {
        if (slots[kind].buffer) drawWave($(`#${kind}-wave`), slots[kind].buffer);
      }
    }, 120);
  });

  refreshGo();
  refreshTreatState();
}

/* ── Recording ─────────────────────────────────────────────────────── */

let recorder = null;

/**
 * Record into a slot, playing the beat if there is one.
 *
 * Monitoring the beat while the vocal is cut is what makes the two
 * genuinely locked, so the take that comes out needs no timing
 * correction at all. The beat is played through the page rather than
 * mixed into the capture; what is recorded is the microphone alone.
 */
async function startRecording(kind) {
  const slot = $(`#slot-${kind}`);
  const btn = slot.querySelector('[data-act="record"]');

  if (recorder) {
    recorder.stop();
    return;
  }

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false,
               autoGainControl: false },
    });
  } catch {
    $(`#${kind}-meta`).textContent = 'microphone unavailable';
    return;
  }

  const chunks = [];
  recorder = new MediaRecorder(stream);
  recorder.ondataavailable = e => e.data.size && chunks.push(e.data);

  // Play the beat underneath, when recording a vocal against one.
  let monitor = null;
  if (kind === 'vocal' && slots.beat.file) {
    monitor = new Audio(URL.createObjectURL(slots.beat.file));
    monitor.play().catch(() => {});
  }

  recorder.onstop = async () => {
    stream.getTracks().forEach(t => t.stop());
    if (monitor) { monitor.pause(); URL.revokeObjectURL(monitor.src); }
    recorder = null;
    slot.classList.remove('is-active');
    btn.textContent = 'Record';
    const blob = new Blob(chunks, { type: chunks[0]?.type || 'audio/webm' });
    await fillSlot(kind, new File([blob], `${kind}-take.webm`, { type: blob.type }));
  };

  recorder.start();
  slot.classList.add('is-active');
  btn.textContent = 'Stop';
  $(`#${kind}-meta`).textContent = monitor ? 'recording over the beat' : 'recording';
}
