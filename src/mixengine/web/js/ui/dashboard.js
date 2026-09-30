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

const FIELDS = ['vocal_state', 'relationship', 'tune', 'timing', 'space',
                'nudge', 'key'];

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

/**
 * What was done to the files, said plainly.
 *
 * The loader's repairs and the analysis' warnings arrive with the
 * prepare step; what the render itself silenced, moved, stretched,
 * looped or trimmed arrives with the render. Shown before the questions
 * and again with the result, so nothing is done to a take unannounced.
 */
function paintNotes(id, lines) {
  const seen = new Set();
  const notes = (lines || []).filter(l => l && !seen.has(l) && seen.add(l));
  const el = $(`#${id}`);
  el.innerHTML = notes.map(n => `<li>${esc(n)}</li>`).join('');
  el.hidden = notes.length === 0;
}

function preparedNotes() {
  const v = prepared?.vocal || {};
  const b = prepared?.beat || {};
  return [...(v.repairs || []), ...(v.warnings || []),
          ...(b.repairs || []), ...(b.warnings || [])];
}

/**
 * Where the vocal was put, and how sure the engine was about it.
 *
 * Every other decision here is measured. This one partly is not: whether
 * a first line is a pickup into the bar or lands on it is a musical
 * reading, not a fact in the audio, and for a take that was never
 * recorded to this beat the engine is choosing. It reports the confidence
 * its own bar-grid reading had, and when that reading failed it says so
 * rather than presenting a guess as a measurement.
 *
 * The nudge sits here rather than in Treatment because this is the moment
 * the person can hear that it is wrong.
 */
export function paintPlacement(render) {
  const t = render.transform || {};
  const align = t.alignment || {};
  const place = align.placement || {};
  const grid = t.vocal_grid || {};
  const panel = $('#placement');

  const where = [];
  if (place.method === 'section_entry' && place.section) {
    where.push(`The first line lands where the beat's ${place.section} arrives, at ${fmt(place.entry_s)}s.`);
  } else if (align.method === 'measured_offset') {
    where.push('The take was recorded to this beat, so it was left where it was performed.');
  } else if (place.reason) {
    where.push(cap(place.reason) + '.');
  } else {
    where.push('The vocal was placed on the beat’s bar lines.');
  }

  // A take recorded to this beat needs no grid reading; anything else
  // rests on one, and the engine knows how well that reading went.
  const locked = align.method === 'measured_offset';
  const conf = Number(grid.confidence);
  const unsure = !locked && (!Number.isFinite(conf) || conf < 0.6);
  panel.classList.toggle('is-unsure', unsure);

  let why = '';
  if (locked) {
    why = 'Measured against the beat it was recorded over, so this is not a guess.';
  } else if (unsure) {
    why = 'The take’s own bar grid could not be read clearly, so where its bars fall is the engine’s best reading rather than a measurement. If it sounds off the bar, move it.';
  } else {
    why = `The take’s own bar grid read clearly${grid.conditioning && grid.conditioning !== 'raw'
      ? ` (after the file was re-read as ${esc(grid.conditioning)}, the take as it arrived being too damaged to track)` : ''}, so its bars were laid over the beat’s.`;
  }

  $('#placement-where').textContent = where.join(' ');
  $('#placement-conf').textContent = locked ? 'recorded to this beat'
    : Number.isFinite(conf) ? `grid ${Math.round(conf * 100)}%` : 'no grid found';
  $('#placement-why').textContent = why;

  const applied = Number(align.nudge_beats) || 0;
  for (const b of $('#placement').querySelectorAll('.nudge-btn')) {
    b.classList.toggle('is-on', Number(b.dataset.nudge) === applied && applied !== 0);
    b.disabled = false;
  }
  panel.hidden = false;
}

const fmt = n => (Number(n) || 0).toFixed(1);
const cap = s => String(s || '').replace(/^./, c => c.toUpperCase());

function paintResult(render) {
  $('#run').hidden = true;
  $('#result').hidden = false;
  paintPlacement(render);
  paintNotes('result-notes', [...preparedNotes(), ...(render.notes || [])]);
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

/**
 * Two steps, not one.
 *
 * The engine first listens to both files and says what it could not
 * settle on its own — a count-in before the first line, a take that
 * reads as rap but not clearly, noise it could not remove. Those are
 * facts the person who made the recording has and the engine does not,
 * so they are asked, with the engine's own answer preselected, before
 * a minute of rendering is spent on the wrong one. When there is
 * nothing to ask the render starts straight away.
 */
let prepared = null;   // { vocal_path, beat_path, questions } from /api/prepare

function stageProgress(j) {
  const i = STAGES.indexOf(j.stage);
  showStage(j.message || j.stage || 'Working…',
            8 + (i < 0 ? 0 : (i + 1) / STAGES.length * 88));
}

function beginRun() {
  $('#go').disabled = true;
  $('#result').hidden = true;
  $('#ask').hidden = true;
  $('#run').hidden = false;
  $('#plan').hidden = true;
  $('#rail-fill').style.background = '';
}

function failRun(err) {
  showStage(err.message, 100);
  $('#rail-fill').style.background = 'var(--red)';
  $('#go').disabled = false;
}

async function run() {
  beginRun();
  showStage('Uploading…', 4);

  const body = new FormData();
  body.append('vocal', slots.vocal.file);
  body.append('beat', slots.beat.file);
  const intents = readIntents();
  if (intents.key) body.append('key', intents.key);

  try {
    const job = await api('/api/prepare', { method: 'POST', body });
    prepared = await pollJob(job.id, { onProgress: stageProgress });
  } catch (err) {
    failRun(err);
    return;
  }

  const questions = prepared.questions || [];
  if (questions.length) {
    showStage('Waiting for you', 40);
    $('#run').hidden = true;
    askQuestions(questions);
    $('#go').disabled = false;
    return;
  }
  await renderPrepared({});
}

function askQuestions(questions) {
  const block = questions.some(q => q.severity === 'block');
  $('#ask-lead').textContent = block
    ? 'One of these decides whether this take can be rendered at all.'
    : 'A few things the engine could not settle from the audio alone. Its own answer is preselected.';
  $('#ask-list').innerHTML = questions.map(q => `
    <fieldset class="ask-q ${esc(q.severity)}" data-id="${esc(q.id)}" data-intent="${esc(q.intent)}">
      <legend>${esc(q.text)}</legend>
      <div class="ask-opts">
        ${q.options.map(o => `
          <label class="ask-opt">
            <input type="radio" name="q-${esc(q.id)}" value="${esc(o.value)}"
                   ${o.value === q.default ? 'checked' : ''}>
            <span>${esc(o.label)}</span>
          </label>`).join('')}
      </div>
      ${q.reason ? `<p class="ask-why">${esc(q.reason)}</p>` : ''}
    </fieldset>`).join('');
  paintNotes('ask-notes', preparedNotes());
  $('#ask').hidden = false;
  syncAskButton();
  $('#ask').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/**
 * The submit button says what it will do.
 *
 * "I'll upload a cleaner take" does not render anything — it frees the
 * slot for the next take. A button that still read "Make the render"
 * underneath that choice described the opposite of what it did.
 */
const REFUSAL_ASKS = {
  noise: 'upload a cleaner take',
  length: 'upload the full take',
  voice: 'upload the vocal take',
};

function syncAskButton() {
  const { refused } = readAnswers();
  const ask = REFUSAL_ASKS[refused] || 'upload a different take';
  $('#ask-go span').textContent = !refused ? 'Make the render'
    : ask[0].toUpperCase() + ask.slice(1);
}

function readAnswers() {
  const out = {};
  let refused = null;                       // id of the question turned down
  for (const fs of $('#ask-list').querySelectorAll('.ask-q')) {
    const picked = fs.querySelector('input:checked');
    if (!picked) continue;
    if (picked.value === 'rerecord') { refused = refused || fs.dataset.id; continue; }
    if (picked.value === 'auto') continue;
    out[fs.dataset.intent] = picked.value;
  }
  return { answers: out, refused };
}

async function submitAnswers(e) {
  e.preventDefault();
  const { answers, refused } = readAnswers();
  if (refused) {
    // The person is going to bring a better take. Free the slot for it and
    // say which kind, rather than rendering the one they just turned down.
    $('#ask').hidden = true;
    clearSlot('vocal');
    $('#vocal-meta').textContent = REFUSAL_ASKS[refused] || 'upload a different take';
    $('#slot-vocal').scrollIntoView({ behavior: 'smooth', block: 'center' });
    return;
  }
  await renderPrepared(answers);
}

// Kept so a nudge re-renders the same decisions with one thing moved,
// rather than dropping the answers the person already gave.
let lastAnswers = {};

async function renderPrepared(answers) {
  lastAnswers = { ...answers };
  beginRun();
  showStage('Starting the render…', 8);

  const body = new FormData();
  body.append('vocal_path', prepared.vocal_path);
  body.append('beat_path', prepared.beat_path);
  for (const [k, v] of Object.entries(readIntents())) body.append(k, v);
  for (const [k, v] of Object.entries(answers)) body.append(k, v);

  try {
    const job = await api('/api/render', { method: 'POST', body });
    const result = await pollJob(job.id, { onProgress: stageProgress });

    const render = result?.renders?.[0];
    if (!render) {
      throw new Error(result?.failures?.[0]?.error
        || 'The engine finished without producing a render.');
    }
    paintPlan(render.transform?.plan);
    showStage('Done', 100);
    paintResult(render);
  } catch (err) {
    failRun(err);
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
  on($('#ask-form'), 'submit', submitAnswers);
  on($('#ask-list'), 'change', syncAskButton);
  on($('#ask-back'), 'click', () => {
    $('#ask').hidden = true;
    $('#slot-vocal').scrollIntoView({ behavior: 'smooth', block: 'center' });
  });
  on($('#again'), 'click', () => {
    $('#result').hidden = true;
    $('#treatment').open = true;
    $('#treatment').scrollIntoView({ behavior: 'smooth', block: 'center' });
  });

  // Nudging re-renders the same decisions with the vocal moved. The files
  // are already analysed, so this costs a render and not an intake.
  on($('#placement'), 'click', e => {
    const btn = e.target.closest('.nudge-btn');
    if (!btn || !prepared) return;
    const step = Number(btn.dataset.nudge) || 0;
    const applied = (Number(lastAnswers.nudge) || 0) + step;
    for (const b of $('#placement').querySelectorAll('.nudge-btn')) b.disabled = true;
    renderPrepared(applied === 0
      ? Object.fromEntries(Object.entries(lastAnswers).filter(([k]) => k !== 'nudge'))
      : { ...lastAnswers, nudge: String(applied) });
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
