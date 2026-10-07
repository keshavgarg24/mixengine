/**
 * The vocal view: upload or receive a take, analyse it, show the matches.
 *
 * The take report is shown here and never during recording. That split is
 * the whole capture design in one place: detail after, silence during.
 */

import { api, pollJob, postForm } from '../core/api.js';
import { $, $$, clock, DASH, esc, meterRow, note, num, on, show } from '../core/dom.js';
import { store, subscribe, update } from '../core/store.js';
import { wireDrop } from './catalog.js';

let chosen = new Set();

let recordedOver = null;

export function init() {
  wireDrop($('#vocal-drop'), $('#vocal-input'), files => analyze(files[0]));
  on($('#vocal-browse'), 'click', () => $('#vocal-input').click());
  subscribe('catalog', fillRecordedOver);
  on($('#vocal-over'), 'change', ev => { recordedOver = ev.target.value || null; });

  // A take handed over from the record view is already on the server, so it
  // is matched by path rather than uploaded a second time.
  window.addEventListener('mixengine:use-take', ev => adoptTake(ev.detail));
}

export async function analyze(file) {
  const status = $('#vocal-status');
  status.innerHTML =
    '<p class="quiet"><span class="spin"></span> Analysing — separation, pitch, phrasing…</p>';
  $('#vocal-readout').innerHTML = '';
  $('#match-panel').innerHTML = '<p class="placeholder">Matching…</p>';

  try {
    const d = await postForm('/api/vocal/match', {
      file,
      bpm: $('#vocal-bpm').value || null,
      key: $('#vocal-key').value.trim() || null,
      recorded_over: recordedOver,
      n: '6',
    });
    update('vocal', d.vocal);
    update('vocalPath', d.path);
    update('matches', d.match);
    chosen = new Set((d.match.matches || []).slice(0, 1).map(m => m.beat_id));
    status.innerHTML = '';
    paintReadout(d.vocal);
    paintMatches(d.match);
    reviewTake(file);
  } catch (err) {
    status.innerHTML = note(err.message, 'bad');
    $('#match-panel').innerHTML = '<p class="placeholder">No matches.</p>';
  }
}

function fillRecordedOver(catalog) {
  const sel = $('#vocal-over');
  if (!sel) return;
  const cur = sel.value;
  sel.innerHTML = '<option value="">nothing / headphones</option>' +
    catalog.map(b => `<option value="${esc(b.beat_id)}">${esc(b.title || b.beat_id)}</option>`).join('');
  if (cur) sel.value = cur;
}

async function adoptTake(take) {
  window.dispatchEvent(new CustomEvent('mixengine:view', { detail: 'vocal' }));
  // A take from the recorder knows which beat was playing. That is the
  // reference the de-bleed and the tempo stage need, so it travels with it.
  if (take.beat_id) {
    recordedOver = take.beat_id;
    const sel = $('#vocal-over');
    if (sel) sel.value = take.beat_id;
  }
  const status = $('#vocal-status');
  status.innerHTML =
    `<p class="quiet"><span class="spin"></span> Analysing ${esc(take.name)}…</p>`;
  try {
    const blob = await fetch(take.url).then(r => r.blob());
    await analyze(new File([blob], take.name, { type: 'audio/wav' }));
  } catch (err) {
    status.innerHTML = note(err.message, 'bad');
  }
}

const LANGUAGE_NAMES = { en: 'English', hi: 'Hindi', pa: 'Punjabi' };

/**
 * What the words were read as, and how it was decided.
 *
 * Said by the person, detected, or defaulted: the three are not equally
 * trustworthy and the engine's own guess is shown as one.
 */
function languageLabel(lyrics) {
  if (!lyrics || !lyrics.n_words) return DASH;
  const name = LANGUAGE_NAMES[lyrics.language] || lyrics.language || DASH;
  if (lyrics.language_source === 'declared') return name;
  if (lyrics.language_source === 'default') return `${name} (a guess)`;
  return `${name} (detected)`;
}

function paintReadout(v) {
  const q = v.quality || {};
  const warnings = v.warnings || [];
  const bleed = v.debleed;
  const tone = (val, good, bad) =>
    val == null ? '' : val >= good ? 'good' : val <= bad ? 'bad' : 'warn';

  $('#vocal-readout').innerHTML = `
    <section class="panel">
      <h2>Take analysis</h2>
      <p class="summary">${esc(v.summary || '')}</p>
      <dl class="facts">
        <div><dt>Key</dt><dd>${esc(v.key?.name || DASH)}
          <span class="muted mono">${num(v.key_confidence, 2)}</span></dd></div>
        <div><dt>Tempo</dt><dd class="mono">${v.bpm ? num(v.bpm, 1, ' BPM') : 'no stable tempo'}</dd></div>
        <div><dt>Tempo source</dt><dd>${esc(v.bpm_source || DASH)}</dd></div>
        <div><dt>Delivery</dt><dd>${esc((v.performance_type || DASH).replace(/_/g, ' '))}</dd></div>
        <div><dt>Language</dt><dd>${esc(languageLabel(v.lyrics))}</dd></div>
        <div><dt>Voice</dt><dd>${esc((v.voice_type || DASH).replace(/_/g, ' '))}</dd></div>
        <div><dt>Range</dt><dd class="mono">${esc(v.range_low_note || '?')}–${esc(v.range_high_note || '?')}</dd></div>
        <div><dt>Notes</dt><dd class="mono">${esc(v.n_notes ?? DASH)}</dd></div>
        <div><dt>Phrases</dt><dd class="mono">${esc(v.n_phrases ?? DASH)}</dd></div>
        <div><dt>Input</dt><dd>${esc((v.input_type || DASH).replace(/_/g, ' '))}</dd></div>
      </dl>
    </section>

    ${bleed ? `
    <section class="panel">
      <h2>Beat bleed</h2>
      ${bleed.applied ? `
        <dl class="facts">
          <div><dt>Removed</dt><dd class="mono">${num(bleed.cancellation_db, 1, ' dB')}</dd></div>
          <div><dt>Offset found</dt><dd class="mono">${num(bleed.offset_s, 2, ' s')}</dd></div>
          <div><dt>Clock drift</dt><dd>${bleed.drift_corrected ? 'corrected' : 'none'}</dd></div>
        </dl>`
      : `<p class="quiet">${esc(bleed.note || 'Not applied.')}</p>`}
    </section>` : ''}

    <section class="panel">
      <h2>Recording quality</h2>
      ${meterRow('Signal-to-noise', q.snr_db ?? 0, 0, 60, num(q.snr_db, 0, ' dB'),
                 tone(q.snr_db, 24, 12))}
      ${meterRow('Room tail', q.estimated_rt60_s ?? 0, 0, 1, num(q.estimated_rt60_s, 2, ' s'),
                 (q.estimated_rt60_s ?? 0) < 0.3 ? 'good' : (q.estimated_rt60_s ?? 0) > 0.6 ? 'bad' : '')}
      ${meterRow('Bandwidth', q.bandwidth_hz ?? 0, 6000, 22050,
                 num((q.bandwidth_hz || 0) / 1000, 1, ' kHz'),
                 (q.bandwidth_hz ?? 0) > 15000 ? 'good' : '')}
      <dl class="facts">
        <div><dt>Peak</dt><dd class="mono">${num(q.peak_db, 1, ' dBFS')}</dd></div>
        <div><dt>Clipping</dt><dd class="mono ${(q.clipping_pct ?? 0) > 0.5 ? 'bad' : ''}">${num(q.clipping_pct, 2, '%')}</dd></div>
        <div><dt>Noise floor</dt><dd class="mono">${num(q.noise_floor_db, 1, ' dB')}</dd></div>
      </dl>
      ${warnings.length ? warnings.map(w => note(w, 'warn')).join('') : ''}
    </section>`;
}

async function reviewTake(file) {
  try {
    const r = await postForm('/api/take/review', {
      file,
      beat_id: store.selectedBeat || null,
    });
    const cues = r.cues || [];
    $('#take-report').innerHTML = `
      <section class="panel">
        <h2>Take report <span class="pill ${r.usable ? 'good' : 'bad'}">${esc(r.grade)}</span></h2>
        <dl class="facts">
          <div><dt>Intonation</dt><dd class="mono">${num(r.median_pitch_error_cents, 0, ' ¢')}</dd></div>
          <div><dt>Level spread</dt><dd class="mono">${num(r.level_spread_db, 1, ' dB')}</dd></div>
          <div><dt>Plosives</dt><dd class="mono">${esc(r.plosive_count ?? DASH)}</dd></div>
          <div><dt>Voiced time</dt><dd class="mono">${num(r.voiced_s, 1, ' s')}</dd></div>
        </dl>
        ${cues.map(c => `
          <div class="cue cue-${esc(c.severity)}">
            <p class="cue-msg">${esc(c.message)}</p>
            ${c.detail ? `<p class="cue-why">${esc(c.detail)}</p>` : ''}
          </div>`).join('') || '<p class="quiet">Nothing worth flagging.</p>'}
        <p class="quiet">Detail is shown after the take, not during it.
           Concurrent feedback measurably degrades the performance it is
           supervising, so the engine stays quiet while you sing.</p>
      </section>`;
  } catch (_) {
    // Supplementary. A failed report must never block the match.
  }
}

/* ── matches ─────────────────────────────────────────────────────────────── */

function paintMatches(report) {
  const panel = $('#match-panel');
  const matches = report?.matches || [];
  if (!matches.length) {
    panel.innerHTML =
      `<p class="placeholder">${esc(report?.message || 'No compatible beats found.')}</p>`;
    return;
  }
  const relaxed = (report.relaxation_level || 0) >= 4;

  panel.innerHTML = `
    <section class="panel">
      <h2>Matches <span class="muted mono">${matches.length} of ${esc(report.catalog_size)}</span></h2>
      <p class="quiet">${esc(report.message || '')}</p>
      ${relaxed ? note(`Filters were relaxed to find these (${report.relaxation_reason || ''}).
        They are the best available rather than a close fit.`, 'warn') : ''}
    </section>
    ${matches.map(m => `
      <article class="match${chosen.has(m.beat_id) ? ' is-on' : ''}" data-id="${esc(m.beat_id)}">
        <header>
          <span class="match-name">${esc(m.title || m.beat_id)}</span>
          <span class="match-score mono">${esc(m.score_pct)}%</span>
        </header>
        <p class="match-xf mono">${esc(m.transform_summary || 'no transformation')}</p>
        <ul>
          ${(m.reasons || []).slice(0, 3).map(r => `<li>${esc(r)}</li>`).join('')}
          ${(m.warnings || []).slice(0, 2).map(w => `<li class="warn">${esc(w)}</li>`).join('')}
        </ul>
      </article>`).join('')}
    <div class="actions">
      <button class="btn primary" id="render-btn">Render selected</button>
      <button class="btn" id="render-all">All five variants</button>
    </div>`;

  $$('.match', panel).forEach(el => on(el, 'click', () => {
    const id = el.dataset.id;
    chosen.has(id) ? chosen.delete(id) : chosen.add(id);
    el.classList.toggle('is-on', chosen.has(id));
  }));
  on($('#render-btn'), 'click', () => startRender(1));
  on($('#render-all'), 'click', () => startRender(5));
}

async function startRender(variants) {
  if (!store.vocalPath) return;
  const host = $('#vocal-status');
  window.dispatchEvent(new CustomEvent('mixengine:view', { detail: 'renders' }));
  const list = $('#render-list');
  list.innerHTML = '<p class="placeholder"><span class="spin"></span> Starting…</p>';

  try {
    const job = await api('/api/render', {
      method: 'POST',
      body: new URLSearchParams({
        vocal_path: store.vocalPath,
        beat_ids: [...chosen].join(','),
        variants: String(variants),
        bpm: $('#vocal-bpm').value || '',
        key: $('#vocal-key').value.trim() || '',
      }),
    });
    const result = await pollJob(job.id, {
      onProgress: j => {
        list.innerHTML = `
          <div class="progress">
            <div class="progress-bar"><span style="width:${((j.progress || 0) * 100).toFixed(0)}%"></span></div>
            <span class="quiet">${esc(j.stage || j.status)} · ${esc(j.elapsed_s)}s</span>
          </div>`;
      },
    });
    window.dispatchEvent(new CustomEvent('mixengine:renders', { detail: result }));
  } catch (err) {
    list.innerHTML = note(err.message, 'bad');
  }
}
