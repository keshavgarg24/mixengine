/**
 * Rendered results, and the transport that auditions them.
 *
 * Each render is shown with the critic's gates and the decisions that
 * produced it — what the tuner corrected and what it left alone, how many
 * onsets moved, which layers the arranger built. A score on its own is not
 * actionable; the numbers behind it are.
 *
 * The waveform is loaded and drawn lazily, only for the render being
 * auditioned. Decoding five three-minute masters to draw five pictures
 * nobody has looked at yet is a second of blocked main thread per master.
 */

import { $, $$, clock, DASH, delegate, esc, note, num, on, show } from '../core/dom.js';
import { store, update } from '../core/store.js';
import { decodeToMono } from '../audio/wav.js';
import { Waveform } from './waveform.js';

let wave = null;
let audioEl = null;
let decodeCtx = null;
let current = null;

export function init() {
  audioEl = $('#player');
  wave = new Waveform($('#render-wave'), {
    onSeek: t => { if (audioEl.duration) audioEl.currentTime = t; },
  });

  on($('#t-play'), 'click', () => {
    audioEl.paused ? audioEl.play().catch(() => {}) : audioEl.pause();
  });
  on(audioEl, 'play', () => { $('#t-play').textContent = '❚❚'; });
  on(audioEl, 'pause', () => { $('#t-play').textContent = '▶'; });
  on(audioEl, 'timeupdate', () => {
    const pct = audioEl.duration ? (audioEl.currentTime / audioEl.duration) * 100 : 0;
    $('#t-fill').style.width = pct.toFixed(2) + '%';
    $('#t-time').textContent = clock(audioEl.currentTime);
    wave.setPlayhead(audioEl.currentTime);
  });
  on($('#t-bar'), 'click', ev => {
    if (!audioEl.duration) return;
    const r = ev.currentTarget.getBoundingClientRect();
    audioEl.currentTime = ((ev.clientX - r.left) / r.width) * audioEl.duration;
  });

  delegate($('#render-list'), 'click', '[data-render]', (_e, el) => {
    play(store.renders[Number(el.dataset.render)]);
  });

  window.addEventListener('mixengine:renders', ev => paint(ev.detail));
}

export function paint(result) {
  const renders = result?.renders || [];
  update('renders', renders);
  const host = $('#render-list');

  if (!renders.length) {
    const failures = result?.failures || [];
    host.innerHTML = `<p class="placeholder">No renders were produced.</p>` +
      failures.map(f => note(`${f.beat_id || ''} ${f.stage || ''}: ${f.error || ''}`, 'bad')).join('');
    return;
  }

  host.innerHTML = renders.map((r, i) => renderCard(r, i)).join('');
  if (renders[0]) play(renders[0], false);
}

function renderCard(r, i) {
  const c = r.critic || {};
  const m = r.master || {};
  const t = r.transform || {};
  const tuning = t.tuning || {};
  const align = t.align || t.quantize || {};
  const arrangement = t.arrangement || {};
  const layers = (t.layers?.built || []).map(l => l.name);
  const structure = arrangement.structure || {};

  return `
    <article class="render">
      <header>
        <button class="btn icon" data-render="${i}" aria-label="Audition">▶</button>
        <span class="render-name">${esc(r.beat_title || r.beat_id)}</span>
        <span class="pill">${esc(r.variant)}</span>
        <span class="render-score mono ${c.passed ? 'good' : 'bad'}">${esc(r.score_pct)}%</span>
      </header>

      <div class="render-grid">
        <section>
          <h3>Master</h3>
          <dl class="facts compact">
            <div><dt>Loudness</dt><dd class="mono">${num(m.output_lufs, 2, ' LUFS')}</dd></div>
            <div><dt>Target</dt><dd class="mono">${num(m.target_lufs, 1, ' LUFS')}</dd></div>
            <div><dt>True peak</dt><dd class="mono ${(m.output_true_peak_db ?? -1) <= -0.9 ? 'good' : 'bad'}">${num(m.output_true_peak_db, 2, ' dBTP')}</dd></div>
            <div><dt>Dynamic range</dt><dd class="mono">${num(m.dynamic_range_db, 1, ' dB')}</dd></div>
            <div><dt>Mono loss</dt><dd class="mono">${num(m.mono_loss_db, 2, ' dB')}</dd></div>
            <div><dt>Limiting</dt><dd class="mono">${num(m.gain_into_limiter_db, 1, ' dB')}</dd></div>
          </dl>
          ${m.loudness_converged === false
            ? note(m.loudness_note || 'Loudness target not reached.', 'warn') : ''}
        </section>

        <section>
          <h3>Decisions</h3>
          <dl class="facts compact">
            <div><dt>Tuned</dt><dd class="mono">${esc(tuning.notes_corrected ?? 0)}/${esc(tuning.notes_considered ?? 0)}</dd></div>
            <div><dt>Gestures kept</dt><dd class="mono">${esc((tuning.notes_skipped_transition ?? 0) + (tuning.notes_skipped_melisma ?? 0))}</dd></div>
            <div><dt>Chord-aware</dt><dd class="mono">${num((tuning.chord_aware_fraction ?? 0) * 100, 0, '%')}</dd></div>
            <div><dt>Onsets moved</dt><dd class="mono">${esc(align.moved ?? align.onsets_moved ?? 0)}/${esc(align.onsets ?? align.onsets_considered ?? 0)}</dd></div>
            <div><dt>Grid error</dt><dd class="mono">${align.error_before_ms != null
              ? `${num(align.error_before_ms, 0)} → ${num(align.error_after_ms, 0)} ms` : DASH}</dd></div>
            <div><dt>Drift removed</dt><dd class="mono">${align.drift_corrected
              ? num(align.drift_removed_ms, 0, ' ms') : 'none'}</dd></div>
          </dl>
        </section>

        <section>
          <h3>Arrangement</h3>
          ${structure.labels?.length ? `
            <div class="chips">
              ${structure.labels.map((l, n) =>
                `<span class="chip${l === 'hook' ? ' amber' : ''}">${esc(l)}</span>`).join('')}
            </div>
            <dl class="facts compact">
              <div><dt>Energy contrast</dt><dd class="mono">${num(arrangement.contour?.contrast, 2)}</dd></div>
              <div><dt>Layers</dt><dd>${layers.length ? esc(layers.join(', ')) : 'none'}</dd></div>
            </dl>`
          : `<p class="quiet">${esc(arrangement.note || structure.note || 'Flat arrangement.')}</p>`}
        </section>
      </div>

      <div class="chips gates">${(c.gates || []).map(gateChip).join('')}</div>
      ${(r.warnings || []).map(w => note(w, 'warn')).join('')}
    </article>`;
}

function gateChip(g) {
  const cls = g.severity === 'info' ? 'info'
    : g.passed ? 'good' : g.severity === 'error' ? 'bad' : 'warn';
  const val = g.value == null ? '' : ` ${num(g.value, 1)}`;
  return `<span class="chip ${cls}" title="${esc(g.message || '')}">${esc(g.name)}${esc(val)}</span>`;
}

async function play(render, autoplay = true) {
  if (!render?.download) return;
  current = render;
  show($('#transport'), true);
  $('#t-title').textContent = render.beat_title || render.beat_id;
  $('#t-sub').textContent =
    `${render.variant} · ${render.score_pct}% · ${num(render.master?.output_lufs, 1)} LUFS`;
  $('#t-dl').href = render.download;
  audioEl.src = render.download;
  if (autoplay) audioEl.play().catch(() => {});

  show($('#render-wave-panel'), true);
  try {
    decodeCtx ||= new (window.AudioContext || window.webkitAudioContext)();
    const buf = await fetch(render.download).then(r => r.arrayBuffer());
    const { samples, sampleRate } = await decodeToMono(buf, decodeCtx);
    if (current !== render) return;      // a newer audition won the race
    wave.setAudio(samples, sampleRate);
    const sections = render.transform?.arrangement?.sections || [];
    wave.setRegions(sections.map(s => ({
      start: s.start, end: s.end, label: s.label,
      accent: s.label === 'hook' || s.label === 'chorus',
    })));
  } catch (_) {
    // A waveform is a nicety; a failed decode must not stop playback.
  }
}
