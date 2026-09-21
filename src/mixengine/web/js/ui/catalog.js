/**
 * The beat catalog: import, listing, and the per-beat analysis panel.
 *
 * The detail panel deliberately shows the engine's *uncertainty* alongside
 * its answers — key confidence, whether a groove template was meaningful
 * enough to use, whether a tempo was verified against a producer tag. Those
 * are the numbers that explain why a render came out the way it did, and
 * hiding them leaves only an unexplainable result.
 */

import { api, pollJob, postForm } from '../core/api.js';
import { $, clock, DASH, delegate, esc, meterRow, note, num, on } from '../core/dom.js';
import { store, update } from '../core/store.js';

export function init() {
  wireDrop($('#beat-drop'), $('#beat-input'), files => importBeats(files));
  on($('#beat-browse'), 'click', () => $('#beat-input').click());
  delegate($('#catalog-body'), 'click', 'tr[data-id]', (_e, tr) => {
    update('selectedBeat', tr.dataset.id);
    paintList(store.catalog);
    showDetail(tr.dataset.id);
  });
}

export function paintList(catalog) {
  const body = $('#catalog-body');
  if (!catalog.length) {
    body.innerHTML =
      '<tr class="empty"><td colspan="8">No beats analysed yet — drop some above.</td></tr>';
    return;
  }
  body.innerHTML = catalog.map(b => `
    <tr data-id="${esc(b.beat_id)}"${store.selectedBeat === b.beat_id ? ' class="is-on"' : ''}>
      <td>${esc(b.title || b.beat_id)}</td>
      <td class="num">${num(b.bpm, 1)}</td>
      <td>${keyCell(b)}</td>
      <td class="num">${num(b.pocket_score, 2)}</td>
      <td class="num">${num(b.grid_stability, 2)}</td>
      <td class="num">${num(b.swing_ratio, 2)}</td>
      <td class="num">${b.duration_s ? clock(b.duration_s) : DASH}</td>
      <td class="muted">${esc(b.genre || DASH)}</td>
    </tr>`).join('');
}

function keyCell(b) {
  if (b.is_atonal) return '<span class="muted">drum-only</span>';
  if (!b.key_name) return `<span class="muted">${DASH}</span>`;
  return esc(b.key_name) +
    (b.camelot ? ` <span class="muted">${esc(b.camelot)}</span>` : '');
}

export async function load() {
  try {
    const d = await api('/api/catalog');
    update('catalog', d.beats || []);
  } catch (err) {
    update('catalog', []);
    $('#catalog-body').innerHTML =
      `<tr class="empty"><td colspan="8">${esc(err.message)}</td></tr>`;
  }
}

async function showDetail(id) {
  const panel = $('#beat-detail');
  panel.innerHTML = '<p class="placeholder"><span class="spin"></span> Loading…</p>';
  try {
    const b = await api('/api/catalog/' + encodeURIComponent(id));
    panel.innerHTML = detailMarkup(b);
  } catch (err) {
    panel.innerHTML = note(err.message, 'bad');
  }
}

function detailMarkup(b) {
  const key = b.key || {};
  const groove = b.groove || {};
  const space = b.space || {};
  const genre = b.genre_detection || {};
  const sections = b.sections || [];
  const counts = sections.reduce((m, s) => (m[s.label] = (m[s.label] || 0) + 1, m), {});

  return `
    <section class="panel">
      <h2>${esc(b.title || b.beat_id)}</h2>
      <dl class="facts">
        <div><dt>Tempo</dt><dd class="mono">${num(b.bpm, 2, ' BPM')}</dd></div>
        <div><dt>Tempo source</dt><dd>${esc(b.bpm_source || DASH)}</dd></div>
        <div><dt>Key</dt><dd>${b.is_atonal ? 'atonal' : esc(key.name || DASH)}</dd></div>
        <div><dt>Key confidence</dt><dd class="mono">${num(b.key_confidence, 2)}</dd></div>
        <div><dt>Camelot</dt><dd class="mono">${esc(b.camelot || DASH)}</dd></div>
        <div><dt>Length</dt><dd class="mono">${clock(b.duration_s)} · ${esc(b.duration_bars ?? DASH)} bars</dd></div>
        <div><dt>Stems</dt><dd>${b.has_stems ? 'separated' : 'none'}</dd></div>
      </dl>
    </section>

    <section class="panel">
      <h2>Genre</h2>
      ${genre.genre ? `
        <dl class="facts">
          <div><dt>Detected</dt><dd>${esc(genre.genre)}${genre.usable ? '' : ' <span class="muted">(not used)</span>'}</dd></div>
          <div><dt>Source</dt><dd>${esc(genre.source || DASH)}</dd></div>
          <div><dt>Confidence</dt><dd class="mono">${num(genre.confidence, 2)}</dd></div>
        </dl>
        ${(genre.evidence || []).map(e => `<p class="quiet">· ${esc(e)}</p>`).join('')}
        ${genre.note ? `<p class="quiet">${esc(genre.note)}</p>` : ''}`
      : '<p class="quiet">Not classified.</p>'}
    </section>

    <section class="panel">
      <h2>Groove</h2>
      ${meterRow('Swing ratio', groove.swing_ratio ?? 0.5, 0.4, 0.75, num(groove.swing_ratio, 3))}
      ${meterRow('Pattern consistency', groove.consistency ?? 0, 0, 1,
                 num(groove.consistency, 2), (groove.consistency ?? 0) > 0.6 ? 'good' : '')}
      ${meterRow('Grid stability', b.grid_stability ?? 0, 0, 1,
                 num(b.grid_stability, 2), (b.grid_stability ?? 0) > 0.85 ? 'good' : '')}
      <p class="quiet">${groove.is_meaningful
        ? `Measured over ${esc(groove.n_bars_observed || 0)} bars. Vocals are aligned to these
           positions rather than to an exact grid — that is what "in the pocket" means.`
        : `Not reliably measurable here, so alignment falls back to a straight grid. That is
           the correct fallback: applying an unreliable pattern would be the random-jitter
           mistake.`}</p>
    </section>

    <section class="panel">
      <h2>Space</h2>
      ${space.measured ? `
        <dl class="facts">
          <div><dt>Decay (RT60)</dt><dd class="mono">${num(space.rt60_mean, 2, ' s')}</dd></div>
          <div><dt>Direct / reverb</dt><dd class="mono">${num(space.drr_db, 1, ' dB')}</dd></div>
          <div><dt>Confidence</dt><dd class="mono">${num(space.confidence, 2)}</dd></div>
        </dl>
        <p class="quiet">The vocal is moved toward this space so it sits in the track
           rather than on top of it.</p>`
      : `<p class="quiet">${esc(space.note || 'Not measured.')}</p>`}
    </section>

    <section class="panel">
      <h2>Room for a vocal</h2>
      ${meterRow('Pocket score', b.pocket_score ?? 0, 0, 1, num(b.pocket_score, 2),
                 (b.pocket_score ?? 0) > 0.6 ? 'good' : (b.pocket_score ?? 0) < 0.4 ? 'bad' : '')}
      <dl class="facts">
        <div><dt>Own vocal content</dt>
          <dd class="${b.has_vocal_content ? 'warn' : ''}">${b.has_vocal_content
            ? 'present — will be ducked' : 'none'}</dd></div>
        <div><dt>Loudness</dt><dd class="mono">${num(b.lufs_integrated, 1, ' LUFS')}</dd></div>
        <div><dt>Dynamic range</dt><dd class="mono">${num(b.dynamic_range_db, 1, ' dB')}</dd></div>
      </dl>
    </section>

    <section class="panel">
      <h2>Structure <span class="muted mono">${sections.length}</span></h2>
      <div class="chips">
        ${Object.entries(counts).map(([l, n]) =>
          `<span class="chip">${esc(l)}${n > 1 ? ` ×${n}` : ''}</span>`).join('') ||
          `<span class="chip">${DASH}</span>`}
      </div>
    </section>`;
}

/* ── import ──────────────────────────────────────────────────────────────── */

async function importBeats(files) {
  const host = $('#import-job');
  const fd = new FormData();
  for (const f of files) fd.append('files', f);
  const genre = $('#beat-genre').value;
  if (genre) fd.append('genre', genre);
  if ($('#beat-separate').checked) fd.append('separate', 'true');

  host.innerHTML = `<p class="quiet"><span class="spin"></span> Uploading ${files.length} file${files.length > 1 ? 's' : ''}…</p>`;
  try {
    const job = await api('/api/beats', { method: 'POST', body: fd });
    await pollJob(job.id, {
      onProgress: j => {
        host.innerHTML = `
          <div class="progress">
            <div class="progress-bar"><span style="width:${((j.progress || 0) * 100).toFixed(0)}%"></span></div>
            <span class="quiet">${esc(j.stage || j.status)}</span>
          </div>`;
      },
    });
    host.innerHTML = note('Analysis complete.', 'good');
    await load();
    paintList(store.catalog);
  } catch (err) {
    host.innerHTML = note(err.message, 'bad');
  }
}

export function wireDrop(zone, input, onFiles) {
  on(input, 'change', () => { if (input.files.length) onFiles(input.files); });
  ['dragenter', 'dragover'].forEach(t =>
    on(zone, t, ev => { ev.preventDefault(); zone.classList.add('is-over'); }));
  ['dragleave', 'drop'].forEach(t =>
    on(zone, t, ev => { ev.preventDefault(); zone.classList.remove('is-over'); }));
  on(zone, 'drop', ev => {
    const files = Array.from(ev.dataTransfer?.files || [])
      .filter(f => f.type.startsWith('audio/') || /\.(wav|mp3|flac|m4a|aiff?|ogg)$/i.test(f.name));
    if (files.length) onFiles(files);
  });
}
