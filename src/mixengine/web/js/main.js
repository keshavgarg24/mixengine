/**
 * Entry point: capabilities, navigation, and wiring the views together.
 *
 * Native ES modules with no build step. A local tool that needs `npm
 * install` before it will show a waveform has added a dependency on the
 * whole Node ecosystem to avoid typing `<script type="module">`, and every
 * browser that can run an AudioWorklet can run modules.
 */

import { api } from './core/api.js';
import { $, $$, esc, on, show } from './core/dom.js';
import { store, subscribe, update } from './core/store.js';
import * as catalog from './ui/catalog.js';
import * as record from './ui/record.js';
import * as renders from './ui/renders.js';
import * as vocal from './ui/vocal.js';

function showView(name) {
  update('view', name);
  $$('.view').forEach(v => v.classList.toggle('is-on', v.dataset.view === name));
  $$('.tab').forEach(t => {
    const on_ = t.dataset.view === name;
    t.classList.toggle('is-on', on_);
    t.setAttribute('aria-selected', String(on_));
  });
  location.hash = name;
}

async function loadCapabilities() {
  const dot = $('#tier .dot');
  const label = $('#tier-label');
  try {
    const c = await api('/api/capabilities');
    update('capabilities', c);
    label.textContent = c.missing.length
      ? `${c.tier} · ${c.missing.length} optional missing`
      : `${c.tier} · complete`;
    dot.className = `dot ${c.tier === 'full' ? 'good' : c.can_render ? 'warn' : 'bad'}`;
    $('#tier').title = c.missing.length
      ? 'Missing: ' + c.missing.map(m => `${m.name} (${m.cost})`).join('; ')
      : 'Every optional component is installed.';
    if (!c.can_render) {
      $('#main').prepend(banner(
        'The engine cannot render: librosa and soundfile are required. ' +
        'Run `pip install -e ".[quality]"`.', 'bad'));
    }
  } catch (err) {
    label.textContent = 'offline';
    dot.className = 'dot bad';
    $('#main').prepend(banner(err.message, 'bad'));
  }
}

function banner(message, tone) {
  const el = document.createElement('div');
  el.className = `banner ${tone}`;
  el.textContent = message;
  return el;
}

function boot() {
  $$('.tab').forEach(t => on(t, 'click', () => showView(t.dataset.view)));
  window.addEventListener('mixengine:view', ev => showView(ev.detail));
  window.addEventListener('hashchange', () => {
    const name = location.hash.slice(1);
    if (name && $(`.view[data-view="${name}"]`)) showView(name);
  });

  catalog.init();
  record.init();
  vocal.init();
  renders.init();

  subscribe('catalog', c => catalog.paintList(c));

  loadCapabilities();
  catalog.load();

  const initial = location.hash.slice(1);
  showView(initial && $(`.view[data-view="${initial}"]`) ? initial : 'catalog');

  // A module-level throw leaves a blank page with the reason only in the
  // console. Surfacing it keeps a broken build debuggable by whoever hits it.
  window.addEventListener('error', ev => {
    console.error(ev.error || ev.message);
  });
  window.addEventListener('unhandledrejection', ev => {
    console.error('unhandled rejection:', ev.reason);
  });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
