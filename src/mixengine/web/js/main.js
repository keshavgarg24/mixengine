/**
 * Entry point.
 *
 * Native ES modules with no build step. A local tool that needs `npm
 * install` before it will show a waveform has taken on the whole Node
 * ecosystem to avoid typing `<script type="module">`, and every browser
 * that can decode audio can run modules.
 */

import { api } from './core/api.js';
import { $ } from './core/dom.js';
import * as dashboard from './ui/dashboard.js';

async function loadCapabilities() {
  const dot = $('#tier .dot');
  const label = $('#tier-label');
  try {
    const c = await api('/api/capabilities');
    label.textContent = c.missing.length ? `${c.tier}` : 'ready';
    dot.className = `dot ${c.tier === 'full' ? 'good' : c.can_render ? 'warn' : 'bad'}`;
    $('#tier').title = c.missing.length
      ? 'Missing: ' + c.missing.map(m => `${m.name} (${m.cost})`).join('; ')
      : 'Every optional component is installed.';
    if (!c.can_render) banner('The engine cannot render: librosa and '
      + 'soundfile are required. Run `pip install -e ".[quality]"`.');
  } catch (err) {
    label.textContent = 'offline';
    dot.className = 'dot bad';
    banner(err.message);
  }
}

function banner(message) {
  const el = document.createElement('div');
  el.className = 'banner bad';
  el.textContent = message;
  $('#main').prepend(el);
}

function boot() {
  dashboard.init();
  loadCapabilities();

  // A module-level throw leaves a blank page with the reason only in the
  // console. Surfacing it keeps a broken build debuggable by whoever hits it.
  window.addEventListener('error', ev => console.error(ev.error || ev.message));
  window.addEventListener('unhandledrejection',
    ev => console.error('unhandled rejection:', ev.reason));
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
