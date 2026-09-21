/**
 * DOM helpers and the formatting discipline.
 *
 * Two rules the rest of the interface depends on.
 *
 * **Everything interpolated into markup goes through `esc`.** Beat titles,
 * filenames and error strings all originate outside this code. There is no
 * template engine here to do it automatically, so it is done by hand, every
 * time, without exception.
 *
 * **Unknown renders as an em dash, never as zero.** The engine is careful to
 * distinguish "measured as zero" from "could not measure" — a take with no
 * silence in it reports an unmeasurable noise floor rather than a wrong one —
 * and an interface that prints 0.0 for both throws that away at the last
 * step. `num` returns the dash for null, undefined and NaN.
 */

export const DASH = '—';

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const ESCAPES = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };

export function esc(value) {
  return String(value == null ? '' : value).replace(/[&<>"']/g, c => ESCAPES[c]);
}

export function num(value, digits = 1, suffix = '') {
  if (value == null || Number.isNaN(value)) return DASH;
  const n = Number(value);
  if (!Number.isFinite(n)) return DASH;
  return n.toFixed(digits) + suffix;
}

export function clock(seconds) {
  if (seconds == null || !Number.isFinite(Number(seconds))) return DASH;
  const s = Math.max(0, Number(seconds));
  const m = Math.floor(s / 60);
  const r = Math.floor(s % 60);
  return `${m}:${String(r).padStart(2, '0')}`;
}

export function clockMs(seconds) {
  if (seconds == null || !Number.isFinite(Number(seconds))) return DASH;
  const s = Math.max(0, Number(seconds));
  const m = Math.floor(s / 60);
  const r = s % 60;
  return `${m}:${r.toFixed(1).padStart(4, '0')}`;
}

/** Build an element from markup. Callers are responsible for escaping. */
export function html(markup) {
  const t = document.createElement('template');
  t.innerHTML = markup.trim();
  return t.content.firstElementChild;
}

export function setHTML(node, markup) {
  if (node) node.innerHTML = markup;
  return node;
}

export function on(node, type, handler, opts) {
  node?.addEventListener(type, handler, opts);
  return () => node?.removeEventListener(type, handler, opts);
}

/** Delegate an event to descendants matching `selector`. */
export function delegate(root, type, selector, handler) {
  return on(root, type, ev => {
    const target = ev.target.closest(selector);
    if (target && root.contains(target)) handler(ev, target);
  });
}

export function show(node, visible = true) {
  if (node) node.hidden = !visible;
}

/**
 * A short status line. Tone drives the accent, not the wording — the wording
 * should stand on its own if the colour is not perceivable.
 */
export function note(message, tone = '') {
  return message ? `<div class="note ${esc(tone)}">${esc(message)}</div>` : '';
}

export function pill(text, tone = '') {
  return `<span class="pill ${esc(tone)}">${esc(text)}</span>`;
}

/** Format a value against a range as a horizontal bar, for readouts. */
export function meterRow(label, value, min, max, valueText, tone = '') {
  const pct = value == null || !Number.isFinite(value)
    ? 0
    : Math.max(0, Math.min(1, (value - min) / (max - min))) * 100;
  return `
    <div class="meter-row">
      <span class="meter-label">${esc(label)}</span>
      <span class="meter-track"><span class="meter-fill ${esc(tone)}"
            style="width:${pct.toFixed(1)}%"></span></span>
      <span class="meter-value mono">${esc(valueText ?? num(value))}</span>
    </div>`;
}

export function debounce(fn, ms = 150) {
  let t = 0;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

/** Device-pixel-ratio aware canvas sizing. Returns the CSS dimensions. */
export function fitCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(1, Math.floor(rect.width));
  const h = Math.max(1, Math.floor(rect.height));
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
    canvas.width = w * dpr;
    canvas.height = h * dpr;
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width: w, height: h };
}

export function cssVar(name, fallback = '#888') {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name);
  return v ? v.trim() : fallback;
}
