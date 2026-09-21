/**
 * Application state, and a subscription mechanism for it.
 *
 * Small on purpose. The alternative that suggests itself — every view owning
 * its own copy of the catalog, the current take, the selected beat — is how
 * the record view ends up offering a beat the catalog has already replaced.
 * One store, one update path, views re-render from it.
 *
 * Subscriptions are keyed by top-level slice so that a metre updating sixty
 * times a second does not re-render the catalog table.
 */

const listeners = new Map();

export const store = {
  capabilities: null,
  catalog: [],
  selectedBeat: null,
  vocal: null,
  vocalPath: null,
  matches: null,
  renders: [],
  takes: [],
  activeTake: null,
  latency: null,
  view: 'catalog',
  jobs: new Map(),
};

export function subscribe(slice, fn) {
  if (!listeners.has(slice)) listeners.set(slice, new Set());
  listeners.get(slice).add(fn);
  return () => listeners.get(slice)?.delete(fn);
}

export function update(slice, value) {
  store[slice] = typeof value === 'function' ? value(store[slice]) : value;
  notify(slice);
}

export function notify(slice) {
  const set = listeners.get(slice);
  if (!set) return;
  for (const fn of set) {
    // One throwing subscriber must not stop the others; a render error in a
    // side panel should not take the transport down with it.
    try { fn(store[slice], store); }
    catch (err) { console.error(`subscriber for "${slice}" failed`, err); }
  }
}

/** Persisted preferences: device choice, monitoring, measured latency. */
const PREF_KEY = 'mixengine.prefs.v1';

export const prefs = {
  read() {
    try { return JSON.parse(localStorage.getItem(PREF_KEY)) || {}; }
    catch (_) { return {}; }
  },
  get(key, fallback = null) {
    const v = this.read()[key];
    return v === undefined ? fallback : v;
  },
  set(key, value) {
    try {
      const all = this.read();
      all[key] = value;
      localStorage.setItem(PREF_KEY, JSON.stringify(all));
    } catch (_) {
      // Private browsing, a full quota, a locked-down profile. A preference
      // that cannot be saved is not worth interrupting anyone over.
    }
  },
};
