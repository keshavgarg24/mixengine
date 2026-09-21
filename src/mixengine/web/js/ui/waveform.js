/**
 * Waveform display.
 *
 * Canvas rather than SVG: at any useful width this is a few thousand
 * vertical lines, and a few thousand DOM nodes that must be laid out and
 * composited is the difference between a display that scrubs smoothly and
 * one that stutters.
 *
 * Three things are drawn that a waveform usually is not, because each
 * answers a question the singer is actually asking:
 *
 *   **The target level band.** A horizontal zone at -6 dB peak. Waveforms
 *   are normally drawn full-height whatever the level, so a take recorded
 *   20 dB too quiet looks identical to one recorded correctly. Drawing
 *   against a fixed scale with the target marked makes the gain setting
 *   visible without reading a number.
 *
 *   **Clipped samples, in red, at their own column.** Because peaks are
 *   kept as min/max per column rather than decimated, a single clipped
 *   sample survives to any zoom level instead of disappearing between
 *   pixels.
 *
 *   **The beat's bar lines.** A vocal's relationship to the grid is the
 *   whole problem the engine solves, and seeing where the bars fall
 *   underneath the phrases is how a person checks it.
 */

import { cssVar, fitCanvas } from '../core/dom.js';
import { peaks } from '../audio/wav.js';

const TARGET_PEAK_DB = -6;

export class Waveform {
  constructor(canvas, { onSeek } = {}) {
    this.canvas = canvas;
    this.onSeek = onSeek;
    this.samples = null;
    this.sampleRate = 48000;
    this.peaks = null;
    this.peakWidth = 0;
    this.playhead = null;
    this.downbeats = [];
    this.regions = [];
    this._raf = 0;

    this._ro = new ResizeObserver(() => this.invalidate());
    this._ro.observe(canvas);

    canvas.addEventListener('pointerdown', ev => {
      if (!this.samples || !this.onSeek) return;
      const rect = canvas.getBoundingClientRect();
      const frac = (ev.clientX - rect.left) / rect.width;
      this.onSeek(Math.max(0, Math.min(1, frac)) * this.duration);
    });
  }

  get duration() {
    return this.samples ? this.samples.length / this.sampleRate : 0;
  }

  setAudio(samples, sampleRate) {
    this.samples = samples;
    this.sampleRate = sampleRate;
    this.peaks = null;
    this.invalidate();
  }

  setDownbeats(times) {
    this.downbeats = times || [];
    this.invalidate();
  }

  /** Labelled spans drawn behind the waveform — sections, phrases. */
  setRegions(regions) {
    this.regions = regions || [];
    this.invalidate();
  }

  setPlayhead(seconds) {
    this.playhead = seconds;
    this.invalidate();
  }

  invalidate() {
    if (this._raf) return;
    this._raf = requestAnimationFrame(() => { this._raf = 0; this.draw(); });
  }

  destroy() {
    this._ro.disconnect();
    cancelAnimationFrame(this._raf);
  }

  draw() {
    const { ctx, width, height } = fitCanvas(this.canvas);
    const mid = height / 2;
    ctx.clearRect(0, 0, width, height);

    const line = cssVar('--line', '#2b2d33');
    const ink3 = cssVar('--ink-3', '#6b6e77');
    const wave = cssVar('--wave', '#7f8894');
    const amber = cssVar('--amber', '#d9a441');
    const red = cssVar('--red', '#d2594f');
    const green = cssVar('--green', '#6fbf73');

    // ── target band ───────────────────────────────────────────────────────
    const targetLin = Math.pow(10, TARGET_PEAK_DB / 20);
    const th = targetLin * mid;
    ctx.fillStyle = green;
    ctx.globalAlpha = 0.07;
    ctx.fillRect(0, mid - th, width, th * 2);
    ctx.globalAlpha = 1;

    // ── section regions ───────────────────────────────────────────────────
    if (this.regions.length && this.duration > 0) {
      ctx.font = '10px ui-monospace, monospace';
      for (const r of this.regions) {
        const x0 = (r.start / this.duration) * width;
        const x1 = (r.end / this.duration) * width;
        ctx.fillStyle = r.accent ? amber : ink3;
        ctx.globalAlpha = r.accent ? 0.10 : 0.05;
        ctx.fillRect(x0, 0, Math.max(1, x1 - x0), height);
        ctx.globalAlpha = 0.85;
        ctx.fillStyle = r.accent ? amber : ink3;
        if (x1 - x0 > 34 && r.label) ctx.fillText(r.label, x0 + 4, 12);
        ctx.globalAlpha = 1;
      }
    }

    // ── bar lines ─────────────────────────────────────────────────────────
    if (this.downbeats.length && this.duration > 0) {
      ctx.strokeStyle = line;
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (const t of this.downbeats) {
        if (t > this.duration) break;
        const x = Math.round((t / this.duration) * width) + 0.5;
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
      }
      ctx.stroke();
    }

    // ── centre line ───────────────────────────────────────────────────────
    ctx.strokeStyle = line;
    ctx.beginPath();
    ctx.moveTo(0, Math.round(mid) + 0.5);
    ctx.lineTo(width, Math.round(mid) + 0.5);
    ctx.stroke();

    if (!this.samples || !this.samples.length) {
      ctx.fillStyle = ink3;
      ctx.font = '11px system-ui, sans-serif';
      ctx.fillText('No audio', 8, mid - 6);
      return;
    }

    // Peaks are recomputed only when the width changes, not on every
    // playhead move — scrubbing a three-minute take otherwise re-reduces
    // eight million samples sixty times a second.
    if (!this.peaks || this.peakWidth !== width) {
      this.peaks = peaks(this.samples, width);
      this.peakWidth = width;
    }

    ctx.strokeStyle = wave;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let x = 0; x < width; x++) {
      const lo = this.peaks[x * 2];
      const hi = this.peaks[x * 2 + 1];
      const y0 = mid - hi * mid;
      const y1 = mid - lo * mid;
      ctx.moveTo(x + 0.5, y0);
      ctx.lineTo(x + 0.5, Math.max(y1, y0 + 0.5));
    }
    ctx.stroke();

    // Clipped columns, drawn over the top so they are never hidden.
    ctx.strokeStyle = red;
    ctx.beginPath();
    let clipped = 0;
    for (let x = 0; x < width; x++) {
      if (this.peaks[x * 2 + 1] >= 0.999 || this.peaks[x * 2] <= -0.999) {
        clipped++;
        ctx.moveTo(x + 0.5, 0);
        ctx.lineTo(x + 0.5, height);
      }
    }
    if (clipped) ctx.stroke();

    // ── playhead ──────────────────────────────────────────────────────────
    if (this.playhead != null && this.duration > 0) {
      const x = Math.round((this.playhead / this.duration) * width) + 0.5;
      ctx.strokeStyle = amber;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
  }
}

/**
 * A live scrolling level history, drawn while recording.
 *
 * Deliberately not a waveform: at 60 fps a real waveform of the last ten
 * seconds is unreadable and invites staring, which is the behaviour the
 * coaching design is built to avoid. This is a coarse envelope with the
 * target band marked — enough to see at a glance that the level is right,
 * not enough to study.
 */
export class LiveMeterTrace {
  constructor(canvas, { seconds = 12, fps = 30 } = {}) {
    this.canvas = canvas;
    this.size = Math.round(seconds * fps);
    this.buf = new Float32Array(this.size).fill(-120);
    this.pos = 0;
    this._raf = 0;
    this._ro = new ResizeObserver(() => this.invalidate());
    this._ro.observe(canvas);
  }

  push(db) {
    this.buf[this.pos] = db;
    this.pos = (this.pos + 1) % this.size;
    this.invalidate();
  }

  reset() {
    this.buf.fill(-120);
    this.pos = 0;
    this.invalidate();
  }

  invalidate() {
    if (this._raf) return;
    this._raf = requestAnimationFrame(() => { this._raf = 0; this.draw(); });
  }

  destroy() { this._ro.disconnect(); cancelAnimationFrame(this._raf); }

  draw() {
    const { ctx, width, height } = fitCanvas(this.canvas);
    ctx.clearRect(0, 0, width, height);

    const toY = db => height - Math.max(0, Math.min(1, (db + 60) / 60)) * height;

    ctx.fillStyle = cssVar('--green', '#6fbf73');
    ctx.globalAlpha = 0.08;
    ctx.fillRect(0, toY(-6), width, toY(-24) - toY(-6));
    ctx.globalAlpha = 1;

    ctx.strokeStyle = cssVar('--line', '#2b2d33');
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (const db of [-6, -18, -38]) {
      const y = Math.round(toY(db)) + 0.5;
      ctx.moveTo(0, y); ctx.lineTo(width, y);
    }
    ctx.stroke();

    ctx.strokeStyle = cssVar('--amber', '#d9a441');
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < this.size; i++) {
      const idx = (this.pos + i) % this.size;
      const x = (i / (this.size - 1)) * width;
      const y = toY(this.buf[idx]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
}
