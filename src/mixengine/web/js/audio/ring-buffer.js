/**
 * Lock-free single-producer single-consumer ring buffer over SharedArrayBuffer.
 *
 * The audio thread writes; the main thread reads. Neither ever blocks, and
 * no memory is allocated on the audio thread — both properties are
 * requirements rather than optimisations. An AudioWorklet's `process()` runs
 * on a real-time thread with a hard deadline of one render quantum (2.7 ms at
 * 48 kHz); allocating there invites a garbage collection pause, and taking a
 * lock invites priority inversion against a main thread that is busy laying
 * out a waveform. Either one is a dropout in the middle of a take.
 *
 * Correctness rests on two invariants that make a mutex unnecessary:
 *
 *   - only the producer ever advances `write`, only the consumer advances
 *     `read`, so there is no contended write anywhere;
 *   - the index store is released *after* the samples it covers are written
 *     (`Atomics.store` is sequentially consistent), so a consumer that sees a
 *     new index is guaranteed to see the data behind it.
 *
 * Capacity is one slot short of the allocation. Full and empty are otherwise
 * the same state (`read === write`) and cannot be told apart without a third
 * variable that both threads would have to write.
 *
 * SharedArrayBuffer needs cross-origin isolation. The server sends the COOP
 * and COEP headers for it; where it is unavailable — an insecure origin, an
 * older browser — `createTransport` falls back to copying blocks through
 * `postMessage`, which works and merely costs an allocation per quantum.
 */

const HEADER_SLOTS = 4;       // write, read, sampleRate, channels
const IDX_WRITE = 0;
const IDX_READ = 1;
const IDX_SR = 2;
const IDX_CH = 3;

export function sharedMemoryAvailable() {
  return typeof SharedArrayBuffer !== 'undefined' &&
         typeof Atomics !== 'undefined' &&
         (typeof crossOriginIsolated === 'undefined' || crossOriginIsolated);
}

export class RingBuffer {
  /**
   * @param {SharedArrayBuffer} sab  allocated by `RingBuffer.alloc`
   */
  constructor(sab) {
    this.sab = sab;
    this.header = new Int32Array(sab, 0, HEADER_SLOTS);
    this.data = new Float32Array(sab, HEADER_SLOTS * 4);
    this.capacity = this.data.length;
  }

  static alloc(frames, sampleRate = 48000, channels = 1) {
    const slots = frames * channels + 1;
    const sab = new SharedArrayBuffer(HEADER_SLOTS * 4 + slots * 4);
    const rb = new RingBuffer(sab);
    Atomics.store(rb.header, IDX_SR, sampleRate | 0);
    Atomics.store(rb.header, IDX_CH, channels | 0);
    return rb;
  }

  get sampleRate() { return Atomics.load(this.header, IDX_SR); }
  get channels() { return Atomics.load(this.header, IDX_CH); }

  get available() {
    const w = Atomics.load(this.header, IDX_WRITE);
    const r = Atomics.load(this.header, IDX_READ);
    return w >= r ? w - r : this.capacity - r + w;
  }

  get free() { return this.capacity - this.available - 1; }

  /**
   * Producer side. Returns the number of samples actually written, which is
   * short only when the consumer has fallen behind.
   *
   * Overrun drops the *newest* samples rather than overwriting the oldest.
   * Dropping the oldest would silently rewrite history the consumer is in
   * the middle of reading; dropping the newest loses the same amount of
   * audio but leaves everything already captured intact and contiguous.
   */
  push(channel) {
    const n = channel.length;
    const free = this.free;
    const take = n < free ? n : free;
    if (take <= 0) return 0;

    let w = Atomics.load(this.header, IDX_WRITE);
    const first = Math.min(take, this.capacity - w);
    this.data.set(channel.subarray(0, first), w);
    if (take > first) this.data.set(channel.subarray(first, take), 0);

    w = (w + take) % this.capacity;
    Atomics.store(this.header, IDX_WRITE, w);
    return take;
  }

  /** Consumer side. Returns a newly allocated Float32Array, or null. */
  pull(max = Infinity) {
    const avail = Math.min(this.available, max);
    if (avail <= 0) return null;

    const out = new Float32Array(avail);
    let r = Atomics.load(this.header, IDX_READ);
    const first = Math.min(avail, this.capacity - r);
    out.set(this.data.subarray(r, r + first), 0);
    if (avail > first) out.set(this.data.subarray(0, avail - first), first);

    r = (r + avail) % this.capacity;
    Atomics.store(this.header, IDX_READ, r);
    return out;
  }

  clear() {
    Atomics.store(this.header, IDX_READ, Atomics.load(this.header, IDX_WRITE));
  }
}

/**
 * Metrics shared back from the audio thread.
 *
 * A separate region from the audio, and deliberately not a queue: the meter
 * only ever wants the newest value, so a torn read costs one stale frame on
 * a display refreshing at 60 Hz and nothing else. The sequence counter is
 * there so a reader can tell whether the worklet is still running at all,
 * which is how a dead input is distinguished from a silent one.
 */
export const METRIC = {
  SEQ: 0, PEAK: 1, RMS: 2, CLIPPED: 3, LOW: 4, MID: 5, HIGH: 6,
  TRUE_PEAK: 7, FRAMES: 8, DROPPED: 9, PITCH_HZ: 10, CLARITY: 11,
  LUFS_M: 12, LUFS_S: 13, COUNT: 14,
};

export class MetricBlock {
  constructor(buffer) {
    this.view = new Float32Array(buffer);
  }

  static alloc() {
    const shared = sharedMemoryAvailable();
    const bytes = METRIC.COUNT * 4;
    return new MetricBlock(shared ? new SharedArrayBuffer(bytes)
                                  : new ArrayBuffer(bytes));
  }

  read() {
    const v = this.view;
    return {
      seq: v[METRIC.SEQ],
      peakDb: v[METRIC.PEAK],
      rmsDb: v[METRIC.RMS],
      truePeakDb: v[METRIC.TRUE_PEAK],
      clipped: v[METRIC.CLIPPED],
      low: v[METRIC.LOW],
      mid: v[METRIC.MID],
      high: v[METRIC.HIGH],
      frames: v[METRIC.FRAMES],
      dropped: v[METRIC.DROPPED],
      pitchHz: v[METRIC.PITCH_HZ],
      clarity: v[METRIC.CLARITY],
      lufsMomentary: v[METRIC.LUFS_M],
      lufsShort: v[METRIC.LUFS_S],
    };
  }
}
