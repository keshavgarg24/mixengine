/**
 * WAV encoding, and the peak summary the waveform display draws from.
 *
 * 24-bit rather than 16. The take is the only thing in the pipeline that
 * cannot be regenerated, and a singer who has left headroom — which is what
 * the recorder asks them to do — is using maybe 14 of 16 bits. Twenty-four
 * costs half again in disk for a local file and removes the question.
 */

export function encodeWav(samples, sampleRate, { bitDepth = 24 } = {}) {
  const bytesPerSample = bitDepth / 8;
  const n = samples.length;
  const dataBytes = n * bytesPerSample;
  const buf = new ArrayBuffer(44 + dataBytes);
  const view = new DataView(buf);

  const str = (off, s) => {
    for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i));
  };

  str(0, 'RIFF');
  view.setUint32(4, 36 + dataBytes, true);
  str(8, 'WAVE');
  str(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);                       // PCM
  view.setUint16(22, 1, true);                       // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * bytesPerSample, true);
  view.setUint16(32, bytesPerSample, true);
  view.setUint16(34, bitDepth, true);
  str(36, 'data');
  view.setUint32(40, dataBytes, true);

  let off = 44;
  if (bitDepth === 24) {
    for (let i = 0; i < n; i++) {
      const v = Math.max(-1, Math.min(1, samples[i]));
      const s = Math.round(v < 0 ? v * 0x800000 : v * 0x7fffff);
      view.setUint8(off++, s & 0xff);
      view.setUint8(off++, (s >> 8) & 0xff);
      view.setUint8(off++, (s >> 16) & 0xff);
    }
  } else {
    for (let i = 0; i < n; i++, off += 2) {
      const v = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(off, v < 0 ? v * 0x8000 : v * 0x7fff, true);
    }
  }
  return new Blob([buf], { type: 'audio/wav' });
}

/**
 * Min/max pairs per horizontal pixel, which is what a waveform actually is.
 *
 * Drawing every sample is both slow and wrong: at any useful zoom there are
 * hundreds of samples behind one pixel, and picking one of them — which is
 * what naive decimation does — makes a signal look quieter than it is and
 * hides exactly the short transient a clip lives in. Keeping both extremes
 * per column shows the true envelope, and a single clipped sample stays
 * visible at any zoom.
 */
export function peaks(samples, buckets) {
  const out = new Float32Array(buckets * 2);
  const step = samples.length / buckets;
  for (let b = 0; b < buckets; b++) {
    const start = Math.floor(b * step);
    const end = Math.min(samples.length, Math.floor((b + 1) * step));
    let lo = 0, hi = 0;
    for (let i = start; i < end; i++) {
      const v = samples[i];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    out[b * 2] = lo;
    out[b * 2 + 1] = hi;
  }
  return out;
}

/** Peak, RMS and clip count over the whole take, for the summary line. */
export function summarise(samples) {
  let peak = 0, sumsq = 0, clipped = 0;
  for (let i = 0; i < samples.length; i++) {
    const v = samples[i];
    const a = v < 0 ? -v : v;
    if (a > peak) peak = a;
    if (a >= 0.999) clipped++;
    sumsq += v * v;
  }
  const db = x => (x > 1e-7 ? 20 * Math.log10(x) : -120);
  return {
    peakDb: db(peak),
    rmsDb: db(Math.sqrt(sumsq / Math.max(samples.length, 1))),
    clippedSamples: clipped,
  };
}

/** Decode a fetched audio file to mono Float32 for display. */
export async function decodeToMono(arrayBuffer, ctx) {
  const buf = await ctx.decodeAudioData(arrayBuffer);
  if (buf.numberOfChannels === 1) return { samples: buf.getChannelData(0),
                                           sampleRate: buf.sampleRate };
  const a = buf.getChannelData(0);
  const b = buf.getChannelData(1);
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = (a[i] + b[i]) * 0.5;
  return { samples: out, sampleRate: buf.sampleRate };
}
