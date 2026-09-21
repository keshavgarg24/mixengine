/**
 * The capture engine.
 *
 * Owns the microphone, the AudioContext, the worklet, and the buffer the
 * take accumulates into. Everything real-time happens in the worklet; this
 * side arms it, drains it, and reports.
 *
 * Three decisions here are load-bearing.
 *
 * **Every browser audio "enhancement" is disabled.** Echo cancellation,
 * noise suppression and automatic gain control are tuned for speech
 * intelligibility on a conference call. Echo cancellation gates and ducks
 * against the far end, which on a take recorded over speakers is the beat —
 * so it chews holes in the vocal wherever the two overlap. Noise suppression
 * is a spectral subtractor that treats sustained tones as noise, which is a
 * fair description of singing. AGC moves the gain *during* the performance,
 * so every level decision the mix makes later is fighting one already
 * applied invisibly. All three are irreversible by the time the file exists.
 *
 * **AudioWorklet, not ScriptProcessor.** ScriptProcessor runs its callback
 * on the main thread, so a layout pass or a long paint drops audio. The
 * worklet runs on the render thread at real-time priority.
 *
 * **Raw PCM, not MediaRecorder.** MediaRecorder gives Opus or AAC. The take
 * is the one artifact in the entire pipeline that cannot be regenerated, and
 * lossy-encoding it before the engine has ever seen it throws away exactly
 * the high-frequency detail that pitch tracking and de-essing depend on.
 */

import { MetricBlock, RingBuffer, sharedMemoryAvailable } from './ring-buffer.js';

export const CAPTURE_STATE = {
  IDLE: 'idle',
  READY: 'ready',
  ARMED: 'armed',
  RECORDING: 'recording',
};

const WORKLET_URL = '/static/worklets/capture-processor.js';
const RING_SECONDS = 12;

export class CaptureEngine extends EventTarget {
  constructor() {
    super();
    this.state = CAPTURE_STATE.IDLE;
    this.ctx = null;
    this.stream = null;
    this.source = null;
    this.node = null;
    this.monitorGain = null;
    this.ring = null;
    this.metrics = null;
    this.shared = false;

    this.chunks = [];
    this.recordedFrames = 0;
    this.startTime = 0;
    this.deviceId = null;
    this.latencySamples = 0;
    this._drainTimer = 0;
    this._lastSeq = -1;
    this._staleFrames = 0;
  }

  get sampleRate() { return this.ctx ? this.ctx.sampleRate : 48000; }

  get recordedSeconds() {
    return this.recordedFrames / this.sampleRate;
  }

  /**
   * Total output-to-input delay the browser knows about, in seconds.
   *
   * `outputLatency` is what the device reports between a sample being
   * scheduled and it leaving the speaker; `baseLatency` is the graph's own
   * buffering. Both are estimates, and neither includes the air gap or the
   * input device's own buffering — which is why `latency.js` exists to
   * measure the real figure by loopback. This is the fallback when the user
   * has not run that.
   */
  get reportedLatency() {
    if (!this.ctx) return 0;
    return (this.ctx.outputLatency || 0) + (this.ctx.baseLatency || 0);
  }

  /**
   * How late the singer hears their own voice when monitoring through the
   * browser, in seconds: input buffering, the graph, output buffering.
   *
   * There is nothing software can do about this from inside a browser --
   * direct monitoring happens in the audio interface's hardware, before
   * the signal reaches the computer. What can be done is to say the
   * number, and to say when it is past the point where it stops feeling
   * like hearing yourself and starts feeling like an echo. Delayed
   * auditory feedback disrupts speech and singing from a few tens of
   * milliseconds; below about twelve it reads as room.
   */
  get monitorLatency() {
    if (!this.ctx) return 0;
    const track = this.stream?.getAudioTracks()[0];
    const input = track?.getSettings?.().latency || 0;
    return input + (this.ctx.baseLatency || 0) + (this.ctx.outputLatency || 0);
  }

  emit(type, detail) {
    this.dispatchEvent(new CustomEvent(type, { detail }));
  }

  async listDevices() {
    if (!navigator.mediaDevices?.enumerateDevices) return [];
    const all = await navigator.mediaDevices.enumerateDevices();
    return all.filter(d => d.kind === 'audioinput')
              .map(d => ({ id: d.deviceId, label: d.label || 'Microphone' }));
  }

  async open({ deviceId = null } = {}) {
    if (this.state !== CAPTURE_STATE.IDLE) await this.close();
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('This browser cannot record audio.');
    }

    const audio = {
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
      channelCount: 1,
    };
    if (deviceId) audio.deviceId = { exact: deviceId };

    try {
      this.stream = await navigator.mediaDevices.getUserMedia({ audio });
    } catch (err) {
      if (err.name === 'NotAllowedError') {
        throw new Error('Microphone access was declined.');
      }
      if (err.name === 'NotFoundError') {
        throw new Error('No microphone was found.');
      }
      throw new Error(`Could not open the microphone: ${err.message}`);
    }
    this.deviceId = deviceId;

    // `interactive` asks for the smallest buffer the device will give, which
    // is what makes monitoring usable. The alternative, `playback`, buys
    // stability the recorder does not need and adds latency it cannot spend.
    this.ctx = new (window.AudioContext || window.webkitAudioContext)({
      latencyHint: 'interactive',
    });
    if (this.ctx.state === 'suspended') await this.ctx.resume();

    try {
      await this.ctx.audioWorklet.addModule(WORKLET_URL);
    } catch (err) {
      await this.close();
      throw new Error(`Could not load the audio engine: ${err.message}`);
    }

    this.shared = sharedMemoryAvailable();
    const processorOptions = {};
    if (this.shared) {
      this.ring = RingBuffer.alloc(
        Math.ceil(RING_SECONDS * this.ctx.sampleRate), this.ctx.sampleRate, 1);
      processorOptions.ringBuffer = this.ring.sab;
    }
    this.metrics = MetricBlock.alloc();
    processorOptions.metrics = this.metrics.view.buffer;

    this.source = this.ctx.createMediaStreamSource(this.stream);
    this.node = new AudioWorkletNode(this.ctx, 'capture-processor', {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [1],
      processorOptions,
    });
    this.node.port.onmessage = ev => this.onWorkletMessage(ev.data);

    // Monitoring starts muted. Routing a microphone to speakers by default
    // is how feedback happens, and the singer usually has headphones.
    this.monitorGain = this.ctx.createGain();
    this.monitorGain.gain.value = 0;

    this.source.connect(this.node);
    this.node.connect(this.monitorGain);
    this.monitorGain.connect(this.ctx.destination);

    this.state = CAPTURE_STATE.READY;
    this.emit('state', { state: this.state });
    this.emit('opened', {
      sampleRate: this.ctx.sampleRate,
      shared: this.shared,
      latency: this.reportedLatency,
      monitorLatency: this.monitorLatency,
      track: this.stream.getAudioTracks()[0]?.getSettings?.() || {},
    });
    return true;
  }

  onWorkletMessage(msg) {
    if (msg.type === 'audio') {
      // Fallback path only: the worklet transfers a filled block.
      this.chunks.push(new Float32Array(msg.samples));
      this.recordedFrames += msg.samples.length;
    } else if (msg.type === 'stopped') {
      this.emit('dropped', { frames: msg.dropped });
    }
  }

  setMonitor(on, gainDb = -6) {
    if (!this.monitorGain) return;
    const target = on ? Math.pow(10, gainDb / 20) : 0;
    // Ramped, because a step on a live microphone path is a click straight
    // into someone's headphones.
    this.monitorGain.gain.setTargetAtTime(target, this.ctx.currentTime, 0.02);
  }

  readMetrics() {
    if (!this.metrics) return null;
    const m = this.metrics.read();
    // A frozen sequence counter means the render thread has stopped calling
    // us. That is a dead input, which looks identical to silence on a level
    // meter and is the single most expensive thing to discover after a take.
    if (m.seq === this._lastSeq) this._staleFrames++;
    else { this._staleFrames = 0; this._lastSeq = m.seq; }
    m.alive = this._staleFrames < 30;
    return m;
  }

  drain() {
    if (!this.ring) return;
    const block = this.ring.pull();
    if (block && block.length) {
      this.chunks.push(block);
      this.recordedFrames += block.length;
    }
  }

  start() {
    if (this.state !== CAPTURE_STATE.READY && this.state !== CAPTURE_STATE.ARMED) {
      throw new Error('The microphone is not open.');
    }
    this.chunks = [];
    this.recordedFrames = 0;
    if (this.ring) this.ring.clear();
    this.node.port.postMessage({ type: 'arm' });
    this.startTime = this.ctx.currentTime;
    this.state = CAPTURE_STATE.RECORDING;
    this.emit('state', { state: this.state });

    // Drained on a timer rather than from requestAnimationFrame: rAF stops
    // in a background tab, and a take does not stop when someone switches
    // window. The ring holds 12 seconds, so a 250 ms period has ample margin.
    clearInterval(this._drainTimer);
    this._drainTimer = setInterval(() => this.drain(), 250);
  }

  async stop() {
    if (this.state !== CAPTURE_STATE.RECORDING) return null;
    this.node.port.postMessage({ type: 'disarm' });
    clearInterval(this._drainTimer);
    this._drainTimer = 0;

    // Let the worklet's last quantum land before the final drain.
    await new Promise(r => setTimeout(r, 60));
    this.drain();

    this.state = CAPTURE_STATE.READY;
    this.emit('state', { state: this.state });

    const total = this.chunks.reduce((n, c) => n + c.length, 0);
    const out = new Float32Array(total);
    let off = 0;
    for (const c of this.chunks) { out.set(c, off); off += c.length; }
    this.chunks = [];

    return {
      samples: this.trimLatency(out),
      sampleRate: this.sampleRate,
      seconds: total / this.sampleRate,
      latencySamples: this.latencySamples,
    };
  }

  /**
   * Remove the round-trip delay from the head of the take.
   *
   * When someone records over playback, what the microphone hears is late
   * relative to what the browser scheduled, by the time the sound took to
   * leave the speaker and come back. The singer performed in time with what
   * they *heard*, so the recording is uniformly late by that amount, and
   * every later alignment stage would be correcting a fixed offset it should
   * never have been given. Cutting it here is exact, free, and cannot be
   * done as well downstream, where it is indistinguishable from the singer
   * being behind the beat.
   */
  trimLatency(samples) {
    const n = Math.round(this.latencySamples);
    if (n <= 0 || n >= samples.length) return samples;
    return samples.subarray(n);
  }

  setLatency(seconds) {
    this.latencySamples = Math.max(0, Math.round(seconds * this.sampleRate));
  }

  async close() {
    clearInterval(this._drainTimer);
    this._drainTimer = 0;
    try { this.node?.port.postMessage({ type: 'disarm' }); } catch (_) {}
    try { this.node?.disconnect(); } catch (_) {}
    try { this.source?.disconnect(); } catch (_) {}
    try { this.monitorGain?.disconnect(); } catch (_) {}
    this.stream?.getTracks().forEach(t => t.stop());
    if (this.ctx && this.ctx.state !== 'closed') { try { await this.ctx.close(); } catch (_) {} }
    this.ctx = null; this.stream = null; this.source = null;
    this.node = null; this.monitorGain = null; this.ring = null;
    this.chunks = [];
    this.state = CAPTURE_STATE.IDLE;
    this.emit('state', { state: this.state });
  }
}
