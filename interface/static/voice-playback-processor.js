/**
 * Gapless jitter-buffered PCM playback for Aura's duplex voice lane.
 *
 * Why a worklet instead of queueing AudioBufferSourceNodes: scheduling
 * separate buffers leaves a sub-millisecond seam at every clause boundary,
 * which on speech is plainly audible as a click or a stutter. A single
 * continuous ring buffer has no seams at all.
 *
 * The other reason is barge-in. When the user cuts in, playback must stop
 * within a frame or two — not after the currently scheduled buffer drains.
 * `flush` empties the ring on the audio thread itself, so the next 128-frame
 * render block is already silent.
 *
 * Runs on the audio thread: no allocation in `process`, ever.
 */
class VoicePlaybackProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();
        const seconds = (options && options.processorOptions && options.processorOptions.bufferSeconds) || 30;
        const requestedRate = options && options.processorOptions
            ? Number(options.processorOptions.sourceSampleRate) : sampleRate;
        this._sourceRate = Number.isFinite(requestedRate) && requestedRate > 0
            ? requestedRate : sampleRate;
        this._capacity = Math.ceil(sampleRate * seconds);
        this._ring = new Float32Array(this._capacity);
        this._read = 0;
        this._write = 0;
        this._available = 0;

        // Frames actually rendered to the speakers. This — not how much the
        // server sent — is the honest answer to "what did the user hear".
        this._playedFrames = 0;
        this._reportCountdown = 0;

        // Short linear fade applied when flushing, so an interruption sounds
        // like someone stopping mid-word rather than a hard digital click.
        this._fadeRemaining = 0;
        this._fadeLength = Math.max(1, Math.round(sampleRate * 0.008));

        this._level = 0;
        this._draining = false;
        this._sourceFrames = 0;
        this._nextOutputSourcePos = 0;
        this._sourceTail = 0;
        this._haveSourceTail = false;

        // Overlap ducking. Her volume drops the moment the user starts
        // talking over her — before anything decides whether that was an
        // interruption or just "mhm" — so the reaction is instant and the
        // irreversible call can wait for evidence. Ramped, because a step
        // change in gain is an audible click.
        this._gain = 1;
        this._gainTarget = 1;
        this._gainStep = 0;

        this.port.onmessage = (e) => this._onMessage(e.data);
    }

    _onMessage(msg) {
        if (!msg) return;
        switch (msg.type) {
            case 'push':
                this._push(new Int16Array(msg.pcm));
                break;
            case 'flush':
                // Preserve the next few buffered waveform samples and ramp
                // them down. Dropping first and fading zero is not a fade.
                this._fadeRemaining = this._fadeLength;
                if (this._available === 0) this._dropAll();
                break;
            case 'duck': {
                const target = typeof msg.gain === 'number' ? msg.gain : 1;
                const rampMs = Math.max(1, msg.ramp_ms || 60);
                this._gainTarget = Math.max(0, Math.min(1, target));
                const frames = Math.max(1, Math.round(sampleRate * rampMs / 1000));
                this._gainStep = (this._gainTarget - this._gain) / frames;
                break;
            }
            case 'reset':
                this._dropAll();
                this._playedFrames = 0;
                this._fadeRemaining = 0;
                this._draining = false;
                this._gain = 1;
                this._gainTarget = 1;
                this._gainStep = 0;
                this._sourceFrames = 0;
                this._nextOutputSourcePos = 0;
                this._sourceTail = 0;
                this._haveSourceTail = false;
                break;
            default:
                break;
        }
    }

    _dropAll() {
        this._read = 0;
        this._write = 0;
        this._available = 0;
    }

    _push(int16) {
        const n = int16.length;
        if (n === 0) return;
        const start = this._sourceFrames;
        const end = start + n;
        const ratio = sampleRate / this._sourceRate;
        const output = [];

        // Linear rate conversion with continuity across WebSocket chunks.
        // The last source sample is retained until the next chunk supplies
        // the right-hand interpolation point, eliminating boundary seams.
        while (this._nextOutputSourcePos < end - 1) {
            const leftIndex = Math.floor(this._nextOutputSourcePos);
            const fraction = this._nextOutputSourcePos - leftIndex;
            let left;
            let right;
            if (leftIndex === start - 1 && this._haveSourceTail) {
                left = this._sourceTail;
                right = int16[0] / 32768;
            } else if (leftIndex >= start && leftIndex + 1 < end) {
                left = int16[leftIndex - start] / 32768;
                right = int16[leftIndex - start + 1] / 32768;
            } else {
                break;
            }
            output.push(left + (right - left) * fraction);
            this._nextOutputSourcePos += 1 / ratio;
        }
        this._sourceFrames = end;
        this._sourceTail = int16[n - 1] / 32768;
        this._haveSourceTail = true;

        // Overflow means the server is far ahead of playback. Dropping the
        // oldest audio would desynchronise the caption track, so drop the
        // newest instead and report it — silently losing audio mid-sentence
        // is the kind of bug that looks like a model failure.
        if (this._available + output.length > this._capacity) {
            this.port.postMessage({ type: 'overflow', dropped: output.length });
            return;
        }
        for (let i = 0; i < output.length; i++) {
            this._ring[this._write] = output[i];
            this._write = (this._write + 1) % this._capacity;
        }
        this._available += output.length;
        this._draining = true;
    }

    process(inputs, outputs) {
        const out = outputs[0];
        if (!out || out.length === 0) return true;
        const channel = out[0];
        const n = channel.length;

        let peak = 0;
        for (let i = 0; i < n; i++) {
            let sample = 0;
            if (this._available > 0) {
                sample = this._ring[this._read];
                this._read = (this._read + 1) % this._capacity;
                this._available--;
                this._playedFrames++;
            } else if (this._fadeRemaining > 0) {
                sample = 0;
            }

            if (this._fadeRemaining > 0) {
                sample *= this._fadeRemaining / this._fadeLength;
                this._fadeRemaining--;
                if (this._fadeRemaining === 0) this._dropAll();
            }

            if (this._gain !== this._gainTarget) {
                this._gain += this._gainStep;
                if ((this._gainStep > 0 && this._gain >= this._gainTarget)
                    || (this._gainStep < 0 && this._gain <= this._gainTarget)) {
                    this._gain = this._gainTarget;
                    this._gainStep = 0;
                }
            }
            sample *= this._gain;

            channel[i] = sample;
            const abs = sample < 0 ? -sample : sample;
            if (abs > peak) peak = abs;
        }

        // Mirror to any additional channels rather than leaving them silent.
        for (let c = 1; c < out.length; c++) out[c].set(channel);

        // Smooth the level for the orb; raw per-block peaks look jittery.
        this._level = this._level * 0.72 + peak * 0.28;

        // Report ~every 50 ms. Every block would flood the main thread.
        this._reportCountdown -= n;
        if (this._draining && this._reportCountdown <= 0) {
            this._reportCountdown = Math.round(sampleRate * 0.05);
            this.port.postMessage({
                type: 'progress',
                playedMs: (this._playedFrames / sampleRate) * 1000,
                bufferedMs: (this._available / sampleRate) * 1000,
                level: this._level,
            });
        }

        if (this._draining && this._available === 0) {
            this._draining = false;
            this.port.postMessage({
                type: 'drained',
                playedMs: (this._playedFrames / sampleRate) * 1000,
            });
        }

        return true;
    }
}

registerProcessor('voice-playback-processor', VoicePlaybackProcessor);
