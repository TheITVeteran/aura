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

        this.port.onmessage = (e) => this._onMessage(e.data);
    }

    _onMessage(msg) {
        if (!msg) return;
        switch (msg.type) {
            case 'push':
                this._push(new Int16Array(msg.pcm));
                break;
            case 'flush':
                // Fade the tail already in flight, then drop everything.
                this._fadeRemaining = this._fadeLength;
                this._dropAll();
                break;
            case 'reset':
                this._dropAll();
                this._playedFrames = 0;
                this._fadeRemaining = 0;
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
        // Overflow means the server is far ahead of playback. Dropping the
        // oldest audio would desynchronise the caption track, so drop the
        // newest instead and report it — silently losing audio mid-sentence
        // is the kind of bug that looks like a model failure.
        if (this._available + n > this._capacity) {
            this.port.postMessage({ type: 'overflow', dropped: n });
            return;
        }
        for (let i = 0; i < n; i++) {
            this._ring[this._write] = int16[i] / 32768;
            this._write = (this._write + 1) % this._capacity;
        }
        this._available += n;
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
            }

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
        if (this._reportCountdown <= 0) {
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
            this.port.postMessage({ type: 'drained' });
        }

        return true;
    }
}

registerProcessor('voice-playback-processor', VoicePlaybackProcessor);
