/**
 * Microphone capture for Aura's duplex voice lane.
 *
 * Separate from the legacy voice-processor.js on purpose: that one is still
 * wired to the half-duplex path, and changing its message shape would break
 * the wake-word lane.
 *
 * Emits 20 ms chunks. Small enough that barge-in detection sees the user's
 * first syllable almost immediately; large enough that the socket is not
 * sending thousands of tiny frames per second.
 *
 * Runs on the audio thread: preallocated buffer, no allocation in `process`
 * except the transferable that leaves.
 */
class VoiceCaptureProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();
        const requested = options && options.processorOptions
            ? Number(options.processorOptions.targetSampleRate) : 16000;
        this._targetRate = Number.isFinite(requested) && requested > 0 ? requested : 16000;
        this._chunkSamples = Math.max(1, Math.round(this._targetRate * 0.02));
        this._buffer = new Int16Array(this._chunkSamples);
        this._filled = 0;
        this._level = 0;
        this._levelCountdown = 0;
        this._resamplePhase = 0;
        this._resampleSum = 0;
        this._resampleCount = 0;
    }

    _emit(sample) {
        let s = sample;
        if (s > 1) s = 1; else if (s < -1) s = -1;
        this._buffer[this._filled++] = s < 0 ? s * 0x8000 : s * 0x7fff;
        if (this._filled === this._chunkSamples) {
            const out = new Int16Array(this._buffer);
            this.port.postMessage({ type: 'pcm', pcm: out.buffer }, [out.buffer]);
            this._filled = 0;
        }
    }

    process(inputs) {
        const input = inputs[0];
        if (!input || !input[0]) return true;
        const samples = input[0];

        let peak = 0;
        for (let i = 0; i < samples.length; i++) {
            const s = samples[i];
            const abs = s < 0 ? -s : s;
            if (abs > peak) peak = abs;

            if (sampleRate >= this._targetRate) {
                // Boxcar decimation is intentionally simple and stable. It
                // also provides the low-pass averaging a raw 48 -> 16 kHz
                // sample drop would omit.
                this._resampleSum += s;
                this._resampleCount++;
                this._resamplePhase += this._targetRate;
                if (this._resamplePhase >= sampleRate) {
                    this._emit(this._resampleSum / this._resampleCount);
                    this._resamplePhase -= sampleRate;
                    this._resampleSum = 0;
                    this._resampleCount = 0;
                }
            } else {
                // Rare low-rate devices: zero-order hold preserves timing
                // and intelligibility until the server's ASR sees 16 kHz.
                this._resamplePhase += this._targetRate;
                while (this._resamplePhase >= sampleRate) {
                    this._emit(s);
                    this._resamplePhase -= sampleRate;
                }
            }
        }

        // Smoothed level for the orb. Reported ~every 50 ms rather than every
        // render block, which would flood the main thread for no visual gain.
        this._level = this._level * 0.7 + peak * 0.3;
        this._levelCountdown -= samples.length;
        if (this._levelCountdown <= 0) {
            this._levelCountdown = Math.round(sampleRate * 0.05);
            this.port.postMessage({ type: 'level', level: this._level });
        }

        return true;
    }
}

registerProcessor('voice-capture-processor', VoiceCaptureProcessor);
