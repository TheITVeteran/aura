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
const CHUNK_SAMPLES = 320; // 20 ms at 16 kHz

class VoiceCaptureProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this._buffer = new Int16Array(CHUNK_SAMPLES);
        this._filled = 0;
        this._level = 0;
        this._levelCountdown = 0;
    }

    process(inputs) {
        const input = inputs[0];
        if (!input || !input[0]) return true;
        const samples = input[0];

        let peak = 0;
        for (let i = 0; i < samples.length; i++) {
            let s = samples[i];
            if (s > 1) s = 1; else if (s < -1) s = -1;
            const abs = s < 0 ? -s : s;
            if (abs > peak) peak = abs;

            this._buffer[this._filled++] = s < 0 ? s * 0x8000 : s * 0x7fff;

            if (this._filled === CHUNK_SAMPLES) {
                // Copy out so the transfer cannot race the next fill.
                const out = new Int16Array(this._buffer);
                this.port.postMessage({ type: 'pcm', pcm: out.buffer }, [out.buffer]);
                this._filled = 0;
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
