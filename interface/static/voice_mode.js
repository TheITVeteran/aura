/**
 * voice_mode.js — Aura's full-duplex voice surface.
 *
 * Design notes worth keeping:
 *
 * Two AudioContexts, not one. Capture runs at 16 kHz because that is what
 * Silero and Whisper want; playback runs at 24 kHz because that is Kokoro's
 * native rate and resampling her voice down to 16 k audibly dulls it. A
 * single context cannot have two rates.
 *
 * Echo cancellation is requested explicitly and is not optional. Without it,
 * on speakers, her own voice re-enters the microphone, trips barge-in, and
 * she interrupts herself — which looks exactly like a model failure. The
 * server has a transcript-level echo guard as the last line, but the browser's
 * WebRTC canceller is what does the real work.
 *
 * The client reports playback position continuously. The server needs it to
 * know what the user actually heard before an interruption, which is what
 * keeps her memory honest about her own half of the conversation.
 */
(function () {
    'use strict';

    const CAPTURE_RATE = 16000;
    const PLAYBACK_RATE = 24000;

    const OPCODE = { SPEECH: 1, BACKCHANNEL: 2, FILLER: 3 };
    const HEADER_BYTES = 8;

    const state = {
        active: false,
        ws: null,
        captureCtx: null,
        playbackCtx: null,
        micStream: null,
        micNode: null,
        speechNode: null,
        asideNode: null,
        micLevel: 0,
        auraLevel: 0,
        sessionState: 'idle',
        muted: false,
        playedMs: 0,
        bufferedMs: 0,
        transcript: [],
        partialStable: '',
        partialTentative: '',
        currentReply: '',
        spokenSoFar: '',
        voices: [],
        currentVoice: '',
        metrics: null,
        reconnectAttempts: 0,
        reconnectTimer: null,
        closingIntentionally: false,
        rafHandle: null,
    };

    // ── DOM ──────────────────────────────────────────────────────────────

    let root = null;
    const el = {};

    function buildSurface() {
        if (root) return root;
        root = document.createElement('div');
        root.className = 'vm-root';
        root.setAttribute('role', 'dialog');
        root.setAttribute('aria-label', 'Voice conversation');
        root.setAttribute('aria-modal', 'true');
        root.innerHTML = `
          <div class="vm-scrim"></div>
          <div class="vm-stage">
            <div class="vm-status" id="vm-status">connecting</div>

            <div class="vm-orb-wrap" id="vm-orb-wrap">
              <div class="vm-orb-glow" id="vm-orb-glow"></div>
              <div class="vm-orb" id="vm-orb">
                <div class="vm-orb-inner"></div>
                <div class="vm-orb-ring vm-ring-1"></div>
                <div class="vm-orb-ring vm-ring-2"></div>
              </div>
              <div class="vm-listening-dots" id="vm-dots">
                <span></span><span></span><span></span>
              </div>
            </div>

            <div class="vm-caption" id="vm-caption">
              <div class="vm-caption-user" id="vm-caption-user"></div>
              <div class="vm-caption-aura" id="vm-caption-aura"></div>
            </div>

            <div class="vm-transcript" id="vm-transcript" aria-live="polite"></div>

            <div class="vm-controls">
              <button type="button" class="vm-btn" id="vm-mute" title="Mute microphone" aria-label="Mute microphone">
                <span class="vm-ico" id="vm-mute-ico"></span>
                <span class="vm-btn-label">Mute</span>
              </button>
              <button type="button" class="vm-btn vm-btn-stop" id="vm-interrupt" title="Stop Aura speaking" aria-label="Stop Aura speaking">
                <span class="vm-ico vm-ico-stop"></span>
                <span class="vm-btn-label">Stop</span>
              </button>
              <button type="button" class="vm-btn" id="vm-transcript-toggle" title="Show transcript" aria-label="Show transcript">
                <span class="vm-ico vm-ico-list"></span>
                <span class="vm-btn-label">Transcript</span>
              </button>
              <button type="button" class="vm-btn" id="vm-voice-btn" title="Change voice" aria-label="Change voice">
                <span class="vm-ico vm-ico-wave"></span>
                <span class="vm-btn-label">Voice</span>
              </button>
              <button type="button" class="vm-btn vm-btn-end" id="vm-end" title="Leave voice mode" aria-label="Leave voice mode">
                <span class="vm-ico vm-ico-end"></span>
                <span class="vm-btn-label">End</span>
              </button>
            </div>

            <div class="vm-voice-picker" id="vm-voice-picker" role="listbox" aria-label="Voice"></div>

            <div class="vm-meta" id="vm-meta"></div>
          </div>`;
        document.body.appendChild(root);

        el.status = root.querySelector('#vm-status');
        el.orbWrap = root.querySelector('#vm-orb-wrap');
        el.orb = root.querySelector('#vm-orb');
        el.glow = root.querySelector('#vm-orb-glow');
        el.dots = root.querySelector('#vm-dots');
        el.captionUser = root.querySelector('#vm-caption-user');
        el.captionAura = root.querySelector('#vm-caption-aura');
        el.transcript = root.querySelector('#vm-transcript');
        el.mute = root.querySelector('#vm-mute');
        el.muteIco = root.querySelector('#vm-mute-ico');
        el.interrupt = root.querySelector('#vm-interrupt');
        el.transcriptToggle = root.querySelector('#vm-transcript-toggle');
        el.end = root.querySelector('#vm-end');
        el.meta = root.querySelector('#vm-meta');
        el.voiceBtn = root.querySelector('#vm-voice-btn');
        el.voicePicker = root.querySelector('#vm-voice-picker');

        el.voiceBtn.addEventListener('click', () => {
            root.classList.toggle('vm-show-voices');
            el.voiceBtn.classList.toggle('vm-active');
            renderVoicePicker();
        });

        el.mute.addEventListener('click', toggleMute);
        el.interrupt.addEventListener('click', () => sendCommand('barge_in', { played_ms: state.playedMs }));
        el.transcriptToggle.addEventListener('click', () => {
            root.classList.toggle('vm-show-transcript');
            el.transcriptToggle.classList.toggle('vm-active');
        });
        el.end.addEventListener('click', () => exitVoiceMode());

        // Tapping the orb interrupts — the fastest possible target, and the
        // gesture people reach for instinctively when they want it to stop.
        el.orbWrap.addEventListener('click', () => {
            if (state.sessionState === 'speaking') {
                sendCommand('barge_in', { played_ms: state.playedMs });
            }
        });

        document.addEventListener('keydown', onKeyDown);
        return root;
    }

    function onKeyDown(e) {
        if (!state.active) return;
        if (e.key === 'Escape') {
            e.preventDefault();
            exitVoiceMode();
        } else if (e.code === 'Space' && e.target === document.body) {
            // Space is the universal "stop talking" key here.
            e.preventDefault();
            if (state.sessionState === 'speaking') {
                sendCommand('barge_in', { played_ms: state.playedMs });
            }
        } else if (e.key.toLowerCase() === 'm' && e.target === document.body) {
            e.preventDefault();
            toggleMute();
        }
    }

    // ── audio capture ────────────────────────────────────────────────────

    async function startCapture() {
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                // Not optional. See the note at the top of this file.
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
                channelCount: 1,
                sampleRate: CAPTURE_RATE,
            },
        });
        state.micStream = stream;

        const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: CAPTURE_RATE });
        state.captureCtx = ctx;
        await ctx.audioWorklet.addModule('/static/voice-capture-processor.js');

        const source = ctx.createMediaStreamSource(stream);
        const node = new AudioWorkletNode(ctx, 'voice-capture-processor');
        state.micNode = node;

        node.port.onmessage = (e) => {
            const d = e.data;
            if (d.type === 'pcm') {
                if (state.ws && state.ws.readyState === WebSocket.OPEN && !state.muted) {
                    state.ws.send(d.pcm);
                }
            } else if (d.type === 'level') {
                state.micLevel = d.level;
            }
        };

        source.connect(node);
        // Do NOT connect the capture node to the destination: routing the
        // microphone to the speakers is a feedback loop.
    }

    async function startPlayback() {
        const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: PLAYBACK_RATE });
        state.playbackCtx = ctx;
        await ctx.audioWorklet.addModule('/static/voice-playback-processor.js');

        // Two independent lanes so a backchannel can overlap speech without
        // either one flushing the other.
        state.speechNode = new AudioWorkletNode(ctx, 'voice-playback-processor', {
            processorOptions: { bufferSeconds: 60 },
            outputChannelCount: [1],
        });
        state.asideNode = new AudioWorkletNode(ctx, 'voice-playback-processor', {
            processorOptions: { bufferSeconds: 10 },
            outputChannelCount: [1],
        });

        state.speechNode.port.onmessage = (e) => {
            const d = e.data;
            if (d.type === 'progress') {
                state.playedMs = d.playedMs;
                state.bufferedMs = d.bufferedMs;
                state.auraLevel = d.level;
                // The server needs this to know what was actually heard.
                sendCommand('playback', { played_ms: d.playedMs });
            } else if (d.type === 'overflow') {
                console.warn('[voice] playback buffer overflow, dropped', d.dropped, 'samples');
            }
        };
        state.asideNode.port.onmessage = (e) => {
            if (e.data.type === 'progress') {
                state.auraLevel = Math.max(state.auraLevel, e.data.level);
            }
        };

        state.speechNode.connect(ctx.destination);
        state.asideNode.connect(ctx.destination);
    }

    // ── websocket ────────────────────────────────────────────────────────

    function wsUrl() {
        const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        return `${proto}//${window.location.host}/ws/voice`;
    }

    function connect() {
        const ws = new WebSocket(wsUrl());
        ws.binaryType = 'arraybuffer';
        state.ws = ws;

        ws.onopen = () => {
            state.reconnectAttempts = 0;
            setStatus('listening');
            const token = (window.AURA_API_TOKEN || '');
            if (token) ws.send(JSON.stringify({ type: 'auth', token }));
            sendCommand('list_voices');
        };

        ws.onmessage = (e) => {
            if (typeof e.data === 'string') {
                let msg;
                try { msg = JSON.parse(e.data); } catch (_err) { return; }
                handleEvent(msg);
            } else {
                handleAudio(new Uint8Array(e.data));
            }
        };

        ws.onclose = () => {
            if (state.closingIntentionally || !state.active) return;
            scheduleReconnect();
        };

        ws.onerror = () => { /* onclose handles recovery */ };
    }

    function scheduleReconnect() {
        // Exponential backoff, capped. A voice session that silently stops
        // reconnecting looks identical to one that is merely listening,
        // so the status line always says which is happening.
        state.reconnectAttempts += 1;
        if (state.reconnectAttempts > 6) {
            setStatus('disconnected');
            showMeta('Connection lost. Press End and try again.');
            return;
        }
        const delay = Math.min(8000, 400 * Math.pow(2, state.reconnectAttempts - 1));
        setStatus('reconnecting');
        clearTimeout(state.reconnectTimer);
        state.reconnectTimer = setTimeout(() => { if (state.active) connect(); }, delay);
    }

    function sendCommand(command, extra) {
        if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return;
        state.ws.send(JSON.stringify(Object.assign({ command }, extra || {})));
    }

    // ── events ───────────────────────────────────────────────────────────

    function handleEvent(msg) {
        switch (msg.type) {
            case 'voice.ready':
                showMeta(`${msg.tts_engine} · ${msg.vad_backend}${msg.asr_available ? '' : ' · ASR unavailable'}`);
                break;

            case 'voice.state':
                setStatus(msg.state);
                if (typeof msg.muted === 'boolean') {
                    state.muted = msg.muted;
                    el.mute.classList.toggle('vm-active', state.muted);
                }
                break;

            case 'voice.partial':
                state.partialStable = msg.stable || '';
                state.partialTentative = msg.tentative || '';
                renderUserCaption();
                break;

            case 'voice.final':
                state.partialStable = '';
                state.partialTentative = '';
                if (msg.text) {
                    addTranscript('user', msg.text);
                    el.captionUser.textContent = msg.text;
                } else {
                    el.captionUser.textContent = '';
                }
                break;

            case 'voice.reply':
                state.currentReply = msg.text || '';
                state.spokenSoFar = '';
                addTranscript('aura', state.currentReply);
                break;

            case 'voice.chunk':
                // Caption follows the audio clause by clause, so the words
                // on screen are the words currently in the air.
                state.spokenSoFar = (state.spokenSoFar ? state.spokenSoFar + ' ' : '') + msg.text;
                el.captionAura.textContent = state.spokenSoFar;
                break;

            case 'voice.backchannel':
                flashAside(msg.text);
                break;

            case 'voice.filler':
                flashAside(msg.text);
                break;

            case 'voice.interrupted':
                if (state.speechNode) state.speechNode.port.postMessage({ type: 'flush' });
                markInterrupted(msg.spoken || '');
                break;

            case 'voice.flush':
                if (state.speechNode) state.speechNode.port.postMessage({ type: 'flush' });
                break;

            case 'voice.style':
                showMeta(`voice: ${msg.change}`);
                break;

            case 'voice.voices':
                state.voices = msg.voices || [];
                state.currentVoice = msg.current || '';
                renderVoicePicker();
                break;

            case 'voice.metrics':
                state.metrics = msg;
                showMeta(`${Math.round(msg.time_to_first_audio_ms)} ms to first word`);
                break;

            case 'voice.error':
                showMeta(msg.message || 'Voice error');
                setStatus('error');
                break;

            default:
                break;
        }
    }

    function handleAudio(bytes) {
        if (bytes.length < HEADER_BYTES) return;
        const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
        const opcode = view.getUint8(0);
        const payload = bytes.subarray(HEADER_BYTES);
        if (payload.length === 0) return;

        // Copy: the worklet takes ownership of the buffer it is handed.
        const pcm = new Int16Array(payload.length / 2);
        for (let i = 0; i < pcm.length; i++) {
            pcm[i] = view.getInt16(HEADER_BYTES + i * 2, true);
        }

        const node = opcode === OPCODE.SPEECH ? state.speechNode : state.asideNode;
        if (!node) return;
        if (opcode === OPCODE.SPEECH && state.speechSeqReset) {
            state.speechSeqReset = false;
        }
        node.port.postMessage({ type: 'push', pcm: pcm.buffer }, [pcm.buffer]);
    }

    // ── rendering ────────────────────────────────────────────────────────

    function setStatus(s) {
        state.sessionState = s;
        if (!el.status) return;
        const labels = {
            idle: 'ready', listening: 'listening', user_speaking: 'listening',
            thinking: 'thinking', speaking: 'speaking', closed: 'ended',
            connecting: 'connecting', reconnecting: 'reconnecting',
            disconnected: 'disconnected', error: 'error',
        };
        el.status.textContent = labels[s] || s;
        if (root) {
            root.dataset.state = s;
        }
    }

    function renderUserCaption() {
        if (!el.captionUser) return;
        const stable = state.partialStable;
        const tail = state.partialTentative;
        // The unstable tail is rendered faintly: Whisper may still revise it,
        // and showing it at full weight makes the caption look like it is
        // correcting itself constantly.
        el.captionUser.innerHTML =
            escapeHtml(stable) + (tail ? ` <span class="vm-tentative">${escapeHtml(tail)}</span>` : '');
    }

    function addTranscript(who, text) {
        if (!text) return;
        state.transcript.push({ who, text, at: Date.now() });
        const line = document.createElement('div');
        line.className = `vm-line vm-line-${who}`;
        line.innerHTML = `<span class="vm-who">${who === 'user' ? 'You' : 'Aura'}</span><span class="vm-text">${escapeHtml(text)}</span>`;
        el.transcript.appendChild(line);
        el.transcript.scrollTop = el.transcript.scrollHeight;
    }

    function markInterrupted(spoken) {
        const lines = el.transcript.querySelectorAll('.vm-line-aura');
        const last = lines[lines.length - 1];
        if (!last) return;
        last.classList.add('vm-interrupted');
        const textEl = last.querySelector('.vm-text');
        if (textEl && spoken) {
            // Show exactly what was heard, with the rest struck through —
            // this mirrors what her memory now holds.
            const full = textEl.textContent || '';
            const rest = full.startsWith(spoken) ? full.slice(spoken.length) : '';
            textEl.innerHTML = escapeHtml(spoken) +
                (rest ? `<span class="vm-unheard">${escapeHtml(rest)}</span>` : '');
        }
        el.captionAura.textContent = spoken;
    }

    // Kokoro's ids are opaque ("af_heart"). Render them as a language/gender
    // grouping plus a readable name, so choosing a voice is a two-second
    // decision instead of a lookup table.
    const VOICE_PREFIX = {
        af: 'US · female', am: 'US · male',
        bf: 'UK · female', bm: 'UK · male',
        ef: 'ES · female', em: 'ES · male',
        ff: 'FR · female', hf: 'HI · female', hm: 'HI · male',
        if: 'IT · female', im: 'IT · male',
        jf: 'JP · female', jm: 'JP · male',
        pf: 'PT · female', pm: 'PT · male',
        zf: 'ZH · female', zm: 'ZH · male',
    };

    function prettyVoice(id) {
        const [prefix, ...rest] = String(id).split('_');
        const name = rest.join('_') || id;
        return {
            group: VOICE_PREFIX[prefix] || 'other',
            name: name.charAt(0).toUpperCase() + name.slice(1),
        };
    }

    function renderVoicePicker() {
        if (!el.voicePicker) return;
        if (!state.voices.length) {
            el.voicePicker.innerHTML = '<div class="vm-voice-empty">No alternate voices available.</div>';
            return;
        }
        const groups = new Map();
        for (const id of state.voices) {
            const { group, name } = prettyVoice(id);
            if (!groups.has(group)) groups.set(group, []);
            groups.get(group).push({ id, name });
        }
        el.voicePicker.innerHTML = '';
        for (const [group, items] of groups) {
            const wrap = document.createElement('div');
            wrap.className = 'vm-voice-group';
            wrap.innerHTML = `<div class="vm-voice-group-label">${escapeHtml(group)}</div>`;
            const row = document.createElement('div');
            row.className = 'vm-voice-row';
            for (const item of items) {
                const chip = document.createElement('button');
                chip.type = 'button';
                chip.className = 'vm-voice-chip' + (item.id === state.currentVoice ? ' vm-active' : '');
                chip.textContent = item.name;
                chip.setAttribute('role', 'option');
                chip.setAttribute('aria-selected', String(item.id === state.currentVoice));
                chip.addEventListener('click', () => {
                    state.currentVoice = item.id;
                    sendCommand('set_voice', { voice: item.id });
                    // Speak a line in the new voice immediately: picking a
                    // voice you cannot hear is guesswork.
                    sendCommand('text', { text: 'Okay — this is how I sound now.' });
                    renderVoicePicker();
                });
                row.appendChild(chip);
            }
            wrap.appendChild(row);
            el.voicePicker.appendChild(wrap);
        }
    }

    function flashAside(text) {
        const node = document.createElement('div');
        node.className = 'vm-aside';
        node.textContent = text;
        el.orbWrap.appendChild(node);
        setTimeout(() => node.remove(), 1800);
    }

    function showMeta(text) {
        if (!el.meta) return;
        el.meta.textContent = text;
        el.meta.classList.add('vm-meta-show');
        clearTimeout(el.metaTimer);
        el.metaTimer = setTimeout(() => el.meta.classList.remove('vm-meta-show'), 3200);
    }

    function escapeHtml(s) {
        return String(s == null ? '' : s)
            .replace(/&/g, '&amp;').replace(/</g, '&lt;')
            .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    // ── animation ────────────────────────────────────────────────────────

    function tick() {
        if (!state.active) return;
        // The orb answers to whoever is talking: the user's mic level while
        // listening, her own output level while speaking. That single rule
        // makes turn-taking legible without any text.
        const speaking = state.sessionState === 'speaking';
        const level = speaking ? state.auraLevel : state.micLevel;
        const eased = Math.min(1, Math.pow(level * (speaking ? 2.6 : 4.2), 0.62));

        const scale = 1 + eased * (speaking ? 0.26 : 0.17);
        const glow = 0.25 + eased * 0.75;

        if (el.orb) el.orb.style.transform = `scale(${scale.toFixed(4)})`;
        if (el.glow) el.glow.style.opacity = glow.toFixed(3);

        state.rafHandle = requestAnimationFrame(tick);
    }

    // ── entry / exit ─────────────────────────────────────────────────────

    async function enterVoiceMode() {
        if (state.active) return true;
        buildSurface();
        state.active = true;
        state.closingIntentionally = false;
        setStatus('connecting');

        // Animate in before the (possibly slow) permission prompt, so the
        // transition never appears to hang on the click.
        requestAnimationFrame(() => root.classList.add('vm-open'));
        document.body.classList.add('vm-active');

        try {
            await startPlayback();
            await startCapture();
        } catch (err) {
            console.error('[voice] audio init failed', err);
            showMeta(err && err.name === 'NotAllowedError'
                ? 'Microphone permission denied.'
                : 'Could not start audio.');
            setStatus('error');
            // Leave the surface up so the message is readable, but tear the
            // half-initialised audio graph down.
            await teardownAudio();
            return false;
        }

        connect();
        state.rafHandle = requestAnimationFrame(tick);
        return true;
    }

    async function teardownAudio() {
        if (state.micStream) {
            state.micStream.getTracks().forEach((t) => t.stop());
            state.micStream = null;
        }
        if (state.micNode) { try { state.micNode.disconnect(); } catch (_e) { /* already gone */ } state.micNode = null; }
        if (state.speechNode) { try { state.speechNode.disconnect(); } catch (_e) { /* already gone */ } state.speechNode = null; }
        if (state.asideNode) { try { state.asideNode.disconnect(); } catch (_e) { /* already gone */ } state.asideNode = null; }
        for (const key of ['captureCtx', 'playbackCtx']) {
            const ctx = state[key];
            if (ctx && ctx.state !== 'closed') { try { await ctx.close(); } catch (_e) { /* already closed */ } }
            state[key] = null;
        }
    }

    async function exitVoiceMode() {
        if (!state.active) return;
        state.active = false;
        state.closingIntentionally = true;
        clearTimeout(state.reconnectTimer);
        if (state.rafHandle) cancelAnimationFrame(state.rafHandle);

        if (state.ws && state.ws.readyState === WebSocket.OPEN) {
            sendCommand('stop');
            state.ws.close(1000, 'user ended voice mode');
        }
        state.ws = null;
        await teardownAudio();

        // Hand the conversation back to the text thread so nothing said out
        // loud is lost when the surface closes.
        exportTranscript();

        root.classList.remove('vm-open');
        document.body.classList.remove('vm-active');
        setTimeout(() => { if (!state.active && root) root.classList.remove('vm-show-transcript'); }, 400);
    }

    function exportTranscript() {
        if (!state.transcript.length) return;
        const sink = window.auraAppendVoiceTranscript;
        if (typeof sink === 'function') {
            try { sink(state.transcript.slice()); } catch (err) { console.warn('[voice] transcript export failed', err); }
        }
        state.transcript = [];
        if (el.transcript) el.transcript.innerHTML = '';
    }

    function toggleMute() {
        state.muted = !state.muted;
        sendCommand(state.muted ? 'mute' : 'unmute');
        el.mute.classList.toggle('vm-active', state.muted);
        el.muteIco.classList.toggle('vm-ico-muted', state.muted);
        showMeta(state.muted ? 'microphone off' : 'microphone on');
    }

    window.AuraVoiceMode = {
        enter: enterVoiceMode,
        exit: exitVoiceMode,
        toggle: async () => (state.active ? exitVoiceMode() : enterVoiceMode()),
        isActive: () => state.active,
        setVoice: (v) => sendCommand('set_voice', { voice: v }),
        voices: () => state.voices.slice(),
        status: () => ({ state: state.sessionState, muted: state.muted, metrics: state.metrics }),
    };
})();
