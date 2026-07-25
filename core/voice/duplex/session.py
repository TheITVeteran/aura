"""core/voice/duplex/session.py — The duplex conversation state machine.

This is where the pieces become a conversation. The shape that matters is
that there are two clocks running at once:

  * a **reflex clock** on every 32 ms audio frame — VAD, barge-in detection,
    backchannel placement, endpointing. Nothing here waits on a model, so it
    stays responsive while her mind is busy.
  * a **cognition clock** per turn — transcription, the governed turn, and
    synthesis, all of which take from hundreds of milliseconds to seconds.

Half-duplex designs collapse these into one loop, which is exactly why they
cannot say "mhm" while you talk or stop when you cut in. Keeping them apart
is the whole architecture.

Audio intake never awaits cognition: :meth:`feed_audio` does bounded work
and hands longer jobs to tasks. If that invariant breaks, the microphone
stutters whenever she thinks.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from core.runtime.errors import record_degradation
from core.voice.duplex import protocol
from core.voice.duplex.audio import (
    FrameSplitter,
    UtteranceBuffer,
    float32_to_pcm16,
    pcm16_to_float32,
)
from core.voice.duplex.backchannel import BackchannelReflex
from core.voice.duplex.clause_chunker import StreamingChunker
from core.voice.duplex.config import (
    CAPTURE_RATE,
    OUTPUT_RATE,
    VAD_FRAME_SAMPLES,
    DuplexConfig,
)
from core.voice.duplex.echo_guard import EchoGuard
from core.voice.duplex.endpointing import Completeness, Endpointer
from core.voice.duplex.fillers import FillerReflex, ThinkingCause
from core.voice.duplex.mind_bridge import MindBridge, SpokenRecord
from core.voice.duplex.overlap import OverlapArbiter, OverlapVerdict
from core.voice.duplex.paralinguistics import (
    DeliveryReading,
    SpeakerBaseline,
    convergence_factors,
)
from core.voice.duplex.paralinguistics import (
    analyze as analyze_delivery,
)
from core.voice.duplex.paralinguistics import (
    interpret as interpret_delivery,
)
from core.voice.duplex.prosody import ProsodyCompiler, live_speech_profile
from core.voice.duplex.streaming_asr import StreamingAsr, looks_hallucinated
from core.voice.duplex.streaming_reply import ClauseValidator, is_streamable
from core.voice.duplex.style import StyleController
from core.voice.duplex.tts_stream import CancellationToken, StreamingTts
from core.voice.duplex.vad_gate import SpeechEvent, VadGate

logger = logging.getLogger("Aura.Voice.Session")

# Callbacks the transport supplies. Returning awaitables lets the session
# apply backpressure if the socket is slow.
JsonSender = Callable[[dict[str, Any]], Awaitable[None]]
BinarySender = Callable[[bytes], Awaitable[None]]


class SessionState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    USER_SPEAKING = "user_speaking"
    THINKING = "thinking"
    SPEAKING = "speaking"
    CLOSED = "closed"


@dataclass(slots=True)
class TurnMetrics:
    """Latency ledger for one turn. Reported so claims stay measurable."""

    speech_end_at: float = 0.0
    final_transcript_at: float = 0.0
    reply_ready_at: float = 0.0
    first_audio_at: float = 0.0
    asr_ms: float = 0.0
    asr_speculative_ms: float = 0.0  # decode that ran inside the endpoint wait
    cognition_ms: float = 0.0
    tts_first_chunk_ms: float = 0.0

    def as_dict(self) -> dict[str, float]:
        ttfa = (
            (self.first_audio_at - self.speech_end_at) * 1000.0
            if self.first_audio_at and self.speech_end_at
            else 0.0
        )
        return {
            "asr_ms": round(self.asr_ms, 1),
            "asr_speculative_ms": round(self.asr_speculative_ms, 1),
            "cognition_ms": round(self.cognition_ms, 1),
            "tts_first_chunk_ms": round(self.tts_first_chunk_ms, 1),
            "time_to_first_audio_ms": round(ttfa, 1),
        }


@dataclass(slots=True)
class _SpeakingTrack:
    """Bookkeeping for the utterance currently being played.

    ``chunks`` records the text and duration of every chunk handed to the
    client, so a barge-in can be mapped back to the exact words the user
    actually heard rather than the whole reply she intended.
    """

    utterance_id: int = 0
    intended: str = ""
    chunks: list[tuple[str, float]] = field(default_factory=list)
    sent_duration_s: float = 0.0
    started_at: float = 0.0
    token: CancellationToken = field(default_factory=CancellationToken)

    def spoken_prefix(self, played_s: float) -> str:
        """The text corresponding to ``played_s`` seconds of playback."""
        if played_s <= 0:
            return ""
        out: list[str] = []
        remaining = played_s
        for text, duration in self.chunks:
            if remaining <= 0:
                break
            if remaining >= duration:
                out.append(text)
                remaining -= duration
                continue
            # Partially played chunk: approximate by word fraction. Speech
            # is not uniform in time, but for the purpose of "what did they
            # hear" a proportional cut is far closer than all-or-nothing.
            words = text.split()
            if words:
                take = max(1, int(len(words) * (remaining / max(duration, 1e-6))))
                out.append(" ".join(words[:take]))
            remaining = 0
        return " ".join(out).strip()


class DuplexVoiceSession:
    """One live full-duplex voice conversation."""

    def __init__(
        self,
        *,
        session_id: str,
        send_json: JsonSender,
        send_binary: BinarySender,
        config: DuplexConfig | None = None,
        mind: MindBridge | None = None,
    ) -> None:
        self._id = session_id
        self._send_json = send_json
        self._send_binary = send_binary
        self._config = config or DuplexConfig()

        self._vad = VadGate(self._config.vad)
        self._asr = StreamingAsr(self._config.asr)
        self._endpointer = Endpointer(self._config.endpoint)
        self._backchannel = BackchannelReflex(self._config.backchannel)
        self._filler = FillerReflex()
        self._echo = EchoGuard()
        self._style = StyleController()
        self._overlap = OverlapArbiter()
        self._overlap_audio: list[np.ndarray] = []
        self._voice_override = ""
        self._tts = StreamingTts(self._config.tts)
        self._prosody = ProsodyCompiler(
            base_voice=self._config.tts.voice,
            base_speed=self._config.tts.speed,
        )
        self._mind = mind or MindBridge(
            session_id=session_id,
            cognition_timeout_s=self._config.cognition_timeout_s,
            spoken_reply_words=self._config.spoken_reply_words,
        )

        self._splitter = FrameSplitter(VAD_FRAME_SAMPLES)
        self._utterance = UtteranceBuffer(
            self._config.asr.max_utterance_s, CAPTURE_RATE
        )

        self._state = SessionState.IDLE
        self._muted = False
        self._closed = False

        self._turn_task: asyncio.Task[None] | None = None
        self._partial_task: asyncio.Task[None] | None = None
        self._filler_task: asyncio.Task[None] | None = None
        self._side_tasks: set[asyncio.Task[Any]] = set()

        self._speaking: _SpeakingTrack | None = None
        self._utterance_counter = 0
        self._utterance_generation = 0
        self._barge_run_ms = 0.0
        self._client_played_s = 0.0
        self._metrics = TurnMetrics()
        self._stable_text = ""
        # A final decode started during the endpoint's silence wait, valid
        # only until the user speaks again.
        self._speculative: Any = None
        self._speculative_task: asyncio.Task[None] | None = None
        # How the user has been sounding, and how this turn deviates from it.
        self._speaker_baseline = SpeakerBaseline()
        self._delivery = DeliveryReading()

    # ── lifecycle ────────────────────────────────────────────────────────

    @property
    def state(self) -> SessionState:
        return self._state

    async def start(self) -> None:
        """Warm every model before the first word so nobody pays cold start.

        Measured cold costs on this host: Kokoro 635 ms vs 190 ms warm, and
        Whisper 13–35 s for weight load plus Metal kernel compilation. Paying
        that inside a live turn would be indistinguishable from a hang.
        """
        await self._set_state(SessionState.LISTENING)
        spec = self._prosody_spec()

        results = await asyncio.gather(
            self._tts.warm_up(spec),
            self._asr.warm_up(),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                record_degradation(
                    "voice_duplex.session",
                    result,
                    action="continued with a cold model; the first turn will be slower",
                    severity="warning",
                )

        await self._mind.start_activity_watch(self._filler.observe_activity)
        await self._mind.publish("session_started", {"engine": self._tts.engine_name})
        await self._send_json(
            {
                "type": protocol.EVT_READY,
                "tts_engine": self._tts.engine_name,
                "vad_backend": self._vad.backend_name,
                "asr_available": self._asr.available,
                "sample_rate_in": CAPTURE_RATE,
                "sample_rate_out": OUTPUT_RATE,
            }
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._cancel_playback()
        for task in (self._turn_task, self._partial_task, self._filler_task):
            if task is not None and not task.done():
                task.cancel()
        for task in list(self._side_tasks):
            if not task.done():
                task.cancel()
        with contextlib.suppress(asyncio.CancelledError, RuntimeError):
            await self._mind.stop_activity_watch()
        self._tts.shutdown()
        self._state = SessionState.CLOSED
        await self._mind.publish("session_ended", {})

    # ── audio intake (the reflex clock) ──────────────────────────────────

    async def feed_audio(self, data: bytes) -> None:
        """Handle one arbitrary-sized PCM16 buffer from the client.

        Must stay bounded. Anything that could take longer than a frame
        period is dispatched to a task instead of awaited here.
        """
        if self._closed or self._muted:
            return
        try:
            samples = pcm16_to_float32(data)
        except (ValueError, TypeError) as exc:
            record_degradation(
                "voice_duplex.session",
                exc,
                action="dropped a malformed audio buffer",
                severity="warning",
            )
            return

        for frame in self._splitter.push(samples):
            await self._process_frame(frame)

    async def _process_frame(self, frame: np.ndarray) -> None:
        try:
            vf = self._vad.process(frame)
        except (ValueError, RuntimeError) as exc:
            record_degradation(
                "voice_duplex.session", exc, action="skipped a VAD frame", severity="debug"
            )
            return

        # While she is speaking, audio feeds barge-in detection and nothing
        # else. This exclusivity is load-bearing: the ordinary onset rule
        # fires after 2 frames above 0.55, while barge-in deliberately
        # demands 4 frames above 0.72. Run both and onset always wins the
        # race, silently converting every interruption into "user started a
        # new turn" — her audio keeps playing and nothing is ever cut off.
        if self._state is SessionState.SPEAKING:
            # Retain the frames anyway, so the words that did the
            # interrupting survive the time it takes to classify them.
            self._utterance.observe_silence(frame)
            await self._handle_overlap(frame, vf)
            return

        if vf.event is SpeechEvent.ONSET:
            await self._begin_user_turn()
        elif not self._vad.in_speech:
            self._utterance.observe_silence(frame)
            return

        if self._vad.in_speech:
            self._utterance.append(frame)

        if vf.event is SpeechEvent.PAUSE:
            await self._on_pause(vf.silence_ms, vf.speech_ms)
        elif vf.event in (SpeechEvent.CONTINUING, SpeechEvent.RESUMED):
            if vf.event is SpeechEvent.RESUMED:
                # They were not finished after all, so any speculative
                # transcript describes an unfinished sentence. Drop it.
                self._discard_speculation()
            self._maybe_schedule_partial()

    async def _begin_user_turn(self) -> None:
        """The user started talking."""
        # Cancel a pending turn: if they started again before she answered,
        # the newer utterance supersedes the older one.
        if self._turn_task is not None and not self._turn_task.done():
            self._turn_task.cancel()
        self._stop_fillers()

        self._utterance.begin()
        self._asr.reset()
        self._discard_speculation()
        self._stable_text = ""
        # A partial decode started for the previous utterance can still be
        # in flight. Bump the generation so its result is discarded instead
        # of overwriting this utterance's transcript with the last one's.
        self._utterance_generation += 1
        self._backchannel.on_user_turn_start()
        await self._set_state(SessionState.USER_SPEAKING)

    async def _on_pause(self, silence_ms: float, speech_ms: float) -> None:
        """A gap inside or at the end of the user's turn."""
        decision = self._endpointer.evaluate(
            transcript=self._stable_text,
            silence_ms=silence_ms,
            speech_ms=speech_ms,
            min_utterance_ms=self._config.vad.min_utterance_ms,
        )

        if decision.should_end:
            await self._end_user_turn(decision.reason)
            return

        # The endpointer is about to spend hundreds of milliseconds waiting
        # to see whether they resume. Spend that time decoding instead.
        self._maybe_speculate_final(decision, silence_ms)

        # Not an endpoint — so this gap is a prosodic boundary, which is
        # exactly where acknowledgement belongs.
        bc = self._backchannel.consider(
            silence_ms=silence_ms,
            speech_ms=speech_ms,
            aura_is_speaking=self._state is SessionState.SPEAKING,
            substrate=self._substrate_snapshot(),
        )
        if bc.should_emit:
            self._spawn(self._speak_aside(bc.text, protocol.AudioOpcode.BACKCHANNEL, bc.gain))
            await self._send_json(
                {"type": protocol.EVT_BACKCHANNEL, "text": bc.text, "register": bc.register}
            )
            await self._mind.publish("backchannel", {"text": bc.text, "register": bc.register})

        self._maybe_schedule_partial()

    def _maybe_speculate_final(self, decision: Any, silence_ms: float) -> None:
        """Start the final decode during the endpoint's wait, not after it.

        The endpointer deliberately waits 340–1500 ms to see whether the user
        resumes. Serially, the ~320 ms final decode then happens *after* that
        wait, so the two costs add. Overlapping them means that when the
        endpoint confirms, the transcript her mind needs already exists.

        Safe because a decode is a pure function of the audio: a wasted one
        costs CPU and nothing else. It is thrown away the instant speech
        resumes, since it would then describe an unfinished sentence.
        """
        cfg = self._config.asr
        if not cfg.speculative_final or self._speculative is not None:
            return
        if self._speculative_task is not None and not self._speculative_task.done():
            return
        if silence_ms < cfg.speculate_after_ms:
            return
        # A trailing "um" means they are audibly still composing; speculating
        # there would burn a decode on a sentence that is about to grow.
        if getattr(decision, "completeness", None) is Completeness.THINKING:
            return
        if self._utterance.sample_count == 0:
            return
        self._speculative_task = self._spawn(self._run_speculative_final())

    async def _run_speculative_final(self) -> None:
        audio = self._utterance.audio()
        if audio.size == 0:
            return
        generation = self._utterance_generation
        result = await self._asr.finalize(audio)
        # Two ways this can be stale: a whole new utterance began, or they
        # resumed mid-pause and _discard_speculation cleared the slot.
        if generation != self._utterance_generation:
            return
        if self._vad.in_speech and self._vad.last_probability >= self._config.vad.speech_threshold:
            return
        self._speculative = result

    def _read_delivery(self, audio: np.ndarray, transcript: str) -> None:
        """Measure how this turn was said, relative to how they usually sound.

        Runs after the transcript exists because speaking rate needs a word
        count. Costs 3–5 ms of numpy on audio already in memory.
        """
        try:
            signature = analyze_delivery(
                audio, CAPTURE_RATE, word_count=len(transcript.split())
            )
            # Interpret against the baseline *before* folding this turn into
            # it — otherwise every utterance partly normalises itself away.
            self._delivery = interpret_delivery(signature, self._speaker_baseline)
            self._speaker_baseline.observe(signature)
            if self._delivery.notable:
                logger.info("Delivery: %s", ", ".join(self._delivery.descriptors))
        except (ValueError, TypeError, FloatingPointError, ZeroDivisionError) as exc:
            record_degradation(
                "voice_duplex.paralinguistics",
                exc,
                action="spoke without paralinguistic context for this turn",
                severity="debug",
            )
            self._delivery = DeliveryReading()

    def _discard_speculation(self) -> None:
        self._speculative = None
        task = self._speculative_task
        self._speculative_task = None
        if task is not None and not task.done():
            task.cancel()

    def _maybe_schedule_partial(self) -> None:
        """Kick off an incremental decode if one is due and none is running."""
        if self._partial_task is not None and not self._partial_task.done():
            return
        audio_s = self._utterance.sample_count / float(CAPTURE_RATE)
        if not self._asr.due_for_partial(time.monotonic(), audio_s):
            return
        self._partial_task = self._spawn(self._run_partial())

    async def _run_partial(self) -> None:
        audio = self._utterance.audio()
        if audio.size == 0:
            return
        generation = self._utterance_generation
        result = await self._asr.partial(audio)
        if result is None:
            return
        if generation != self._utterance_generation:
            # The utterance was reset while this decode ran. Its text
            # describes speech that is no longer the current turn.
            return
        self._stable_text = result.stable
        await self._send_json(
            {
                "type": protocol.EVT_PARTIAL,
                "stable": result.stable,
                "tentative": result.tentative,
                "decode_ms": round(result.decode_ms, 1),
            }
        )

    # ── barge-in ─────────────────────────────────────────────────────────

    async def _handle_overlap(self, frame: np.ndarray, vf: Any) -> None:
        """User speech while she is speaking. Duck first, decide second.

        The two cases — "mhm" and "no, stop" — are acoustically identical at
        onset, so any classifier forced to decide immediately is reliably
        wrong. Ducking is the correct response to *both*, is instant, and is
        reversible, which lets the irreversible call wait for real evidence.
        """
        cfg = self._config.barge_in
        track = self._speaking
        if not cfg.enabled or track is None:
            return

        # Grace window: without it, the tail of the user's own question or a
        # scrap of residual echo kills the reply the instant it begins.
        if (time.monotonic() - track.started_at) * 1000.0 < cfg.grace_ms:
            return

        is_speech = vf.probability >= cfg.threshold
        if not self._overlap.active:
            if not is_speech:
                return
            self._overlap.begin()
            self._overlap_audio = []

        self._overlap_audio.append(frame)
        verdict = self._overlap.observe(
            frame_ms=VAD_FRAME_SAMPLES / CAPTURE_RATE * 1000.0,
            is_speech=is_speech,
            energy=float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)))),
        )

        if self._overlap.should_duck():
            await self._send_json(
                {
                    "type": protocol.EVT_DUCK,
                    "gain": self._overlap.duck_gain,
                    "ramp_ms": 60,
                }
            )

        if verdict is OverlapVerdict.PENDING:
            return

        if verdict is OverlapVerdict.BARGE_IN:
            self._overlap.reset()
            self._overlap_audio = []
            await self._interrupt(reason="user_barge_in")
            await self._begin_user_turn()
            return

        # Backchannel: they were listening, not interrupting. Come back up
        # and keep the sentence going.
        overlap_audio = (
            np.concatenate(self._overlap_audio) if self._overlap_audio else None
        )
        self._overlap.reset()
        self._overlap_audio = []
        await self._send_json(
            {"type": protocol.EVT_DUCK, "gain": 1.0, "ramp_ms": 140}
        )
        await self._mind.publish("user_backchannel", {})
        logger.info("User backchannel over her speech — continuing")

        # Optimistic resume, verified a beat later. Timing alone can mistake
        # a short sharp objection ("no—") for acknowledgement, so transcribe
        # the overlap and undo the decision if it turns out to be words.
        if overlap_audio is not None and overlap_audio.size:
            self._spawn(self._verify_backchannel(overlap_audio, track))

    async def _verify_backchannel(
        self, audio: np.ndarray, track: _SpeakingTrack
    ) -> None:
        """Confirm a backchannel verdict against what was actually said."""
        try:
            result = await self._asr.partial(audio)
        except (RuntimeError, ValueError, OSError, AttributeError) as exc:
            record_degradation(
                "voice_duplex.overlap",
                exc,
                action="kept the timing-based backchannel verdict",
                severity="debug",
            )
            return
        if result is None or self._speaking is not track:
            return

        from core.voice.duplex.overlap import looks_like_backchannel

        text = result.full.strip()
        if not text or looks_like_backchannel(text):
            return
        if len(text.split()) < 2:
            return

        logger.info("Late barge-in: overlap was %r, not acknowledgement", text[:50])
        await self._interrupt(reason="late_barge_in")
        await self._begin_user_turn()

    async def _check_barge_in(self, probability: float) -> bool:
        """Cut her off when the user starts talking over her."""
        cfg = self._config.barge_in
        track = self._speaking
        if not cfg.enabled or track is None:
            return False

        # Grace window: without it, the tail of the user's own question or a
        # scrap of acoustic echo kills the reply the instant it begins.
        if (time.monotonic() - track.started_at) * 1000.0 < cfg.grace_ms:
            self._barge_run_ms = 0.0
            return False

        if probability < cfg.threshold:
            self._barge_run_ms = 0.0
            return False

        self._barge_run_ms += VAD_FRAME_SAMPLES / CAPTURE_RATE * 1000.0
        if self._barge_run_ms < cfg.trigger_ms:
            return False

        await self._interrupt(reason="user_barge_in")
        return True

    async def _interrupt(self, *, reason: str, played_s: float | None = None) -> None:
        """Stop speaking now and record what was actually heard."""
        track = self._speaking
        if track is None:
            return

        self._barge_run_ms = 0.0
        self._overlap.reset()
        self._overlap_audio = []
        track.token.cancel()

        # The client owns the playback buffer, so its reported position is
        # the only real answer to "what did they hear". Wall-clock is the
        # last resort and is always an over-estimate, because the server
        # finishes sending long before the client finishes playing.
        if played_s is None:
            if self._client_played_s > 0:
                played_s = self._client_played_s
            else:
                elapsed = time.monotonic() - track.started_at
                played_s = min(elapsed, track.sent_duration_s)
        played_s = max(0.0, min(played_s, track.sent_duration_s))

        spoken = track.spoken_prefix(played_s)
        record = SpokenRecord(
            intended=track.intended,
            spoken=spoken,
            interrupted=True,
            started_at=track.started_at,
            ended_at=time.monotonic(),
        )
        self._mind.record_spoken(record)
        self._speaking = None

        # Tell the client to drop everything buffered — otherwise it keeps
        # playing audio the server has already stopped generating.
        await self._send_json({"type": protocol.EVT_FLUSH, "utterance_id": track.utterance_id})
        await self._send_json(
            {
                "type": protocol.EVT_INTERRUPTED,
                "reason": reason,
                "spoken": spoken,
                "unheard": record.unheard,
            }
        )
        await self._mind.publish(
            "interrupted",
            {"reason": reason, "spoken_chars": len(spoken), "unheard_chars": len(record.unheard)},
        )
        logger.info("Barge-in (%s): heard %d chars, dropped %d", reason, len(spoken), len(record.unheard))

        # Leave the session listening. A real barge-in continues into
        # _begin_user_turn, which opens the utterance; an explicit "stop"
        # command has no follow-on speech and correctly ends here.
        await self._set_state(SessionState.LISTENING)

    # ── turn handling (the cognition clock) ──────────────────────────────

    async def _end_user_turn(self, reason: str) -> None:
        self._vad.close_utterance()
        audio = self._utterance.audio()
        self._utterance.clear()
        self._backchannel.on_user_turn_end()

        if audio.size == 0:
            await self._set_state(SessionState.LISTENING)
            return

        self._metrics = TurnMetrics(speech_end_at=time.monotonic())
        await self._set_state(SessionState.THINKING)
        self._turn_task = self._spawn(self._run_turn(audio, reason))

    async def _run_turn(self, audio: np.ndarray, reason: str) -> None:
        try:
            speculative = self._speculative
            self._speculative = None
            if speculative is not None:
                # Already decoded during the endpoint's wait — the turn's
                # transcript costs nothing here.
                final = speculative
                self._metrics.asr_ms = 0.0
                self._metrics.asr_speculative_ms = speculative.decode_ms
            else:
                final = await self._asr.finalize(audio)
                self._metrics.asr_ms = final.decode_ms
            self._metrics.final_transcript_at = time.monotonic()

            transcript = final.stable.strip()
            if not transcript or looks_hallucinated(transcript):
                # Silence or ASR noise. Say nothing — inventing a turn here
                # is how a voice agent starts talking to an empty room.
                await self._send_json(
                    {"type": protocol.EVT_FINAL, "text": "", "discarded": True}
                )
                await self._set_state(SessionState.LISTENING)
                return

            # Last line against her own voice coming back through the
            # speakers. If this "turn" is mostly her own recent words, it is
            # echo — answering it would have her talking to herself.
            verdict = self._echo.evaluate(transcript)
            if verdict.is_echo:
                await self._send_json(
                    {
                        "type": protocol.EVT_FINAL,
                        "text": "",
                        "discarded": True,
                        "reason": "echo_rejected",
                        "overlap": round(verdict.overlap, 3),
                    }
                )
                await self._mind.publish(
                    "echo_rejected", {"overlap": verdict.overlap, "text": transcript[:120]}
                )
                await self._set_state(SessionState.LISTENING)
                return

            await self._send_json(
                {
                    "type": protocol.EVT_FINAL,
                    "text": transcript,
                    "endpoint_reason": reason,
                    "decode_ms": round(final.decode_ms, 1),
                }
            )

            # Delivery requests take effect on this very reply, not the next
            # one — that immediacy is most of what makes it feel responsive.
            style_change = self._style.observe(transcript)
            if style_change:
                await self._send_json(
                    {"type": protocol.EVT_STYLE, "change": style_change}
                )
                await self._mind.publish("style_changed", {"change": style_change})

            self._read_delivery(audio, transcript)

            self._mind.notify_user_spoke()
            await self._mind.publish(
                "user_turn",
                {
                    "transcript": transcript,
                    "reason": reason,
                    "delivery": list(self._delivery.descriptors),
                },
            )

            # Fillers start now and run until the first real audio goes out.
            self._filler.begin_turn()
            self._filler_task = self._spawn(self._run_fillers(time.monotonic()))

            cognition_started = time.perf_counter()

            # Narrow streaming carve-out. Only conversational turns, every
            # clause validated before it is spoken, and any doubt at all
            # falls through to the fully governed buffered path below.
            if self._config.stream_reply:
                eligibility = is_streamable(transcript)
                if eligibility.ok:
                    if await self._speak_streaming(transcript):
                        self._metrics.cognition_ms = (
                            time.perf_counter() - cognition_started
                        ) * 1000.0
                        return
                else:
                    logger.debug("Streaming declined: %s", eligibility.reason)

            reply = await self._mind.respond(
                transcript, delivery_context=self._delivery.as_context()
            )
            self._metrics.cognition_ms = (time.perf_counter() - cognition_started) * 1000.0
            self._metrics.reply_ready_at = time.monotonic()

            if not reply:
                self._stop_fillers()
                await self._speak_text(
                    "Something went wrong in my reasoning lane before I had an answer. "
                    "I'd rather say that than make something up.",
                    cause=ThinkingCause.UNCERTAINTY,
                )
                await self._set_state(SessionState.LISTENING)
                return

            await self._speak_reply(reply)
        except asyncio.CancelledError:
            # Superseded by a newer utterance; not an error.
            raise
        except (RuntimeError, ValueError, AttributeError, TypeError, OSError) as exc:
            record_degradation(
                "voice_duplex.session",
                exc,
                action="voice turn aborted; returned the session to listening",
            )
            await self._send_json(
                {"type": protocol.EVT_ERROR, "message": "The voice turn failed before a reply formed."}
            )
            await self._set_state(SessionState.LISTENING)
        finally:
            self._stop_fillers()

    async def _run_fillers(self, started_at: float) -> None:
        """Emit thinking sounds while cognition runs."""
        cfg = self._config.filler
        if not cfg.enabled:
            return
        try:
            while True:
                await asyncio.sleep(0.1)
                elapsed = (time.monotonic() - started_at) * 1000.0
                utterance = self._filler.due(
                    elapsed,
                    first=cfg.first_delay_ms,
                    second=cfg.second_delay_ms,
                    third=cfg.third_delay_ms,
                )
                if utterance is None:
                    continue
                await self._send_json(
                    {
                        "type": protocol.EVT_FILLER,
                        "text": utterance.text,
                        "tier": utterance.tier,
                        "cause": utterance.cause.value,
                    }
                )
                await self._speak_aside(
                    utterance.text, protocol.AudioOpcode.FILLER, utterance.gain
                )
                await self._mind.publish(
                    "filler", {"text": utterance.text, "cause": utterance.cause.value}
                )
        except asyncio.CancelledError:
            raise

    def _stop_fillers(self) -> None:
        task = self._filler_task
        self._filler_task = None
        if task is not None and not task.done():
            task.cancel()

    # ── speech output ────────────────────────────────────────────────────

    async def _speak_streaming(self, transcript: str) -> bool:
        """Speak clauses as they are generated. Returns True if it handled the turn.

        Returning False means nothing irreversible happened and the caller
        should run the fully governed buffered path instead. That is the
        default outcome for every kind of doubt: an empty stream, an
        unavailable engine, or a clause that fails validation before any
        audio has gone out.
        """
        validator = ClauseValidator()
        chunker = StreamingChunker(
            first_max_chars=self._config.tts.first_chunk_max_chars,
            max_chars=self._config.tts.chunk_max_chars,
        )
        spec = self._prosody_spec()
        self._utterance_counter += 1
        track = _SpeakingTrack(
            utterance_id=self._utterance_counter,
            intended="",
            started_at=time.monotonic(),
        )

        spoken_any = False
        seq = 0

        async def _emit(clause: str) -> bool:
            """Validate, synthesise and send one clause. False = abort."""
            nonlocal spoken_any, seq
            verdict = validator.check(clause)
            if not verdict.ok:
                logger.warning("Streaming clause rejected (%s): %r", verdict.reason, clause[:60])
                await self._mind.publish(
                    "stream_clause_rejected", {"reason": verdict.reason}
                )
                return False

            result = await self._tts.synthesize(clause, spec, track.token)
            if result is None or track.token.cancelled:
                return False

            if not spoken_any:
                # First audio: only now is the turn committed to streaming.
                self._stop_fillers()
                self._speaking = track
                self._client_played_s = 0.0
                self._overlap.reset()
                self._overlap_audio = []
                self._metrics.first_audio_at = time.monotonic()
                self._metrics.tts_first_chunk_ms = result.synth_ms
                await self._set_state(SessionState.SPEAKING)
                spoken_any = True

            track.chunks.append((result.text, result.duration_s))
            track.sent_duration_s += result.duration_s
            track.intended = f"{track.intended} {result.text}".strip()
            self._echo.note_spoken(result.text)

            await self._send_json(
                {"type": protocol.EVT_SPEAKING_CHUNK, "text": result.text, "seq": seq}
            )
            await self._send_binary(
                protocol.encode_audio(
                    float32_to_pcm16(result.samples),
                    opcode=protocol.AudioOpcode.SPEECH,
                    seq=seq,
                    utterance_id=track.utterance_id,
                )
            )
            seq += 1
            return True

        try:
            async for piece in self._mind.stream_response(
                transcript, delivery_context=self._delivery.as_context()
            ):
                if track.token.cancelled:
                    break
                for clause in chunker.push(piece):
                    if not await _emit(clause):
                        return await self._abandon_stream(track, spoken_any)
            if not track.token.cancelled:
                for clause in chunker.flush():
                    if not await _emit(clause):
                        return await self._abandon_stream(track, spoken_any)
        except asyncio.CancelledError:
            raise
        except (RuntimeError, ValueError, AttributeError, TypeError, OSError) as exc:
            record_degradation(
                "voice_duplex.streaming",
                exc,
                action="abandoned the streamed reply; falling back to the governed turn",
                severity="warning",
            )
            return await self._abandon_stream(track, spoken_any)

        if not spoken_any:
            # The stream produced nothing usable. Nothing was said, so the
            # buffered path can answer cleanly.
            return False

        await self._send_binary(
            protocol.encode_audio(
                b"",
                opcode=protocol.AudioOpcode.SPEECH,
                seq=seq,
                utterance_id=track.utterance_id,
                last=True,
            )
        )
        await self._send_json({"type": protocol.EVT_REPLY, "text": track.intended})
        await self._await_playback_drain(track)
        if track.token.cancelled or self._speaking is not track:
            return True

        self._mind.record_spoken(
            SpokenRecord(
                intended=track.intended,
                spoken=track.intended,
                interrupted=False,
                started_at=track.started_at,
                ended_at=time.monotonic(),
            )
        )
        self._speaking = None
        await self._send_json({"type": protocol.EVT_METRICS, **self._metrics.as_dict()})
        await self._mind.publish(
            "spoke", {"chars": len(track.intended), "streamed": True, **self._metrics.as_dict()}
        )
        await self._set_state(SessionState.LISTENING)
        return True

    async def _abandon_stream(self, track: _SpeakingTrack, spoken_any: bool) -> bool:
        """Give up on a streamed reply. Returns True if the turn is finished.

        If nothing was spoken this is invisible — the caller re-runs the
        governed path and the user never knows. If audio already went out we
        cannot unsay it, so she stops and says so, and the governed path then
        answers properly. Honest and slightly awkward beats fluent and wrong.
        """
        track.token.cancel()
        if not spoken_any:
            self._speaking = None
            return False

        await self._send_json({"type": protocol.EVT_FLUSH, "utterance_id": track.utterance_id})
        self._mind.record_spoken(
            SpokenRecord(
                intended=track.intended,
                spoken=track.intended,
                interrupted=True,
                started_at=track.started_at,
                ended_at=time.monotonic(),
            )
        )
        self._speaking = None
        await self._speak_text(
            "Sorry — that came out wrong. Let me say it properly.",
            cause=ThinkingCause.UNCERTAINTY,
        )
        return False

    async def _speak_reply(self, reply: str) -> None:
        await self._send_json({"type": protocol.EVT_REPLY, "text": reply})
        await self._speak_text(reply, cause=None)

    async def _speak_text(self, text: str, *, cause: ThinkingCause | None) -> None:
        """Chunk, synthesise and stream one full utterance."""
        spec = self._prosody_spec()
        self._utterance_counter += 1
        track = _SpeakingTrack(
            utterance_id=self._utterance_counter,
            intended=text,
            started_at=time.monotonic(),
        )
        self._speaking = track
        self._client_played_s = 0.0
        self._overlap.reset()
        self._overlap_audio = []
        await self._set_state(SessionState.SPEAKING)

        chunker = StreamingChunker(
            first_max_chars=self._config.tts.first_chunk_max_chars,
            max_chars=self._config.tts.chunk_max_chars,
        )
        pieces = chunker.push(text) + chunker.flush()

        async def _iter_chunks():
            for piece in pieces:
                if track.token.cancelled:
                    return
                yield piece

        seq = 0
        first_audio = True
        try:
            async for result in self._tts.stream(_iter_chunks(), spec, track.token):
                if track.token.cancelled or self._speaking is not track:
                    break
                if first_audio:
                    self._stop_fillers()
                    self._metrics.first_audio_at = time.monotonic()
                    self._metrics.tts_first_chunk_ms = result.synth_ms
                    first_audio = False

                track.chunks.append((result.text, result.duration_s))
                track.sent_duration_s += result.duration_s
                # Register the words so that if they come back through the
                # microphone a moment later, the echo guard recognises them.
                self._echo.note_spoken(result.text)

                await self._send_json(
                    {"type": protocol.EVT_SPEAKING_CHUNK, "text": result.text, "seq": seq}
                )
                await self._send_binary(
                    protocol.encode_audio(
                        float32_to_pcm16(result.samples),
                        opcode=protocol.AudioOpcode.SPEECH,
                        seq=seq,
                        utterance_id=track.utterance_id,
                    )
                )
                seq += 1
        except asyncio.CancelledError:
            raise
        except (RuntimeError, ValueError, OSError, AttributeError) as exc:
            record_degradation(
                "voice_duplex.session",
                exc,
                action="stopped synthesis for this utterance",
            )

        if self._speaking is track and not track.token.cancelled:
            await self._send_binary(
                protocol.encode_audio(
                    b"",
                    opcode=protocol.AudioOpcode.SPEECH,
                    seq=seq,
                    utterance_id=track.utterance_id,
                    last=True,
                )
            )

            # Sending is not speaking. Kokoro synthesises 6–8x faster than
            # realtime, so a 14 s reply is fully transmitted in ~2 s while
            # the client is only two seconds into playing it. Returning to
            # LISTENING here would disarm barge-in for the remaining twelve
            # seconds — the user would talk over her and nothing would stop.
            # So the turn stays open until the audio has actually been heard.
            await self._await_playback_drain(track)
            if track.token.cancelled or self._speaking is not track:
                return

            self._mind.record_spoken(
                SpokenRecord(
                    intended=text,
                    spoken=text,
                    interrupted=False,
                    started_at=track.started_at,
                    ended_at=time.monotonic(),
                )
            )
            self._speaking = None
            await self._send_json({"type": protocol.EVT_METRICS, **self._metrics.as_dict()})
            await self._mind.publish("spoke", {"chars": len(text), **self._metrics.as_dict()})
            await self._set_state(SessionState.LISTENING)

    async def _await_playback_drain(self, track: _SpeakingTrack) -> None:
        """Stay in SPEAKING until the client has actually played the audio.

        The client's reported position is authoritative. The wall-clock
        deadline is a fallback for a client that stops reporting: without it
        a dropped report would strand the session in SPEAKING forever, which
        is a worse failure than ending the turn slightly early.
        """
        total = track.sent_duration_s
        if total <= 0:
            return
        deadline = time.monotonic() + total + 3.0
        while not track.token.cancelled and self._speaking is track:
            if self._client_played_s >= total - 0.05:
                return
            if time.monotonic() >= deadline:
                logger.debug(
                    "Playback drain fell back to wall clock (played %.2fs of %.2fs)",
                    self._client_played_s,
                    total,
                )
                return
            await asyncio.sleep(0.05)

    async def _speak_aside(self, text: str, opcode: protocol.AudioOpcode, gain: float) -> None:
        """Synthesise a backchannel or filler.

        Asides deliberately do not touch ``_speaking``: they are not turns,
        they must not be interruptible as if they were, and they must not
        overwrite the record of what she actually said.
        """
        spec = self._prosody_spec()
        spec = spec.scaled(gain=spec.gain * gain)
        token = CancellationToken()
        try:
            result = await self._tts.synthesize(text, spec, token)
        except (RuntimeError, ValueError, OSError, AttributeError) as exc:
            record_degradation(
                "voice_duplex.session",
                exc,
                action=f"skipped a {opcode.name.lower()} utterance",
                severity="warning",
            )
            return
        if result is None or self._closed:
            return
        await self._send_binary(
            protocol.encode_audio(
                float32_to_pcm16(result.samples),
                opcode=opcode,
                seq=0,
                utterance_id=0,
                last=True,
            )
        )

    def _cancel_playback(self) -> None:
        track = self._speaking
        if track is not None:
            track.token.cancel()
        self._speaking = None

    # ── client commands ──────────────────────────────────────────────────

    async def handle_command(self, message: dict[str, Any]) -> None:
        command = str(message.get("command") or message.get("type") or "")

        if command == protocol.CMD_STOP:
            await self._interrupt(reason="user_stop")
        elif command == protocol.CMD_MUTE:
            self._muted = True
            await self._send_json({"type": protocol.EVT_STATE, "state": self._state.value, "muted": True})
        elif command == protocol.CMD_UNMUTE:
            self._muted = False
            self._splitter.reset()
            self._vad.reset()
            await self._send_json({"type": protocol.EVT_STATE, "state": self._state.value, "muted": False})
        elif command == protocol.CMD_BARGE_IN:
            played_ms = float(message.get("played_ms") or 0.0)
            await self._interrupt(reason="client_barge_in", played_s=played_ms / 1000.0)
        elif command == protocol.CMD_PLAYBACK:
            self._client_played_s = float(message.get("played_ms") or 0.0) / 1000.0
        elif command == protocol.CMD_LIST_VOICES:
            await self._send_json(
                {"type": protocol.EVT_VOICES, "voices": self._tts.available_voices(),
                 "current": self._voice_override or self._config.tts.voice}
            )
        elif command == protocol.CMD_SET_VOICE:
            requested = str(message.get("voice") or "").strip()
            if requested and requested in self._tts.available_voices():
                self._voice_override = requested
                await self._send_json(
                    {"type": protocol.EVT_VOICES, "voices": self._tts.available_voices(),
                     "current": requested}
                )
                await self._mind.publish("voice_changed", {"voice": requested})
        elif command == protocol.CMD_TEXT:
            text = str(message.get("text") or "").strip()
            if text:
                self._metrics = TurnMetrics(speech_end_at=time.monotonic())
                await self._set_state(SessionState.THINKING)
                self._turn_task = self._spawn(self._run_typed_turn(text))

    async def _run_typed_turn(self, text: str) -> None:
        """A typed message while in voice mode still gets a spoken answer."""
        try:
            await self._send_json({"type": protocol.EVT_FINAL, "text": text, "typed": True})
            self._filler.begin_turn()
            self._filler_task = self._spawn(self._run_fillers(time.monotonic()))
            reply = await self._mind.respond(text)
            if reply:
                await self._speak_reply(reply)
            else:
                await self._set_state(SessionState.LISTENING)
        except asyncio.CancelledError:
            raise
        except (RuntimeError, ValueError, AttributeError, TypeError, OSError) as exc:
            record_degradation(
                "voice_duplex.session", exc, action="typed voice turn failed"
            )
            await self._set_state(SessionState.LISTENING)
        finally:
            self._stop_fillers()

    # ── helpers ──────────────────────────────────────────────────────────

    def _prosody_spec(self) -> Any:
        """Her compiled prosody, plus any delivery the user asked for.

        Substrate first, user override second: if she is low-energy the
        voice is still slower, but "talk faster" moves it from wherever it
        currently is rather than snapping to a fixed rate.
        """
        spec = self._prosody.compile(live_speech_profile())

        # Move partway toward how the user is speaking. Full mirroring is
        # mimicry; none at all is the flat-affect problem this exists to fix.
        conv_speed, conv_gain = convergence_factors(self._delivery)
        if conv_speed != 1.0 or conv_gain != 1.0:
            spec = spec.scaled(
                speed=max(0.7, min(1.4, spec.speed * conv_speed)),
                gain=max(0.25, min(1.3, spec.gain * conv_gain)),
            )

        adjustment = self._style.adjustment
        if adjustment.active:
            spec = spec.scaled(
                speed=max(0.7, min(1.4, spec.speed * (1.0 + adjustment.rate_delta))),
                gain=max(0.25, min(1.3, spec.gain * (1.0 + adjustment.gain_delta))),
            )
        if self._voice_override:
            spec.voice = self._voice_override
        return spec

    async def _set_state(self, state: SessionState) -> None:
        if state is self._state:
            return
        self._state = state
        await self._send_json({"type": protocol.EVT_STATE, "state": state.value})

    def _substrate_snapshot(self) -> dict[str, float]:
        """Current affect, for choosing a backchannel register."""
        try:
            profile = live_speech_profile()
            if profile is None:
                return {}
            return {
                "valence": float(getattr(profile, "warmth", 0.5)) * 2.0 - 1.0,
                "arousal": float(getattr(profile, "energy", 0.5)),
                "curiosity": float(getattr(profile, "playfulness", 0.3)),
            }
        except (AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "voice_duplex.session",
                exc,
                action="used a neutral backchannel register",
                severity="debug",
            )
            return {}

    def _spawn(self, coro: Any) -> asyncio.Task[Any]:
        """Track background tasks so close() can cancel every one."""
        task = asyncio.ensure_future(coro)
        self._side_tasks.add(task)
        task.add_done_callback(self._side_tasks.discard)
        return task

    def status(self) -> dict[str, Any]:
        return {
            "session_id": self._id,
            "state": self._state.value,
            "muted": self._muted,
            "tts_engine": self._tts.engine_name,
            "vad_backend": self._vad.backend_name,
            "asr_available": self._asr.available,
            "metrics": self._metrics.as_dict(),
        }


SessionEvent = protocol  # re-exported for callers that want the event names
