"""Contract tests for the full-duplex voice lane.

These pin the behaviours that the end-to-end harness caught regressions in.
Three of them correspond to bugs that shipped-looking code actually had, and
each one silently defeated a headline feature rather than raising:

  * a trailing-word check that let Whisper's speculative full stop cut the
    user off mid-sentence,
  * a barge-in rule that could never fire because the ordinary onset rule
    always won the race,
  * a turn that closed when audio was *sent* rather than *heard*, disarming
    barge-in for most of the time the user was actually listening.

No model weights, no audio device, no network: everything here is either
pure logic or driven through injected fakes, so it runs in the offline suite.
"""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from core.voice.duplex.audio import (
    FrameSplitter,
    UtteranceBuffer,
    float32_to_pcm16,
    pcm16_to_float32,
)
from core.voice.duplex.backchannel import BackchannelReflex
from core.voice.duplex.clause_chunker import StreamingChunker, first_chunk, split_for_speech
from core.voice.duplex.config import VAD_FRAME_SAMPLES, BackchannelConfig
from core.voice.duplex.echo_guard import EchoGuard
from core.voice.duplex.endpointing import Completeness, Endpointer, classify
from core.voice.duplex.fillers import FillerReflex, ThinkingCause
from core.voice.duplex.mind_bridge import MindBridge, SpokenRecord
from core.voice.duplex.protocol import AudioOpcode, decode_audio, encode_audio
from core.voice.duplex.session import _SpeakingTrack
from core.voice.duplex.streaming_asr import looks_hallucinated
from core.voice.duplex.style import StyleController

# ── endpointing ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text,expected",
    [
        ("What time is it?", Completeness.COMPLETE),
        ("Hello.", Completeness.COMPLETE),
        ("Tell me about the way you handle interruptions", Completeness.COMPLETE),
        ("yeah", Completeness.COMPLETE),
        ("I was thinking we could maybe", Completeness.INCOMPLETE),
        ("I went to the", Completeness.INCOMPLETE),
        ("I think that's right but", Completeness.INCOMPLETE),
        ("so the thing is, um", Completeness.THINKING),
        ("wait...", Completeness.THINKING),
        ("the cat", Completeness.NEUTRAL),
    ],
)
def test_completeness_classification(text, expected):
    assert classify(text) is expected


def test_whisper_period_does_not_override_a_dangling_word():
    """The regression that cut users off mid-sentence.

    Whisper's full stops are a language-model prior, not an acoustic
    reading; it writes "So I was thinking that maybe." for someone who is
    plainly still talking. Trusting that period ends the turn early, which
    is the single worst failure this module can have.
    """
    assert classify("So I was thinking that maybe.") is Completeness.INCOMPLETE
    # A question mark is trustworthy and must still win.
    assert classify("What time is it?") is Completeness.COMPLETE


def test_silence_budget_adapts_to_completeness():
    ep = Endpointer()
    finished = ep.evaluate(
        transcript="What time is it?", silence_ms=400, speech_ms=2000, min_utterance_ms=220
    )
    midthought = ep.evaluate(
        transcript="I was thinking we could maybe",
        silence_ms=400,
        speech_ms=2000,
        min_utterance_ms=220,
    )
    assert finished.should_end is True
    # Same 400 ms pause, opposite decision — this is the whole point.
    assert midthought.should_end is False
    assert midthought.required_silence_ms > finished.required_silence_ms


def test_short_noise_burst_is_not_a_turn():
    ep = Endpointer()
    decision = ep.evaluate(
        transcript="hm", silence_ms=5000, speech_ms=80, min_utterance_ms=220
    )
    assert decision.should_end is False
    assert decision.reason == "below_min_utterance"


def test_turn_always_ends_eventually():
    ep = Endpointer()
    decision = ep.evaluate(
        transcript="and then I", silence_ms=5000, speech_ms=3000, min_utterance_ms=220
    )
    assert decision.should_end is True
    assert decision.reason == "max_silence"


# ── barge-in accounting ──────────────────────────────────────────────────


def test_spoken_prefix_reflects_only_what_played():
    """Her memory must hold what was heard, not what was sent."""
    track = _SpeakingTrack(intended="A B C. D E F. G H I.")
    track.chunks = [("A B C.", 1.0), ("D E F.", 1.0), ("G H I.", 1.0)]
    track.sent_duration_s = 3.0

    assert track.spoken_prefix(0.0) == ""
    assert track.spoken_prefix(1.0) == "A B C."
    assert track.spoken_prefix(2.0) == "A B C. D E F."
    assert track.spoken_prefix(3.0) == "A B C. D E F. G H I."

    # Mid-chunk interruption keeps a proportional share of the words.
    partial = track.spoken_prefix(1.5)
    assert partial.startswith("A B C.")
    assert "G H I." not in partial


def test_interruption_hands_the_unheard_tail_to_the_next_turn():
    record = SpokenRecord(
        intended="Yeah, I think so. The tricky part is the rest of this.",
        spoken="Yeah, I think so.",
        interrupted=True,
    )
    assert record.unheard == "The tricky part is the rest of this."

    bridge = MindBridge(session_id="t")
    bridge.record_spoken(record)
    effective = bridge._compose_effective_message("wait, what?")

    # The engine is told what the user did and did not hear...
    assert "did not hear" in effective
    assert "The tricky part" in effective
    # ...and the user's own words survive verbatim at the end.
    assert effective.endswith("wait, what?")


def test_uninterrupted_turn_adds_no_interruption_note():
    bridge = MindBridge(session_id="t")
    bridge.record_spoken(SpokenRecord(intended="All of it.", spoken="All of it.", interrupted=False))
    effective = bridge._compose_effective_message("next question")

    assert "did not hear" not in effective
    assert effective.endswith("next question")


def test_every_voice_turn_carries_the_spoken_length_directive():
    """Reply length *is* time-to-first-audio on this path.

    The governed turn returns one finished string, so nothing can be spoken
    until the last token is decoded. Dropping this directive silently costs
    seconds per reply, so it is pinned.
    """
    bridge = MindBridge(session_id="t", spoken_reply_words=45)
    effective = bridge._compose_effective_message("what do you think?")

    assert "spoken turn" in effective
    assert "45 words" in effective
    assert "No markdown" in effective
    # The user's own words still arrive last and verbatim.
    assert effective.endswith("what do you think?")


# ── echo rejection ───────────────────────────────────────────────────────


def test_echo_guard_rejects_her_own_words_returning():
    guard = EchoGuard()
    guard.note_spoken("The tricky part is that interruption handling has to edit what I said")
    verdict = guard.evaluate("the tricky part is that interruption handling has to edit")
    assert verdict.is_echo is True


def test_echo_guard_lets_a_real_interruption_through():
    guard = EchoGuard()
    guard.note_spoken("The tricky part is that interruption handling has to edit what I said")
    assert guard.evaluate("wait stop that is not what I meant").is_echo is False
    # Short interjections are exactly what a real barge-in looks like.
    assert guard.evaluate("no").is_echo is False


def test_echo_guard_is_inert_before_she_speaks():
    assert EchoGuard().evaluate("hello can you hear me").is_echo is False


# ── style control ────────────────────────────────────────────────────────


def test_delivery_requests_change_prosody():
    style = StyleController()
    assert style.observe("can you talk a bit slower")
    assert style.adjustment.rate_delta < 0
    assert style.observe("you're too loud")
    assert style.adjustment.gain_delta < 0


def test_topic_mention_of_speed_is_not_a_delivery_request():
    """"We should speed up the release" must not retune her voice."""
    style = StyleController()
    assert style.observe("we should speed up the release") == ""
    assert style.adjustment.active is False


def test_style_adjustments_stay_in_a_speakable_range():
    style = StyleController()
    for _ in range(20):
        style.observe("talk much faster")
    assert style.adjustment.rate_delta <= 0.28


# ── backchannels ─────────────────────────────────────────────────────────


def test_backchannel_needs_a_prosodic_boundary():
    reflex = BackchannelReflex(BackchannelConfig(fire_probability=1.0))
    reflex.on_user_turn_start(now=0.0)
    common = dict(speech_ms=6000.0, aura_is_speaking=False, now=100.0)

    # Too short to be a boundary — this is just inter-word silence.
    assert reflex.consider(silence_ms=50.0, **common).should_emit is False
    # Long enough to be an endpoint, not a boundary.
    assert reflex.consider(silence_ms=900.0, **common).should_emit is False
    # In the window.
    assert reflex.consider(silence_ms=250.0, **common).should_emit is True


def test_backchannel_never_talks_over_her_own_speech():
    reflex = BackchannelReflex(BackchannelConfig(fire_probability=1.0))
    reflex.on_user_turn_start(now=0.0)
    decision = reflex.consider(
        silence_ms=250.0, speech_ms=6000.0, aura_is_speaking=True, now=100.0
    )
    assert decision.should_emit is False
    assert decision.reason == "aura_speaking"


def test_backchannel_requires_the_user_to_have_held_the_floor():
    reflex = BackchannelReflex(BackchannelConfig(fire_probability=1.0))
    reflex.on_user_turn_start(now=0.0)
    decision = reflex.consider(
        silence_ms=250.0, speech_ms=500.0, aura_is_speaking=False, now=100.0
    )
    assert decision.should_emit is False


# ── fillers ──────────────────────────────────────────────────────────────


def test_filler_tiers_escalate_and_never_repeat_a_tier():
    reflex = FillerReflex()
    reflex.begin_turn()
    bounds = dict(first=380.0, second=1900.0, third=6500.0)

    assert reflex.due(100.0, **bounds) is None
    tier1 = reflex.due(400.0, **bounds)
    assert tier1 is not None and tier1.tier == 1
    assert reflex.due(500.0, **bounds) is None  # tier 1 already spent
    tier2 = reflex.due(2000.0, **bounds)
    assert tier2 is not None and tier2.tier == 2


def test_filler_words_follow_the_real_activity():
    """"Let me look that up" should mean a search is genuinely running."""
    reflex = FillerReflex()
    reflex.begin_turn()
    reflex.observe_activity("sovereign_browser")
    assert reflex.cause is ThinkingCause.WEB_SEARCH
    filler = reflex.due(2000.0, first=380.0, second=1900.0, third=6500.0)
    assert filler is not None
    assert filler.cause is ThinkingCause.WEB_SEARCH


# ── chunking ─────────────────────────────────────────────────────────────


def test_abbreviations_and_decimals_do_not_split_sentences():
    assert split_for_speech("Dr. Chen said it was 3.5 seconds. Then he left.", max_chars=200) == [
        "Dr. Chen said it was 3.5 seconds. Then he left."
    ]


def test_first_chunk_is_short_but_not_a_fragment():
    head, rest = first_chunk(
        "Yeah, I think that's basically right, though there's a wrinkle in it.", max_chars=48
    )
    assert len(head.split()) >= 2
    assert rest
    assert head.endswith(",") or head.endswith(".")


def test_streaming_chunker_preserves_word_boundaries():
    """Regression: a stripped remainder fused onto the next token.

    "median." + "Most" became "median.Most", which the TTS then pronounces
    as one mangled word.
    """
    chunker = StreamingChunker(first_max_chars=40, max_chars=90)
    out: list[str] = []
    for token in ["Okay, so ", "the answer is yes. ", "But it depends on the tail latency. ", "Most people say median."]:
        out += chunker.push(token)
    out += chunker.flush()
    joined = " ".join(out)
    assert "median.Most" not in joined
    assert ".M" not in joined.replace(". M", "")
    assert all(chunk.strip() for chunk in out)


# ── audio plumbing ───────────────────────────────────────────────────────


def test_pcm_roundtrip_preserves_signal():
    original = np.linspace(-0.9, 0.9, 480, dtype=np.float32)
    restored = pcm16_to_float32(float32_to_pcm16(original))
    assert restored.shape == original.shape
    assert np.max(np.abs(restored - original)) < 1e-3


def test_torn_frame_does_not_raise():
    """An odd trailing byte is a torn socket read, not a reason to die."""
    assert pcm16_to_float32(b"\x01\x02\x03").size == 1


def test_frame_splitter_emits_exact_frames_and_keeps_remainder():
    splitter = FrameSplitter(VAD_FRAME_SAMPLES)
    assert splitter.push(np.zeros(VAD_FRAME_SAMPLES - 10, dtype=np.float32)) == []
    frames = splitter.push(np.zeros(20, dtype=np.float32))
    assert len(frames) == 1
    assert frames[0].size == VAD_FRAME_SAMPLES
    assert splitter.pending_samples == 10


def test_utterance_buffer_keeps_preroll_so_the_first_word_survives():
    buffer = UtteranceBuffer(max_seconds=10, sample_rate=16000, preroll_ms=100)
    for _ in range(10):
        buffer.observe_silence(np.ones(160, dtype=np.float32))
    buffer.begin()
    # VAD always confirms speech a frame or two late; without preroll the
    # leading consonant is gone and Whisper mis-hears the first word.
    assert buffer.sample_count > 0


# ── protocol ─────────────────────────────────────────────────────────────


def test_audio_frame_roundtrip():
    payload = b"\x01\x02\x03\x04"
    frame = encode_audio(payload, opcode=AudioOpcode.BACKCHANNEL, seq=9, utterance_id=1234, last=True)
    opcode, last, seq, utterance, body = decode_audio(frame)
    assert opcode is AudioOpcode.BACKCHANNEL
    assert (last, seq, utterance, body) == (True, 9, 1234, payload)


def test_truncated_frame_is_rejected():
    with pytest.raises(ValueError):
        decode_audio(b"\x01\x02")


# ── ASR guards ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text", ["", "  ", "Thank you for watching!", "please subscribe", "♪", "[BLANK_AUDIO]"]
)
def test_silence_hallucinations_are_discarded(text):
    """Whisper emits these confidently on silence. Answering one has her
    talking to an empty room."""
    assert looks_hallucinated(text) is True


@pytest.mark.parametrize("text", ["hey can you hear me", "yes", "no", "stop"])
def test_real_speech_is_not_discarded(text):
    assert looks_hallucinated(text) is False


# ── mind bridge ──────────────────────────────────────────────────────────


# ── coqui compatibility shim ─────────────────────────────────────────────


def test_shim_restores_the_symbol_coqui_needs():
    """coqui-TTS imports a helper transformers 5.x removed.

    Pinning transformers back would satisfy the TTS package at the cost of
    the version mlx-lm and the resident 32B are built against — risking the
    mind to gain a voice option. The shim reinstates the one symbol instead.
    """
    from core.voice.duplex import coqui_compat

    assert coqui_compat.apply() is True
    import transformers.pytorch_utils as pytorch_utils

    assert hasattr(pytorch_utils, "isin_mps_friendly")


def test_shim_is_idempotent_and_does_not_clobber_a_real_symbol():
    import transformers.pytorch_utils as pytorch_utils

    from core.voice.duplex import coqui_compat

    coqui_compat.apply()
    sentinel = pytorch_utils.isin_mps_friendly
    coqui_compat._applied = False  # force a second pass
    coqui_compat.apply()
    assert pytorch_utils.isin_mps_friendly is sentinel


def test_shim_matches_torch_isin_semantics():
    import torch

    from core.voice.duplex.coqui_compat import _isin_mps_friendly

    elements = torch.tensor([1, 2, 3, 4, 5])
    test = torch.tensor([2, 4])
    assert torch.equal(_isin_mps_friendly(elements, test), torch.isin(elements, test))


def test_cloned_voice_refuses_without_licence_acceptance(monkeypatch, tmp_path):
    """XTTS-v2 is CPML-licensed; accepting on the operator's behalf is not
    this code's call, so it must fail closed with a reason."""
    from core.voice.duplex.config import TtsConfig
    from core.voice.duplex.tts_stream import _ClonedVoiceEngine

    for name in ("AURA_COQUI_CPML_ACCEPTED", "AURA_COQUI_COMMERCIAL_LICENSED", "COQUI_TOS_AGREED"):
        monkeypatch.delenv(name, raising=False)

    clip = tmp_path / "ref.wav"
    clip.write_bytes(b"RIFF")
    config = TtsConfig()
    config.clone_reference = str(clip)

    assert _ClonedVoiceEngine(config).load() is False


def test_responder_failure_returns_none_rather_than_inventing_a_reply():
    async def broken(transcript, *, effective_message, session_id, timeout_s):
        raise RuntimeError("cognition lane down")

    bridge = MindBridge(session_id="t", responder=broken)
    assert asyncio.run(bridge.respond("hello")) is None


def test_empty_transcript_never_reaches_cognition():
    calls: list[str] = []

    async def responder(transcript, *, effective_message, session_id, timeout_s):
        calls.append(transcript)
        return "hi"

    bridge = MindBridge(session_id="t", responder=responder)
    assert asyncio.run(bridge.respond("   ")) is None
    assert calls == []


# ── overlap: "mhm" is not an interruption ────────────────────────────────


def _drive_overlap(arbiter, *, speech_ms, then_silence_ms, energy=0.1):
    """Feed frames until a verdict settles."""
    from core.voice.duplex.overlap import OverlapVerdict as V

    frame = 32.0
    arbiter.begin()
    verdict = V.PENDING
    for _ in range(int(speech_ms / frame)):
        verdict = arbiter.observe(frame_ms=frame, is_speech=True, energy=energy)
        if verdict is not V.PENDING:
            return verdict
    for _ in range(int(then_silence_ms / frame)):
        verdict = arbiter.observe(frame_ms=frame, is_speech=False, energy=0.0)
        if verdict is not V.PENDING:
            return verdict
    return verdict


def test_short_acknowledgement_does_not_stop_her():
    """The defect this module exists to fix.

    Saying "mhm" while she talks previously killed her mid-sentence, which
    punishes the user for being a good listener.
    """
    from core.voice.duplex.overlap import OverlapArbiter, OverlapVerdict

    verdict = _drive_overlap(OverlapArbiter(), speech_ms=220, then_silence_ms=400)
    assert verdict is OverlapVerdict.BACKCHANNEL


def test_sustained_speech_takes_the_floor():
    from core.voice.duplex.overlap import OverlapArbiter, OverlapVerdict

    verdict = _drive_overlap(OverlapArbiter(), speech_ms=1200, then_silence_ms=0)
    assert verdict is OverlapVerdict.BARGE_IN


def test_ducking_happens_before_any_verdict():
    """Volume must drop while the decision is still pending — that is what
    makes the response instant without making it irreversible."""
    from core.voice.duplex.overlap import OverlapArbiter, OverlapVerdict

    arbiter = OverlapArbiter()
    arbiter.begin()
    ducked_at = None
    for i in range(20):
        verdict = arbiter.observe(frame_ms=32.0, is_speech=True, energy=0.1)
        if arbiter.should_duck() and ducked_at is None:
            ducked_at = (i + 1) * 32.0
            assert verdict is OverlapVerdict.PENDING
    assert ducked_at is not None and ducked_at <= 200.0


def test_duck_fires_only_once():
    from core.voice.duplex.overlap import OverlapArbiter

    arbiter = OverlapArbiter()
    arbiter.begin()
    ducks = 0
    for _ in range(30):
        arbiter.observe(frame_ms=32.0, is_speech=True, energy=0.1)
        if arbiter.should_duck():
            ducks += 1
    assert ducks == 1


@pytest.mark.parametrize("text", ["mhm", "yeah", "right", "okay", "haha", "yeah yeah"])
def test_acknowledgement_tokens_recognised(text):
    from core.voice.duplex.overlap import looks_like_backchannel

    assert looks_like_backchannel(text) is True


@pytest.mark.parametrize(
    "text", ["yeah but no", "wait that's wrong", "no I meant the other one", "stop talking"]
)
def test_real_objections_are_not_acknowledgement(text):
    from core.voice.duplex.overlap import looks_like_backchannel

    assert looks_like_backchannel(text) is False


def test_transcript_overrides_a_timing_misread():
    """A short sharp objection ("no—") has backchannel *timing*. The words
    are the tiebreaker."""
    from core.voice.duplex.overlap import OverlapArbiter, OverlapVerdict

    arbiter = OverlapArbiter()
    _drive_overlap(arbiter, speech_ms=220, then_silence_ms=400)
    assert arbiter.resolve("no wait that's wrong") is OverlapVerdict.BARGE_IN
    arbiter2 = OverlapArbiter()
    _drive_overlap(arbiter2, speech_ms=220, then_silence_ms=400)
    assert arbiter2.resolve("mhm") is OverlapVerdict.BACKCHANNEL


# ── paralinguistics ──────────────────────────────────────────────────────


def test_pitch_tracking_recovers_a_known_tone():
    from core.voice.duplex.paralinguistics import estimate_f0

    t = np.arange(16000 * 0.5) / 16000
    tone = (0.3 * np.sin(2 * np.pi * 150.0 * t)).astype(np.float32)
    voiced = estimate_f0(tone, 16000)
    voiced = voiced[~np.isnan(voiced)]
    assert voiced.size > 5
    assert abs(float(np.median(voiced)) - 150.0) < 8.0


def test_delivery_stays_quiet_without_a_baseline():
    """Reporting a mood from an absolute number is invention. With fewer
    than three samples there is no baseline, so there is nothing to say."""
    from core.voice.duplex.paralinguistics import SpeakerBaseline, analyze, interpret

    audio = (0.1 * np.sin(2 * np.pi * 140 * np.arange(16000) / 16000)).astype(np.float32)
    reading = interpret(analyze(audio, 16000, word_count=4), SpeakerBaseline())
    assert reading.as_context() == ""


def test_imperceptible_change_is_not_reported():
    """A tiny baseline variance makes an inaudible difference score many
    sigma; without a perceptibility floor she announces "quieter than usual"
    on a turn that sounded identical."""
    from core.voice.duplex.paralinguistics import SpeakerBaseline, VoiceSignature

    baseline = SpeakerBaseline()
    for value in (0.100, 0.101, 0.099, 0.100):
        sig = VoiceSignature(energy_rms=value, duration_s=2.0, voiced_ratio=0.6)
        baseline.observe(sig)
    nearly_identical = VoiceSignature(energy_rms=0.102, duration_s=2.0, voiced_ratio=0.6)
    assert baseline.energy_z(nearly_identical) == 0.0


def test_convergence_is_partial_not_mimicry():
    """Full mirroring reads as mockery; none at all is the flat-affect
    problem. Bounded partial movement is the point."""
    from core.voice.duplex.paralinguistics import DeliveryReading, convergence_factors

    speed, gain = convergence_factors(DeliveryReading(rate_z=5.0, energy_z=5.0))
    assert 1.0 < speed <= 1.15
    assert 1.0 < gain <= 1.2
    slow_speed, _ = convergence_factors(DeliveryReading(rate_z=-5.0, energy_z=-5.0))
    assert 0.85 <= slow_speed < 1.0


def test_neutral_delivery_leaves_her_voice_alone():
    from core.voice.duplex.paralinguistics import DeliveryReading, convergence_factors

    assert convergence_factors(DeliveryReading()) == (1.0, 1.0)


# ── adaptive reply length ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "question,ceiling",
    [
        ("is the build green?", 25),
        ("what time is it", 25),
        ("how many are left", 25),
    ],
)
def test_closed_questions_get_short_answers(question, ceiling):
    bridge = MindBridge(session_id="t", spoken_reply_words=45)
    words, offer = bridge._reply_budget_for(question)
    assert words <= ceiling
    assert offer is False


def test_explanatory_questions_get_room_and_an_offer():
    bridge = MindBridge(session_id="t", spoken_reply_words=45)
    words, offer = bridge._reply_budget_for("explain the tradeoff between the two")
    assert words >= 80
    assert offer is True


def test_conversational_default_is_unchanged():
    bridge = MindBridge(session_id="t", spoken_reply_words=45)
    words, offer = bridge._reply_budget_for("so I was thinking about the queue again")
    assert words == 45
    assert offer is False


# ── predictive fillers ───────────────────────────────────────────────────


def test_known_slow_cause_announces_itself_immediately():
    """"Let me look that up" at 300ms beats "uh…" then the same sentence at
    1.9s — knowing *why* she is slow is better than knowing *that* she is."""
    reflex = FillerReflex()
    reflex.begin_turn()
    reflex.observe_activity("sovereign_browser")

    filler = reflex.due(50.0, first=380.0, second=1900.0, third=6500.0)
    assert filler is not None
    assert filler.tier == 2
    assert filler.cause is ThinkingCause.WEB_SEARCH
    # And she must not then say "uh…" after already explaining herself.
    assert reflex.due(500.0, first=380.0, second=1900.0, third=6500.0) is None


def test_unknown_cause_still_waits_its_turn():
    reflex = FillerReflex()
    reflex.begin_turn()
    assert reflex.due(50.0, first=380.0, second=1900.0, third=6500.0) is None
    assert reflex.due(400.0, first=380.0, second=1900.0, third=6500.0) is not None


# ── streaming carve-out: the safety boundary ─────────────────────────────


@pytest.mark.parametrize(
    "question",
    [
        "what do you think about that",
        "how are you feeling today",
        "that's interesting, tell me more",
        "do you agree with me",
    ],
)
def test_conversational_turns_may_stream(question):
    from core.voice.duplex.streaming_reply import is_streamable

    assert is_streamable(question).ok is True


@pytest.mark.parametrize(
    "question",
    [
        "look up the release date",
        "how many tests are failing",
        "what did the log say",
        "run the build",
        "delete that file",
        "cite your source for that",
        "what's your memory usage",
        "show me the code for the parser",
        "what happened in 1996",
    ],
)
def test_evidence_critical_turns_never_stream(question):
    """Streaming speaks before validation. These are exactly the turns where
    that matters, so they take the fully governed buffered path."""
    from core.voice.duplex.streaming_reply import is_streamable

    verdict = is_streamable(question)
    assert verdict.ok is False, f"{question!r} must not stream ({verdict.reason})"


def test_eligibility_fails_closed_on_empty():
    from core.voice.duplex.streaming_reply import is_streamable

    assert is_streamable("").ok is False
    assert is_streamable("   ").ok is False


@pytest.mark.parametrize(
    "clause",
    [
        "[spoken turn: answer in 45 words]",
        "[voice context: the user interrupted]",
        "System: you are Aura",
        "As an AI language model, I cannot",
        "### Heading",
    ],
)
def test_prompt_scaffolding_is_never_spoken(clause):
    from core.voice.duplex.streaming_reply import ClauseValidator

    assert ClauseValidator().check(clause).ok is False


@pytest.mark.parametrize(
    "clause",
    ["```python", "- first bullet", "1. first item", "# Title"],
)
def test_written_structure_is_rejected(clause):
    """Markdown read aloud is a monotone run of fragments; headings vanish
    entirely. If the model switches to writing, stop streaming."""
    from core.voice.duplex.streaming_reply import ClauseValidator

    assert ClauseValidator().check(clause).ok is False


def test_repetition_loop_is_caught():
    """The classic local-model failure. Speaking it aloud is worse than any
    latency it would have saved."""
    from core.voice.duplex.streaming_reply import ClauseValidator

    validator = ClauseValidator()
    assert validator.check("I think so.").ok is True
    assert validator.check("I think so.").ok is True
    assert validator.check("I think so.").ok is False


def test_ordinary_spoken_clauses_pass():
    from core.voice.duplex.streaming_reply import ClauseValidator

    validator = ClauseValidator()
    for clause in ("Yeah, I think so.", "The tricky part is the overlap.", "Right."):
        assert validator.check(clause).ok is True


def test_overlong_clause_is_rejected():
    from core.voice.duplex.streaming_reply import ClauseValidator

    assert ClauseValidator().check("word " * 200).ok is False
