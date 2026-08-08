"""The compatibility facade claimed four things it did not do.

CP126 on core/brain/local_llm.py, all four critical:

* ``check_health`` excluded exactly two lane states, so a cold lane with no
  model resident, one still spawning, one mid-handshake, one warming, one
  recovering from a kill and one already closed all reported healthy;
* ``generate``/``chat`` called ``_record_success`` the moment the unified
  engine RETURNED. That engine reports failures in-band, so an explicitly
  failed generation reset the failure streak and healed the circuit breaker
  — which therefore could not trip on the one failure mode it exists for;
* ``self.timeout`` was read from config and never passed, wrapped, or
  turned into a deadline. The advertised bound was a field on an object;
* both "stream" methods checked cancellation once, before calling the
  buffered path, then yielded the finished answer as one chunk. No
  incremental text, and a cancel raised during generation reached nothing.
"""
from __future__ import annotations

import asyncio

import pytest

from core.brain.local_llm import LocalBrain


@pytest.fixture
def brain():
    instance = LocalBrain.__new__(LocalBrain)
    instance.model = "test-model"
    instance.timeout = 5.0
    instance._consecutive_failures = 0
    instance._circuit_open = False
    instance._circuit_open_until = 0.0
    return instance


# --------------------------------------------------------------- health


@pytest.mark.parametrize(
    "state",
    ["cold", "spawning", "handshaking", "warming", "recovering", "closed", "fenced"],
)
def test_a_lane_that_cannot_serve_a_turn_is_not_healthy(brain, monkeypatch, state):
    monkeypatch.setattr(brain, "_lane_state", lambda: state)
    assert brain.check_health() is False, (
        f"a lane in state {state!r} reported healthy; a caller routing a turn "
        "here waits for a token that cannot come"
    )


def test_a_ready_lane_is_healthy(brain, monkeypatch):
    monkeypatch.setattr(brain, "_lane_state", lambda: "ready")
    assert brain.check_health() is True


@pytest.mark.parametrize("state", ["failed", "retired", "closed"])
def test_a_terminal_lane_cannot_become_ready(brain, monkeypatch, state):
    monkeypatch.setattr(brain, "_lane_state", lambda: state)
    assert brain.can_become_ready() is False


@pytest.mark.parametrize("state", ["cold", "warming", "recovering"])
def test_a_loadable_lane_can_become_ready(brain, monkeypatch, state):
    """The distinction the single predicate was hiding."""
    monkeypatch.setattr(brain, "_lane_state", lambda: state)
    assert brain.can_become_ready() is True
    assert brain.check_health() is False


def test_an_unreachable_client_fails_closed_both_ways(brain, monkeypatch):
    monkeypatch.setattr(brain, "_lane_state", lambda: "")
    assert brain.check_health() is False
    assert brain.can_become_ready() is False


# ------------------------------------------------- outcomes feed the breaker


def test_an_in_band_failure_is_not_a_success(brain):
    assert brain._is_usable_result({"response": "", "error": "model_unavailable"}) is False


def test_an_empty_response_is_not_a_success(brain):
    assert brain._is_usable_result({"response": "   ", "thought": "hmm"}) is False


def test_a_non_dict_is_not_a_success(brain):
    assert brain._is_usable_result("a string") is False
    assert brain._is_usable_result(None) is False


def test_a_real_answer_is_a_success(brain):
    assert brain._is_usable_result({"response": "Here is the answer."}) is True


def test_repeated_in_band_failures_open_the_circuit(brain, monkeypatch):
    """The whole point of the breaker, and what the bug disabled.

    Five failed generations in a row must open it. Before the fix each of
    these RESET the streak, because the engine returned rather than raised.
    """

    class _Engine:
        async def generate_unified(self, **kwargs):
            return {"response": "", "error": "worker_died"}

    monkeypatch.setattr(
        "core.brain.unified_inference.UnifiedInferenceEngine", lambda: _Engine()
    )

    async def _drive():
        for _ in range(5):
            await brain.generate("anything")

    asyncio.run(_drive())

    assert brain._consecutive_failures >= 5
    assert brain._circuit_open is True, (
        "five consecutive failed generations left the breaker closed; it "
        "cannot trip on inference that keeps answering, emptily"
    )


def test_a_good_generation_clears_the_streak(brain, monkeypatch):
    class _Engine:
        async def generate_unified(self, **kwargs):
            return {"response": "an actual answer", "thought": ""}

    monkeypatch.setattr(
        "core.brain.unified_inference.UnifiedInferenceEngine", lambda: _Engine()
    )
    brain._consecutive_failures = 3

    asyncio.run(brain.generate("anything"))

    assert brain._consecutive_failures == 0


# ----------------------------------------------------------- the deadline


def test_a_generation_that_overruns_the_timeout_is_abandoned(brain, monkeypatch):
    """`self.timeout` was read from config and never used anywhere."""

    class _Engine:
        async def generate_unified(self, **kwargs):
            await asyncio.sleep(30)
            return {"response": "far too late"}

    monkeypatch.setattr(
        "core.brain.unified_inference.UnifiedInferenceEngine", lambda: _Engine()
    )
    brain.timeout = 0.05

    result = asyncio.run(brain.generate("anything"))

    assert result["error"] == "internal_mlx_timeout"
    assert result["response"] == ""


def test_the_timeout_counts_as_a_failure(brain, monkeypatch):
    class _Engine:
        async def generate_unified(self, **kwargs):
            await asyncio.sleep(30)

    monkeypatch.setattr(
        "core.brain.unified_inference.UnifiedInferenceEngine", lambda: _Engine()
    )
    brain.timeout = 0.05

    asyncio.run(brain.generate("anything"))

    assert brain._consecutive_failures == 1


def test_chat_is_bounded_by_the_same_deadline(brain, monkeypatch):
    """Two entry points, one enforcement point; the bug was in both."""

    class _Engine:
        async def generate_unified(self, **kwargs):
            await asyncio.sleep(30)

    monkeypatch.setattr(
        "core.brain.unified_inference.UnifiedInferenceEngine", lambda: _Engine()
    )
    brain.timeout = 0.05

    result = asyncio.run(brain.chat([{"role": "user", "content": "hi"}]))

    assert result["error"] == "internal_mlx_timeout"


@pytest.mark.parametrize("bad", [0, -1, None, "soon", float("nan"), float("inf")])
def test_an_unusable_timeout_does_not_invent_a_number(brain, bad):
    """"Unspecified" stays unspecified; the engine's own bound applies."""
    brain.timeout = bad
    assert brain._deadline_seconds() == 0.0


def test_a_usable_timeout_is_the_configured_one(brain):
    brain.timeout = 42.5
    assert brain._deadline_seconds() == 42.5


# ------------------------------------------------------------- streaming


def _collect(agen) -> list[str]:
    async def _run():
        return [chunk async for chunk in agen]

    return asyncio.run(_run())


class _Nucleus:
    """A streaming lane that records whether it was stopped."""

    def __init__(self, chunks, closed_flag):
        self._chunks = chunks
        self._closed = closed_flag

    async def generate_stream_async(self, prompt, system_prompt=None, **kwargs):
        try:
            for chunk in self._chunks:
                yield chunk
                await asyncio.sleep(0)
        finally:
            self._closed["stopped"] = True


def test_the_stream_delivers_incremental_chunks(brain, monkeypatch):
    """It yielded the completed generation as a single chunk."""
    closed = {"stopped": False}
    monkeypatch.setattr(
        LocalBrain, "_streaming_lane",
        staticmethod(lambda: _Nucleus(["Hel", "lo ", "there"], closed)),
    )

    chunks = _collect(brain.generate_text_stream_async("hi"))

    assert chunks == ["Hel", "lo ", "there"], (
        "the generation arrived whole; there is no stream, only a delayed "
        "single yield"
    )


def test_cancellation_midflight_stops_the_model(brain, monkeypatch):
    """Cancellation was checked once, before generation, and never again."""
    closed = {"stopped": False}
    cancel = asyncio.Event()

    class _SlowNucleus:
        async def generate_stream_async(self, prompt, system_prompt=None, **kwargs):
            try:
                for index in range(100):
                    if index == 2:
                        cancel.set()
                    yield f"chunk{index}"
                    await asyncio.sleep(0)
            finally:
                closed["stopped"] = True

    monkeypatch.setattr(
        LocalBrain, "_streaming_lane", staticmethod(lambda: _SlowNucleus())
    )

    chunks = _collect(brain.generate_text_stream_async("hi", cancel_event=cancel))

    assert len(chunks) < 100, "cancellation during generation was ignored"
    assert closed["stopped"] is True, (
        "the stream was abandoned without closing it, so the resident model "
        "keeps generating into a queue nobody reads"
    )


def test_cancellation_before_the_first_token_yields_nothing(brain, monkeypatch):
    cancel = asyncio.Event()
    cancel.set()
    closed = {"stopped": False}
    monkeypatch.setattr(
        LocalBrain, "_streaming_lane", staticmethod(lambda: _Nucleus(["never"], closed))
    )

    assert _collect(brain.generate_text_stream_async("hi", cancel_event=cancel)) == []


def test_no_streaming_lane_still_answers(brain, monkeypatch):
    """One chunk is a poor stream but a correct answer; a refusal is not."""

    monkeypatch.setattr(LocalBrain, "_streaming_lane", staticmethod(lambda: None))

    class _Engine:
        async def generate_unified(self, **kwargs):
            return {"response": "buffered answer", "thought": ""}

    monkeypatch.setattr(
        "core.brain.unified_inference.UnifiedInferenceEngine", lambda: _Engine()
    )

    assert _collect(brain.generate_text_stream_async("hi")) == ["buffered answer"]


def test_chat_streaming_uses_the_buffered_path_and_still_honours_cancel(
    brain, monkeypatch
):
    """The nucleus stream is prompt-shaped; a message list falls back."""
    cancel = asyncio.Event()

    class _Engine:
        async def generate_unified(self, **kwargs):
            cancel.set()
            return {"response": "answer nobody will see", "thought": ""}

    monkeypatch.setattr(
        "core.brain.unified_inference.UnifiedInferenceEngine", lambda: _Engine()
    )

    chunks = _collect(
        brain.chat_stream_async(
            [{"role": "user", "content": "hi"}], cancel_event=cancel
        )
    )

    assert chunks == [], (
        "cancellation raised while the buffered call was in flight was "
        "ignored, and the abandoned answer was delivered anyway"
    )


def test_a_streaming_error_still_reaches_the_caller(brain, monkeypatch):
    monkeypatch.setattr(LocalBrain, "_streaming_lane", staticmethod(lambda: None))

    class _Engine:
        async def generate_unified(self, **kwargs):
            return {"response": "", "error": "worker_died"}

    monkeypatch.setattr(
        "core.brain.unified_inference.UnifiedInferenceEngine", lambda: _Engine()
    )

    chunks = _collect(brain.generate_text_stream_async("hi"))

    assert any("worker_died" in chunk for chunk in chunks)
