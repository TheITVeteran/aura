"""Unified inference: option collisions, process-random 'token ids', no deadline."""
from __future__ import annotations

import asyncio

import pytest

import core.brain.unified_inference as ui

pytestmark = pytest.mark.unit


# ── caller options must not collide with explicit arguments ────────────────


@pytest.mark.parametrize("reserved", sorted(ui._RESERVED_CALL_ARGS))
def test_reserved_option_names_are_dropped(reserved):
    """Options are splatted as **final_options AFTER explicit
    messages/max_tokens/foreground_request/origin, so a caller passing any of
    those names produced a 'multiple values for argument' TypeError from deep
    inside the client — a request-shape error surfacing as a crash."""
    assert reserved in ui._RESERVED_CALL_ARGS


def test_reserved_names_cover_the_explicit_call_arguments():
    """If the call site gains an explicit argument, it must be reserved too."""
    import inspect

    source = inspect.getsource(ui.UnifiedInferenceEngine.generate_unified)
    call = source.split("generate_text_async(", 1)[1].split("timeout=", 1)[0]
    explicit = {
        line.split("=", 1)[0].strip()
        for line in call.splitlines()
        if "=" in line and not line.strip().startswith(("#", "**"))
    }
    explicit = {name for name in explicit if name.isidentifier()}

    missing = explicit - ui._RESERVED_CALL_ARGS
    assert not missing, f"explicit call args not reserved: {missing}"


# ── surrogate ids must at least be stable ──────────────────────────────────


def test_lexical_fallback_ids_are_deterministic():
    """Python's hash() is salted per process, so the same word produced a
    different 'token id' on every run and none corresponded to anything in the
    vocabulary — the homeostatic loop was fed noise that looked like
    measurement."""
    engine = ui.UnifiedInferenceEngine()
    captured = []
    engine.feedback_loop = type(
        "FB", (), {"process_output": lambda self, **kw: (
            captured.append(kw["token_ids"]),
            {"surprise": 0.0, "coherence": 0.0},
        )[1]}
    )()

    for _ in range(2):
        engine._process_feedback("the quick brown fox", modulation=None)

    assert captured[0] == captured[1], "surrogate ids must be stable"
    assert all(isinstance(i, int) for i in captured[0])


def test_lexical_fallback_is_bounded():
    engine = ui.UnifiedInferenceEngine()
    captured = []
    engine.feedback_loop = type(
        "FB", (), {"process_output": lambda self, **kw: (
            captured.append(kw["token_ids"]),
            {"surprise": 0.0, "coherence": 0.0},
        )[1]}
    )()

    engine._process_feedback(" ".join(["word"] * 10_000), modulation=None)

    assert len(captured[0]) <= ui._MAX_FALLBACK_TOKENS


# ── a request must be bounded in wall-clock time ───────────────────────────


def test_request_deadline_is_bounded_and_sane(monkeypatch):
    """max_tokens was the ONLY bound, so a wedged generation held the lane
    indefinitely with no cancellation point."""
    monkeypatch.delenv("AURA_UNIFIED_INFERENCE_TIMEOUT_S", raising=False)
    assert 5.0 <= ui._request_deadline_s() <= 3600.0

    monkeypatch.setenv("AURA_UNIFIED_INFERENCE_TIMEOUT_S", "0.001")
    assert ui._request_deadline_s() >= 5.0, "an absurd override must not disable bounding"

    monkeypatch.setenv("AURA_UNIFIED_INFERENCE_TIMEOUT_S", "not-a-number")
    assert ui._request_deadline_s() == 300.0


def test_a_hung_generation_raises_rather_than_hanging(monkeypatch):
    engine = ui.UnifiedInferenceEngine()
    monkeypatch.setenv("AURA_UNIFIED_INFERENCE_TIMEOUT_S", "5")
    monkeypatch.setattr(ui, "_request_deadline_s", lambda: 0.05)

    class _HungClient:
        async def generate_text_async(self, *a, **k):
            await asyncio.sleep(30)

    monkeypatch.setattr("core.brain.llm.mlx_client.get_mlx_client",
                        lambda **kw: _HungClient())
    monkeypatch.setattr("core.brain.llm.model_registry.get_lane_context_window",
                        lambda name: 4096)
    monkeypatch.setattr("core.brain.llm.model_registry.get_lane_runtime_model_path",
                        lambda name: "/tmp/model")

    with pytest.raises(RuntimeError, match="deadline_exceeded"):
        asyncio.run(engine.generate_unified("hello"))
