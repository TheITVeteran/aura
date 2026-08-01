"""AGI integration: a federated body, a false-running service, and a fabricated
head vector."""
from __future__ import annotations

import asyncio

import pytest

import core.agi.agi_integration as agi

pytestmark = pytest.mark.unit


# ── one body, not a federation ─────────────────────────────────────────────


def test_canonical_organs_are_reused_not_reconstructed(monkeypatch):
    """Constructing private instances forked the system: inference feedback and
    runtime telemetry accumulated in two places, so whichever one a consumer
    held saw a different picture of the same body."""
    canonical_mod = object()
    canonical_fb = object()
    monkeypatch.setattr(
        agi.ServiceContainer, "get",
        staticmethod(lambda name, default=None: {
            "homeostatic_modulator": canonical_mod,
            "inference_feedback_loop": canonical_fb,
        }.get(name, default)),
    )

    layer = agi.AGIIntegrationLayer()

    assert layer.modulator is canonical_mod
    assert layer.feedback_loop is canonical_fb


def test_a_created_organ_is_published_so_only_one_fork_can_exist(monkeypatch):
    """If nothing is registered we must build one — but publish it, so the NEXT
    consumer lands on the same instance instead of forking again."""
    registered = {}
    monkeypatch.setattr(
        agi.ServiceContainer, "get",
        staticmethod(lambda name, default=None: registered.get(name, default)),
    )
    monkeypatch.setattr(
        agi.ServiceContainer, "register_instance",
        staticmethod(lambda name, instance, required=False: registered.__setitem__(name, instance)),
    )

    layer = agi.AGIIntegrationLayer()

    assert registered.get("homeostatic_modulator") is layer.modulator
    assert registered.get("inference_feedback_loop") is layer.feedback_loop


# ── start must not advertise a service it failed to start ──────────────────


def test_failed_start_leaves_the_layer_stopped(monkeypatch):
    """The running flag and container registration were set BEFORE the task
    existed, so any failure left the instance advertised as running with no
    loop — permanently, since a later start() returns early on that flag."""
    layer = agi.AGIIntegrationLayer()

    def _boom(*a, **k):
        raise RuntimeError("registration refused")

    monkeypatch.setattr(agi.ServiceContainer, "register", staticmethod(_boom))

    with pytest.raises(RuntimeError):
        asyncio.run(layer.start())

    assert layer._running is False, "a failed start must not report running"
    assert layer._loop_task is None


def test_a_failed_start_can_be_retried(monkeypatch):
    """The consequence that made the flag bug permanent."""
    layer = agi.AGIIntegrationLayer()
    calls = {"n": 0}

    def _flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")

    monkeypatch.setattr(agi.ServiceContainer, "register", staticmethod(_flaky))

    with pytest.raises(RuntimeError):
        asyncio.run(layer.start())
    asyncio.run(layer.start())
    try:
        assert layer._running is True
    finally:
        asyncio.run(layer.stop())


# ── stop must join before saving ───────────────────────────────────────────


def test_stop_awaits_the_tick_before_the_final_save():
    """stop() cancelled, dropped the reference, and saved immediately — so an
    in-flight tick could still be mutating the state being written."""
    layer = agi.AGIIntegrationLayer()
    saved_while_running = {}

    async def _slow_tick():
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            saved_while_running["tick_ended"] = True
            raise

    async def _run():
        layer._running = True
        layer._loop_task = asyncio.ensure_future(_slow_tick())
        await asyncio.sleep(0)
        original = layer._save_projection_weights
        layer._save_projection_weights = lambda: saved_while_running.setdefault(
            "saved_after_tick", saved_while_running.get("tick_ended", False)
        )
        try:
            await layer.stop()
        finally:
            layer._save_projection_weights = original

    asyncio.run(_run())

    assert saved_while_running.get("saved_after_tick") is True


def test_stop_tolerates_a_task_handle_that_is_not_a_real_task():
    """The tracker may return a stub or wrapper; shutdown must not depend on
    its exact shape."""
    from types import SimpleNamespace

    layer = agi.AGIIntegrationLayer()
    layer._running = True
    layer._loop_task = SimpleNamespace()  # no done(), no cancel()

    asyncio.run(layer.stop())  # must not raise

    assert layer._running is False


# ── a fallback must not assert an architecture it never read ───────────────


def test_fallback_head_weights_are_not_fabricated(monkeypatch):
    """A hardcoded 32-element vector is shape-incompatible with any model that
    does not have 32 heads — and it LOOKS usable, so it reaches the steering
    path."""
    monkeypatch.setattr(agi.ServiceContainer, "get",
                        staticmethod(lambda name, default=None: default))
    layer = agi.AGIIntegrationLayer()

    assert layer._neutral_head_weights() is None


def test_fallback_head_weights_use_the_real_head_count(monkeypatch):
    """When the count IS discoverable, use it rather than declining."""
    class _Registry:
        active_config = {"num_attention_heads": 40}

    monkeypatch.setattr(
        agi.ServiceContainer, "get",
        staticmethod(lambda name, default=None: _Registry()
                     if name == "model_registry" else default),
    )
    layer = agi.AGIIntegrationLayer()

    weights = layer._neutral_head_weights()

    assert weights is not None and weights.shape == (40,)


def test_absurd_head_counts_are_refused(monkeypatch):
    class _Registry:
        active_config = {"num_attention_heads": 10_000}

    monkeypatch.setattr(
        agi.ServiceContainer, "get",
        staticmethod(lambda name, default=None: _Registry()
                     if name == "model_registry" else default),
    )

    assert agi.AGIIntegrationLayer()._neutral_head_weights() is None
