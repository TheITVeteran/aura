"""When the thing that would have said no cannot answer, do not proceed.

Two CP126 findings where an unavailable guard read as a passed guard.

21c6730b — an ImportError in the crash-loop breaker returned "no blocker", so a
crash storm kept respawning workers unchecked. The absence of the breaker
module is not the absence of evidence: this process records every worker death
it sees.

aa66b0ac — the background quiet policy was read off a PRIVATE inference-gate
method, and any lookup or runtime failure returned "no deferral", letting a
stale background request respawn a worker that had just been unloaded to
protect a user turn.
"""
from __future__ import annotations

import time

import pytest

from core.brain.llm import mlx_client

pytestmark = pytest.mark.unit


class _Client:
    def __init__(self, model_path="/models/aura-32b"):
        self.model_path = model_path


@pytest.fixture(autouse=True)
def _clean_ledger():
    with mlx_client._LOCAL_DEATH_LEDGER_LOCK:
        mlx_client._LOCAL_DEATH_LEDGER.clear()
    yield
    with mlx_client._LOCAL_DEATH_LEDGER_LOCK:
        mlx_client._LOCAL_DEATH_LEDGER.clear()


# --- an absent breaker still has this process's own evidence (21c6730b) --


def test_a_quiet_lane_is_not_blocked():
    assert mlx_client._local_crash_loop_block(_Client()) is None


def test_a_burst_of_deaths_blocks_the_next_spawn():
    for _ in range(mlx_client._LOCAL_CRASH_LOOP_DEATHS):
        mlx_client._note_local_worker_death("/models/aura-32b")

    blocked = mlx_client._local_crash_loop_block(_Client())

    assert blocked and "local_crash_loop" in blocked


def test_old_deaths_fall_out_of_the_window():
    stale = time.time() - (mlx_client._LOCAL_CRASH_LOOP_WINDOW_S + 60.0)
    with mlx_client._LOCAL_DEATH_LEDGER_LOCK:
        mlx_client._LOCAL_DEATH_LEDGER["/models/aura-32b"] = [stale] * 10

    assert mlx_client._local_crash_loop_block(_Client()) is None


def test_deaths_are_counted_per_model():
    for _ in range(mlx_client._LOCAL_CRASH_LOOP_DEATHS):
        mlx_client._note_local_worker_death("/models/aura-32b")

    assert mlx_client._local_crash_loop_block(_Client("/models/other")) is None


def test_the_ledger_is_bounded():
    for _ in range(200):
        mlx_client._note_local_worker_death("/models/aura-32b")

    with mlx_client._LOCAL_DEATH_LEDGER_LOCK:
        assert len(mlx_client._LOCAL_DEATH_LEDGER["/models/aura-32b"]) <= 16


def test_a_missing_breaker_module_consults_the_local_ledger(monkeypatch):
    """It used to return None — an unchecked respawn during a storm."""
    import builtins

    for _ in range(mlx_client._LOCAL_CRASH_LOOP_DEATHS):
        mlx_client._note_local_worker_death("/models/aura-32b")

    real_import = builtins.__import__

    def _no_breaker(name, *args, **kwargs):
        if name == "core.runtime.lane_reconciler":
            raise ImportError("absent in this build")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_breaker)

    blocked = mlx_client._crash_loop_blocks_worker_spawn(_Client())

    assert blocked and "local_crash_loop" in blocked


# --- a gate that cannot be asked means defer (aa66b0ac) -----------------


class _Gate:
    def __init__(self, reason=None, raises=None):
        self._reason = reason
        self._raises = raises

    def background_local_deferral_reason(self, *, origin=None):
        if self._raises:
            raise self._raises
        return self._reason


class _GateWithoutReader:
    pass


def _install_gate(monkeypatch, gate):
    from core.container import ServiceContainer

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        classmethod(
            lambda cls, name, default=None: gate if name == "inference_gate" else default
        ),
    )


def test_no_gate_at_all_is_not_a_deferral(monkeypatch):
    """A build without the gate has no background policy to honour."""
    _install_gate(monkeypatch, None)

    assert mlx_client._background_deferral_active("background_sweep") is None


def test_a_quiet_gate_permits_background_work(monkeypatch):
    _install_gate(monkeypatch, _Gate(reason=None))

    assert mlx_client._background_deferral_active("background_sweep") is None


def test_a_deferring_gate_is_honoured(monkeypatch):
    _install_gate(monkeypatch, _Gate(reason="foreground_reserved"))

    assert mlx_client._background_deferral_active("background_sweep") == (
        "foreground_reserved"
    )


def test_a_gate_that_cannot_be_read_defers(monkeypatch):
    """The gate's PRESENCE says this runtime has a background policy; being
    unable to read it is a reason to defer, not to proceed."""
    _install_gate(monkeypatch, _GateWithoutReader())

    assert mlx_client._background_deferral_active("background_sweep") == (
        "background_deferral_policy_unreadable"
    )


def test_a_gate_that_raises_defers(monkeypatch):
    _install_gate(monkeypatch, _Gate(raises=RuntimeError("gate is wedged")))

    assert mlx_client._background_deferral_active("background_sweep") == (
        "background_deferral_policy_unavailable"
    )


def test_the_public_reader_exists_on_the_gate():
    """The boundary reached into a private method; a rename there would have
    silently turned this check into 'no deferral'."""
    from core.brain.inference_gate import InferenceGate

    assert callable(InferenceGate.background_local_deferral_reason)
