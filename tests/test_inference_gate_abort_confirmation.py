"""A requested abort is not a confirmed stop without supervision evidence."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.brain.inference_gate import InferenceGate, _generation_actually_stopped


@pytest.mark.parametrize(
    ("client", "expected"),
    [
        (SimpleNamespace(), None),
        (SimpleNamespace(get_supervision_status=lambda: {}), None),
        (SimpleNamespace(get_supervision_status=lambda: {"active_generations": 1}), False),
        (SimpleNamespace(get_supervision_status=lambda: {"active_generations": 0}), True),
    ],
)
def test_generation_stop_confirmation_is_tristate(client, expected) -> None:
    assert _generation_actually_stopped(client) is expected


def _gate_with(client, monkeypatch: pytest.MonkeyPatch) -> InferenceGate:
    gate = InferenceGate.__new__(InferenceGate)
    gate._mlx_client = client
    monkeypatch.setattr(gate, "_iter_local_clients", lambda: {})
    return gate


def test_abort_without_supervision_is_not_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = SimpleNamespace(
        model_path="test-model",
        force_abort_active_generation=lambda **_kwargs: True,
    )
    gate = _gate_with(client, monkeypatch)

    assert gate.force_abort_active_generation("watchdog") == 0


def test_abort_with_explicit_zero_active_is_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = SimpleNamespace(
        model_path="test-model",
        force_abort_active_generation=lambda **_kwargs: True,
        get_supervision_status=lambda: {"active_generations": 0},
    )
    gate = _gate_with(client, monkeypatch)

    assert gate.force_abort_active_generation("watchdog") == 1
