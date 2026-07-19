from __future__ import annotations

from pathlib import Path

import pytest

from tools import verify_recurrence_v2_smoke as verifier


def _activation(*, active: bool, calls: int, positions: int) -> dict[str, object]:
    return {
        "schema": "aura.recurrence_adapter_activation.v1",
        "scope": "latent_slots_only",
        "active": active,
        "calls": calls,
        "adapted_positions": positions,
        "observed_positions": positions,
    }


def test_activation_contract_distinguishes_base_and_adapter_arms() -> None:
    assert verifier._activation(
        _activation(active=False, calls=0, positions=0),
        expected_active=False,
    )["active"] is False
    assert verifier._activation(
        _activation(active=True, calls=4, positions=8),
        expected_active=True,
    )["adapted_positions"] == 8


@pytest.mark.parametrize(
    ("value", "expected_active", "reason"),
    [
        (_activation(active=True, calls=1, positions=1), False, "activation_invalid"),
        (_activation(active=False, calls=0, positions=0), True, "activation_invalid"),
        (_activation(active=True, calls=0, positions=1), True, "did_not_run"),
        (
            {
                **_activation(active=True, calls=1, positions=1),
                "observed_positions": 0,
            },
            True,
            "did_not_run",
        ),
    ],
)
def test_activation_contract_fails_closed(
    value: dict[str, object],
    expected_active: bool,
    reason: str,
) -> None:
    with pytest.raises(verifier.SmokeVerificationError, match=reason):
        verifier._activation(value, expected_active=expected_active)


def test_verdict_publication_is_create_once_and_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "verdict.json"
    payload = b'{"passed":true}\n'
    verifier._atomic_create_or_verify(path, payload)
    verifier._atomic_create_or_verify(path, payload)
    assert path.read_bytes() == payload

    with pytest.raises(verifier.SmokeVerificationError, match="existing_verdict_differs"):
        verifier._atomic_create_or_verify(path, b'{"passed":false}\n')
