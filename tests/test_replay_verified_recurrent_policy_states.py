"""File-protocol and recovery tests for detached policy-state replay."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec
from core.learning.verified_transition_episode import canonical_json_bytes
from tools import replay_verified_recurrent_policy_states as worker


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _runner_sha256(value: Any) -> str:
    """Hash exactly as ``tools/run_detached_step.py`` does, independently.

    Reproduced here rather than imported so a drift in either canonicaliser
    fails this test instead of cancelling out against itself.
    """
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _request() -> dict[str, Any]:
    body = {
        "schema": worker.REQUEST_SCHEMA,
        "purpose": worker.REQUEST_PURPOSE,
        "evidence_manifest": {
            "manifest_sha256": "a" * 64,
            "updated_replay_sequences": [0],
        },
        "policy_state_replay_contract": {"contract_sha256": "b" * 64},
        "campaign_trust_policy": {"policy_sha256": "c" * 64},
        "verifier_identity": "external-replay-verifier",
        "verified_at_unix": 1_800_000_000,
    }
    return {**body, "request_sha256": _digest(body)}


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_bytes(canonical_json_bytes(value))
    os.chmod(path, 0o600)


def test_worker_publishes_one_immutable_result_and_reuses_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    _write(request_path, request)
    spec = RLCExecutionSpec(
        n_slots=2,
        branch_roles=("constructive_solution", "critical_audit"),
        exchange_interval=1,
        recurrent_steps=2,
        alpha=0.35,
        prelude_frac=0.25,
        coda_frac=0.25,
    )
    contract = {
        "contract_sha256": "b" * 64,
        "model": {"path": str(tmp_path / "model")},
        "execution_spec": {"document_json": json.dumps(spec.to_dict())},
        "initial_policy_state_custody": {
            "adapter_initialization": {
                "rank": 2,
                "layers": 1,
                "targets": ["q_proj"],
                "seed": 7,
            }
        },
    }
    monkeypatch.setattr(
        worker,
        "validate_policy_state_replay_contract",
        lambda *_args, **_kwargs: contract,
    )
    monkeypatch.setattr(
        worker,
        "campaign_trust_policy_from_verifier_material",
        lambda _value: SimpleNamespace(policy_sha256="c" * 64),
    )
    import mlx_lm

    monkeypatch.setattr(mlx_lm, "load", lambda _path: (object(), object()))
    monkeypatch.setattr(
        worker,
        "attach_recurrent_policy_adapters",
        lambda *_args, **_kwargs: ("site",),
    )
    transition = {
        "sequence": 0,
        "receipt_sha256": "d" * 64,
    }
    calls = 0

    def replay(*_args: Any, **_kwargs: Any) -> tuple[dict[str, Any], ...]:
        nonlocal calls
        calls += 1
        return (transition,)

    monkeypatch.setattr(
        worker,
        "replay_recurrent_evidence_manifest_policy_states",
        replay,
    )

    first = worker._execute(request_path, result_path)
    second = worker._execute(request_path, result_path)

    assert first == second
    assert calls == 1
    assert first["transition_results"] == [transition]
    assert first["transition_result_root_sha256"] == _digest(
        [{"sequence": 0, "receipt_sha256": "d" * 64}]
    )
    assert result_path.stat().st_mode & 0o077 == 0


@pytest.mark.parametrize(
    ("result_state", "expected"),
    (
        ("absent", "safe_to_resume"),
        ("complete", "safe_to_resume"),
        ("invalid", "indeterminate"),
    ),
)
def test_resume_verdict_is_cheap_and_fails_closed_on_bad_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    result_state: str,
    expected: str,
) -> None:
    request = _request()
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    evidence_path = tmp_path / "resume-evidence.json"
    _write(request_path, request)
    if result_state == "complete":
        body = {
            "schema": worker.RESULT_SCHEMA,
            "request_sha256": request["request_sha256"],
            "policy_state_replay_contract_sha256": "b" * 64,
            "evidence_manifest_sha256": "a" * 64,
            "verifier_identity": "external-replay-verifier",
            "verified_at_unix": 1_800_000_000,
            "transition_results": [{"sequence": 0, "receipt_sha256": "d" * 64}],
            "transition_result_root_sha256": _digest([{"sequence": 0, "receipt_sha256": "d" * 64}]),
            "completed_at_unix": 1_800_000_001,
        }
        _write(result_path, {**body, "result_sha256": _digest(body)})
    elif result_state == "invalid":
        _write(result_path, {"not": "a replay result"})
    environment = {
        "AURA_DETACHED_PLAN_SHA256": "1" * 64,
        "AURA_DETACHED_COMMAND_SHA256": "2" * 64,
        "AURA_DETACHED_PRIOR_ATTEMPT": "1",
        "AURA_DETACHED_PRIOR_JOURNAL_HEAD_SHA256": "3" * 64,
        "AURA_DETACHED_RESUME_EVIDENCE_TRANSPORT": "stdout-v3",
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)

    verdict = worker._resume_verdict(request_path, result_path)

    assert verdict["verdict"] == expected
    assert verdict["evidence"]["result_state"] == result_state
    assert verdict["checkpoint_sequence"] == (1 if result_state == "complete" else 0)
    # Evidence rides inline on stdout-v3; the runner hashes the object itself.
    assert "evidence_path" not in verdict
    assert verdict["evidence_sha256"] == _runner_sha256(verdict["evidence"])
    assert not evidence_path.exists()


def test_request_rejects_resealed_schema_or_digest_drift() -> None:
    request = _request()
    request["verifier_identity"] = "substituted"
    with pytest.raises(
        worker.ExternalPolicyStateReplayError,
        match="request_invalid",
    ):
        worker.validate_request(request)
