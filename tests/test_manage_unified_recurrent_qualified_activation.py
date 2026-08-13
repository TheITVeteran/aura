from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.brain.llm.unified_recurrent_shadow_battery import seal_shadow_canary_battery
from tools import manage_unified_recurrent_qualified_activation as command


def _arguments(tmp_path: Path, package: Path, lifecycle: Path) -> argparse.Namespace:
    root = tmp_path / "authority"
    return argparse.Namespace(
        package=package,
        lifecycle_result=lifecycle,
        pointer=root / "active.json",
        releases_root=root / "releases",
        activation=root / "qualified-active.json",
        expected_current_pointer_sha256=None,
        expected_current_activation_sha256=None,
    )


def _lifecycle(tmp_path: Path) -> tuple[Path, dict]:
    value = {"controller_sha256": "c" * 64, "result_sha256": "d" * 64}
    path = tmp_path / "lifecycle.json"
    path.write_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    )
    path.chmod(0o600)
    return path, value


def _manifest() -> dict:
    return {
        "package_id": "fixture",
        "manifest_sha256": "a" * 64,
        "checkpoint_sha256": "b" * 64,
        "domain_contract": {
            "families": ["khop"],
            "task_depths": [2],
            "recurrence_depth": 4,
        },
    }


def _unit_battery() -> dict:
    return seal_shadow_canary_battery(
        [
            {
                "task_id": "fresh-khop-1",
                "family": "khop",
                "task_depth": 2,
                "prompt_sha256": "1" * 64,
                "expected_sha256": "2" * 64,
                "public_token_ids": [10],
                "expected_token_ids": [20],
                "max_tokens": 1,
            }
        ],
        seed=7,
        replication_plan_sha256="5" * 64,
        replication_verdict_sha256="6" * 64,
        excluded_task_ids_sha256="7" * 64,
        excluded_prompt_sha256s_sha256="8" * 64,
        generator_source_sha256s={"generator.py": "9" * 64},
    )


def _canary(activation: dict, *, serving: bool) -> dict:
    battery = _unit_battery()
    case = battery["cases"][0]
    expected = command._canonical_sha256(case["expected_token_ids"])  # noqa: SLF001
    evidence = [
        {
            "index": 0,
            "task_id": "fresh-khop-1",
            "family": "khop",
            "task_depth": 2,
            "request_sha256": case["request_sha256"],
            "expected_token_ids_sha256": expected,
            "generated_token_ids_sha256": expected,
            "qualified_result_sha256": "9" * 64,
            "latency_ms": 7,
            "exact": True,
        }
    ]
    body = {
        "schema": command.QUALIFIED_CANARY_SCHEMA,
        "package_id": activation["package_id"],
        "manifest_sha256": activation["manifest_sha256"],
        "checkpoint_sha256": activation["checkpoint_sha256"],
        "controller_sha256": activation["controller_sha256"],
        "activation_sha256": activation["activation_sha256"],
        "battery_sha256": battery["battery_sha256"],
        "started_at_unix": 1.0,
        "completed_at_unix": 2.0,
        "case_count": 1,
        "exact_count": 1,
        "total_latency_ms": 7,
        "maximum_latency_ms": 7,
        "evidence": evidence,
        "supported": True,
        "serving_authority": serving,
        "authority_remains_active": serving,
        "canary_authority_was_request_scoped": not serving,
        "output_exposed": False,
    }
    return {**body, "result_sha256": command._canonical_sha256(body)}  # noqa: SLF001


def _candidate_activation(
    *,
    families: list[str] | None = None,
    task_depths: list[int] | None = None,
) -> dict:
    body = {
        "schema": "aura.unified_intrinsic.qualified_activation.v2",
        "package_id": "qualified-fixture",
        "manifest_sha256": "c" * 64,
        "checkpoint_sha256": "d" * 64,
        "controller_sha256": "b" * 64,
        "pointer_sha256": "1" * 64,
        "lifecycle_result_sha256": "2" * 64,
        "canary_plan_sha256": "3" * 64,
        "candidate_canary_sha256": "",
        "qualified_canary_sha256": "",
        "families": families or ["khop"],
        "task_depths": task_depths or [2],
        "recurrence_depth": 4,
        "mode": "qualified_canary_only",
        "ordinary_chat_authorized": False,
        "arbitrary_reasoning_authorized": False,
        "serving_authority": False,
    }
    return {**body, "activation_sha256": command._canonical_sha256(body)}  # noqa: SLF001


def test_unverified_activation_entrypoint_is_retired(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    lifecycle_path, _lifecycle_value = _lifecycle(tmp_path)

    with pytest.raises(
        command.UnifiedRecurrentQualifiedActivationCommandError,
        match="unverified activation is disabled",
    ):
        command._activate(_arguments(tmp_path, package, lifecycle_path))

    parser = command._parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["activate", str(package)])


def test_status_refuses_orphaned_qualified_authority(tmp_path: Path) -> None:
    root = tmp_path / "authority"
    root.mkdir()
    (root / "qualified-active.json").write_text("present", encoding="ascii")
    arguments = argparse.Namespace(
        pointer=root / "active.json",
        releases_root=root / "releases",
        activation=root / "qualified-active.json",
    )

    with pytest.raises(
        command.UnifiedRecurrentQualifiedActivationCommandError,
        match="without a shadow pointer",
    ):
        command._status(arguments)


def test_promotion_and_revocation_share_one_transaction_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "package"
    package.mkdir()
    lifecycle_path, _lifecycle_value = _lifecycle(tmp_path)
    arguments = _arguments(tmp_path, package, lifecycle_path)
    promotion_entered = threading.Event()
    release_promotion = threading.Event()
    revocation_entered = threading.Event()
    failures: list[BaseException] = []

    def fake_activate(_arguments, *, paths):
        promotion_entered.set()
        assert release_promotion.wait(timeout=5.0)
        return {"action": "activate_verified", "paths": paths}

    def fake_deactivate(_arguments, *, paths):
        revocation_entered.set()
        return {"action": "deactivate", "paths": paths}

    monkeypatch.setattr(command, "_activate_verified_locked", fake_activate)
    monkeypatch.setattr(command, "_deactivate_locked", fake_deactivate)

    def run(operation):
        try:
            operation(arguments)
        except BaseException as exc:  # noqa: BLE001 - preserve thread failure
            failures.append(exc)

    promotion = threading.Thread(target=run, args=(command._activate_verified,))
    revocation = threading.Thread(target=run, args=(command._deactivate,))
    promotion.start()
    assert promotion_entered.wait(timeout=5.0)
    revocation.start()
    assert not revocation_entered.wait(timeout=0.1)
    release_promotion.set()
    promotion.join(timeout=5.0)
    revocation.join(timeout=5.0)

    assert not promotion.is_alive()
    assert not revocation.is_alive()
    assert revocation_entered.is_set()
    assert failures == []


@pytest.mark.asyncio
async def test_verified_activation_canary_uses_live_qualified_path_without_publishing_tokens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cases = [
        {
            "task_id": "fresh-khop-1",
            "family": "khop",
            "task_depth": 2,
            "prompt_sha256": "1" * 64,
            "expected_sha256": "2" * 64,
            "public_token_ids": [10, 11],
            "expected_token_ids": [20, 21],
            "max_tokens": 2,
        },
        {
            "task_id": "fresh-modular-1",
            "family": "modular",
            "task_depth": 1,
            "prompt_sha256": "3" * 64,
            "expected_sha256": "4" * 64,
            "public_token_ids": [12, 13],
            "expected_token_ids": [22],
            "max_tokens": 1,
        },
    ]
    battery = seal_shadow_canary_battery(
        cases,
        seed=7,
        replication_plan_sha256="5" * 64,
        replication_verdict_sha256="6" * 64,
        excluded_task_ids_sha256="7" * 64,
        excluded_prompt_sha256s_sha256="8" * 64,
        generator_source_sha256s={"generator.py": "9" * 64},
    )
    candidate = _candidate_activation(
        families=["khop", "modular"],
        task_depths=[1, 2],
    )
    pending = command.seal_verified_qualified_activation(
        candidate,
        _canary(candidate, serving=False),
        expected_battery=_unit_battery(),
    )
    activation = command.seal_serving_qualified_activation(
        pending,
        _canary(pending, serving=False),
        expected_battery=_unit_battery(),
    )
    manifest = {
        "package_id": "qualified-fixture",
        "manifest_sha256": "c" * 64,
        "checkpoint_sha256": "d" * 64,
    }
    monkeypatch.setattr(
        command,
        "inspect_shadow_package",
        lambda _path: {"manifest": manifest, "canary_battery": battery},
    )
    monkeypatch.setattr(command, "read_qualified_activation", lambda _path: activation)

    expected_by_public = {
        tuple(case["public_token_ids"]): case["expected_token_ids"] for case in cases
    }

    class Client:
        _unified_recurrent_qualified_activation_status = {
            "loaded": True,
            "serving_authority": True,
            "activation": activation,
        }

        async def warmup(self, **_kwargs):
            return True

        async def unified_recurrent_qualified_decode_async(
            self,
            public_token_ids,
            *,
            family,
            task_depth,
            max_tokens,
            timeout_s,
        ):
            assert timeout_s == 45.0
            expected = expected_by_public[tuple(public_token_ids)]
            assert len(expected) == max_tokens
            return {
                "ok": True,
                "status": "completed",
                "reason": "qualified_decode_completed",
                "receipt": {
                    "generated_token_ids": expected,
                    "family": family,
                    "task_depth": task_depth,
                    "qualified_activation_sha256": activation["activation_sha256"],
                    "result_sha256": "e" * 64,
                },
            }

        async def aclose(self):
            return None

    from core.brain.llm import mlx_client

    monkeypatch.setattr(mlx_client, "get_mlx_client", lambda _model: Client())
    arguments = SimpleNamespace(
        package=tmp_path / "package",
        model=tmp_path / "model",
        case_timeout=45.0,
        canary_output=tmp_path / "private" / "qualified-canary.json",
    )
    activated = {
        "activation_path": str(tmp_path / "qualified-active.json"),
        "activation_sha256": activation["activation_sha256"],
    }

    result = await command._run_qualified_canary(arguments, activated)

    assert result["supported"] is True
    assert result["exact_count"] == result["case_count"] == 2
    assert result["authority_remains_active"] is True
    assert all(row["exact"] for row in result["evidence"])
    published = arguments.canary_output.read_text(encoding="ascii")
    progress = capsys.readouterr().out
    assert "qualified_canary_case_started" in progress
    assert "qualified_canary_case_completed" in progress
    assert "public_token_ids" not in progress
    assert "expected_token_ids" not in progress
    assert "generated_token_ids" not in progress
    assert "public_token_ids" not in published
    assert '"expected_token_ids":' not in published
    assert '"generated_token_ids":' not in published


@pytest.mark.asyncio
async def test_verified_activation_canary_uses_only_request_scoped_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = {
        "task_id": "fresh-khop-1",
        "family": "khop",
        "task_depth": 2,
        "prompt_sha256": "1" * 64,
        "expected_sha256": "2" * 64,
        "public_token_ids": [10, 11],
        "expected_token_ids": [20, 21],
        "max_tokens": 2,
    }
    battery = seal_shadow_canary_battery(
        [case],
        seed=7,
        replication_plan_sha256="5" * 64,
        replication_verdict_sha256="6" * 64,
        excluded_task_ids_sha256="7" * 64,
        excluded_prompt_sha256s_sha256="8" * 64,
        generator_source_sha256s={"generator.py": "9" * 64},
    )
    activation = _candidate_activation()
    manifest = {
        "package_id": "qualified-fixture",
        "manifest_sha256": "c" * 64,
        "checkpoint_sha256": "d" * 64,
    }
    monkeypatch.setattr(
        command,
        "inspect_shadow_package",
        lambda _path: {"manifest": manifest, "canary_battery": battery},
    )
    observed: list[dict] = []

    class Client:
        _unified_recurrent_qualified_activation_status = {
            "loaded": False,
            "serving_authority": False,
            "activation": None,
        }

        async def warmup(self, **_kwargs):
            return True

        async def unified_recurrent_qualified_canary_decode_async(
            self,
            public_token_ids,
            *,
            family,
            task_depth,
            max_tokens,
            activation,
            battery_sha256,
            case_index,
            nonce,
            timeout_s,
        ):
            assert timeout_s == 45.0
            assert battery_sha256 == battery["battery_sha256"]
            assert case_index == 0
            assert len(nonce) == 64
            observed.append(activation)
            return {
                "ok": True,
                "status": "completed",
                "reason": "qualified_decode_completed",
                "receipt": {
                    "generated_token_ids": case["expected_token_ids"],
                    "family": family,
                    "task_depth": task_depth,
                    "qualified_activation_sha256": activation["activation_sha256"],
                    "result_sha256": "e" * 64,
                },
            }

        async def unified_recurrent_qualified_decode_async(self, *_args, **_kwargs):
            pytest.fail("durable serving path used before publication")

        async def aclose(self):
            return None

    from core.brain.llm import mlx_client

    monkeypatch.setattr(mlx_client, "get_mlx_client", lambda _model: Client())
    arguments = SimpleNamespace(
        package=tmp_path / "package",
        model=tmp_path / "model",
        case_timeout=45.0,
        canary_output=tmp_path / "private" / "qualified-canary.json",
    )
    staged = {
        "activation_sha256": activation["activation_sha256"],
    }

    result = await command._run_qualified_canary(
        arguments,
        staged,
        candidate_activation=activation,
    )

    assert observed == [activation]
    assert result["supported"] is True
    assert result["serving_authority"] is False
    assert result["authority_remains_active"] is False
    assert result["canary_authority_was_request_scoped"] is True
    pending = command.seal_verified_qualified_activation(
        activation,
        result,
        expected_battery=battery,
    )
    assert pending["candidate_canary_sha256"] == result["result_sha256"]
    assert pending["serving_authority"] is False


def test_verified_activation_failure_never_publishes_authority_and_retires_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "package"
    package.mkdir()
    pointer = {"pointer_sha256": "a" * 64}
    candidate = {
        "activation_sha256": "b" * 64,
        "mode": "unified_intrinsic_recurrent",
        "families": ["modular", "khop", "register_trace"],
        "task_depths": [1, 2, 4],
    }
    retired: list[str] = []
    monkeypatch.setattr(
        command,
        "inspect_shadow_package",
        lambda _path: {"manifest": _manifest(), "canary_battery": _unit_battery()},
    )
    monkeypatch.setattr(command, "_read_lifecycle", lambda _path: {})
    monkeypatch.setattr(
        command,
        "publish_shadow_pointer",
        lambda *_args, **_kwargs: pointer,
    )
    monkeypatch.setattr(command, "seal_qualified_activation", lambda *_args: candidate)
    monkeypatch.setattr(
        command,
        "publish_qualified_activation",
        lambda *_args, **_kwargs: pytest.fail("authority published before canary"),
    )

    async def fail(_arguments, _activated, **_kwargs):
        raise RuntimeError("live serving path refuted")

    monkeypatch.setattr(command, "_run_qualified_canary", fail)
    monkeypatch.setattr(
        command,
        "deactivate_shadow_pointer",
        lambda **kwargs: retired.append(kwargs["expected_current_sha256"]),
    )
    root = tmp_path / "authority"
    arguments = SimpleNamespace(
        package=package,
        lifecycle_result=tmp_path / "lifecycle.json",
        model=tmp_path / "model",
        canary_output=tmp_path / "canary.json",
        case_timeout=45.0,
        pointer=root / "active.json",
        releases_root=root / "releases",
        activation=root / "qualified-active.json",
        expected_current_pointer_sha256=None,
        expected_current_activation_sha256=None,
    )

    with pytest.raises(RuntimeError, match="refuted"):
        command._activate_verified(arguments)

    assert retired == ["a" * 64]


def test_pending_cold_load_failure_revokes_pending_authority_and_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "package"
    package.mkdir()
    pointer = {"pointer_sha256": "a" * 64}
    candidate = {
        "activation_sha256": "b" * 64,
        "mode": "qualified_canary_only",
        "families": ["khop"],
        "task_depths": [2],
        "package_id": "fixture",
        "manifest_sha256": "c" * 64,
        "checkpoint_sha256": "d" * 64,
        "controller_sha256": "e" * 64,
    }
    candidate_canary = _canary(candidate, serving=False)
    pending = {
        **candidate,
        "activation_sha256": "c" * 64,
        "mode": "qualified_typed_pending",
        "candidate_canary_sha256": candidate_canary["result_sha256"],
    }
    revoked: list[str] = []
    retired: list[str] = []
    calls = 0
    monkeypatch.setattr(
        command,
        "inspect_shadow_package",
        lambda _path: {"manifest": _manifest(), "canary_battery": _unit_battery()},
    )
    monkeypatch.setattr(command, "_read_lifecycle", lambda _path: {})
    monkeypatch.setattr(command, "publish_shadow_pointer", lambda *_args, **_kwargs: pointer)
    monkeypatch.setattr(command, "seal_qualified_activation", lambda *_args: candidate)
    monkeypatch.setattr(
        command,
        "seal_verified_qualified_activation",
        lambda received, receipt, **_kwargs: (
            pending
            if received == candidate and receipt == candidate_canary
            else pytest.fail("pending authority sealed from different evidence")
        ),
    )
    monkeypatch.setattr(
        command,
        "publish_qualified_activation",
        lambda received, **_kwargs: pending if received == pending else pytest.fail("serving published"),
    )
    monkeypatch.setattr(command, "resolve_shadow_pointer", lambda *_args, **_kwargs: package)
    monkeypatch.setattr(command, "activation_matches_shadow_receipt", lambda *_args: True)

    async def canary(_arguments, _staged, *, candidate_activation=None):
        nonlocal calls
        calls += 1
        if calls == 1 and candidate_activation == candidate:
            return candidate_canary
        assert candidate_activation == pending
        raise RuntimeError("pending cold load refuted")

    monkeypatch.setattr(command, "_run_qualified_canary", canary)
    monkeypatch.setattr(
        command,
        "deactivate_qualified_activation",
        lambda **kwargs: revoked.append(kwargs["expected_current_sha256"]),
    )
    monkeypatch.setattr(
        command,
        "deactivate_shadow_pointer",
        lambda **kwargs: retired.append(kwargs["expected_current_sha256"]),
    )
    root = tmp_path / "authority"
    arguments = SimpleNamespace(
        package=package,
        lifecycle_result=tmp_path / "lifecycle.json",
        model=tmp_path / "model",
        canary_output=tmp_path / "canary.json",
        case_timeout=45.0,
        pointer=root / "active.json",
        releases_root=root / "releases",
        activation=root / "qualified-active.json",
        expected_current_pointer_sha256=None,
        expected_current_activation_sha256=None,
    )

    with pytest.raises(RuntimeError, match="pending cold load refuted"):
        command._activate_verified(arguments)

    assert calls == 2
    assert revoked == [pending["activation_sha256"]]
    assert retired == [pointer["pointer_sha256"]]


def test_verified_activation_publishes_authority_only_after_request_scoped_canary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "package"
    package.mkdir()
    pointer = {"pointer_sha256": "a" * 64}
    candidate = {
        "activation_sha256": "b" * 64,
        "mode": "qualified_canary_only",
        "families": ["khop"],
        "task_depths": [2],
        "package_id": "fixture",
        "manifest_sha256": "a" * 64,
        "checkpoint_sha256": "b" * 64,
        "controller_sha256": "c" * 64,
        "candidate_canary_sha256": "",
        "qualified_canary_sha256": "",
    }
    candidate_canary = _canary(candidate, serving=False)
    pending = {
        **candidate,
        "activation_sha256": "c" * 64,
        "mode": "qualified_typed_pending",
        "candidate_canary_sha256": candidate_canary["result_sha256"],
    }
    pending_canary = _canary(pending, serving=False)
    durable = {
        **pending,
        "activation_sha256": "d" * 64,
        "mode": "qualified_typed_only",
        "qualified_canary_sha256": pending_canary["result_sha256"],
        "serving_authority": True,
    }
    observed: list[str] = []
    monkeypatch.setattr(
        command,
        "inspect_shadow_package",
        lambda _path: {"manifest": _manifest(), "canary_battery": _unit_battery()},
    )
    monkeypatch.setattr(command, "_read_lifecycle", lambda _path: {})

    def publish_pointer(*_args, **_kwargs):
        observed.append("pointer")
        return pointer

    async def canary(_arguments, staged, *, candidate_activation=None):
        if candidate_activation == candidate:
            assert staged["active"] is False
            assert candidate_activation == candidate
            assert observed == ["pointer"]
            observed.append("candidate_canary")
            return candidate_canary
        assert candidate_activation == pending
        assert staged["active"] is False
        assert staged["activation_sha256"] == pending["activation_sha256"]
        assert observed == ["pointer", "candidate_canary", "authority"]
        observed.append("pending_canary")
        return pending_canary

    def publish_authority(received, **_kwargs):
        assert received in (pending, durable)
        if received == pending:
            assert observed == ["pointer", "candidate_canary"]
            observed.append("authority")
            return pending
        assert observed == ["pointer", "candidate_canary", "authority", "pending_canary"]
        observed.append("serving")
        return durable

    monkeypatch.setattr(command, "publish_shadow_pointer", publish_pointer)
    monkeypatch.setattr(command, "seal_qualified_activation", lambda *_args: candidate)
    monkeypatch.setattr(
        command,
        "seal_verified_qualified_activation",
        lambda received, receipt, **_kwargs: (
            pending
            if received == candidate and receipt == candidate_canary
            else pytest.fail("durable authority sealed from different evidence")
        ),
    )
    monkeypatch.setattr(command, "_run_qualified_canary", canary)
    monkeypatch.setattr(
        command,
        "seal_serving_qualified_activation",
        lambda received, receipt, **_kwargs: (
            durable
            if received == pending and receipt == pending_canary
            else pytest.fail("serving authority sealed from different evidence")
        ),
    )
    monkeypatch.setattr(command, "publish_qualified_activation", publish_authority)
    monkeypatch.setattr(command, "resolve_shadow_pointer", lambda *_args, **_kwargs: package)
    monkeypatch.setattr(command, "activation_matches_shadow_receipt", lambda *_args: True)
    root = tmp_path / "authority"
    arguments = SimpleNamespace(
        package=package,
        lifecycle_result=tmp_path / "lifecycle.json",
        model=tmp_path / "model",
        canary_output=tmp_path / "canary.json",
        case_timeout=45.0,
        pointer=root / "active.json",
        releases_root=root / "releases",
        activation=root / "qualified-active.json",
        expected_current_pointer_sha256=None,
        expected_current_activation_sha256=None,
    )

    result = command._activate_verified(arguments)

    assert observed == ["pointer", "candidate_canary", "authority", "pending_canary", "serving"]
    assert result["active"] is True
    assert result["candidate_canary"]["canary_authority_was_request_scoped"] is True
    assert result["canary"]["canary_authority_was_request_scoped"] is True
