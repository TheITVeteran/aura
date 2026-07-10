"""Behavioral contracts for the canonical consequential-action transaction."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


class _FakeWill:
    def __init__(self) -> None:
        self.decisions: list[dict[str, Any]] = []
        self.outcomes: list[tuple[str, Any]] = []

    def decide(self, **kwargs: Any) -> Any:
        from core.governance.will import WillDecision, WillOutcome

        self.decisions.append(dict(kwargs))
        return WillDecision(
            receipt_id=f"will-test-{len(self.decisions)}",
            outcome=WillOutcome.PROCEED,
            domain=kwargs["domain"],
            reason="test_approved",
            source=str(kwargs.get("source") or "test"),
        )

    def record_outcome(self, receipt_id: str, outcome: Any) -> None:
        self.outcomes.append((receipt_id, outcome))


@pytest.fixture
def action_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    import core.runtime.action_executor as executor
    from core.runtime.post_action_receipt import PostActionReceiptStore

    fake_will = _FakeWill()
    body = SimpleNamespace(snapshot=lambda: None)
    welfare = SimpleNamespace(last_outputs=None)
    store = PostActionReceiptStore(tmp_path / "post_action.jsonl")

    monkeypatch.setattr(executor, "get_will", lambda: fake_will)
    monkeypatch.setattr(executor.BodyStateService, "get", classmethod(lambda _cls: body))
    monkeypatch.setattr(executor.WelfareState, "get", classmethod(lambda _cls: welfare))
    monkeypatch.setattr(executor, "get_post_action_receipt_store", lambda: store)
    return executor, fake_will, store


@pytest.mark.asyncio
async def test_file_write_requires_readback_and_emits_complete_causal_receipt(
    action_runtime: Any,
    tmp_path: Path,
) -> None:
    executor, fake_will, store = action_runtime
    target = tmp_path / "artifact.txt"

    result = await executor.ActionExecutor.execute(
        domain="file_write",
        action_name="write_artifact",
        params={"path": str(target), "text": "verified body"},
        source="effect_test",
        action_id="action-file-write-1",
    )

    assert target.read_text(encoding="utf-8") == "verified body"
    assert result["ok"] is True
    assert result["status"] == "success_verified"
    assert result["transport_succeeded"] is True
    assert result["effect_verified"] is True
    assert result["welfare_transaction_completed"] is True
    assert result["receipt_persisted"] is True
    assert result["manual_reconciliation_required"] is False
    assert result["retry_safe"] is False
    assert result["action_id"] == "action-file-write-1"
    assert result["request_digest"].startswith("sha256:")
    assert len(fake_will.decisions) == 1
    assert len(fake_will.outcomes) == 1

    receipt = store.get_receipt(result["post_action_receipt_id"])
    assert receipt is not None
    assert receipt.action_id == result["action_id"]
    assert receipt.request_digest == result["request_digest"]
    assert receipt.domain == "file_write"
    assert receipt.transport_succeeded is True
    assert receipt.effect_verified is True
    assert receipt.welfare_transaction_completed is True


@pytest.mark.asyncio
async def test_malformed_file_action_fails_before_dispatch_but_is_receipted(
    action_runtime: Any,
) -> None:
    executor, _fake_will, store = action_runtime

    result = await executor.ActionExecutor.execute(
        domain="file_write",
        action_name="missing_path",
        params={"text": "must not write"},
        source="effect_test",
    )

    assert result["ok"] is False
    assert result["transport_succeeded"] is False
    assert result["effect_verified"] is False
    assert result["status"] == "failed_recoverable"
    assert result["receipt_persisted"] is True
    assert "file action path" in result["error"]
    receipt = store.get_receipt(result["post_action_receipt_id"])
    assert receipt is not None and receipt.actual_outcome == "failure"


@pytest.mark.asyncio
async def test_desktop_transport_success_is_not_completion_without_observed_effect(
    action_runtime: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, _fake_will, store = action_runtime

    class Gateway:
        async def run_applescript_async(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {"ok": True, "exit_code": 0, "stdout": "fired", "stderr": ""}

    monkeypatch.setattr(executor, "get_desktop_action_gateway", lambda: Gateway())
    result = await executor.ActionExecutor.execute(
        domain="environment_action",
        action_name="focus_window",
        params={"script": "tell application \"Finder\" to activate"},
        source="effect_test",
    )

    assert result["transport_succeeded"] is True
    assert result["effect_verified"] is False
    assert result["ok"] is False
    assert result["status"] == "partial_success"
    assert result["manual_reconciliation_required"] is True
    assert result["retry_safe"] is False
    receipt = store.get_receipt(result["post_action_receipt_id"])
    assert receipt is not None and receipt.actual_outcome == "partial"


@pytest.mark.asyncio
async def test_custom_verifier_must_supply_evidence_and_receives_full_context(
    action_runtime: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, _fake_will, _store = action_runtime

    class Gateway:
        async def run_applescript_async(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {"ok": True, "exit_code": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr(executor, "get_desktop_action_gateway", lambda: Gateway())

    async def bare_claim(context: dict[str, Any]) -> dict[str, Any]:
        assert context["domain"] == "external_action"
        assert context["params"]["script"] == "do something"
        assert context["result"]["ok"] is True
        return {"effect_verified": True}

    rejected = await executor.ActionExecutor.execute(
        domain="external_action",
        action_name="bare_claim",
        params={"script": "do something"},
        effect_verifier=bare_claim,
        source="effect_test",
    )
    assert rejected["effect_verified"] is False
    assert rejected["verification_evidence"]["custom_verifier"]["error"] == (
        "verifier_evidence_missing"
    )

    async def empty_evidence(_context: dict[str, Any]) -> dict[str, Any]:
        return {"effect_verified": True, "evidence": {}}

    still_rejected = await executor.ActionExecutor.execute(
        domain="external_action",
        action_name="empty_evidence",
        params={"script": "do something"},
        effect_verifier=empty_evidence,
        source="effect_test",
    )
    assert still_rejected["effect_verified"] is False

    async def observed(context: dict[str, Any]) -> dict[str, Any]:
        return {
            "effect_verified": context["result"]["exit_code"] == 0,
            "observed_state": {"frontmost_app": "Finder"},
        }

    accepted = await executor.ActionExecutor.execute(
        domain="external_action",
        action_name="observed_effect",
        params={"script": "do something"},
        effect_verifier=observed,
        source="effect_test",
    )
    assert accepted["ok"] is True
    assert accepted["effect_verified"] is True
    verifier_evidence = accepted["verification_evidence"]["custom_verifier"]
    assert verifier_evidence["observed_state"]["frontmost_app"] == "Finder"
    assert verifier_evidence["verifier"].endswith("observed")


@pytest.mark.asyncio
async def test_capability_self_assertion_needs_passed_durable_expectation_receipt(
    action_runtime: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, _fake_will, _store = action_runtime
    from core.container import ServiceContainer

    class Engine:
        def __init__(self, result: dict[str, Any]) -> None:
            self.result = result

        async def execute(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return dict(self.result)

    current = {
        "engine": Engine({"ok": True, "effect_verified": True}),
    }
    monkeypatch.setattr(
        ServiceContainer,
        "get",
        classmethod(lambda _cls, _name, default=None: current["engine"]),
    )

    claimed = await executor.ActionExecutor.execute(
        domain="tool_execution",
        action_name="self_claiming_skill",
        params={},
        source="effect_test",
    )
    assert claimed["ok"] is False
    observation = claimed["verification_evidence"]["observation"]
    assert observation["downstream_effect_claimed"] is True
    assert observation["effect_verified"] is False

    current["engine"] = Engine(
        {
            "ok": True,
            "effect_verified": True,
            "expectation_receipt_id": "tool-receipt-1",
            "expectation_verdict": {"passed": True},
        }
    )
    receipted = await executor.ActionExecutor.execute(
        domain="tool_execution",
        action_name="receipted_skill",
        params={},
        source="effect_test",
    )
    assert receipted["ok"] is True
    assert receipted["effect_verified"] is True


@pytest.mark.asyncio
async def test_mutating_network_transport_requires_readback_verifier(
    action_runtime: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, _fake_will, _store = action_runtime

    class Gateway:
        async def request_async(self, **_kwargs: Any) -> dict[str, Any]:
            return {"ok": True, "status_code": 204, "content": b""}

    monkeypatch.setattr(executor, "get_network_gateway", lambda: Gateway())
    mutation = await executor.ActionExecutor.execute(
        domain="network_call",
        action_name="update_remote_record",
        params={"method": "POST", "url": "https://example.test/items/1"},
        source="effect_test",
    )
    assert mutation["transport_succeeded"] is True
    assert mutation["effect_verified"] is False
    assert mutation["status"] == "partial_success"

    read = await executor.ActionExecutor.execute(
        domain="network_call",
        action_name="read_remote_record",
        params={"method": "GET", "url": "https://example.test/items/1"},
        source="effect_test",
    )
    assert read["ok"] is True
    assert read["effect_verified"] is True


@pytest.mark.asyncio
async def test_custom_effect_handler_runs_inside_canonical_transaction(
    action_runtime: Any,
) -> None:
    executor, fake_will, store = action_runtime
    handler_contexts: list[dict[str, Any]] = []

    async def browser_handler(context: dict[str, Any]) -> dict[str, Any]:
        handler_contexts.append(context)
        return {
            "ok": True,
            "observed_url": "https://example.test/final",
            "navigation_confirmed": True,
        }

    def browser_verifier(context: dict[str, Any]) -> dict[str, Any]:
        observed_url = context["result"].get("observed_url")
        return {
            "effect_verified": bool(observed_url),
            "observation": {"observed_url": observed_url},
        }

    result = await executor.ActionExecutor.execute(
        domain="network_call",
        action_name="browser_navigation",
        params={"url": "https://example.test/start"},
        source="effect_test",
        effect_handler=browser_handler,
        effect_verifier=browser_verifier,
        execution_timeout_s=2.0,
    )

    assert result["ok"] is True
    assert result["effect_verified"] is True
    assert result["receipt_persisted"] is True
    assert len(handler_contexts) == 1
    assert handler_contexts[0]["will_receipt_id"] == result["will_receipt_id"]
    assert handler_contexts[0]["params"]["url"].endswith("/start")
    assert len(fake_will.decisions) == 1
    receipt = store.get_receipt(result["post_action_receipt_id"])
    assert receipt is not None and receipt.effect_verified is True


@pytest.mark.asyncio
async def test_custom_effect_handler_requires_verifier_and_permitted_domain(
    action_runtime: Any,
    tmp_path: Path,
) -> None:
    executor, fake_will, _store = action_runtime

    async def handler(_context: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True}

    with pytest.raises(ValueError, match="independent effect_verifier"):
        await executor.ActionExecutor.execute(
            domain="network_call",
            action_name="unverified_handler",
            params={},
            effect_handler=handler,
        )

    with pytest.raises(ValueError, match="not permitted"):
        await executor.ActionExecutor.execute(
            domain="file_write",
            action_name="bypass_file_gateway",
            params={"path": str(tmp_path / "forbidden")},
            effect_handler=handler,
            effect_verifier=lambda _context: {
                "effect_verified": True,
                "observation": {"claimed": True},
            },
        )

    assert fake_will.decisions == []


@pytest.mark.asyncio
async def test_custom_effect_handler_timeout_is_failed_and_receipted(
    action_runtime: Any,
) -> None:
    executor, _fake_will, store = action_runtime

    async def slow_handler(_context: dict[str, Any]) -> dict[str, Any]:
        import asyncio

        await asyncio.sleep(0.1)
        return {"ok": True}

    result = await executor.ActionExecutor.execute(
        domain="external_action",
        action_name="bounded_handler",
        params={},
        effect_handler=slow_handler,
        effect_verifier=lambda _context: {
            "effect_verified": True,
            "observation": {"completed": True},
        },
        execution_timeout_s=0.01,
        source="effect_test",
    )

    assert result["ok"] is False
    assert result["transport_succeeded"] is False
    assert result["effect_verified"] is False
    assert result["receipt_persisted"] is True
    assert "TimeoutError" in result["verification_evidence"]["observation"]["error_type"]
    receipt = store.get_receipt(result["post_action_receipt_id"])
    assert receipt is not None and receipt.actual_outcome == "failure"


@pytest.mark.asyncio
async def test_copy_verification_uses_gateway_final_destination(
    action_runtime: Any,
    tmp_path: Path,
) -> None:
    executor, _fake_will, _store = action_runtime
    source = tmp_path / "source.txt"
    destination_dir = tmp_path / "destination"
    source.write_text("copy me", encoding="utf-8")
    destination_dir.mkdir()

    result = await executor.ActionExecutor.execute(
        domain="file_write",
        action_name="copy_into_directory",
        params={
            "op": "copy",
            "path": str(source),
            "destination": str(destination_dir),
        },
        source="effect_test",
    )

    final_path = destination_dir / source.name
    assert final_path.read_text(encoding="utf-8") == "copy me"
    assert result["destination"] == str(final_path)
    assert result["effect_verified"] is True
    observation = result["verification_evidence"]["observation"]
    assert observation["destination"] == str(final_path)
    assert observation["requested_destination"] == str(destination_dir)


@pytest.mark.asyncio
async def test_delete_removes_broken_symlink_and_reports_real_change(
    action_runtime: Any,
    tmp_path: Path,
) -> None:
    executor, _fake_will, _store = action_runtime
    link = tmp_path / "broken-link"
    link.symlink_to(tmp_path / "missing-target")

    result = await executor.ActionExecutor.execute(
        domain="file_write",
        action_name="delete_broken_symlink",
        params={"op": "delete", "path": str(link)},
        source="effect_test",
    )

    assert result["ok"] is True
    assert not link.is_symlink()
    observation = result["verification_evidence"]["observation"]
    assert observation["existed_before"] is True
    assert observation["changed"] is True


@pytest.mark.asyncio
async def test_copy_preserves_symlink_identity_for_effect_verification(
    action_runtime: Any,
    tmp_path: Path,
) -> None:
    executor, _fake_will, _store = action_runtime
    target = tmp_path / "target.txt"
    target.write_text("target", encoding="utf-8")
    source_link = tmp_path / "source-link"
    source_link.symlink_to(target.name)
    destination = tmp_path / "copied-link"

    result = await executor.ActionExecutor.execute(
        domain="file_write",
        action_name="copy_symlink",
        params={
            "op": "copy",
            "path": str(source_link),
            "destination": str(destination),
        },
        source="effect_test",
    )

    assert result["ok"] is True
    assert destination.is_symlink()
    assert destination.readlink() == Path(target.name)
    assert result["verification_evidence"]["observation"]["content_equivalent"] is True


@pytest.mark.asyncio
async def test_state_mutation_verifies_fresh_domain_specific_durable_readback(
    action_runtime: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executor, _fake_will, _store = action_runtime
    from core.state.state_gateway import ConcreteStateGateway

    gateway = ConcreteStateGateway(root=tmp_path / "state")
    monkeypatch.setattr(executor, "get_state_gateway", lambda: gateway)

    result = await executor.ActionExecutor.execute(
        domain="state_mutation",
        action_name="set_focus_mode",
        params={
            "key": "mode",
            "new_value": "focused",
            "state_domain": "cognition",
        },
        source="effect_test",
    )

    assert result["ok"] is True
    assert result["effect_verified"] is True
    assert result["receipt_id"].startswith("statemut-")
    assert await gateway.read("mode", domain="cognition", fresh=True) == "focused"
    assert await gateway.read("mode", domain="world_state", fresh=True) is None


@pytest.mark.asyncio
async def test_receipt_persistence_failure_is_non_retryable_partial_after_effect(
    action_runtime: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executor, _fake_will, _store = action_runtime

    class FailingStore:
        async def record_async(self, _receipt: Any) -> None:
            raise OSError("disk unavailable")

    monkeypatch.setattr(executor, "get_post_action_receipt_store", lambda: FailingStore())
    target = tmp_path / "effect_happened.txt"
    result = await executor.ActionExecutor.execute(
        domain="file_write",
        action_name="write_without_receipt",
        params={"path": str(target), "text": "persisted effect"},
        source="effect_test",
    )

    assert target.read_text(encoding="utf-8") == "persisted effect"
    assert result["transport_succeeded"] is True
    assert result["effect_verified"] is True
    assert result["ok"] is False
    assert result["status"] == "partial_success"
    assert result["receipt_persisted"] is False
    assert result["retry_safe"] is False
    assert result["manual_reconciliation_required"] is True
    assert result["post_action_receipt_attempt_id"].startswith("post-")


def test_action_summary_redacts_nested_command_secrets_and_url_credentials() -> None:
    from core.runtime.action_executor import _safe_action_summary

    summary = _safe_action_summary(
        "remote_call",
        {
            "password": "top-secret-password",
            "url": "https://alice:pw@example.test/path?token=query-secret",
            "argv": [
                "curl",
                "--token",
                "argv-secret",
                "https://bob:pw@example.test/private?api_key=hidden",
            ],
            "actions": [
                {
                    "type": "type",
                    "selector": "input[type=password]",
                    "value": "nested-action-secret",
                }
            ],
        },
    )

    assert "top-secret-password" not in summary
    assert "query-secret" not in summary
    assert "argv-secret" not in summary
    assert "alice" not in summary and "bob" not in summary
    assert "pw" not in summary
    assert "nested-action-secret" not in summary
    assert "[REDACTED]" in summary


def test_post_action_store_snapshots_bounds_and_tail_loads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import core.runtime.post_action_receipt as receipts

    monkeypatch.setattr(receipts._HOT_LIMIT_FLAG, "value", lambda: 64)
    path = tmp_path / "post_action.jsonl"
    store = receipts.PostActionReceiptStore(path)
    template = receipts.PostActionReceipt(
        receipt_id="post-0",
        will_receipt_id="will-0",
        executor_name="unit",
        actual_outcome="success",
        output_hash="sha256:" + "0" * 64,
        error_status="",
        welfare_transaction_id="tx-0",
        verification_evidence={"observation": {"effect_verified": True}},
    )
    for index in range(70):
        store.record(
            replace(
                template,
                receipt_id=f"post-{index}",
                will_receipt_id=f"will-{index}",
                welfare_transaction_id=f"tx-{index}",
            )
        )

    template.verification_evidence["observation"]["effect_verified"] = False
    assert len(store.list_receipts()) == 64
    assert store.get_receipt("post-0") is None
    retained = store.get_receipt("post-69")
    assert retained is not None
    assert retained.verification_evidence["observation"]["effect_verified"] is True

    reloaded = receipts.PostActionReceiptStore(path)
    assert len(reloaded.list_receipts()) == 64
    assert reloaded.get_receipt("post-6") is not None
    assert reloaded.get_receipt("post-5") is None


def test_post_action_store_rejects_oversized_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import core.runtime.post_action_receipt as receipts

    monkeypatch.setattr(receipts._MAX_BYTES_FLAG, "value", lambda: 16_384)
    store = receipts.PostActionReceiptStore(tmp_path / "post_action.jsonl")
    receipt = receipts.PostActionReceipt(
        receipt_id="post-large",
        will_receipt_id="will-large",
        executor_name="unit",
        actual_outcome="success",
        output_hash="sha256:" + "0" * 64,
        error_status="",
        welfare_transaction_id="tx-large",
        verification_evidence={"blob": "x" * 20_000},
    )

    with pytest.raises(ValueError, match="exceeds maximum serialized size"):
        store.record(receipt)
    assert not store.persist_path.exists()
