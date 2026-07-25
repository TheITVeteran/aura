"""Contract tests: pre-action cortex loop.

One cognitive thread across a consequential action: rehearsal before,
objective-discrepancy reconciliation after, continuity carried as typed
action_thread slot items, and receipted skips whenever the latent organ
cannot run.
"""

import asyncio

from core.brain import preaction_cortex as pac


class FakeLatentService:
    def __init__(self, ok=True, text="Predicted: file appears; precondition: dir exists.", reason=""):
        self.ok = ok
        self.text = text
        self.reason = reason
        self.calls: list[dict] = []

    async def deep_reason(self, objective, **kwargs):
        self.calls.append({"objective": objective, **kwargs})
        if not self.ok:
            return {"ok": False, "reason": self.reason or "generation_gate_busy"}
        return {
            "ok": True,
            "text": self.text,
            "receipt": {
                "episode_id": f"ep-{len(self.calls)}",
                "steps_taken": 3,
                "honest_flags": [],
                "workspace_broadcast": {"submitted": True, "accepted": True},
            },
        }


def _thread():
    return pac.PreActionCortexThread(
        domain="network_call", action_name="post_webhook", request_digest="d" * 8
    )


def test_only_consequential_domains_buy_deliberation():
    assert pac.deliberation_worthy("network_call")
    assert pac.deliberation_worthy("SELF_MODIFICATION")
    assert not pac.deliberation_worthy("reflection")
    assert not pac.deliberation_worthy("response")


def test_rehearsal_runs_and_seeds_the_thread(monkeypatch):
    service = FakeLatentService()
    monkeypatch.setattr(pac, "_latent_service", lambda: service)
    thread = _thread()
    receipt = asyncio.run(
        thread.rehearse(
            action_summary="POST https://example.test/hook",
            expectation_objective="the webhook endpoint records the event",
        )
    )
    assert receipt["ran"] is True
    assert "Predicted" in receipt["prediction"]
    assert receipt["episode_id"] == "ep-1"
    call = service.calls[0]
    assert call["domain"] == "action_rehearsal"
    assert call["foreground_request"] is True
    assert call["cognitive_context"] is None  # nothing to carry yet
    assert thread.thread_items[0]["source"] == "action_thread"
    assert thread.thread_items[0]["text"].startswith("[rehearsal]")


def test_rehearsal_skips_are_receipted(monkeypatch):
    monkeypatch.setattr(pac, "_latent_service", lambda: None)
    thread = _thread()
    receipt = asyncio.run(
        thread.rehearse(action_summary="x", expectation_objective="y")
    )
    assert receipt["ran"] is False
    assert receipt["skip_reason"] == "latent_cortex_absent"

    busy = FakeLatentService(ok=False, reason="generation_gate_busy")
    monkeypatch.setattr(pac, "_latent_service", lambda: busy)
    receipt = asyncio.run(
        _thread().rehearse(action_summary="x", expectation_objective="y")
    )
    assert receipt["ran"] is False
    assert receipt["skip_reason"] == "availability_failure:generation_gate_busy"

    monkeypatch.setenv("AURA_PREACTION_RLC", "0")
    receipt = asyncio.run(
        _thread().rehearse(action_summary="x", expectation_objective="y")
    )
    assert receipt["skip_reason"] == "disabled:AURA_PREACTION_RLC=0"


def test_confirmed_prediction_is_recorded_not_redeliberated(monkeypatch):
    service = FakeLatentService()
    monkeypatch.setattr(pac, "_latent_service", lambda: service)
    thread = _thread()
    receipt = asyncio.run(
        thread.reconcile(
            {"transport_succeeded": True, "effect_verified": True}
        )
    )
    assert receipt["discrepancy"] is False
    assert receipt["ran"] is False
    assert receipt["skip_reason"] == "prediction_confirmed"
    assert service.calls == []  # no episode spent on success


def test_discrepancy_replans_with_the_same_thread(monkeypatch):
    service = FakeLatentService(text="Retry changed: add auth header first.")
    monkeypatch.setattr(pac, "_latent_service", lambda: service)
    thread = _thread()
    asyncio.run(
        thread.rehearse(
            action_summary="POST https://example.test/hook",
            expectation_objective="event recorded",
        )
    )
    receipt = asyncio.run(
        thread.reconcile(
            {
                "transport_succeeded": False,
                "effect_verified": False,
                "status": "failed_recoverable",
                "error": "HTTP 401 unauthorized",
            }
        )
    )
    assert receipt["discrepancy"] is True
    assert receipt["ran"] is True
    assert "Retry changed" in receipt["replan"]
    assert receipt["workspace_broadcast_submitted"] is True
    reconcile_call = service.calls[1]
    assert reconcile_call["domain"] == "action_reconciliation"
    carried = [item["text"] for item in reconcile_call["cognitive_context"]]
    assert any(text.startswith("[rehearsal]") for text in carried)
    assert any("HTTP 401" in text for text in carried)
    # Thread now holds rehearsal → observed → replan, newest-bounded.
    assert len(thread.thread_items) <= 3
    assert thread.thread_items[-1]["text"].startswith("[replan]")


def test_unverified_effect_after_transport_success_is_a_discrepancy(monkeypatch):
    service = FakeLatentService(text="Verify manually; effect unobserved.")
    monkeypatch.setattr(pac, "_latent_service", lambda: service)
    thread = _thread()
    receipt = asyncio.run(
        thread.reconcile(
            {
                "transport_succeeded": True,
                "effect_verified": False,
                "status": "success_unverified",
            }
        )
    )
    assert receipt["discrepancy"] is True
    assert receipt["ran"] is True
    objective = service.calls[0]["objective"]
    assert "never verified" in objective


def test_full_receipt_shape():
    thread = _thread()
    receipt = thread.to_receipt()
    assert receipt["schema"] == pac.PREACTION_SCHEMA
    assert receipt["action_name"] == "post_webhook"
    assert receipt["rehearsal"] == {} and receipt["reconciliation"] == {}


def test_request_digest_preserves_full_scheme_prefixed_identity():
    digest = "sha256:" + "a" * 64
    thread = pac.PreActionCortexThread(
        domain="network_call",
        action_name="post_webhook",
        request_digest=digest,
    )
    assert thread.request_digest == digest
