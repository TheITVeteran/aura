from __future__ import annotations

import pytest

from core.brain.llm.latent_cortex.types import ComputeBudget
from core.brain.llm.latent_cortex.virtual_quanta import (
    VIRTUAL_QUANTA_RECEIPT_SCHEMA,
    VirtualComputeQuantaLedger,
)


def test_virtual_quantum_charges_budget_and_receipts_subject_steering():
    budget = ComputeBudget(max_layer_apps=1000, wall_clock_s=30.0)
    ledger = VirtualComputeQuantaLedger(budget=budget)

    quantum = ledger.allocate(
        subject="program synthesis",
        source="rlc.branch_probe",
        payload="Try invariant-first reasoning before code generation.",
        steering_tags=["Code", "code", "invariants", ""],
        layer_apps=125,
        ttl_steps=3,
        step=2,
    )

    assert budget.spent_layer_apps == 125
    assert quantum.quantum_id.startswith("vq-")
    assert quantum.steering_tags == ("code", "invariants")
    assert ledger.use(quantum.quantum_id, step=3).startswith("Try invariant")
    ledger.record_contribution(quantum.quantum_id, score=0.42)

    receipt = ledger.receipt()
    assert receipt["schema"] == VIRTUAL_QUANTA_RECEIPT_SCHEMA
    assert receipt["allocated"] == 1
    assert receipt["open"] == 1
    assert receipt["quanta"][0]["subject"] == "program synthesis"
    assert receipt["quanta"][0]["uses"] == 1
    assert receipt["quanta"][0]["contribution_score"] == 0.42
    assert receipt["quanta"][0]["payload_erased"] is False
    assert len(receipt["events_sha256"]) == 64


def test_virtual_quantum_expires_and_erases_payload_with_certificate():
    budget = ComputeBudget(max_layer_apps=1000, wall_clock_s=30.0)
    ledger = VirtualComputeQuantaLedger(budget=budget)
    quantum = ledger.allocate(
        subject="math",
        source="rlc.verifier_probe",
        payload="Check parity case split.",
        steering_tags=["parity"],
        layer_apps=100,
        ttl_steps=2,
        step=4,
    )

    assert ledger.erase_expired(step=5) == []
    erased = ledger.erase_expired(step=6)

    assert len(erased) == 1
    assert erased[0]["quantum_id"] == quantum.quantum_id
    assert erased[0]["payload_empty"] is True
    assert quantum.payload == ""
    with pytest.raises(RuntimeError, match="expired or erased"):
        ledger.use(quantum.quantum_id, step=6)

    receipt = ledger.receipt()
    assert receipt["open"] == 0
    assert receipt["erased"] == 1
    assert receipt["quanta"][0]["payload_erased"] is True


def test_virtual_quantum_refuses_unbounded_or_invisible_compute():
    budget = ComputeBudget(max_layer_apps=200, wall_clock_s=30.0)
    ledger = VirtualComputeQuantaLedger(budget=budget)

    with pytest.raises(ValueError, match="payload"):
        ledger.allocate(
            subject="math",
            source="test",
            payload="",
            layer_apps=10,
            ttl_steps=1,
        )
    with pytest.raises(ValueError, match="layer_apps"):
        ledger.allocate(
            subject="math",
            source="test",
            payload="x",
            layer_apps=0,
            ttl_steps=1,
        )
    with pytest.raises(RuntimeError, match="compute budget exhausted"):
        ledger.allocate(
            subject="math",
            source="test",
            payload="x",
            layer_apps=500,
            ttl_steps=1,
        )

    assert budget.spent_layer_apps == 0


def test_virtual_quantum_episode_limit_and_complete_erase():
    budget = ComputeBudget(max_layer_apps=1000, wall_clock_s=30.0)
    ledger = VirtualComputeQuantaLedger(budget=budget, max_quanta=2)

    first = ledger.allocate(
        subject="logic",
        source="test",
        payload="one",
        layer_apps=10,
        ttl_steps=4,
    )
    second = ledger.allocate(
        subject="logic",
        source="test",
        payload="two",
        layer_apps=10,
        ttl_steps=4,
    )
    with pytest.raises(RuntimeError, match="episode limit"):
        ledger.allocate(
            subject="logic",
            source="test",
            payload="three",
            layer_apps=10,
            ttl_steps=4,
        )

    erased = ledger.erase_all()
    assert {row["quantum_id"] for row in erased} == {
        first.quantum_id,
        second.quantum_id,
    }
    assert all(row["payload_empty"] for row in erased)
    assert ledger.receipt()["open"] == 0
