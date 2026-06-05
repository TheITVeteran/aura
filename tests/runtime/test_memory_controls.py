"""tests/runtime/test_memory_controls.py — Unit tests for memory control operations.
"""
from __future__ import annotations

import asyncio
import tempfile
import pytest

from core.brain.semantic_memory import SemanticMemory
from core.governance.will import ActionDomain, WillDecision, WillOutcome
from core.governance_context import GovernanceViolation, governed_scope


def _approved_memory_decision() -> WillDecision:
    return WillDecision(
        receipt_id="will-memory-control-test",
        outcome=WillOutcome.PROCEED,
        domain=ActionDomain.MEMORY_WRITE,
        reason="test approval",
        source="test_memory_controls",
    )


def test_semantic_memory_controls():
    with tempfile.TemporaryDirectory() as tmp_dir:
        mem = SemanticMemory(memory_dir=tmp_dir)
        
        # Add a test memory
        mem.add_memory("Aura is a digital assistant designed by Google DeepMind.", {"will_receipt_id": "test_receipt_123"})
        assert mem.memory_count == 1
        
        # Test search
        results = mem.search_memories("assistant")
        assert len(results) == 1
        record_id = results[0]["id"]
        
        # Test edit
        edited = mem.edit_memory(record_id, "Aura is a digital partner designed by Google DeepMind.")
        assert edited is True
        assert mem.metadata[0]["text"] == "Aura is a digital partner designed by Google DeepMind."
        
        # Test freeze
        froze = mem.freeze_memory(record_id, True)
        assert froze is True
        assert mem.metadata[0]["tags"].get("frozen") is True
        
        # Edit should fail when frozen
        edited_fail = mem.edit_memory(record_id, "Aura is a digital partner.")
        assert edited_fail is False
        assert mem.metadata[0]["text"] == "Aura is a digital partner designed by Google DeepMind."
        
        # Unfreeze
        mem.freeze_memory(record_id, False)
        
        # Test contest
        contested = mem.contest_memory(record_id, True)
        assert contested is True
        assert mem.metadata[0]["tags"].get("contested") is True
        
        # Search should now exclude the memory because it's contested
        results_contested = mem.search_memories("partner")
        assert len(results_contested) == 0
        
        # Uncontest
        mem.contest_memory(record_id, False)
        results_uncontested = mem.search_memories("partner")
        assert len(results_uncontested) == 1
        
        # Test delete
        deleted = mem.delete_memory(record_id)
        assert deleted is True
        
        # Search should now exclude deleted memory
        results_deleted = mem.search_memories("partner")
        assert len(results_deleted) == 0
        
        # Test provenance
        provenance = mem.get_provenance(record_id)
        assert provenance["id"] == record_id
        assert provenance["will_receipt_id"] == "test_receipt_123"


def test_semantic_memory_controls_fail_closed_under_strict_governance(monkeypatch):
    async def scenario() -> None:
        monkeypatch.setenv("AURA_REQUIRE_GOVERNANCE", "1")
        with tempfile.TemporaryDirectory() as tmp_dir:
            mem = SemanticMemory(memory_dir=tmp_dir)
            mem.add_memory("Aura should only mutate memories through governed control.", {})
            record_id = mem.metadata[0]["id"]

            with pytest.raises(GovernanceViolation):
                mem.edit_memory(record_id, "unguarded edit")

            async with governed_scope(_approved_memory_decision()):
                assert mem.edit_memory(record_id, "governed edit") is True

            assert mem.metadata[0]["text"] == "governed edit"
            assert mem.metadata[0]["tags"]["last_control_receipt_id"] == "will-memory-control-test"

    asyncio.run(scenario())
