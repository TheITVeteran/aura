"""Symmetric rollback for enacted self-improvements (July external review).

Promotion was one-way: the improver enacted a verified fix but kept no
durable pre-image — an improvement that cannot be undone with the same
rigor it was applied with was never a governed improvement. These
contracts pin the write-ahead ledger and the verified restore.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from core.capabilities import self_code_improver as sci

pytestmark = pytest.mark.unit

ORIGINAL_MODULE = '''"""Test module."""


def add_numbers(case):
    return case["a"] - case["b"]  # BUG: subtracts


def untouched(case):
    return "leave me alone"
'''

IMPROVED_FUNC = '''def add_numbers(case):
    return case["a"] + case["b"]
'''


@pytest.fixture
def ledger_dir(tmp_path, monkeypatch):
    ledger = tmp_path / "enactments"
    monkeypatch.setattr(sci, "_ENACTMENT_LEDGER_DIR", ledger)
    return ledger


@pytest.fixture
def target(tmp_path, monkeypatch):
    # Self-improvement confines rewrites to AURA_SELF_CODE_ROOT; point it at the
    # temp dir so this fixture's victim module is a legitimate in-root target.
    monkeypatch.setenv("AURA_SELF_CODE_ROOT", str(tmp_path))
    module = tmp_path / "victim_module.py"
    module.write_text(ORIGINAL_MODULE, encoding="utf-8")
    return module


def _enact(target: Path) -> str:
    """Apply the improvement through the real ledger + write path."""
    original = target.read_text(encoding="utf-8")
    new_src = sci._replace_function(original, "add_numbers", IMPROVED_FUNC)
    record_id = asyncio.run(
        sci._record_enactment(
            path=target,
            func_name="add_numbers",
            goal="fix the subtraction bug",
            file_before=original,
            file_after=new_src,
            original_function=sci._extract_function_source(original, "add_numbers")[0],
            improved_function=IMPROVED_FUNC,
        )
    )
    target.write_text(new_src, encoding="utf-8")
    return record_id


class TestWriteAheadLedger:
    def test_record_carries_full_preimage(self, ledger_dir, target):
        record_id = _enact(target)
        record = json.loads((ledger_dir / f"{record_id}.json").read_text(encoding="utf-8"))
        assert 'case["a"] - case["b"]' in record["original_function_source"], (
            "the FULL pre-image must be durable, not a truncated echo"
        )
        assert record["file_sha_before"] != record["file_sha_after"]
        assert record["target_file"] == str(target)

    def test_latest_enactment_lookup(self, ledger_dir, target):
        record_id = _enact(target)
        record = sci.latest_enactment_for(str(target))
        assert record is not None and record["id"] == record_id
        assert sci.latest_enactment_for("/nonexistent/file.py") is None


class TestSymmetricRollback:
    def test_rollback_restores_byte_identical_function(self, ledger_dir, target):
        record_id = _enact(target)
        assert 'case["a"] + case["b"]' in target.read_text(encoding="utf-8")

        outcome = asyncio.run(sci.rollback_enactment(record_id))
        assert outcome["ok"] is True and outcome["status"] == "rolled_back"
        assert outcome["effect_verified"] is True
        assert outcome["receipt_persisted"] is True
        assert outcome["post_action_receipt_id"]
        restored = target.read_text(encoding="utf-8")
        assert 'case["a"] - case["b"]' in restored, "original behavior restored"
        assert "leave me alone" in restored, "unrelated code untouched"

    def test_rollback_without_id_uses_latest_record(self, ledger_dir, target):
        _enact(target)
        outcome = asyncio.run(sci.rollback_enactment(target_file=str(target)))
        assert outcome["ok"] is True

    def test_rollback_refuses_on_file_drift(self, ledger_dir, target):
        """A blind restore over someone else's edits would destroy work."""
        record_id = _enact(target)
        drifted = target.read_text(encoding="utf-8") + "\n\nEXTERNAL_EDIT = True\n"
        target.write_text(drifted, encoding="utf-8")

        outcome = asyncio.run(sci.rollback_enactment(record_id))
        assert outcome["ok"] is False
        assert outcome["status"] == "refused_file_drift"
        assert "EXTERNAL_EDIT" in target.read_text(encoding="utf-8"), "file untouched"

    def test_forced_rollback_overrides_drift_guard(self, ledger_dir, target):
        record_id = _enact(target)
        target.write_text(
            target.read_text(encoding="utf-8") + "\n\nEXTERNAL_EDIT = True\n",
            encoding="utf-8",
        )
        outcome = asyncio.run(sci.rollback_enactment(record_id, force=True))
        assert outcome["ok"] is True
        assert 'case["a"] - case["b"]' in target.read_text(encoding="utf-8")

    def test_missing_record_is_a_named_refusal(self, ledger_dir):
        outcome = asyncio.run(sci.rollback_enactment("nope-does-not-exist"))
        assert outcome == {"ok": False, "status": "no_enactment_record"}


def test_enactment_receipt_failure_triggers_symmetric_compensation(
    target,
    monkeypatch,
):
    async def _research(_goal):
        return []

    async def _generate(_prompt):
        return IMPROVED_FUNC

    async def _retain(*_args):
        return "retained"

    async def _record(**_kwargs):
        return "record-1"

    async def _write(**_kwargs):
        return {
            "ok": False,
            "status": "partial_success",
            "effect_verified": True,
            "receipt_persisted": False,
            "manual_reconciliation_required": True,
        }

    async def _rollback(*_args, **_kwargs):
        return {"ok": True, "status": "rolled_back"}

    def _verify(source, _func_name, _checks):
        return (1, []) if 'case["a"] + case["b"]' in source else (0, [])

    monkeypatch.setattr(sci, "_research", _research)
    monkeypatch.setattr(sci, "_generate", _generate)
    monkeypatch.setattr(sci, "_retain", _retain)
    monkeypatch.setattr(sci, "_record_enactment", _record)
    monkeypatch.setattr(sci, "_execute_self_code_write", _write)
    monkeypatch.setattr(sci, "rollback_enactment", _rollback)
    monkeypatch.setattr(sci, "_verify", _verify)

    result = asyncio.run(
        sci.improve_function(
            target_file=str(target),
            func_name="add_numbers",
            goal="fix addition",
            checks=[{"args": [{"a": 2, "b": 3}], "expected": 5}],
            max_iters=1,
            enact=True,
        )
    )

    assert result.enacted is False
    assert result.status == "enactment_receipt_failed_rolled_back"
    assert result.compensation == {"ok": True, "status": "rolled_back"}
    assert "receipt did not persist" in result.error
