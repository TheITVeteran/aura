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
def target(tmp_path):
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
