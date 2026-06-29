from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.self_modification.repair_import_policy import RepairImportPolicy


def test_builtin_safe_imports_remain_available(tmp_path: Path) -> None:
    policy = RepairImportPolicy(tmp_path / "policy.json")
    assert policy.resolve("Path") == "from pathlib import Path"
    assert policy.resolve("subprocess") is None


def test_policy_can_grow_only_with_governance_receipt(tmp_path: Path) -> None:
    path = tmp_path / "policy.json"
    policy = RepairImportPolicy(path)
    approved = policy.approve(
        "Counter",
        "from collections import Counter",
        governance_receipt_id="will-import-1",
    )
    assert approved.symbol == "Counter"
    assert RepairImportPolicy(path).resolve("Counter") == "from collections import Counter"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["imports"][0]["governance_receipt_id"] == "will-import-1"


@pytest.mark.parametrize(
    "symbol,statement",
    [
        ("subprocess", "import subprocess"),
        ("system", "from os import system"),
        ("socket", "import socket"),
        ("Counter", "value = 1"),
    ],
)
def test_policy_refuses_capability_escalation(tmp_path: Path, symbol: str, statement: str) -> None:
    policy = RepairImportPolicy(tmp_path / "policy.json")
    with pytest.raises(ValueError):
        policy.approve(symbol, statement, governance_receipt_id="will-import-2")
