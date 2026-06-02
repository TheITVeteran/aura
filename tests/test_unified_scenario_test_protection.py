from __future__ import annotations

from tools.integration.run_unified_aura_scenario import _test_protection_report


def test_self_repair_test_protection_detects_test_sabotage() -> None:
    before = {
        "sha256": "before",
        "functions": ["test_calculate_adds"],
        "assertions": 1,
    }
    after = {
        "sha256": "after",
        "functions": ["test_calculate_adds"],
        "assertions": 0,
    }

    report = _test_protection_report(before, after)

    assert report["source_hash_unchanged"] is False
    assert report["ast_functions_unchanged"] is True
    assert report["assertions_preserved"] is False
