"""tests/runtime/test_welfare_transaction_coverage.py — Verification of WelfareTransaction/ActionExecutor coverage.
"""
from __future__ import annotations

import ast
import asyncio
from pathlib import Path


def test_welfare_transaction_and_executor_coverage():
    """Assert that consequential/effectful production modules utilize WelfareTransaction or ActionExecutor."""
    root = Path(__file__).resolve().parent.parent.parent
    skills_dir = root / "core" / "skills"
    
    # We collect files that are expected to perform real tool/OS effects
    effect_files = []
    if skills_dir.exists():
        effect_files = [f for f in skills_dir.rglob("*.py") if f.name != "__init__.py"]

    assert len(effect_files) > 0, "No skill files found to check for coverage"

    # Exact list of consequential executor modules that perform raw external effects
    consequential_executors = {
        "computer_use.py",
        "code_repl.py",
        "file_operation.py",
        "reddit_adapter.py",
        "email_adapter.py",
        "sovereign_browser.py",
        "web_search.py",
        "memory_ops.py",
        "self_repair.py",
    }

    found_files = []
    for file_path in effect_files:
        if file_path.name in consequential_executors:
            found_files.append(file_path)

    assert len(found_files) == len(consequential_executors), "Could not find all consequential executor files"

    for file_path in found_files:
        content = file_path.read_text(encoding="utf-8")
        
        # Check if the file imports or calls WelfareTransaction or ActionExecutor
        has_transaction = "WelfareTransaction" in content
        has_executor = "ActionExecutor" in content

        assert has_transaction or has_executor, (
            f"Consequential executor module {file_path.name} does not import "
            "or use WelfareTransaction or ActionExecutor to wrap executions."
        )


def test_action_executor_lifecycle():
    """Prove that consequential action execution creates Will decision, receipt, welfare transaction, and post-action outcome."""
    async def scenario() -> None:
        from core.runtime.action_executor import ActionExecutor
        from core.runtime.post_action_receipt import get_post_action_receipt_store
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=True) as tmp:
            params = {"path": tmp.name, "text": "test_welfare_lifecycle"}

            result = await ActionExecutor.execute(
                domain="file_write",
                action_name="write_test_file",
                params=params,
                source="test_suite_welfare",
            )

            assert result.get("ok") is True

            store = get_post_action_receipt_store()
            receipts = store.list_receipts()
            assert len(receipts) > 0

            matching = [r for r in receipts if r.executor_name == "write_test_file"]
            assert len(matching) > 0, "No post-action receipt found for our executor"

            receipt = matching[-1]
            assert receipt.actual_outcome == "success"
            assert receipt.will_receipt_id is not None
            assert receipt.welfare_transaction_id is not None
            assert receipt.output_hash.startswith("sha256:")

    asyncio.run(scenario())
