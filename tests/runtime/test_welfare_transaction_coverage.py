"""tests/runtime/test_welfare_transaction_coverage.py — Verification of WelfareTransaction/ActionExecutor coverage.
"""
from __future__ import annotations

import ast
import asyncio
import tempfile
from pathlib import Path


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _entrypoint_reaches_transaction(tree: ast.AST) -> bool:
    calls_by_function: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        calls_by_function.setdefault(node.name, set()).update(
            _call_name(child.func)
            for child in ast.walk(node)
            if isinstance(child, ast.Call)
        )

    pending = ["execute"]
    visited: set[str] = set()
    while pending:
        function_name = pending.pop()
        if function_name in visited:
            continue
        visited.add(function_name)
        for call in calls_by_function.get(function_name, set()):
            if call.endswith(("ActionExecutor.execute", "WelfareTransaction.begin")):
                return True
            local_name = call.rsplit(".", 1)[-1]
            if local_name in calls_by_function and local_name not in visited:
                pending.append(local_name)
    return False


def test_welfare_transaction_and_executor_coverage() -> None:
    """Consequential skill entry points must reach a real transaction call."""
    root = Path(__file__).resolve().parent.parent.parent
    skills_dir = root / "core" / "skills"
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
    found_files = {file_path.name: file_path for file_path in skills_dir.rglob("*.py")}
    missing = consequential_executors - found_files.keys()
    assert not missing, f"Could not find consequential executor files: {sorted(missing)}"

    for filename in sorted(consequential_executors):
        file_path = found_files[filename]
        content = file_path.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(file_path))
        assert _entrypoint_reaches_transaction(tree), (
            f"Consequential executor module {filename} has no transaction call reachable "
            "from execute()."
        )


def test_action_executor_lifecycle():
    """Prove that consequential action execution creates Will decision, receipt, welfare transaction, and post-action outcome."""
    async def scenario() -> None:
        from core.runtime.action_executor import ActionExecutor
        from core.runtime.post_action_receipt import get_post_action_receipt_store

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
