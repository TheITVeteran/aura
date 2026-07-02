"""Ratchet: no NEW blocking gateway/atomic writes inside async functions.

Every one of the 12 recorded live loop-wedge crashes came from a synchronous
fsync running on the event loop. The async write lane exists for this
(`FileWriteGateway.*_async`, `core.runtime.atomic_writer.async_atomic_*`);
this test freezes the historical offenders and fails when a new one appears,
or when a fixed one lingers in the allowlist.

If this test fails on code you just wrote: use the *_async gateway methods or
async_atomic_* writers instead of calling the sync writers from async code.
If you just FIXED an entry, delete it from the allowlist below.
"""
from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_SYNC_WRITE_CALLS = {
    "atomic_write_bytes",
    "atomic_write_text",
    "atomic_write_json",
    "atomic_append_text",
    "write_text",
    "write_bytes",
    "append_text",
}

# (file, async function, callee) — historical debt only. This list may only
# shrink. The hot per-turn paths (memory_write_gateway, episode_store,
# state_gateway, self_model, conversation_persistence) are already converted.
ALLOWED_LEGACY_OFFENDERS: frozenset[tuple[str, str, str]] = frozenset(
    {
        ("core/adaptation/nightly_lora.py", "run", "write_text"),
        ("core/adaptation/post_training_validator.py", "_write_validation_log", "write_text"),
        ("core/adaptation/post_training_validator.py", "quarantine_adapter", "write_text"),
        ("core/adaptation/safe_optimizer.py", "_backup_current_weights", "write_bytes"),
        ("core/adaptation/safe_optimizer.py", "_rollback", "write_bytes"),
        ("core/adaptation/safe_optimizer.py", "_run_training_command", "write_bytes"),
        ("core/adaptation/safe_optimizer.py", "_run_training_command", "write_text"),
        ("core/adaptation/self_optimizer.py", "optimize", "append_text"),
        ("core/adaptation/self_optimizer.py", "optimize", "write_text"),
        ("core/adaptation/star_reasoner.py", "_flush_accepted", "append_text"),
        ("core/agency/sandboxed_modifier.py", "_apply_via_worktree", "atomic_write_text"),
        ("core/agency/tool_orchestrator.py", "execute_python", "atomic_write_text"),
        ("core/body/file_motor.py", "actuate", "write_text"),
        ("core/brain/cognitive_patch.py", "apply", "write_text"),
        ("core/brain/symbolic_sandbox.py", "run", "atomic_write_text"),
        ("core/brain/verifiers/code_engine.py", "_ruff", "atomic_write_text"),
        ("core/capabilities/document_service.py", "create_pdf", "write_bytes"),
        ("core/capabilities/document_service.py", "create_pdf", "write_text"),
        ("core/capabilities/document_service.py", "create_text", "write_text"),
        ("core/capabilities/file_broker.py", "write_bytes", "write_bytes"),
        ("core/capabilities/file_broker.py", "write_file", "write_text"),
        ("core/capabilities/web_asset_handler.py", "download_image", "write_bytes"),
        ("core/collective/probe_manager.py", "deploy_probe", "atomic_write_text"),
        ("core/daemon.py", "start", "atomic_write_text"),
        ("core/embodiment/world_bridge.py", "_file_workspace_handler", "atomic_write_text"),
        ("core/environments/terminal_grid/nethack_adapter.py", "start", "atomic_write_text"),
        ("core/guardians/airlock.py", "process_mutation", "atomic_write_text"),
        ("core/knowledge/bottling.py", "bottle", "atomic_write_text"),
        ("core/learning/rsi_gauntlet.py", "run", "atomic_write_text"),
        ("core/mutate.py", "apply_mutation", "atomic_write_text"),
        ("core/planning/mission_state.py", "_execute_node", "write_text"),
        ("core/resilience/startup_validator.py", "_check_sys_02", "atomic_write_text"),
        ("core/runtime/action_executor.py", "execute", "write_bytes"),
        ("core/runtime/action_executor.py", "execute", "write_text"),
        ("core/safety/self_preservation_safe.py", "create_backup", "atomic_write_text"),
        ("core/self/mind_state_export.py", "export_mind", "write_bytes"),
        ("core/self_improvement/deterministic_comparator.py", "compare", "atomic_write_text"),
        ("core/self_modification/code_repair.py", "run_custom_probe", "atomic_write_text"),
        ("core/self_modification/safe_pipeline.py", "run", "atomic_write_text"),
        ("core/self_modification/self_modification_engine.py", "trigger_synthetic_test", "atomic_write_text"),
        ("core/self_modification/shadow_runtime.py", "_run_in_subprocess", "write_text"),
        ("core/self_modification/shadow_runtime.py", "test_mutation", "write_text"),
        ("core/shell.py", "write_file_safe", "write_text"),
        ("core/skills/branching_futures.py", "execute", "write_text"),
        ("core/skills/code_repl.py", "_execute_via_subprocess", "write_text"),
        ("core/skills/manifest_to_device.py", "execute", "write_bytes"),
        ("core/skills/manim_renderer.py", "execute", "write_text"),
        ("core/skills/reddit_adapter.py", "_save_session", "write_text"),
        ("core/skills/self_evolution.py", "execute", "write_text"),
        ("core/skills/train_self.py", "_collect_high_value_memories", "append_text"),
        ("core/skills/train_self.py", "_trigger_finetuning", "append_text"),
        ("core/skills/train_self.py", "_trigger_finetuning", "atomic_write_text"),
        ("core/social/community.py", "send", "append_text"),
        ("core/sovereign/local_sandbox.py", "run_code", "atomic_write_text"),
    }
)


def _scan_offenders() -> set[tuple[str, str, str]]:
    offenders: set[tuple[str, str, str]] = set()
    for path in (PROJECT_ROOT / "core").rglob("*.py"):
        if "__pycache__" in str(path):
            continue
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue
        rel = str(path.relative_to(PROJECT_ROOT))

        class Visitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.async_stack: list[str] = []

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                self.async_stack.append(node.name)
                self.generic_visit(node)
                self.async_stack.pop()

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                # A sync def nested in an async def runs wherever it is
                # called from (thread executors included) — not our target.
                saved = self.async_stack
                self.async_stack = []
                self.generic_visit(node)
                self.async_stack = saved

            def visit_Call(self, node: ast.Call) -> None:
                if self.async_stack:
                    if isinstance(node.func, ast.Name):
                        name = node.func.id
                    elif isinstance(node.func, ast.Attribute):
                        name = node.func.attr
                    else:
                        name = ""
                    if name in _SYNC_WRITE_CALLS:
                        offenders.add((rel, self.async_stack[-1], name))
                self.generic_visit(node)

        Visitor().visit(tree)
    return offenders


def test_no_new_blocking_writes_in_async_functions():
    offenders = _scan_offenders()
    new = offenders - ALLOWED_LEGACY_OFFENDERS
    assert not new, (
        "NEW blocking write(s) inside async functions — this exact pattern "
        "froze the live event loop for 20 minutes and crash-cycled the "
        "runtime. Use FileWriteGateway.*_async or async_atomic_* instead:\n"
        + "\n".join(f"  {f}:{fn}() -> {call}" for f, fn, call in sorted(new))
    )


def test_allowlist_contains_no_fixed_entries():
    offenders = _scan_offenders()
    stale = ALLOWED_LEGACY_OFFENDERS - offenders
    assert not stale, (
        "These allowlist entries are fixed — delete them from "
        "ALLOWED_LEGACY_OFFENDERS so the ratchet only tightens:\n"
        + "\n".join(f"  {f}:{fn}() -> {call}" for f, fn, call in sorted(stale))
    )


def test_hot_paths_stay_converted():
    """The per-turn lanes must never regress to on-loop fsyncs."""
    offenders = _scan_offenders()
    hot_files = {
        "core/memory/memory_write_gateway.py",
        "core/memory/episode_store.py",
        "core/state/state_gateway.py",
        "core/self_model.py",
    }
    regressions = {o for o in offenders if o[0] in hot_files}
    assert not regressions, f"hot path regressed to blocking writes: {sorted(regressions)}"
