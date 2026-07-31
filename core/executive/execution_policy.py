"""Canonical effect and risk classification for governed tool execution.

The capability engine, orchestrator, and authority gateway must reason about the
same operation.  This module resolves a concrete invocation into one effect
scope and one conservative risk class without treating an unknown tool as safe.
"""

from __future__ import annotations

import json
import os
import shlex
from pathlib import Path
from typing import Any

from core.skills.catalog_policy import resolve_skill_policy

RISK_ORDER = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "critical": 3,
}

_GENERIC_EFFECT_SCOPES = {
    "browser": "external_io",
    "command": "privileged_mutation",
    "curiosity_web_search": "read_only",
    "edit_file": "read_write_artifacts",
    "execute": "privileged_mutation",
    # No generic "file_write" mapping: the canonical file skill is
    # file_operation (scoped per-invocation above). A bare unregistered
    # file_write stays "unknown" → classified critical by default —
    # presuming a write scope for an unowned tool under-gated it.
    "get_time": "status",
    "grep_search": "read_only",
    "list_dir": "read_only",
    "multi_replace_file_content": "read_write_artifacts",
    "notify_user": "external_io",
    "read_file": "read_only",
    "replace_file_content": "read_write_artifacts",
    "run_command": "privileged_mutation",
    "run_python": "sandboxed_compute",
    "search_web": "read_only",
    "self_diagnosis": "status",
    "sensory_motor_browser_research": "read_only",
    "status": "status",
    "subconscious_sandbox_probe": "sandboxed_compute",
    "swarm_debate": "pure_compute",
    "system_health": "status",
    "terminal": "privileged_mutation",
    "view_file": "read_only",
    "write_file": "read_write_artifacts",
    "write_to_file": "read_write_artifacts",
}

_CRITICAL_TOOLS = frozenset(
    {
        "auto_refactor",
        "command",
        "computer_use",
        "desktop_task",
        "execute",
        "install_package",
        "manage_abilities",
        "os_automation",
        "os_manipulation",
        "run_command",
        "self_evolution",
        "self_improvement",
        "self_modify",
        "self_repair",
        "shell",
        "sovereign_terminal",
        "terminal",
        "train_self",
        "web_interlocutor",
    }
)


def normalize_tool_name(value: Any) -> str:
    return str(value or "").strip().lower()


def normalize_risk(value: Any, *, default: str = "critical") -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in RISK_ORDER else default


def risk_at_most(actual: Any, maximum: Any) -> bool:
    return RISK_ORDER[normalize_risk(actual)] <= RISK_ORDER[normalize_risk(maximum)]


def _is_path_within_workspace(path: Any) -> bool:
    raw = str(path or "").strip()
    if not raw:
        return False
    try:
        root = Path.cwd().resolve()
        target = Path(raw).expanduser()
        if not target.is_absolute():
            target = root / target
        resolved = target.resolve()
        return os.path.commonpath([str(root), str(resolved)]) == str(root)
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def _workspace_file_scope(params: dict[str, Any]) -> str | None:
    action = str((params or {}).get("action") or "").strip().lower()
    if action not in {"append", "copy", "exists", "list", "patch", "read", "write"}:
        return None
    if not _is_path_within_workspace((params or {}).get("path")):
        return None
    if action == "copy" and not _is_path_within_workspace((params or {}).get("destination")):
        return None
    return "workspace_file_io"


def _computer_use_scope(params: dict[str, Any]) -> str | None:
    action = str((params or {}).get("action") or "").strip().lower()
    if action in {"get_clipboard", "read_menu_clock", "read_screen_text", "wait"}:
        return "read_only"
    if action in {
        "click",
        "hotkey",
        "open_app",
        "open_url",
        "run_applescript",
        "scroll",
        "set_clipboard",
        "type",
    }:
        return "foreground_desktop_control"
    if action in {"move_file", "render_text_pdf", "write_text_file"}:
        return "desktop_file_io"
    if action != "run_command":
        return None
    try:
        argv = shlex.split(str((params or {}).get("target") or ""))
    except ValueError:
        return None
    if not argv:
        return None
    binary = Path(argv[0]).name
    if binary in {"cat", "echo", "find", "grep", "ls", "pwd", "tree"}:
        return "sandboxed_compute"
    if binary == "git":
        subcommand = argv[1] if len(argv) > 1 else ""
        if subcommand in {"branch", "diff", "log", "rev-parse", "show", "status"}:
            return "sandboxed_compute"
        return "subprocess"
    if binary in {"pip", "python3"} and len(argv) == 2 and argv[1] in {"--version", "-V"}:
        return "sandboxed_compute"
    return "subprocess"


def _auto_refactor_scope(params: dict[str, Any]) -> str:
    params = params or {}
    mode = str(params.get("mode") or params.get("action") or "scan").strip().lower()
    if bool(
        params.get("apply")
        or params.get("write")
        or params.get("commit")
        or params.get("promote")
        or params.get("allow_mutation")
        or mode in {"apply", "commit", "promote", "rewrite", "write"}
    ):
        return "privileged_mutation"
    return "sandboxed_compute" if bool(params.get("run_tests")) else "read_only"


def _test_generator_scope(params: dict[str, Any]) -> str:
    """A read-only request writes only to an ephemeral sandbox."""

    return (
        "sandboxed_compute"
        if bool((params or {}).get("read_only"))
        else "read_write_artifacts"
    )


def _self_evolution_scope(params: dict[str, Any]) -> str:
    """Proposal-only inspection is distinct from applying a code mutation."""

    arguments = params or {}
    action = str(arguments.get("action") or "propose").strip().lower()
    if action in {"", "propose"} and bool(arguments.get("read_only")):
        return "read_only"
    return "privileged_mutation"



# Effect scope belongs to what an invocation actually does, not to what its
# skill is capable of. desktop_task is the composite case: it can write files,
# drive AppleScript and read the screen, so its blanket declaration is the
# widest thing any of its steps might do. Governing every invocation at that
# width blocks a screen-reading objective as if it were a filesystem write, and
# governing at the parent's width also misses a child that does more. The
# honest scope of a plan is the widest scope among the steps it actually
# contains — computed here, from the steps, at the layer that knows them.
_DESKTOP_ACTION_SCOPES: dict[str, str] = {
    "get_clipboard": "read_only",
    "inspect_screen": "read_only",
    "read_menu_clock": "read_only",
    "read_screen_text": "read_only",
    "wait": "read_only",
    "click": "foreground_desktop_control",
    "hotkey": "foreground_desktop_control",
    "open_app": "foreground_desktop_control",
    "open_url": "foreground_desktop_control",
    "run_applescript": "foreground_desktop_control",
    "scroll": "foreground_desktop_control",
    "set_clipboard": "foreground_desktop_control",
    "system_control": "foreground_desktop_control",
    "type": "foreground_desktop_control",
    "create_folder": "desktop_file_io",
    "fetch_topic_image": "desktop_file_io",
    "move_file": "desktop_file_io",
    "render_text_pdf": "desktop_file_io",
    "write_text_file": "desktop_file_io",
}
# Widest last: a plan is governed by the most consequential thing in it.
_DESKTOP_SCOPE_RANK: tuple[str, ...] = (
    "read_only",
    "foreground_desktop_control",
    "desktop_file_io",
)


def _desktop_task_scope(params: dict[str, Any]) -> str | None:
    """The widest scope among a desktop plan's declared steps, or None."""
    raw_steps = (params or {}).get("steps")
    if isinstance(raw_steps, str):
        try:
            raw_steps = json.loads(raw_steps)
        except (TypeError, ValueError):
            return None
    if not isinstance(raw_steps, list) or not raw_steps:
        return None

    widest = -1
    for step in raw_steps:
        action = ""
        if isinstance(step, dict):
            action = str(step.get("action") or "").strip().lower()
        elif isinstance(step, str):
            action = step.strip().lower()
        scope = _DESKTOP_ACTION_SCOPES.get(action)
        if scope is None:
            # An unrecognised step could do anything; refuse to narrow.
            return None
        widest = max(widest, _DESKTOP_SCOPE_RANK.index(scope))
    if widest < 0:
        return None
    return _DESKTOP_SCOPE_RANK[widest]


def resolve_execution_effect_scope(
    tool_name: Any,
    params: dict[str, Any] | None = None,
    *,
    declared_effect_scope: Any = "",
) -> str:
    """Resolve invocation-specific scope, returning ``unknown`` on ambiguity."""

    name = normalize_tool_name(tool_name)
    arguments = dict(params or {})
    if name == "file_operation":
        scoped = _workspace_file_scope(arguments)
        if scoped:
            return scoped
    elif name == "computer_use":
        scoped = _computer_use_scope(arguments)
        if scoped:
            return scoped
    elif name == "desktop_task":
        scoped = _desktop_task_scope(arguments)
        if scoped:
            return scoped
    elif name == "auto_refactor":
        return _auto_refactor_scope(arguments)
    elif name == "test_generator":
        return _test_generator_scope(arguments)
    elif name == "self_evolution":
        return _self_evolution_scope(arguments)
    elif name == "web_search":
        # Autonomous web research is READ-ONLY by contract: fetching and
        # summarizing never writes artifacts or posts. A broader scope
        # declared by a skill wrapper must not widen what autonomy may do
        # with a search (test_fictional_ai_runtime_contract pins this).
        return "read_only"
    elif name == "email_adapter":
        mode = str(arguments.get("mode") or "check").strip().lower()
        if mode in {"check", "read", "search"}:
            return "read_only"
    elif name == "reddit_adapter":
        mode = str(arguments.get("mode") or "browse").strip().lower()
        if mode in {
            "browse",
            "check_inbox",
            "check_shadowban",
            "read_post",
            "read_rules",
        }:
            return "read_only"

    declared = str(declared_effect_scope or "").strip().lower()
    policy = resolve_skill_policy(name, declared)
    if policy is not None:
        return policy.effect_scope
    return _GENERIC_EFFECT_SCOPES.get(name, "unknown")


def classify_execution_risk(
    tool_name: Any,
    params: dict[str, Any] | None = None,
    *,
    effect_scope: Any = "",
    metabolic_cost: int = 1,
) -> str:
    """Classify a concrete execution; unknown effects are critical by default."""

    name = normalize_tool_name(tool_name)
    arguments = dict(params or {})
    scope = str(effect_scope or "").strip().lower() or resolve_execution_effect_scope(
        name, arguments
    )
    if name == "run_code" or name == "run_python":
        stateful = bool(arguments.get("stateful", True))
        return "critical" if stateful else "high"
    if name == "auto_refactor":
        if scope == "privileged_mutation":
            return "critical"
        return "high" if scope == "sandboxed_compute" else "low"
    if name == "test_generator" and scope == "sandboxed_compute":
        return "high"
    if name == "self_evolution" and scope == "read_only":
        return "low"
    if name in _CRITICAL_TOOLS:
        if name == "computer_use" and scope == "read_only":
            return "low"
        return "critical"
    if scope in {"status", "read_only", "pure_compute"}:
        return "low"
    if scope == "sandboxed_compute":
        return "high"
    if scope in {"external_io", "state_mutation"}:
        return "medium"
    if scope in {
        "desktop_file_io",
        "foreground_browser_dialogue",
        "foreground_desktop_control",
        "read_write_artifacts",
        "workspace_file_io",
    }:
        return "high"
    if scope in {"privileged_mutation", "subprocess", "unknown"}:
        return "critical"
    if int(metabolic_cost or 1) >= 3:
        return "high"
    if int(metabolic_cost or 1) >= 2:
        return "medium"
    return "critical"


__all__ = [
    "RISK_ORDER",
    "classify_execution_risk",
    "normalize_risk",
    "normalize_tool_name",
    "resolve_execution_effect_scope",
    "risk_at_most",
]
