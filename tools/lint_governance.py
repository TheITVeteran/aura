#!/usr/bin/env python3
"""Inventory and ratchet consequential effect ownership.

The legacy governance lint matched a small list of historic method names and
therefore returned green while raw subprocess, network, filesystem, desktop,
gateway, and Will calls remained distributed across the runtime. This scanner
does not pretend the existing debt is already gone. It records every recognized
effect call by category, file, lexical scope, and resolved callee, then enforces
an exact checked-in baseline:

* a new bucket or an increased count is a governance regression;
* a removed/decreased bucket makes the baseline stale and requires an explicit
  refresh, preserving an auditable record of debt reduction;
* unreadable or unparsable production files fail the analyzer;
* canonical primitive owners are reported separately from migration debt.

The baseline is a ratchet, not an allow-list. Updating it is an explicit review
act and must not be used to normalize unexplained growth.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE = ROOT / "config" / "aura_effect_ownership_baseline.json"
BASELINE_SCHEMA_VERSION = 1

SCAN_ROOTS = ("core", "interface", "skills", "tools/longevity", "tools/chaos")
SKIP_DIR_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "archive",
    "aura_bench",
    "node_modules",
    "tests",
}

# Kept as compatibility exports for older proof tooling. The new scanner does
# not depend on these narrow suffixes or skip files in this list.
CONSEQUENTIAL_CALLS = (
    "memory_facade.write",
    "memory_facade.add",
    "memory_facade.persist_unsafe",
    "execute_tool",
    "shell_exec",
    "run_shell",
    "post_external",
    "modify_code",
    "fine_tune",
    "self_modify",
    "social_post",
    "structural_mutator.apply_patch",
    "shadow_ast_healer.repair",
    "wallet.execute",
)
ALLOW_LIST = {
    "core/agency/agency_orchestrator.py",
    "core/agency/autonomous_task_engine.py",
    "core/agency/capability_token.py",
    "core/agency/skill_library.py",
    # curiosity_daemon calls orchestrator.execute_tool — the CANONICAL
    # governed chain (origin, standing authority, Will, capability,
    # execution, closure), not a bypass of it (e593f5f2).
    "core/agi/curiosity_daemon.py",
    "core/agi/curiosity_explorer.py",
    "core/autonomy/behavior_controller.py",
    "core/autonomy/proactive_presence.py",
    "core/autonomy/research_cycle.py",
    "core/brain/llm/local_agent_client.py",
    "core/brain/react_loop.py",
    "core/cognitive/state_machine.py",
    "core/collective/delegator.py",
    "core/coordinators/cognitive_coordinator.py",
    "core/coordinators/metabolic_coordinator.py",
    "core/coordinators/tool_executor.py",
    "core/curiosity_engine.py",
    "core/embodiment/world_bridge.py",
    "core/executive/authority_gateway.py",
    "core/kernel/upgrades_10x.py",
    "core/orchestrator/main.py",
    "core/orchestrator/mixins/autonomy.py",
    "core/orchestrator/mixins/incoming_logic.py",
    "core/orchestrator/mixins/message_pipeline.py",
    "core/orchestrator/mixins/response_processing.py",
    "core/orchestrator/mixins/tool_execution.py",
    "core/phases/response_generation_unitary.py",
    "core/self_modification/safe_pipeline.py",
    "core/self_modification/self_modification_engine.py",
    "core/self_modification/shadow_ast_healer.py",
    "core/self_modification/structural_mutator.py",
    "core/soul.py",
    "core/sovereignty/migration.py",
    "core/sovereignty/wallet.py",
    "core/will.py",
    # /api/imagination/visualize renders Aura's own mental canvas via
    # execute_tool("image_gen", ...) — the CANONICAL governed chain
    # (owner-authenticated route, desktop execution contract, standing
    # authority, Will, constitution, capability token), not a bypass.
    "interface/routes/system.py",
}

CANONICAL_PRIMITIVE_OWNERS: dict[str, frozenset[str]] = {
    "raw_subprocess": frozenset({"core/runtime/subprocess_gateway.py"}),
    "raw_network": frozenset({"core/runtime/network_gateway.py"}),
    "raw_file_mutation": frozenset(
        {
            "core/runtime/atomic_writer.py",
            "core/brain/llm/latent_cortex/campaign_journal.py",
            # Private paired-action snapshots require no-follow opens,
            # owner/mode/link checks, directory fsync, key-first destruction,
            # and staged crash recovery that the general writer deliberately
            # does not expose. This owner accepts only its fixed internal
            # namespace and schema-bound campaign state, never caller paths.
            "core/brain/llm/latent_cortex/action_state_capture.py",
            # Detached campaign evidence is an immutable, no-follow verifier
            # and staged import is the sole transactional owner of its bounded
            # private arm artifacts. Neither accepts arbitrary user paths.
            "core/brain/llm/latent_cortex/detached_campaign_evidence.py",
            "core/brain/llm/latent_cortex/worker_attempt_import.py",
            # Verified-replay SFT publication owns one fixed, owner-private
            # candidate/evaluator namespace. Its sole raw mutation is a
            # no-follow, single-link, inode-bound pair-publication lock; all
            # payload and commit bytes still traverse FileWriteGateway.
            "core/learning/verified_replay_sft_publication.py",
            "core/runtime/file_read_gateway.py",
            "core/runtime/file_write_gateway.py",
            "core/runtime/shutdown_artifact_store.py",
        }
    ),
    "raw_desktop": frozenset({"core/runtime/desktop_action_gateway.py"}),
    "raw_browser": frozenset(),
    "direct_atomic_file_write": frozenset(
        {
            "core/memory/memory_write_gateway.py",
            # Immutable recurrence-training generations are a purpose-built
            # atomic evidence store, analogous to campaign_journal. It owns no
            # arbitrary user path and advances only a digest-bound pointer.
            "core/learning/recurrence_training_state.py",
            "core/runtime/atomic_writer.py",
            "core/runtime/file_write_gateway.py",
            "core/runtime/post_action_receipt.py",
            "core/runtime/receipts.py",
            "core/runtime/shutdown_artifact_store.py",
            "core/state/state_gateway.py",
        }
    ),
    "subprocess_gateway": frozenset(
        {
            "core/runtime/action_executor.py",
            "core/runtime/desktop_action_gateway.py",
            "core/runtime/skill_catalog_probe.py",
        }
    ),
    "network_gateway": frozenset({"core/runtime/action_executor.py"}),
    "file_write_gateway": frozenset(
        {
            "core/agency/tool_orchestrator.py",
            "core/agency/self_repair_backlog.py",
            # Independently checked verifier outcomes are append-only,
            # schema-bound calibration evidence under Aura's data directory.
            # The ledger accepts no arbitrary runtime action or user path and
            # writes only from its named internal governance scope.
            "core/brain/llm/latent_cortex/verifier_fusion.py",
            # External-effect transactions are digest-sealed, path-derived
            # records under Aura's data directory. The coordinator accepts no
            # caller-selected file path and writes only from its named scope.
            "core/brain/external_execute_coordinator.py",
            "core/brain/llm/latent_cortex/persistence.py",
            # Ontogeny owns only its schema-bound experience, reservoir,
            # learned-head, and authority records under the configured Aura
            # data root (or an explicitly injected test store). Every write
            # remains inside a named state-mutation governance scope.
            "core/ontogeny/authority.py",
            "core/ontogeny/experience.py",
            "core/ontogeny/service.py",
            "core/ontogeny/state.py",
            # The learned world model owns one fixed, schema-bound VRNN
            # checkpoint under Aura's data root. It accepts no caller path,
            # and every publication executes in its named state-mutation
            # scope through FileWriteGateway.
            "core/world_model/learned_world_model.py",
            # Singleton owns one fixed boot-refusal marker under Aura's private
            # run directory. It accepts no caller-selected path or payload and
            # publishes/clears only that bounded launcher coordination record.
            "core/utils/singleton.py",
            # Bus bags and cross-process leases are bounded internal evidence
            # stores. Their paths are runtime-derived, never user-selected,
            # and each write executes inside a named governed scope.
            "core/observability/bus_recorder.py",
            "core/observability/trace_events.py",
            "core/runtime/lease.py",
            # Task-disjoint prefix-stability calibration owns one exact,
            # content-addressed artifact schema. It accepts no runtime action
            # or arbitrary effect and is the sole writer for that evidence.
            "core/learning/prefix_stability.py",
            # Verified replay projection writes only its fixed custody commit
            # and exact candidate/evaluator artifact sets under Aura's private
            # RLC root, inside named internal governance scopes. It grants no
            # trainer authority and accepts no arbitrary artifact filenames.
            "core/learning/verified_replay_sft_publication.py",
            # Combined lineage publication owns one fixed candidate/evaluator
            # custody namespace. All payload and commit bytes traverse
            # FileWriteGateway and the result grants no training authority.
            "core/learning/combined_sft_lineage_publication.py",
            # The safe optimizer writes only its configured adapter state and
            # byte-identical backup through FileWriteGateway; rollback restores
            # that same bounded artifact inside the optimizer's governed lane.
            "core/adaptation/safe_optimizer.py",
            "core/runtime/action_executor.py",
            "core/runtime/detached_subprocess_broker.py",
            "core/runtime/flight_recorder.py",
            "core/security/tls_local.py",
            "core/self_improvement/program_dna.py",
            "infrastructure/rollback.py",
        }
    ),
    "desktop_action_gateway": frozenset({"core/runtime/action_executor.py"}),
    "memory_write_gateway": frozenset({"core/runtime/action_executor.py"}),
    "state_gateway": frozenset({"core/runtime/action_executor.py"}),
    "will_decision": frozenset(
        {
            "core/runtime/action_executor.py",
            "core/executive/authority_gateway.py",
            "core/governance/authority_gateway.py",
        }
    ),
    "action_executor": frozenset(),
}

_SUBPROCESS_CALLS = frozenset(
    {
        "asyncio.create_subprocess_exec",
        "asyncio.create_subprocess_shell",
        "multiprocessing.Process",
        "os.popen",
        "os.posix_spawn",
        "os.posix_spawnp",
        "os.spawnl",
        "os.spawnle",
        "os.spawnlp",
        "os.spawnlpe",
        "os.spawnv",
        "os.spawnve",
        "os.spawnvp",
        "os.spawnvpe",
        "os.system",
        "subprocess.Popen",
        "subprocess.call",
        "subprocess.check_call",
        "subprocess.check_output",
        "subprocess.run",
    }
)
_NETWORK_EXACT_CALLS = frozenset(
    {
        "http.client.HTTPConnection",
        "http.client.HTTPSConnection",
        "socket.create_connection",
        "urllib.request.build_opener",
        "urllib.request.urlopen",
        "websockets.connect",
    }
)
_NETWORK_PREFIXES = (
    "aiohttp.ClientSession().",
    "httpx.",
    "requests.",
    "urllib3.",
)
_BROWSER_EXACT_CALLS = frozenset(
    {
        "selenium.webdriver.Chrome",
        "selenium.webdriver.Edge",
        "selenium.webdriver.Firefox",
        "selenium.webdriver.Safari",
        "webbrowser.open",
        "webbrowser.open_new",
        "webbrowser.open_new_tab",
    }
)
_BROWSER_ACTION_METHODS = frozenset(
    {
        "check",
        "click",
        "evaluate",
        "fill",
        "get",
        "goto",
        "launch",
        "new_page",
        "press",
        "select_option",
        "set_input_files",
        "type",
        "uncheck",
    }
)
_RAW_FILE_EXACT_CALLS = frozenset(
    {
        "os.chmod",
        "os.link",
        "os.makedirs",
        "os.mkdir",
        "os.open",
        "os.remove",
        "os.removedirs",
        "os.rename",
        "os.renames",
        "os.replace",
        "os.rmdir",
        "os.symlink",
        "os.truncate",
        "os.unlink",
        "shutil.copy",
        "shutil.copy2",
        "shutil.copyfile",
        "shutil.copytree",
        "shutil.move",
        "shutil.rmtree",
    }
)
_PATH_MUTATION_METHODS = frozenset(
    {
        "chmod",
        "hardlink_to",
        "mkdir",
        "rmdir",
        "symlink_to",
        "touch",
        "unlink",
        "write_bytes",
        "write_text",
    }
)
_AMBIGUOUS_PATH_MUTATION_METHODS = frozenset({"rename", "replace"})
_MODED_FILE_OPEN_CALLS = frozenset(
    {
        "aiofiles.open",
        "builtins.open",
        "bz2.open",
        "codecs.open",
        "gzip.open",
        "io.open",
        "lzma.open",
        "open",
        "tarfile.open",
        "wave.open",
    }
)
_ATOMIC_FILE_CALL_SUFFIXES = (
    "atomic_append_text",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_text",
    "async_atomic_append_text",
    "async_atomic_write_bytes",
    "async_atomic_write_json",
    "async_atomic_write_text",
    "async_durable_replace",
    "async_durable_unlink",
    "durable_replace",
    "durable_unlink",
    "ensure_private_directory",
)
_DESKTOP_MUTATION_METHODS = frozenset(
    {
        "click",
        "doubleClick",
        "dragRel",
        "dragTo",
        "hscroll",
        "hotkey",
        "keyDown",
        "keyUp",
        "leftClick",
        "middleClick",
        "mouseDown",
        "mouseUp",
        "move",
        "moveRel",
        "moveTo",
        "press",
        "release",
        "rightClick",
        "scroll",
        "tripleClick",
        "typewrite",
        "vscroll",
        "write",
    }
)
_GATEWAY_FACTORIES = {
    "get_subprocess_gateway": "subprocess_gateway",
    "get_network_gateway": "network_gateway",
    "get_file_write_gateway": "file_write_gateway",
    "get_desktop_action_gateway": "desktop_action_gateway",
    "get_memory_write_gateway": "memory_write_gateway",
    "get_state_gateway": "state_gateway",
    "get_will": "will_decision",
}
_GATEWAY_METHODS = {
    "subprocess_gateway": frozenset({"run", "run_async", "spawn"}),
    "network_gateway": frozenset({"request", "request_async"}),
    "file_write_gateway": frozenset(
        {
            "append_text",
            "append_text_async",
            "copy_path_async",
            "delete_file",
            "delete_path_async",
            "drain_text",
            "ensure_directory",
            "ensure_directory_async",
            "move_path_async",
            "open_owned_binary",
            "replace_file",
            "write_bytes",
            "write_bytes_batch",
            "write_bytes_async",
            "write_json",
            "write_json_async",
            "write_text",
            "write_text_async",
        }
    ),
    "desktop_action_gateway": frozenset({"run_applescript", "run_applescript_async"}),
    "memory_write_gateway": frozenset({"quarantine", "write"}),
    "state_gateway": frozenset({"mutate"}),
    "will_decision": frozenset({"decide", "decide_async"}),
}


@dataclass(frozen=True, order=True)
class EffectBucket:
    category: str
    path: str
    scope: str
    callee: str
    count: int
    canonical_owner: bool

    def key(self) -> tuple[str, str, str, str]:
        return (self.category, self.path, self.scope, self.callee)


@dataclass(frozen=True)
class ScanProblem:
    path: str
    problem: str


class EffectVisitor(ast.NodeVisitor):
    def __init__(self, *, relative_path: str) -> None:
        self.relative_path = relative_path
        self.aliases: dict[str, str] = {}
        self.binding_scopes: list[dict[str, str]] = [{}]
        self.scope_parts: list[str] = ["<module>"]
        self.calls: list[tuple[str, str, int]] = []

    @property
    def scope(self) -> str:
        return ".".join(self.scope_parts)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            local = alias.asname or alias.name.split(".", 1)[0]
            self.aliases[local] = alias.name if alias.asname else local

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module is None:
            return
        for alias in node.names:
            if alias.name == "*":
                continue
            local = alias.asname or alias.name
            self.aliases[local] = f"{node.module}.{alias.name}"

    def visit_Assign(self, node: ast.Assign) -> None:
        resolved = self._binding_value(node.value)
        if resolved:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.binding_scopes[-1][target.id] = resolved
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None and isinstance(node.target, ast.Name):
            resolved = self._binding_value(node.value)
            if resolved:
                self.binding_scopes[-1][node.target.id] = resolved
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_scoped(node, node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_scoped(node, node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_scoped(node, node.name)

    def _visit_scoped(self, node: ast.AST, name: str) -> None:
        self.scope_parts.append(name)
        self.binding_scopes.append({})
        for child in ast.iter_child_nodes(node):
            self.visit(child)
        self.binding_scopes.pop()
        self.scope_parts.pop()

    def visit_Call(self, node: ast.Call) -> None:
        for category, callee in self._classified_effect_calls(node):
            self.calls.append((category, callee, node.lineno))
        self.generic_visit(node)

    def _classified_effect_calls(self, node: ast.Call) -> list[tuple[str, str]]:
        classified: list[tuple[str, str]] = []
        callee = self._resolve_expr(node.func)
        category = self._classify_call(node, callee)
        if category:
            classified.append((category, callee or "<dynamic>"))
        delegated = _delegated_call(node, callee)
        if delegated is not None:
            delegated_callee = self._resolve_expr(delegated.func)
            delegated_category = self._classify_call(delegated, delegated_callee)
            if delegated_category:
                classified.append((delegated_category, delegated_callee))
        return classified

    def _lookup_binding(self, name: str) -> str | None:
        for scope in reversed(self.binding_scopes):
            if name in scope:
                return scope[name]
        return None

    def _resolve_expr(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return self._lookup_binding(node.id) or self.aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            base = self._resolve_expr(node.value)
            return f"{base}.{node.attr}" if base else node.attr
        if isinstance(node, ast.Call):
            base = self._resolve_expr(node.func)
            return f"{base}()" if base else ""
        return ""

    def _binding_value(self, node: ast.AST) -> str | None:
        if not isinstance(node, ast.Call):
            return None
        callee = self._resolve_expr(node.func)
        factory = _factory_category(callee)
        if factory:
            return f"<{factory}>"
        if callee.endswith("pathlib.Path") or callee == "Path":
            return "<pathlib.Path>"
        if callee.endswith("aiohttp.ClientSession"):
            return "aiohttp.ClientSession()"
        return None

    def _classify_call(self, node: ast.Call, callee: str) -> str | None:
        method = callee.rsplit(".", 1)[-1] if callee else ""

        for category, methods in _GATEWAY_METHODS.items():
            if method not in methods:
                continue
            if f"<{category}>." in callee:
                return category
            if _callee_uses_factory(callee, category):
                return category

        if method in {"execute"} and (
            "ActionExecutor." in callee or callee.endswith("action_executor.ActionExecutor.execute")
        ):
            return "action_executor"

        if _matches_exact(callee, _SUBPROCESS_CALLS):
            return "raw_subprocess"
        if _matches_exact(callee, _NETWORK_EXACT_CALLS) or any(
            _strip_project_prefix(callee).startswith(prefix) for prefix in _NETWORK_PREFIXES
        ):
            return "raw_network"
        stripped_callee = _strip_project_prefix(callee)
        if (
            (stripped_callee.startswith("pyautogui.") and method in _DESKTOP_MUTATION_METHODS)
            or (
                stripped_callee.startswith(
                    ("pynput.keyboard.Controller().", "pynput.mouse.Controller().")
                )
                and method in _DESKTOP_MUTATION_METHODS
            )
            or stripped_callee.startswith("Quartz.CGEventPost")
        ):
            return "raw_desktop"
        if _matches_exact(callee, _BROWSER_EXACT_CALLS) or (
            method in _BROWSER_ACTION_METHODS and _looks_browser_receiver(callee)
        ):
            return "raw_browser"
        if callee.endswith(_ATOMIC_FILE_CALL_SUFFIXES):
            return "direct_atomic_file_write"
        if _matches_exact(callee, _RAW_FILE_EXACT_CALLS):
            return "raw_file_mutation"
        if method in _PATH_MUTATION_METHODS:
            return "raw_file_mutation"
        if method in _AMBIGUOUS_PATH_MUTATION_METHODS and _looks_path_receiver(callee):
            return "raw_file_mutation"
        if method == "open" and _open_call_mutates(node, callee):
            return "raw_file_mutation"
        return None


def _strip_project_prefix(value: str) -> str:
    for prefix in ("core.", "interface.", "skills."):
        if value.startswith(prefix):
            return value[len(prefix) :]
    return value


def _matches_exact(value: str, options: Sequence[str] | frozenset[str]) -> bool:
    stripped = _strip_project_prefix(value)
    return stripped in options or any(stripped.endswith(f".{option}") for option in options)


def _factory_category(callee: str) -> str | None:
    stripped = callee.removesuffix("()")
    leaf = stripped.rsplit(".", 1)[-1]
    return _GATEWAY_FACTORIES.get(leaf)


def _callee_uses_factory(callee: str, category: str) -> bool:
    for factory, factory_category in _GATEWAY_FACTORIES.items():
        if factory_category == category and f"{factory}()." in callee:
            return True
    return False


def _delegated_call(node: ast.Call, callee: str) -> ast.Call | None:
    stripped = _strip_project_prefix(callee)
    if stripped in {"asyncio.to_thread", "anyio.to_thread.run_sync"}:
        callable_index = 0
    elif stripped.endswith(".run_in_executor"):
        callable_index = 1
    else:
        return None
    if len(node.args) <= callable_index:
        return None
    target = node.args[callable_index]
    if not isinstance(target, (ast.Attribute, ast.Name)):
        return None
    return ast.Call(
        func=target,
        args=list(node.args[callable_index + 1 :]),
        keywords=list(node.keywords),
    )


def _open_call_mutates(node: ast.Call, callee: str) -> bool:
    if callee not in {"open", "builtins.open"} and not callee.endswith(".open"):
        return False
    mode_node: ast.AST | None = None
    stripped_callee = _strip_project_prefix(callee)
    is_function_open = stripped_callee in _MODED_FILE_OPEN_CALLS
    is_path_method_open = not is_function_open and _looks_path_receiver(callee)
    if not is_function_open and not is_path_method_open:
        return False
    if is_function_open and len(node.args) >= 2:
        mode_node = node.args[1]
    elif is_path_method_open and node.args:
        mode_node = node.args[0]
    for keyword in node.keywords:
        if keyword.arg == "mode":
            mode_node = keyword.value
    if mode_node is None:
        return False
    try:
        mode = ast.literal_eval(mode_node)
    except (ValueError, TypeError):
        return True
    if not isinstance(mode, str):
        return True
    normalized = mode.strip().lower()
    if not normalized:
        return False
    if stripped_callee in {"tarfile.open"}:
        return normalized[0] in {"a", "w", "x"}
    if not _is_file_mode(normalized):
        return False
    return any(marker in normalized for marker in ("a", "w", "x", "+"))


def _is_file_mode(value: str) -> bool:
    return bool(value) and all(character in "rwaxbt+" for character in value)


def _looks_path_receiver(callee: str) -> bool:
    receiver = callee.rsplit(".", 1)[0].removesuffix("()")
    if "<pathlib.Path>" in receiver or receiver.endswith(".parent"):
        return True
    leaf = receiver.rsplit(".", 1)[-1].casefold().removeprefix("_")
    return leaf in {
        "destination",
        "directory",
        "dir",
        "file",
        "folder",
        "ledger",
        "path",
        "root",
        "source",
        "target",
    } or leaf.endswith(("_dir", "_file", "_path", "_root"))


def _looks_browser_receiver(callee: str) -> bool:
    if "." not in callee:
        return False
    receiver = callee.rsplit(".", 1)[0].removesuffix("()")
    leaf = receiver.rsplit(".", 1)[-1].casefold().removeprefix("_")
    method = callee.rsplit(".", 1)[-1].casefold()
    if method == "get":
        return leaf == "driver" or leaf.endswith("_driver")
    return leaf in {
        "browser",
        "button",
        "driver",
        "element",
        "input",
        "keyboard",
        "link",
        "locator",
        "page",
        "window",
    } or leaf.endswith(
        (
            "_box",
            "_browser",
            "_btn",
            "_button",
            "_driver",
            "_element",
            "_input",
            "_link",
            "_locator",
            "_page",
            "_window",
        )
    )


def _canonical_owner(category: str, relative_path: str) -> bool:
    owners = CANONICAL_PRIMITIVE_OWNERS.get(category, frozenset())
    if category == "action_executor":
        return True
    return relative_path in owners


def _iter_source_files(root: Path) -> Iterable[Path]:
    for top in SCAN_ROOTS:
        base = root / top
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            relative_parts = path.relative_to(root).parts
            if any(part in SKIP_DIR_PARTS for part in relative_parts):
                continue
            yield path


def scan_repository(root: Path = ROOT) -> tuple[list[EffectBucket], list[ScanProblem]]:
    counts: dict[tuple[str, str, str, str], int] = {}
    problems: list[ScanProblem] = []
    for path in _iter_source_files(root):
        relative = path.relative_to(root).as_posix()
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            problems.append(ScanProblem(relative, f"read_failed:{type(exc).__qualname__}:{exc}"))
            continue
        try:
            tree = ast.parse(source, filename=relative)
        except SyntaxError as exc:
            problems.append(
                ScanProblem(relative, f"parse_failed:{exc.lineno}:{exc.offset}:{exc.msg}")
            )
            continue
        scoped_counts = _scan_tree_scoped(tree, relative)
        for key, count in scoped_counts.items():
            counts[key] = counts.get(key, 0) + count

    buckets = [
        EffectBucket(
            category=category,
            path=path,
            scope=scope,
            callee=callee,
            count=count,
            canonical_owner=_canonical_owner(category, path),
        )
        for (category, path, scope, callee), count in counts.items()
    ]
    return sorted(buckets), sorted(problems, key=lambda problem: problem.path)


class _ScopedEffectVisitor(EffectVisitor):
    def visit_Call(self, node: ast.Call) -> None:
        for category, callee in self._classified_effect_calls(node):
            self.calls.append((category, f"{self.scope}\0{callee}", node.lineno))
        self.generic_visit(node)


def _scan_tree_scoped(
    tree: ast.AST,
    relative_path: str,
) -> dict[tuple[str, str, str, str], int]:
    visitor = _ScopedEffectVisitor(relative_path=relative_path)
    visitor.visit(tree)
    counts: dict[tuple[str, str, str, str], int] = {}
    for category, encoded, _line in visitor.calls:
        scope, callee = encoded.split("\0", 1)
        key = (category, relative_path, scope, callee)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _baseline_payload(buckets: Sequence[EffectBucket]) -> dict[str, Any]:
    rows = [asdict(bucket) for bucket in buckets]
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "description": "Exact AST effect-ownership debt ratchet; refresh only after reviewed change",
        "inventory_sha256": hashlib.sha256(canonical).hexdigest(),
        "buckets": rows,
    }


def write_baseline(path: Path, buckets: Sequence[EffectBucket]) -> None:
    payload = _baseline_payload(buckets)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_baseline(path: Path) -> list[EffectBucket]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"effect ownership baseline is missing: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"effect ownership baseline is unreadable: {path}: {exc}") from exc
    if not isinstance(payload, Mapping) or payload.get("schema_version") != BASELINE_SCHEMA_VERSION:
        raise ValueError("effect ownership baseline schema_version is invalid")
    raw_rows = payload.get("buckets")
    if not isinstance(raw_rows, list):
        raise ValueError("effect ownership baseline buckets must be a list")
    buckets: list[EffectBucket] = []
    for index, row in enumerate(raw_rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"effect ownership baseline bucket {index} is not a mapping")
        try:
            buckets.append(
                EffectBucket(
                    category=str(row["category"]),
                    path=str(row["path"]),
                    scope=str(row["scope"]),
                    callee=str(row["callee"]),
                    count=int(row["count"]),
                    canonical_owner=bool(row["canonical_owner"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"effect ownership baseline bucket {index} is invalid: {exc}") from exc
    expected = str(payload.get("inventory_sha256") or "")
    actual = _baseline_payload(sorted(buckets))["inventory_sha256"]
    if expected != actual:
        raise ValueError("effect ownership baseline inventory_sha256 does not match its buckets")
    return sorted(buckets)


def compare_inventory(
    current: Sequence[EffectBucket],
    baseline: Sequence[EffectBucket],
) -> tuple[list[str], list[str]]:
    current_by_key = {bucket.key(): bucket for bucket in current}
    baseline_by_key = {bucket.key(): bucket for bucket in baseline}
    regressions: list[str] = []
    stale: list[str] = []
    for key in sorted(set(current_by_key) | set(baseline_by_key)):
        observed = current_by_key.get(key)
        expected = baseline_by_key.get(key)
        label = " | ".join(key)
        if expected is None and observed is not None:
            regressions.append(f"NEW {label} count={observed.count}")
        elif observed is None and expected is not None:
            stale.append(f"REMOVED {label} baseline={expected.count}")
        elif observed is not None and expected is not None:
            if observed.count > expected.count:
                regressions.append(
                    f"INCREASED {label} baseline={expected.count} current={observed.count}"
                )
            elif observed.count < expected.count:
                stale.append(
                    f"DECREASED {label} baseline={expected.count} current={observed.count}"
                )
            if expected.canonical_owner and not observed.canonical_owner:
                regressions.append(f"OWNER_DEMOTED {label} baseline=True current=False")
            elif observed.canonical_owner and not expected.canonical_owner:
                stale.append(f"OWNER_PROMOTED {label} baseline=False current=True")
    return regressions, stale


def _summary(buckets: Sequence[EffectBucket]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for bucket in buckets:
        row = summary.setdefault(bucket.category, {"calls": 0, "buckets": 0, "debt_calls": 0})
        row["calls"] += bucket.count
        row["buckets"] += 1
        if not bucket.canonical_owner:
            row["debt_calls"] += bucket.count
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Replace the baseline with the current reviewed inventory",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    return parser


def main(argv: Iterable[str] = ()) -> int:
    args = _parser().parse_args(list(argv))
    root = args.root.expanduser().resolve()
    baseline_path = args.baseline.expanduser()
    if not baseline_path.is_absolute():
        baseline_path = root / baseline_path

    buckets, problems = scan_repository(root)
    report: dict[str, Any] = {
        "ok": False,
        "root": str(root),
        "baseline": str(baseline_path),
        "summary": _summary(buckets),
        "bucket_count": len(buckets),
        "problems": [asdict(problem) for problem in problems],
        "regressions": [],
        "stale_baseline": [],
    }
    if problems:
        print(f"governance effect ownership: analyzer failed on {len(problems)} file(s)")
        for problem in problems[:40]:
            print(f"  {problem.path}: {problem.problem}")
        _write_report(args.json_out, report)
        return 2

    if args.write_baseline:
        write_baseline(baseline_path, buckets)
        report["ok"] = True
        report["baseline_written"] = True
        _write_report(args.json_out, report)
        debt_calls = sum(bucket.count for bucket in buckets if not bucket.canonical_owner)
        print(
            "governance effect ownership baseline written: "
            f"{len(buckets)} buckets, {debt_calls} migration-debt calls"
        )
        return 0

    try:
        baseline = load_baseline(baseline_path)
    except ValueError as exc:
        print(f"governance effect ownership: configuration error: {exc}")
        _write_report(args.json_out, report)
        return 2

    regressions, stale = compare_inventory(buckets, baseline)
    report["regressions"] = regressions
    report["stale_baseline"] = stale
    report["ok"] = not regressions and not stale
    _write_report(args.json_out, report)

    summary = _summary(buckets)
    debt_calls = sum(row["debt_calls"] for row in summary.values())
    total_calls = sum(row["calls"] for row in summary.values())
    if regressions or stale:
        print(
            "governance effect ownership: baseline drift "
            f"({len(regressions)} regression(s), {len(stale)} stale bucket(s))"
        )
        for issue in (regressions + stale)[:80]:
            print(f"  {issue}")
        print(
            "Review the call-site changes. After debt reductions or approved canonical-owner "
            "changes, refresh with tools/lint_governance.py --write-baseline."
        )
        return 1

    print(
        "governance effect ownership: baseline matched; "
        f"{total_calls} recognized calls in {len(buckets)} buckets, "
        f"{debt_calls} calls remain migration debt"
    )
    for category, row in sorted(summary.items()):
        print(
            f"  {category}: calls={row['calls']} buckets={row['buckets']} "
            f"debt_calls={row['debt_calls']}"
        )
    return 0


def _write_report(path: Path | None, report: Mapping[str, Any]) -> None:
    if path is None:
        return
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
