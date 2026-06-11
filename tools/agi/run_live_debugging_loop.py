#!/usr/bin/env python3
"""Real-World Coding & Debugging Loop Runner.

This script orchestrates a live coding-agent debugging loop:
1. Discover files and run tests to diagnose failures.
2. Read files to understand code structure.
3. Request an agent-owned patch proposal and apply it.
4. Verify code changes by re-running tests until success is achieved.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.governance_context import (
    governance_runtime_active,
    local_internal_governed_scope,
)
from core.runtime.atomic_writer import atomic_write_text
from core.runtime.subprocess_gateway import get_subprocess_gateway

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("LiveDebuggingRunner")

_COMMAND_RECOVERABLE_ERRORS = (
    RuntimeError,
    AttributeError,
    TypeError,
    ValueError,
    OSError,
    TimeoutError,
)


def _path_exists(path: Path) -> bool:
    return path.exists()


def _discover_python_files(path: Path) -> list[Path]:
    return list(path.glob("**/*.py"))


def _read_text(path: Path) -> str:
    return path.read_text()


def _write_text(path: Path, content: str) -> None:
    atomic_write_text(path, content)


@dataclass(frozen=True)
class DebugObservation:
    repo_path: Path
    code_file: Path
    test_file: Path
    code_content: str
    test_content: str
    initial_stdout: str
    initial_stderr: str


@dataclass(frozen=True)
class PatchProposal:
    file: Path
    content: str
    rationale: str = ""


PatchProvider = Callable[[DebugObservation], PatchProposal | Awaitable[PatchProposal | None] | None]


async def run_terminal_command(
    cmd: list[str], cwd: Path, timeout_s: float = 180.0
) -> dict[str, Any]:
    """Execute a real terminal command inside the specified directory."""
    logger.info("Executing command: %s in %s", " ".join(cmd), cwd)
    try:
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        # Offline claim is only honest when this loop runs as a CLI proof
        # tool; embedded in a live runtime (external validation battery)
        # the spawn must carry a local-internal receipt or the gateway
        # rightly denies it and every repair round fails before pytest
        # can run. Same contract as local_sandbox and the server spawn.
        with local_internal_governed_scope(
            "live_debugging_loop.run_terminal_command",
            domain="tool_execution",
        ):
            proc = await get_subprocess_gateway().spawn_async(
                cmd,
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                offline_tooling=not governance_runtime_active(),
                source="proof_tooling:live_debugging_loop",
            )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_s
            )
        except TimeoutError:
            proc.kill()
            await proc.communicate()
            logger.error("Command timed out after %.0fs: %s", timeout_s, " ".join(cmd))
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": f"command timed out after {timeout_s:.0f}s",
            }
        return {
            "exit_code": proc.returncode,
            "stdout": stdout.decode("utf-8", errors="replace"),
            "stderr": stderr.decode("utf-8", errors="replace"),
        }
    except _COMMAND_RECOVERABLE_ERRORS as e:
        logger.error("Command execution failed: %s", e)
        return {"exit_code": -1, "stdout": "", "stderr": str(e)}


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    try:
        value = json.loads(stripped)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(stripped[start : end + 1])
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None
    return None


async def _default_patch_provider(observation: DebugObservation) -> PatchProposal | None:
    """Ask the governed model router for a structured repair proposal.

    This intentionally uses the router's proof-repair lane rather than a
    full conversational cognitive cycle: the dialogue stack (persona,
    reliability contracts, answer-quality gates, self-critique) is built
    to judge conversational replies and rejects a raw JSON patch as a
    non-answer. purpose="proof_evaluation_repair" is the recognized
    isolated proof lane, still governed at the router/gate.
    """
    try:
        from core.brain.llm_health_router import get_llm_router
    except (ImportError, AttributeError, RuntimeError):
        return None

    try:
        router = get_llm_router()
    except _COMMAND_RECOVERABLE_ERRORS:
        router = None
    if router is None or not hasattr(router, "think"):
        return None

    body = (
        f"Repository: {observation.repo_path}\n"
        f"Source file: {observation.code_file.relative_to(observation.repo_path)}\n"
        f"Test file: {observation.test_file.relative_to(observation.repo_path)}\n\n"
        f"Initial pytest stdout:\n{observation.initial_stdout[-4000:]}\n\n"
        f"Initial pytest stderr:\n{observation.initial_stderr[-4000:]}\n\n"
        f"Source content:\n```python\n{observation.code_content}\n```\n\n"
        f"Test content:\n```python\n{observation.test_content}\n```"
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are repairing a small local Python repository. Return "
                "only JSON with keys path, content, rationale. The content "
                "must be the complete replacement text for the file you "
                "edit. Do not invent test results; the runner will apply "
                "your patch and rerun pytest."
            ),
        },
        {"role": "user", "content": body},
    ]
    # Budget covers the generation gate queue (gated turns measure
    # 31-35s, serialized at 2) plus a full-file generation at 32B speeds.
    content = str(
        await asyncio.wait_for(
            router.think(
                messages=messages,
                origin="external_live_debugging_loop",
                purpose="proof_evaluation_repair",
                foreground_request=True,
                protected_foreground_lane=True,
                allow_cloud_fallback=False,
                temperature=0.2,
                max_tokens=4096,
            ),
            timeout=240.0,
        )
        or ""
    )
    payload = _extract_json_object(content)
    if not payload:
        return None
    path_value = str(payload.get("path") or payload.get("file") or "").strip()
    patch_content = payload.get("content")
    if not path_value or not isinstance(patch_content, str) or not patch_content.strip():
        return None
    return PatchProposal(
        file=observation.repo_path / path_value,
        content=patch_content,
        rationale=str(payload.get("rationale") or ""),
    )


async def _call_patch_provider(
    provider: PatchProvider,
    observation: DebugObservation,
) -> PatchProposal | None:
    proposed = provider(observation)
    if inspect.isawaitable(proposed):
        proposed = await proposed
    return proposed if isinstance(proposed, PatchProposal) else None


def _validate_patch_target(repo_path: Path, proposal: PatchProposal) -> Path:
    target = proposal.file
    if not target.is_absolute():
        target = repo_path / target
    target = target.resolve()
    repo_root = repo_path.resolve()
    if not target.is_relative_to(repo_root):
        raise ValueError(f"Patch target escapes repository: {target}")
    if target.suffix != ".py":
        raise ValueError(f"Patch target must be a Python file, got {target.name}")
    return target


async def run_debugging_loop(
    repo_path: Path,
    *,
    patch_provider: PatchProvider | None = None,
) -> dict[str, Any]:
    """Run a complete diagnostic, patching, and verification loop on the target repository."""
    # Resolve once: macOS tempdirs live under the /var → /private/var
    # symlink, and _validate_patch_target returns resolved paths. Mixing
    # resolved and unresolved bases made patch_target.relative_to(repo_path)
    # raise ValueError right after the patch write — an exception that
    # asyncio.run never surfaced because runtime teardown hung on
    # cancellation-ignoring loops. Three batteries froze on this line.
    repo_path = await asyncio.to_thread(lambda: Path(repo_path).resolve())
    logger.info("Starting live debugging loop for repository: %s", repo_path)
    
    if not await asyncio.to_thread(_path_exists, repo_path):
        return {"ok": False, "error": f"Repository path {repo_path} does not exist"}

    trace = []
    
    # Step 1: Diagnose (Run initial test suite)
    logger.info("Step 1: Diagnosing initial state...")
    initial_test = await run_terminal_command(["pytest"], repo_path)
    trace.append({
        "stage": "diagnose",
        "exit_code": initial_test["exit_code"],
        "stdout": initial_test["stdout"][:500],
    })
    
    if initial_test["exit_code"] == 0:
        logger.info("Tests are already passing! Nothing to debug.")
        return {"ok": True, "status": "already_passing", "trace": trace}
        
    logger.warning("Initial test suite failed as expected. exit_code: %d", initial_test["exit_code"])

    # Step 2: Discover and read codebase files
    logger.info("Step 2: Discovering source and test files...")
    py_files = await asyncio.to_thread(_discover_python_files, repo_path)
    code_file: Path | None = None
    test_file: Path | None = None

    for f in py_files:
        if f.name.startswith("test_"):
            test_file = f
        elif f.name != "conftest.py" and not f.name.startswith("__"):
            code_file = f

    if not code_file or not test_file:
        return {
            "ok": False,
            "error": f"Could not find code and test files in {repo_path}. Found: {py_files}",
        }

    logger.info("Found code file: %s", code_file.relative_to(repo_path))
    logger.info("Found test file: %s", test_file.relative_to(repo_path))

    # Read the files
    code_content = await asyncio.to_thread(_read_text, code_file)
    test_content = await asyncio.to_thread(_read_text, test_file)
    
    trace.append({
        "stage": "read_files",
        "code_file": code_file.name,
        "test_file": test_file.name,
        "code_content": code_content,
        "test_content": test_content,
    })

    # Step 3: Agent-owned diagnostic reasoning and patching. The runner does
    # not contain task-specific fixes; it only applies and verifies proposals.
    logger.info("Step 3: Requesting agent patch proposal...")
    observation = DebugObservation(
        repo_path=repo_path,
        code_file=code_file,
        test_file=test_file,
        code_content=code_content,
        test_content=test_content,
        initial_stdout=initial_test["stdout"],
        initial_stderr=initial_test["stderr"],
    )
    provider = patch_provider or _default_patch_provider
    try:
        proposal = await _call_patch_provider(provider, observation)
    except _COMMAND_RECOVERABLE_ERRORS as exc:
        logger.error("Patch provider failed: %s: %s", type(exc).__name__, exc)
        return {
            "ok": False,
            "error": f"Patch provider failed: {type(exc).__name__}: {exc}",
            "trace": trace,
        }
    if proposal is None:
        return {
            "ok": False,
            "error": "No agent patch proposal was produced.",
            "trace": trace,
        }

    try:
        patch_target = _validate_patch_target(repo_path, proposal)
    except ValueError as exc:
        return {
            "ok": False,
            "error": str(exc),
            "trace": trace,
        }

    await asyncio.to_thread(_write_text, patch_target, proposal.content)
    logger.info("Successfully wrote agent patch to %s", patch_target.name)

    trace.append({
        "stage": "patch",
        "patched_file": str(patch_target.relative_to(repo_path)),
        "new_content": proposal.content,
        "rationale": proposal.rationale,
    })

    # Step 4: Verify (Re-run test suite)
    logger.info("Step 4: Re-running test suite for verification...")
    await asyncio.sleep(0.1)
    post_test = await run_terminal_command(["pytest"], repo_path)
    trace.append({
        "stage": "verify",
        "exit_code": post_test["exit_code"],
        "stdout": post_test["stdout"][:500],
    })

    if post_test["exit_code"] != 0:
        logger.error("Verification failed! exit_code: %d", post_test["exit_code"])
        return {
            "ok": False,
            "status": "failed_verification",
            "error_output": post_test["stdout"] + "\n" + post_test["stderr"],
            "trace": trace,
        }

    logger.info("🎉 Verification succeeded! Test suite is now 100% green.")
    return {
        "ok": True,
        "status": "success_verified",
        "trace": trace,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ./run_live_debugging_loop.py <repository_path>")
        sys.exit(1)
        
    path = Path(sys.argv[1]).resolve()
    res = asyncio.run(run_debugging_loop(path))
    print("\nResult:")
    print(res)
    sys.exit(0 if res["ok"] else 1)
