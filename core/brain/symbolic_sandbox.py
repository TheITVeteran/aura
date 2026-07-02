"""Symbolic Sandbox — Aura's deterministic cognitive scratchpad.

Frontier reasoning models do not calculate in their head; they write code, run it,
read the error, and self-correct. This is that co-processor. Given a piece of
candidate Python (a calculation, a property test, a counterexample search) it:

1. statically vets the script with the existing AST safety analyzer — any
   dangerous import/call (os, subprocess, socket, open, eval, …) is refused
   *before* execution, so the executed script can only do pure computation + print;
2. writes it to the session scratch directory and runs it in CPython's isolated
   mode (``-I``) under a wall-clock timeout, through the governed subprocess gateway
   as a read-only probe;
3. captures stdout / stderr / traceback;
4. on failure, feeds the traceback back to a caller-supplied ``repair`` generator
   ("[SANDBOX] your script failed with: … fix it") and retries, up to a bounded
   number of rounds.

It is the execution spine the code/math truth engines and the amplifier's
``expand_or_repair_failed_candidates`` step call when prose verification is not
enough and the answer can simply be *run*.
"""
from __future__ import annotations

import logging
import os
import re
import sys
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import async_atomic_write_text, atomic_write_text
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.SymbolicSandbox")

_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def _strip_fence(code: str) -> str:
    m = _CODE_FENCE_RE.search(code or "")
    return (m.group(1) if m else str(code or "")).strip()


@dataclass
class SandboxResult:
    ok: bool
    stdout: str = ""
    stderr: str = ""
    traceback: str = ""
    timed_out: bool = False
    refused: bool = False
    rounds: int = 0
    final_code: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "timed_out": self.timed_out,
            "refused": self.refused,
            "rounds": self.rounds,
            "stdout": self.stdout[-500:],
            "stderr": self.stderr[-500:],
            "warnings": self.warnings[:6],
        }


class SymbolicSandbox:
    """Isolated, AST-vetted Python execution with a self-correction loop."""

    def __init__(self, *, workspace: str | os.PathLike[str] | None = None, timeout: float = 12.0) -> None:
        self._workspace = Path(workspace) if workspace else None
        self._timeout = float(timeout)

    def _resolve_workspace(self) -> Path:
        if self._workspace is not None:
            self._workspace.mkdir(parents=True, exist_ok=True)
            return self._workspace
        # Prefer the session scratchpad if one is configured.
        scratch = os.getenv("CLAUDE_SCRATCHPAD") or os.getenv("AURA_SCRATCH_DIR")
        base = Path(scratch) if scratch else Path(tempfile.gettempdir())
        target = base / "aura_symbolic_sandbox"
        target.mkdir(parents=True, exist_ok=True)
        return target

    def vet(self, code: str) -> tuple[bool, list[str]]:
        """Static safety gate. True ⇒ safe to execute (pure computation only)."""
        body = _strip_fence(code)
        if not body:
            return False, ["empty script"]
        try:
            from core.resilience.code_verifier import CodeVerifier

            if not CodeVerifier.verify_syntax(body):
                return False, ["syntax error"]
            safety = CodeVerifier.analyze_safety(body)
            return bool(safety["safe"]), [str(w) for w in safety["warnings"]]
        except (ImportError, RuntimeError) as exc:  # pragma: no cover - import guard
            record_degradation("symbolic_sandbox_vet", exc)
            return False, ["safety analyzer unavailable"]

    async def run(self, code: str) -> SandboxResult:
        """Vet then execute a single script in CPython isolated mode."""
        body = _strip_fence(code)
        safe, warnings = self.vet(body)
        if not safe:
            logger.info("🔒 [Sandbox] refused unsafe/invalid script: %s", warnings)
            return SandboxResult(ok=False, refused=True, warnings=warnings, final_code=body)

        try:
            from core.runtime.subprocess_gateway import get_subprocess_gateway

            workspace = self._resolve_workspace()
            with tempfile.TemporaryDirectory(prefix="aura_sbx_", dir=str(workspace)) as td:
                script = Path(td) / "scratch.py"
                await async_atomic_write_text(script, body, encoding="utf-8")
                # Isolated mode: ignore env, user site, PYTHON* vars. Restricted PATH.
                env = {"PATH": "/usr/bin:/bin", "HOME": td, "TMPDIR": td}
                res = await get_subprocess_gateway().run_async(
                    (sys.executable, "-I", "-B", str(script)),
                    timeout=self._timeout,
                    cwd=td,
                    env=env,
                    read_only=True,
                    source="symbolic_sandbox:exec",
                )
            tb = ""
            if res.returncode != 0 and res.stderr:
                tb = res.stderr.strip()
            return SandboxResult(
                ok=(res.returncode == 0),
                stdout=res.stdout or "",
                stderr=res.stderr or "",
                traceback=tb,
                final_code=body,
                warnings=warnings,
            )
        except TimeoutError:
            return SandboxResult(ok=False, timed_out=True, final_code=body, warnings=warnings)
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            record_degradation("symbolic_sandbox_run", exc)
            return SandboxResult(ok=False, stderr=str(exc), final_code=body, warnings=warnings)

    async def run_with_self_correction(
        self,
        code: str,
        repair: Callable[[str, str], Awaitable[str]],
        *,
        max_rounds: int = 3,
    ) -> SandboxResult:
        """Run; on failure feed the traceback to ``repair(code, traceback)`` and retry.

        ``repair`` is the generator hook — typically a thin wrapper over the Cortex
        that receives the failing code plus the captured traceback and returns a
        corrected script. The loop stops on first success or when ``max_rounds`` is
        spent. This is the ReAct "reason → run → observe → revise" loop made local
        and deterministic for code/math.
        """
        current = _strip_fence(code)
        last = SandboxResult(ok=False, final_code=current)
        for round_idx in range(max(1, int(max_rounds))):
            last = await self.run(current)
            last.rounds = round_idx + 1
            if last.ok:
                logger.info("✅ [Sandbox] script succeeded on round %d", last.rounds)
                return last
            failure = last.traceback or last.stderr or ("refused: " + "; ".join(last.warnings))
            if round_idx == max_rounds - 1:
                break
            try:
                repaired = await repair(current, failure)
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation("symbolic_sandbox_repair", exc)
                break
            repaired = _strip_fence(repaired)
            if not repaired or repaired == current:
                break
            logger.info("🔁 [Sandbox] round %d failed (%s) → repairing", last.rounds, failure[:80])
            current = repaired
        return last


_instance: SymbolicSandbox | None = None


def get_symbolic_sandbox() -> SymbolicSandbox:
    global _instance
    if _instance is None:
        _instance = SymbolicSandbox()
    return _instance
