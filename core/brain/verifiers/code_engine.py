"""Code truth engine — compile, AST-safety, and lint a candidate's code blocks.

Wraps the existing :class:`core.resilience.code_verifier.CodeVerifier` (isolated
``py_compile`` + AST safety) and adds an optional ``ruff`` static pass through the
governed subprocess gateway as a *read-only probe*. It never executes candidate
code — runtime smoke tests belong to :mod:`core.brain.symbolic_sandbox`.
"""
from __future__ import annotations

import re
from typing import Any

from core.runtime.errors import record_degradation

from .base import VerificationResult
from core.runtime.atomic_writer import async_atomic_write_text

_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
# A line that looks like Python even without a fence (heuristic for inline code answers).
_PY_HINT_RE = re.compile(r"^\s*(?:def |class |import |from \w+ import |async def )", re.MULTILINE)


def extract_code_blocks(text: str) -> list[str]:
    """Pull fenced python blocks; fall back to the whole text if it parses as code."""
    blocks = [m.group(1).strip() for m in _FENCE_RE.finditer(text or "") if m.group(1).strip()]
    if blocks:
        return blocks
    body = str(text or "").strip()
    if body and _PY_HINT_RE.search(body):
        return [body]
    return []


class CodeTruthEngine:
    name = "code"
    domains = ("code", "code_audit", "code_patch", "debug")

    def __init__(self, *, run_ruff: bool = True) -> None:
        self._run_ruff = run_ruff

    def handles(self, task_type: str) -> bool:
        return task_type in self.domains

    async def verify(self, candidate: str, *, context: dict[str, Any] | None = None) -> VerificationResult:
        blocks = extract_code_blocks(candidate)
        if not blocks:
            return VerificationResult(domain="code", ok=True, checked=False, engine=self.name)

        issues: list[str] = []
        evidence: list[str] = []
        compiled_ok = 0
        try:
            from core.resilience.code_verifier import CodeVerifier
        except (ImportError, RuntimeError) as exc:  # pragma: no cover - import guard
            record_degradation("code_truth_engine", exc)
            return VerificationResult(domain="code", ok=True, checked=False, engine=self.name)

        for idx, block in enumerate(blocks):
            report = CodeVerifier.verify_importability_report(block, module_name=f"candidate_{idx}")
            if not report.syntax_ok:
                issues.append(f"block#{idx}: syntax error")
                continue
            if not report.ok:
                stderr = (report.stderr or report.error or "compile failed").strip().splitlines()
                issues.append(f"block#{idx}: {stderr[-1] if stderr else 'compile failed'}")
            else:
                compiled_ok += 1
                evidence.append(f"block#{idx}: compiles clean")
            if report.warnings:
                issues.extend(f"block#{idx} unsafe: {w}" for w in report.warnings)
            if self._run_ruff:
                ruff_issues = await self._ruff(block)
                issues.extend(f"block#{idx} lint: {m}" for m in ruff_issues[:3])

        ok = not any(i for i in issues if "syntax" in i or "compile" in i or "unsafe" in i)
        # Score rewards clean-compiling blocks and penalises lint noise.
        score = (compiled_ok / max(1, len(blocks))) * (0.9 if not issues else 0.6)
        return VerificationResult(
            domain="code",
            ok=ok,
            checked=True,
            score=round(min(0.98, max(0.05, score)), 4),
            engine=self.name,
            issues=issues,
            evidence=evidence,
            detail={"blocks": len(blocks), "compiled_ok": compiled_ok},
        )

    async def _ruff(self, code: str) -> list[str]:
        """Static lint via ruff as a governed read-only probe; best-effort."""
        try:
            import shutil
            import tempfile
            from pathlib import Path

            from core.runtime.atomic_writer import atomic_write_text
            from core.runtime.subprocess_gateway import get_subprocess_gateway

            if shutil.which("ruff") is None:
                return []
            with tempfile.TemporaryDirectory(prefix="aura_ruff_") as td:
                path = Path(td) / "candidate.py"
                await async_atomic_write_text(path, code, encoding="utf-8")
                res = await get_subprocess_gateway().run_async(
                    ("ruff", "check", "--quiet", "--no-cache", str(path)),
                    timeout=15.0,
                    read_only=True,
                    source="reasoning_verifier:code_ruff",
                )
            out = (res.stdout or "") + (res.stderr or "")
            return [ln.strip() for ln in out.splitlines() if ln.strip() and ":" in ln][:5]
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            record_degradation("code_truth_engine_ruff", exc)
            return []
