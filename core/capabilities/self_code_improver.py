"""core/capabilities/self_code_improver.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Recursive self-improvement on her OWN code, with a real standard.

She takes a function in her own source, RESEARCHES how to do it better, generates
an improved version with the un-steered code model, and VERIFIES it against
behavioral checks — the improved function must pass ALL of them while the current
function fails at least one (so the change is a genuine improvement, not a
rewrite). Only then does she ENACT it: rewrite that function in the real file.
The lesson is retained. A caller re-runs the test suite to confirm no regression,
and commits so the change survives the integrity guardian.

This is deliberately narrow-waisted and verifiable — no "looks better," only
"passes checks the old code failed, and breaks none it passed."
"""
from __future__ import annotations

import ast
import asyncio
import json
import logging
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.SelfCodeImprover")


@dataclass
class ImproveResult:
    ok: bool
    target_file: str
    func_name: str
    goal: str
    original_passed: int = 0
    improved_passed: int = 0
    total_checks: int = 0
    enacted: bool = False
    iterations: int = 0
    research_used: list[str] = field(default_factory=list)
    improved_source: str = ""
    original_source: str = ""
    lesson_retained: str = ""
    status: str = "ok"
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["improved_source"] = self.improved_source[:6000]
        d["original_source"] = self.original_source[:4000]
        return d


def _extract_function_source(source: str, func_name: str) -> tuple[str, int, int] | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            lines = source.splitlines()
            start = node.lineno - 1
            end = node.end_lineno
            # include a leading decorator line span if present
            if node.decorator_list:
                start = min(d.lineno - 1 for d in node.decorator_list)
            return "\n".join(lines[start:end]), start, end
    return None


def _extract_function_from_response(raw: str, func_name: str) -> str:
    """Pull just the target function definition out of a model response."""
    import re

    text = str(raw or "")
    fence = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1)
    found = _extract_function_source(text, func_name)
    if found:
        return found[0]
    # fall back: from 'def <name>' to the next top-level line
    m = re.search(rf"(^|\n)(\s*)(async def|def)\s+{re.escape(func_name)}\s*\(", text)
    if not m:
        return ""
    start = m.start(3)
    indent = m.group(2)
    rest = text[start:].splitlines()
    body = [rest[0]]
    for line in rest[1:]:
        if line.strip() and not line.startswith(indent + " ") and not line.startswith(indent + "\t") and line.strip() != "":
            if not line.startswith((" ", "\t")):
                break
        body.append(line)
    return "\n".join(body).rstrip()


def _verify(func_source: str, func_name: str, checks: list[dict[str, Any]]) -> tuple[int, list[dict[str, Any]]]:
    """Run behavioral checks against a function in an isolated subprocess.
    Each check: {"args": [...], "expected": <value>}. Returns (passed, details)."""
    runner = (
        "from typing import Any, Optional, List, Dict, Tuple, Sequence, Iterable, Union\n"
        + func_source
        + "\n\nimport json\n_out=[]\n_CHECKS=" + json.dumps(checks) + "\n"
        + "for _c in _CHECKS:\n"
        + "    try:\n"
        + f"        _got={func_name}(*_c['args'])\n"
        + "        _out.append({'ok': _got==_c['expected'], 'got': _got, 'expected': _c['expected']})\n"
        + "    except Exception as _e:\n"
        + "        _out.append({'ok': False, 'error': str(_e), 'expected': _c['expected']})\n"
        + "print(json.dumps(_out))\n"
    )
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write(runner)
            path = fh.name
        proc = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=15)
        Path(path).unlink(missing_ok=True)
        out = proc.stdout.strip().splitlines()
        for line in reversed(out):
            if line.strip().startswith("["):
                details = json.loads(line)
                return sum(1 for d in details if d.get("ok")), details
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, ValueError) as exc:
        logger.debug("verify failed: %s", exc)
    return 0, [{"ok": False, "error": "verification could not run"}]


async def _research(goal: str, max_notes: int = 4) -> list[str]:
    notes: list[str] = []
    try:
        from core.knowledge.local_corpus import get_local_corpus_store

        for hit in get_local_corpus_store().search(goal, limit=3):
            s = f"{hit.title}: {hit.snippet}".strip()
            if s:
                notes.append("[corpus] " + s[:300])
    except (ImportError, RuntimeError, OSError, TypeError, ValueError):
        pass
    try:
        from core.skills.web_search import WebSearchSkill

        res = await WebSearchSkill().execute({"query": goal, "max_results": 2})
        for item in (res.get("results") or [])[:2]:
            t = str(item.get("snippet") or item.get("content") or item.get("title") or "")
            if t.strip():
                notes.append("[web] " + t.strip()[:300])
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError, KeyError):
        pass
    return notes[:max_notes]


async def _generate(prompt: str, *, max_tokens: int = 1200) -> str:
    try:
        from core.brain.llm.local_code_model import get_local_code_model

        model = get_local_code_model()
        if model is not None:
            return await model.generate(
                prompt,
                system_prompt="You improve one Python function. Output only the corrected function, standard library only.",
                max_tokens=max_tokens, temperature=0.1,
            )
    except (ImportError, RuntimeError, OSError):
        pass
    try:
        from core.brain.llm.code_generator import LLMCodeGenerator

        return await LLMCodeGenerator(max_tokens=max_tokens, temperature=0.1).generate_async(
            prompt, context={"origin": "self_code_improver"})
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError, OSError):
        return ""


async def _retain(func_name: str, goal: str, outcome: str, lesson: str) -> str:
    text = f"Self-improvement ({outcome}) of {func_name} — goal '{goal[:60]}': {lesson}"
    try:
        from core.memory.memory_write_gateway import get_memory_write_gateway
        from core.runtime.gateways import MemoryWriteRequest

        await get_memory_write_gateway().write(MemoryWriteRequest(
            content=text,
            metadata={"family": "learned_rsi_lesson", "source": "self_code_improver", "outcome": outcome},
            cause="self_code_improver.retain",
        ))
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
        record_degradation("self_code_improver.retain", exc, severity="warning",
                           action="kept the improvement after lesson retention failed")
    return text


async def improve_function(
    *,
    target_file: str,
    func_name: str,
    goal: str,
    checks: list[dict[str, Any]],
    max_iters: int = 3,
    enact: bool = True,
) -> ImproveResult:
    """Improve one function in her own source, verified by behavioral checks."""
    path = Path(target_file)
    result = ImproveResult(ok=False, target_file=str(path), func_name=func_name, goal=goal)
    result.total_checks = len(checks)
    src = path.read_text(encoding="utf-8")
    extracted = _extract_function_source(src, func_name)
    if not extracted:
        result.status = "function_not_found"
        result.error = f"{func_name} not found in {target_file}"
        return result
    original_src, _, _ = extracted
    result.original_source = original_src

    original_passed, _ = await asyncio.to_thread(_verify, original_src, func_name, checks)
    result.original_passed = original_passed
    if original_passed == len(checks):
        result.status = "already_meets_standard"
        result.error = "current function already passes every check — no improvement to make"
        return result

    result.research_used = await _research(goal)
    failure = ""
    improved_src = ""
    for attempt in range(1, max_iters + 1):
        result.iterations = attempt
        prompt = _improve_prompt(func_name, original_src, goal, result.research_used, checks, failure)
        raw = await _generate(prompt)
        candidate = _extract_function_from_response(raw, func_name)
        if not candidate:
            failure = "no function returned; output the complete corrected function only"
            continue
        passed, details = await asyncio.to_thread(_verify, candidate, func_name, checks)
        if passed == len(checks):
            improved_src = candidate
            result.improved_passed = passed
            break
        fails = [d for d in details if not d.get("ok")]
        failure = f"passed {passed}/{len(checks)}; failing checks: {json.dumps(fails)[:400]}"
        result.improved_passed = max(result.improved_passed, passed)

    if not improved_src:
        result.status = "no_verified_improvement"
        result.error = f"could not produce a function passing all {len(checks)} checks in {result.iterations} iterations"
        result.lesson_retained = await _retain(func_name, goal, "PARTIAL",
                                               f"best {result.improved_passed}/{len(checks)}; {failure}")
        return result

    result.improved_source = improved_src
    result.ok = True
    result.status = "verified_improvement"

    if enact:
        try:
            new_src = _replace_function(src, func_name, improved_src)
            # Blessed async write lane — never a sync fsync on the live loop.
            from core.runtime.atomic_writer import async_atomic_write_text

            await async_atomic_write_text(path, new_src)
            result.enacted = True
        except (OSError, ValueError) as exc:
            result.error = f"verified but enactment failed: {exc}"
            result.status = "verified_not_enacted"

    result.lesson_retained = await _retain(
        func_name, goal, "SUCCESS",
        f"the fix passed all {len(checks)} checks the original failed {len(checks)-original_passed} of; "
        "verified in isolation before enacting.",
    )
    return result


def _improve_prompt(func_name, original, goal, research, checks, failure) -> str:
    parts = [
        f"Improve this Python function so that: {goal}\n",
        "Keep the same name, signature, and all existing correct behavior. "
        "Output ONLY the complete corrected function.\n",
        f"Current function:\n{original}\n",
    ]
    if research:
        parts.append("Reference knowledge:\n- " + "\n- ".join(research))
    parts.append(
        "It must satisfy these input->output checks exactly:\n"
        + "\n".join(f"  {func_name}({', '.join(map(repr, c['args']))}) == {c['expected']!r}" for c in checks)
    )
    if failure:
        parts.append("Your previous attempt did not pass:\n" + failure)
    return "\n\n".join(parts)


def _replace_function(source: str, func_name: str, new_func_src: str) -> str:
    extracted = _extract_function_source(source, func_name)
    if not extracted:
        raise ValueError(f"{func_name} not found for replacement")
    _, start, end = extracted
    lines = source.splitlines()
    # preserve the original indentation of the def line
    orig_indent = lines[start][: len(lines[start]) - len(lines[start].lstrip())]
    new_lines = new_func_src.splitlines()
    new_indent = new_lines[0][: len(new_lines[0]) - len(new_lines[0].lstrip())] if new_lines else ""
    if new_indent != orig_indent:
        shift = orig_indent
        rebased = []
        for ln in new_lines:
            rebased.append(shift + ln[len(new_indent):] if ln.startswith(new_indent) else shift + ln.lstrip())
        new_lines = rebased
    return "\n".join(lines[:start] + new_lines + lines[end:]) + ("\n" if source.endswith("\n") else "")


__all__ = ["ImproveResult", "improve_function"]
