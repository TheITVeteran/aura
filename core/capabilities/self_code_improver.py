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

Rollback is SYMMETRIC with promotion (July external review): before any
enactment, a write-ahead ledger record persists the FULL pre-image (original
function source + file hashes). ``rollback_enactment`` restores the original
with the same atomic write lane, refuses on file drift (the file changed
since the enactment — a blind restore would destroy someone else's work),
and verifies the restored function matches the ledger byte-for-byte. An
improvement that cannot be undone with the same rigor it was applied with
was never a governed improvement.
"""
from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import logging
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.governance.will import ActionDomain
from core.runtime.action_executor import ActionExecutor
from core.runtime.errors import record_degradation
from core.runtime.skill_contract import ActionExpectation

logger = logging.getLogger("Aura.SelfCodeImprover")

_ENACTMENT_LEDGER_DIR = Path("~/.aura/data/self_improvement/enactments").expanduser()


def _sha(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8", "replace"), digest_size=16).hexdigest()


def _self_code_write_completed(result: dict[str, Any]) -> bool:
    return bool(
        result.get("ok") is True
        and result.get("effect_verified") is True
        and result.get("receipt_persisted") is True
        and result.get("post_action_receipt_id")
    )


async def _execute_self_code_write(
    *,
    path: Path,
    text: str,
    action_name: str,
    action_id: str,
    rollback_target: str | None,
) -> dict[str, Any]:
    result = await ActionExecutor.execute(
        domain=ActionDomain.FILE_WRITE,
        action_name=action_name,
        params={"path": str(path), "text": text, "encoding": "utf-8"},
        source="self_code_improver",
        rollback_target=rollback_target,
        action_id=action_id,
        expectation=ActionExpectation(
            objective=f"durably write and hash-verify {path}",
            acceptance_criteria=["effect_verified"],
            required_evidence=[
                "verification_evidence.observation.state.sha256",
            ],
            user_visible_effect=f"{path} matches the intended source bytes",
            repair_hint="read back the target hash or compensate from the enactment ledger",
            rollback_hint=rollback_target or "restore the pre-image from the enactment ledger",
            allow_partial=False,
        ),
    )
    if not isinstance(result, dict):
        raise TypeError("ActionExecutor returned a non-mapping self-code result")
    return {str(key): value for key, value in result.items()}


async def _record_enactment(
    *,
    path: Path,
    func_name: str,
    goal: str,
    file_before: str,
    file_after: str,
    original_function: str,
    improved_function: str,
) -> str:
    """Write-ahead rollback record: durable BEFORE the file mutates."""
    record_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{_sha(file_after)[:8]}"
    record = {
        "schema": "aura.self_code_enactment.v1",
        "id": record_id,
        "at": time.time(),
        "target_file": str(path),
        "func_name": func_name,
        "goal": goal,
        "file_sha_before": _sha(file_before),
        "file_sha_after": _sha(file_after),
        "original_function_source": original_function,
        "improved_function_source": improved_function,
    }
    action = await _execute_self_code_write(
        path=_ENACTMENT_LEDGER_DIR / f"{record_id}.json",
        text=json.dumps(record, indent=1),
        action_name="record_self_code_enactment",
        action_id=f"self-code-ledger:{record_id}",
        rollback_target=None,
    )
    if not _self_code_write_completed(action):
        raise OSError(
            "enactment ledger write did not complete its effect/receipt contract: "
            f"{action.get('status') or action.get('error') or 'unknown'}"
        )
    return record_id


def _load_enactment(record_id: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(
            (_ENACTMENT_LEDGER_DIR / f"{record_id}.json").read_text(encoding="utf-8")
        )
        if not isinstance(payload, dict):
            return None
        return {str(key): value for key, value in payload.items()}
    except (OSError, ValueError):
        return None


def latest_enactment_for(target_file: str) -> dict[str, Any] | None:
    """Most recent ledger record for a file (rollback without an id)."""
    try:
        records = sorted(_ENACTMENT_LEDGER_DIR.glob("*.json"), reverse=True)
    except OSError:
        return None
    for record_path in records:
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(record, dict):
            continue
        if str(record.get("target_file")) == str(target_file):
            return {str(key): value for key, value in record.items()}
    return None


async def rollback_enactment(
    record_id: str = "",
    *,
    target_file: str = "",
    force: bool = False,
) -> dict[str, Any]:
    """Undo an enacted improvement with promotion-grade rigor.

    Refuses when the file has drifted since the enactment (someone else's
    edits would be destroyed) unless ``force``. Verifies the restored
    function matches the ledger pre-image byte-for-byte.
    """
    record = _load_enactment(record_id) if record_id else latest_enactment_for(target_file)
    if not record:
        return {"ok": False, "status": "no_enactment_record"}

    path = Path(str(record["target_file"]))
    try:
        current = await asyncio.to_thread(path.read_text, encoding="utf-8")
    except OSError as exc:
        return {"ok": False, "status": "target_unreadable", "error": str(exc)}

    if _sha(current) != record["file_sha_after"] and not force:
        return {
            "ok": False,
            "status": "refused_file_drift",
            "detail": "the file changed since this enactment; pass force=True only "
            "after confirming the drift is safe to overwrite",
            "record_id": record["id"],
        }

    restored = _replace_function(
        current, str(record["func_name"]), str(record["original_function_source"])
    )
    action = await _execute_self_code_write(
        path=path,
        text=restored,
        action_name="rollback_self_code_enactment",
        action_id=f"self-code-rollback:{record['id']}",
        rollback_target=str(record["id"]),
    )

    # Symmetric verification: the restored function must equal the pre-image.
    extracted = _extract_function_source(
        await asyncio.to_thread(path.read_text, encoding="utf-8"),
        str(record["func_name"]),
    )
    restored_ok = bool(
        extracted and extracted[0].strip() == str(record["original_function_source"]).strip()
    )
    transaction_complete = _self_code_write_completed(action)
    outcome = {
        "ok": restored_ok and transaction_complete,
        "status": (
            "rolled_back"
            if restored_ok and transaction_complete
            else "rollback_effect_verified_receipt_failed"
            if restored_ok and action.get("effect_verified") is True
            else "rollback_verification_failed"
        ),
        "record_id": record["id"],
        "target_file": str(path),
        "func_name": record["func_name"],
        "effect_verified": action.get("effect_verified") is True,
        "post_action_receipt_id": action.get("post_action_receipt_id"),
        "receipt_persisted": action.get("receipt_persisted") is True,
        "manual_reconciliation_required": bool(
            action.get("manual_reconciliation_required")
            or (restored_ok and not transaction_complete)
        ),
    }
    logger.warning("Self-code rollback %s: %s", outcome["status"], outcome)
    return outcome


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
    # Write-ahead rollback ledger id — set before any enactment mutates the
    # file, so every applied improvement is symmetrically undoable.
    enactment_record: str = ""
    enactment_receipt_id: str = ""
    compensation: dict[str, Any] = field(default_factory=dict)

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
            end = node.end_lineno or node.lineno
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
        # Parse checks with json.loads at runtime — do NOT paste json.dumps output
        # as Python source: JSON null/true/false are not Python literals and would
        # raise NameError, silently zeroing the whole verification.
        + "\n\nimport json\n_out=[]\n_CHECKS=json.loads(" + repr(json.dumps(checks)) + ")\n"
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
        from core.runtime.subprocess_gateway import get_subprocess_gateway

        proc = get_subprocess_gateway().run(
            [sys.executable, path],
            capture_output=True,
            timeout=15,
            # The verifier runner only computes function outputs and prints JSON —
            # no writes, no network. read_only keeps it off the effect-governance
            # path (which would else raise GovernanceViolation outside a governed
            # scope), while still requiring a specific single-line source label.
            read_only=True,
            source="tool_execution:self_code_improver.verify_checks",
        )
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
            return str(
                await model.generate(
                    prompt,
                    system_prompt=(
                        "You improve one Python function. Output only the corrected "
                        "function, standard library only."
                    ),
                    max_tokens=max_tokens,
                    temperature=0.1,
                )
            )
    except (ImportError, RuntimeError, OSError):
        pass
    try:
        from core.brain.llm.code_generator import LLMCodeGenerator

        return str(
            await LLMCodeGenerator(
                max_tokens=max_tokens,
                temperature=0.1,
            ).generate_async(
                prompt,
                context={"origin": "self_code_improver"},
            )
        )
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
    src = await asyncio.to_thread(path.read_text, encoding="utf-8")
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
            # Write-ahead rollback record FIRST: the pre-image is durable
            # before the file mutates, so the undo path always exists.
            result.enactment_record = await _record_enactment(
                path=path,
                func_name=func_name,
                goal=goal,
                file_before=src,
                file_after=new_src,
                original_function=original_src,
                improved_function=improved_src,
            )
            action = await _execute_self_code_write(
                path=path,
                text=new_src,
                action_name="enact_self_code_improvement",
                action_id=f"self-code-enact:{result.enactment_record}",
                rollback_target=result.enactment_record,
            )
            result.enactment_receipt_id = str(
                action.get("post_action_receipt_id") or ""
            )
            if _self_code_write_completed(action):
                result.enacted = True
            elif action.get("effect_verified") is True:
                result.compensation = await rollback_enactment(
                    result.enactment_record,
                    force=True,
                )
                result.status = (
                    "enactment_receipt_failed_rolled_back"
                    if result.compensation.get("ok") is True
                    else "enactment_partial_requires_manual_reconciliation"
                )
                result.error = (
                    "target source changed but the enactment receipt did not persist; "
                    f"compensation={result.compensation.get('status')}"
                )
            else:
                result.status = "verified_not_enacted"
                result.error = str(
                    action.get("error")
                    or action.get("status")
                    or "self-code action did not verify"
                )
        except (OSError, ValueError) as exc:
            result.error = f"verified but enactment failed: {exc}"
            result.status = "verified_not_enacted"

    result.lesson_retained = await _retain(
        func_name, goal, "SUCCESS",
        f"the fix passed all {len(checks)} checks the original failed {len(checks)-original_passed} of; "
        "verified in isolation before enacting.",
    )
    return result


def _improve_prompt(
    func_name: str,
    original: str,
    goal: str,
    research: list[str],
    checks: list[dict[str, Any]],
    failure: str,
) -> str:
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
