"""Evidence provider — the ReAct layer that grounds reasoning in real data.

This is the difference between "prompt the model harder" and actually *knowing*.
Before the amplifier reasons about a verifiable question it should gather real
evidence by acting on the world it can read:

* repo / architecture / self-claim questions → search the actual codebase
  (ripgrep when available, else an in-process scan), then **read the matching
  source spans** (``path:line: code``). The model then answers *from* real spans,
  and the repo/citation truth engines check the answer *against* the same spans.
* factual / memory questions → recall from Aura's live memory facade.

So generation is conditioned on retrieved fact, not vibes, and verification has
something concrete to check. Pure read-only: it never mutates anything.
"""
from __future__ import annotations

import asyncio
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation
from core.utils.paths import PROJECT_ROOT

logger = logging.getLogger("Aura.EvidenceProvider")

_IDENT_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]{3,})\b")
_PATH_RE = re.compile(r"\b([A-Za-z_][\w./-]*\.(?:py|md|json|toml|yaml|yml))\b")
_CAMEL_OR_SNAKE = re.compile(r"^(?:[A-Z][a-z0-9]+){2,}$|^[a-z][a-z0-9]*_[a-z0-9_]+$")
_STOP = frozenset(
    "the and that this with from which would should there their about into your where when what "
    "does done make made work works using used your you aura have here only just like also both "
    "explain describe implement function module method class".split()
)
_SKIP_DIRS = {".venv", "__pycache__", "node_modules", ".git", "archive", "dev_archive", ".mypy_cache",
              ".ruff_cache", ".pytest_cache", "dist", "build",
              # Worktrees are COPIES of this repository. Measured 2026-08-06:
              # 326,885 of the 334,558 scannable .py files were under
              # .claude/worktrees — 98% — so the 4,000-file scan cap was spent
              # almost entirely on duplicates and core/ was never reached. Two
              # separate harms: evidence gathering silently found nothing for
              # most symbols, and anything it did find could cite
              # `.claude/worktrees/<branch>/core/x.py:42`, which is a real line
              # in a file that is not the running code.
              ".claude", "worktrees",
              # Generated evidence bundles, not source. Aura citing her own
              # past output as evidence for a claim about her source is a
              # circularity nobody asked for.
              "artifacts"}

#: A hit scoring at least this much is admitted no matter how full the
#: candidate bucket is: it means the file is named after the symbol, or the
#: line defines it. Those are exactly the hits a cap must never discard —
#: without this, asking about SubprocessGateway filled up on aura_main.py and
#: interface/server.py and never reached core/runtime/subprocess_gateway.py.
_STRONG_EVIDENCE_SCORE = 5.0


@dataclass
class EvidenceSpan:
    source: str          # "repo" | "memory"
    ref: str             # "core/x.py:42" or memory id
    text: str

    def render(self) -> str:
        return f"{self.ref}: {self.text}" if self.ref else self.text


def _salient_terms(objective: str, *, limit: int = 6) -> list[str]:
    """Pull the identifier-shaped, content-bearing terms worth searching for."""
    terms: list[str] = []
    # Explicit code-ish identifiers (CamelCase / snake_case) rank first.
    for m in _IDENT_RE.finditer(objective or ""):
        w = m.group(1)
        if w.lower() in _STOP:
            continue
        if _CAMEL_OR_SNAKE.match(w) and w not in terms:
            terms.append(w)
    if len(terms) < limit:
        for m in _IDENT_RE.finditer(objective or ""):
            w = m.group(1)
            if len(w) >= 5 and w.lower() not in _STOP and w not in terms:
                terms.append(w)
            if len(terms) >= limit:
                break
    return terms[:limit]


class EvidenceProvider:
    def __init__(self, root: Path | None = None, *, memory_facade: Any | None = None) -> None:
        self._root = Path(root or PROJECT_ROOT)
        self._memory = memory_facade

    # -------------------------------------------------------------- public
    async def gather(
        self,
        objective: str,
        *,
        task_type: str,
        limit: int = 6,
    ) -> list[EvidenceSpan]:
        spans: list[EvidenceSpan] = []
        tt = (task_type or "generic").lower()
        if tt in {"repo", "repo_audit", "architecture", "code_audit", "self_claim", "code"}:
            spans.extend(await self._repo_evidence(objective, limit=limit))
        if tt in {"factual", "self_claim", "generic", "architecture"} or not spans:
            spans.extend(await self._memory_evidence(objective, limit=max(2, limit - len(spans))))
        # De-dupe by rendered text, keep order.
        seen: set[str] = set()
        out: list[EvidenceSpan] = []
        for s in spans:
            key = s.render()[:160]
            if key and key not in seen:
                seen.add(key)
                out.append(s)
        return out[:limit]

    async def render_pack(self, objective: str, *, task_type: str, limit: int = 6) -> list[str]:
        return [s.render() for s in await self.gather(objective, task_type=task_type, limit=limit)]

    # -------------------------------------------------------------- repo
    async def _repo_evidence(self, objective: str, *, limit: int) -> list[EvidenceSpan]:
        terms = _salient_terms(objective)
        # Always honor explicitly named paths.
        paths = [m.group(1) for m in _PATH_RE.finditer(objective or "")]
        spans: list[EvidenceSpan] = []
        for ref in paths[:3]:
            spans.extend(self._read_named_path(ref))
        if not terms:
            return spans[:limit]
        try:
            hits = await self._ripgrep(terms, limit=limit) if shutil.which("rg") else await asyncio.to_thread(
                self._inprocess_search, terms, limit
            )
            spans.extend(hits)
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            record_degradation("evidence_repo", exc)
        return spans[:limit]

    def _read_named_path(self, ref: str) -> list[EvidenceSpan]:
        candidate = self._root / ref
        if not candidate.exists():
            matches = list(self._root.rglob(Path(ref).name))[:1]
            if not matches:
                return []
            candidate = matches[0]
        try:
            lines = candidate.read_text(encoding="utf-8", errors="ignore").splitlines()
        except (OSError, ValueError):
            return []
        rel = candidate.relative_to(self._root) if candidate.is_relative_to(self._root) else candidate
        # First few non-trivial lines as a span (signature/docstring region).
        out: list[EvidenceSpan] = []
        for i, ln in enumerate(lines[:40], start=1):
            if ln.strip() and (ln.lstrip().startswith(("def ", "class ", "async def ")) or i == 1):
                out.append(EvidenceSpan("repo", f"{rel}:{i}", ln.strip()[:200]))
            if len(out) >= 4:
                break
        return out

    async def _ripgrep(self, terms: list[str], *, limit: int) -> list[EvidenceSpan]:
        """Search terms in priority order (most specific identifier first).

        ``terms`` is already ranked by :func:`_salient_terms`, so searching them in
        order — rather than OR-ing equally — surfaces the *defining* file of the most
        informative symbol before generic keyword hits crowd it out.
        """
        from core.runtime.subprocess_gateway import get_subprocess_gateway

        gateway = get_subprocess_gateway()
        spans: list[EvidenceSpan] = []
        seen: set[str] = set()
        # Prefer a definition line for the lead identifier when there is one.
        for prefer_def in (True, False):
            for term in terms:
                if len(spans) >= limit:
                    return spans
                pattern = rf"(?:def|class)\s+{re.escape(term)}\b" if prefer_def else re.escape(term)
                argv = (
                    "rg", "--no-heading", "--line-number", "--max-count", "2",
                    "--glob", "*.py",
                    "--glob", "!**/{.venv,__pycache__,archive,dev_archive,node_modules}/**",
                    "-e", pattern, str(self._root),
                )
                try:
                    res = await gateway.run_async(
                        argv, timeout=10.0, read_only=True, source="evidence_provider:ripgrep",
                        # "none", not "auto". ripgrep is a text search and uses
                        # no accelerator, and "auto" could not infer that: the
                        # gateway raised
                        # subprocess_accelerator_capability_unresolved on EVERY
                        # call, the handler below swallowed it, and this
                        # provider returned zero spans. Silently. Every ReAct
                        # answer that was supposed to be grounded in real source
                        # spans has been grounded in nothing.
                        accelerator_capability="none",
                    )
                except (OSError, RuntimeError, ValueError) as exc:
                    # Recorded, not swallowed. The failure that hid here
                    # disabled evidence grounding entirely while every caller
                    # kept reporting success, which is the exact shape this
                    # codebase keeps having to dig out.
                    record_degradation(
                        "evidence_repo",
                        exc,
                        severity="degraded",
                        action="repo evidence search failed; answering without source spans",
                    )
                    continue
                for line in (res.stdout or "").splitlines():
                    parts = line.split(":", 2)
                    if len(parts) < 3:
                        continue
                    path, lineno, code = parts
                    try:
                        rel = Path(path).relative_to(self._root)
                    except ValueError:
                        rel = Path(path).name
                    ref = f"{rel}:{lineno}"
                    if ref in seen:
                        continue
                    seen.add(ref)
                    spans.append(EvidenceSpan("repo", ref, code.strip()[:200]))
                    if len(spans) >= limit:
                        return spans
        return spans

    def _inprocess_search(self, terms: list[str], limit: int) -> list[EvidenceSpan]:
        """Priority scan: collect hits for the most specific term first.

        Mirrors the ripgrep priority so the defining file of the lead identifier
        surfaces before generic keyword matches. Reads every candidate file once
        and buckets matched lines by which term hit, then drains buckets in the
        ranked order of ``terms``.
        """
        needles = [t for t in terms if t]
        if not needles:
            return []
        import re as _re

        def_re = {t: _re.compile(rf"\b(?:def|class)\s+{_re.escape(t)}\b") for t in needles}
        # SubprocessGateway -> subprocess_gateway, so a file named after the
        # symbol can be recognised as its home.
        snake = {
            t: _re.sub(r"(?<!^)(?=[A-Z])", "_", t).lower().strip("_") for t in needles
        }
        # (score, term_rank, tie) -> span. Candidates are SCORED and sorted
        # rather than taken in walk order: the previous version filled a small
        # per-term bucket first-come-first-served, so six test files could
        # crowd out the module that defines the symbol. Measured: asking about
        # SubprocessGateway returned six test spans and not
        # core/runtime/subprocess_gateway.py.
        candidates: list[tuple[float, int, int, EvidenceSpan]] = []
        per_term_cap = max(2, limit)
        per_term_candidates = max(per_term_cap * 8, 40)
        counts = {t: 0 for t in needles}
        scanned = 0
        tie = 0
        for py in self._root.rglob("*.py"):
            if scanned >= 4000:
                break
            if set(py.parts) & _SKIP_DIRS:
                continue
            scanned += 1
            try:
                rel = py.relative_to(self._root)
                parts = set(rel.parts)
                stem = py.stem.lower()
                # What a person does when asked where a symbol lives: look at
                # the file named after it, in the implementation tree, at the
                # line that defines it.
                is_test = bool(parts & {"tests", "test"}) or stem.startswith("test_")
                is_tooling = bool(parts & {"tools", "scripts", "training"})
                path_score = 0.0
                if not is_test and not is_tooling:
                    path_score += 2.0
                elif is_tooling:
                    path_score += 0.5
                for i, ln in enumerate(
                    py.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1
                ):
                    for rank, t in enumerate(needles):
                        if t not in ln:
                            continue
                        score = path_score
                        if def_re[t].search(ln):
                            score += 4.0
                        if stem == snake[t] or stem == t.lower():
                            score += 5.0
                        # A cap that admits candidates in walk order still lets
                        # early files crowd out the answer: with the cap alone,
                        # asking about SubprocessGateway filled up on aura_main
                        # and interface/server before ever reaching
                        # core/runtime/subprocess_gateway.py. A hit in the file
                        # named after the symbol, or one that defines it, is
                        # admitted regardless of how full the bucket is —
                        # that is precisely the hit worth displacing others.
                        if score < _STRONG_EVIDENCE_SCORE and counts[t] >= per_term_candidates:
                            break
                        counts[t] += 1
                        tie += 1
                        candidates.append(
                            (
                                -score,
                                rank,
                                tie,
                                EvidenceSpan("repo", f"{rel}:{i}", ln.strip()[:200]),
                            )
                        )
                        break
            except (OSError, ValueError):
                continue

        candidates.sort()
        ordered: list[EvidenceSpan] = []
        seen: set[str] = set()
        for _score, _rank, _tie, span in candidates:
            if span.ref in seen:
                continue
            seen.add(span.ref)
            ordered.append(span)
            if len(ordered) >= limit:
                break
        return ordered

    # -------------------------------------------------------------- memory
    def _ensure_memory(self) -> Any:
        if self._memory is not None:
            return self._memory
        try:
            from core.container import ServiceContainer

            self._memory = ServiceContainer.get("memory_facade", default=None)
        except (ImportError, RuntimeError, AttributeError):
            self._memory = None
        return self._memory

    async def _memory_evidence(self, objective: str, *, limit: int) -> list[EvidenceSpan]:
        facade = self._ensure_memory()
        if facade is None or not hasattr(facade, "search"):
            return []
        try:
            results = await facade.search(objective, limit=limit)
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("evidence_memory", exc)
            return []
        spans: list[EvidenceSpan] = []
        for item in list(results or [])[:limit]:
            if isinstance(item, dict):
                content = str(item.get("content") or item.get("text") or "").strip()
                ref = str(item.get("id", "") or "memory")
            else:
                content = str(item or "").strip()
                ref = "memory"
            if content:
                spans.append(EvidenceSpan("memory", ref, content[:240]))
        return spans


def get_evidence_provider() -> EvidenceProvider:
    return EvidenceProvider()
