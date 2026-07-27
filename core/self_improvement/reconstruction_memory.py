"""What she learned building the last thing, available while building the next.

Each reconstruction currently starts from nothing. That is the wrong shape for
this problem: most software is mostly other software. A board game she has
already built teaches the next board game what a move looks like, that captures
are usually mandatory somewhere, that a render surface is separate from a rule
engine, and — most usefully — which mistakes the gate caught last time.

So every attempt is kept: the plan, the outcome, and the findings. When a new
target is planned, the closest prior attempts come back with it. Even a small
transfer helps; the point is not to have seen this exact program before, it is
to stop rediscovering that pieces have to move.

Three kinds of transfer, in increasing order of how much they are worth:

* **shape** — the entry points and components that recurred, so the
  decomposition starts from something rather than a blank page;
* **invariants** — properties that held for a similar target are usually worth
  asserting again;
* **corrections** — what the gate rejected last time, which is the only record
  of what she actually gets wrong.

Kept as one JSONL ledger, appended through the write gateway, bounded, and
never fatal: a reconstruction must not fail because its memory is unavailable.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation

_RECOVERABLE = (RuntimeError, AttributeError, TypeError, ValueError, OSError, KeyError)

_LEDGER = Path("artifacts/current/reconstruction_memory.jsonl")
_MAX_ENTRIES = 400
_STOPWORDS = frozenset(
    {"a", "an", "the", "game", "of", "app", "clone", "program", "tool", "for", "and", "in"}
)


@dataclass(frozen=True)
class PriorAttempt:
    """One thing she built before, and how it went."""

    target: str
    summary: str = ""
    entry_points: tuple[str, ...] = ()
    components: tuple[str, ...] = ()
    invariants: tuple[str, ...] = ()
    corrections: tuple[str, ...] = ()
    succeeded: bool = False
    at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "summary": self.summary,
            "entry_points": list(self.entry_points),
            "components": list(self.components),
            "invariants": list(self.invariants),
            "corrections": list(self.corrections),
            "succeeded": self.succeeded,
            "at": self.at or time.time(),
        }


@dataclass
class TransferBrief:
    """What the past has to offer this target, ready to put in a prompt."""

    priors: list[PriorAttempt] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.priors

    def as_prompt_block(self, *, limit: int = 3) -> str:
        """Prior experience as evidence, explicitly not as an answer."""
        if self.is_empty:
            return ""
        lines = [
            "## WHAT YOU LEARNED BUILDING SIMILAR THINGS",
            "Your own earlier reconstructions. Use them to shape the "
            "decomposition and to avoid repeating a correction — not as the "
            "answer, which may well differ for this target.",
        ]
        for prior in self.priors[:limit]:
            verdict = "shipped" if prior.succeeded else "was rejected"
            lines.append(f"- {prior.target} ({verdict})")
            if prior.entry_points:
                lines.append(f"    entry points: {', '.join(prior.entry_points[:8])}")
            if prior.components:
                lines.append(f"    components: {', '.join(prior.components[:6])}")
            if prior.invariants:
                lines.append(f"    properties that held: {'; '.join(prior.invariants[:3])}")
            if prior.corrections:
                lines.append(
                    f"    what the gate caught: {'; '.join(prior.corrections[:3])}"
                )
        return "\n".join(lines)


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", str(text or "").lower())
    return {word for word in words if word not in _STOPWORDS and len(word) > 1}


def _similarity(a: str, b: str) -> float:
    """Jaccard over content words. Crude, cheap, and enough to rank neighbours."""
    left, right = _tokens(a), _tokens(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _ledger_path(root: Path | None = None) -> Path:
    return (root or Path.cwd()) / _LEDGER


def load_attempts(root: Path | None = None) -> list[PriorAttempt]:
    path = _ledger_path(root)
    if not path.exists():
        return []
    attempts: list[PriorAttempt] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            attempts.append(
                PriorAttempt(
                    target=str(raw.get("target") or ""),
                    summary=str(raw.get("summary") or ""),
                    entry_points=tuple(str(x) for x in (raw.get("entry_points") or [])),
                    components=tuple(str(x) for x in (raw.get("components") or [])),
                    invariants=tuple(str(x) for x in (raw.get("invariants") or [])),
                    corrections=tuple(str(x) for x in (raw.get("corrections") or [])),
                    succeeded=bool(raw.get("succeeded")),
                    at=float(raw.get("at") or 0.0),
                )
            )
    except _RECOVERABLE as exc:
        record_degradation(
            "reconstruction_memory", exc, severity="info", action="read no prior attempts"
        )
        return []
    return attempts


def recall_for(target: str, *, summary: str = "", root: Path | None = None, limit: int = 3) -> TransferBrief:
    """The closest things she has built before.

    Ranks by similarity, and prefers a rejected attempt at the same target over
    a successful one at a distant target — the corrections from a failure are
    the most transferable thing in the ledger.
    """
    query = f"{target} {summary}".strip()
    scored: list[tuple[float, PriorAttempt]] = []
    for prior in load_attempts(root):
        score = _similarity(query, f"{prior.target} {prior.summary}")
        if prior.corrections:
            score += 0.15  # a recorded mistake is worth more than a clean run
        if score > 0.08:
            scored.append((score, prior))
    scored.sort(key=lambda pair: (-pair[0], -pair[1].at))
    return TransferBrief(priors=[prior for _, prior in scored[:limit]])


async def remember_attempt(attempt: PriorAttempt, *, root: Path | None = None) -> bool:
    """Append one attempt. Never fatal — memory failing must not fail a build."""
    path = _ledger_path(root)
    try:
        from core.runtime.file_write_gateway import get_file_write_gateway

        gateway = get_file_write_gateway()
        await gateway.ensure_directory_async(path.parent, source="reconstruction_memory")
        existing = load_attempts(root)
        entries = [*existing, attempt][-_MAX_ENTRIES:]
        payload = "\n".join(
            json.dumps(entry.to_dict(), sort_keys=True) for entry in entries
        )
        await gateway.write_text_async(path, payload + "\n", source="reconstruction_memory")
        return True
    except _RECOVERABLE as exc:
        record_degradation(
            "reconstruction_memory",
            exc,
            severity="warning",
            action="continued without recording this reconstruction attempt",
        )
        return False


def attempt_from_outcome(
    plan: Any,
    *,
    succeeded: bool,
    findings: list[str] | tuple[str, ...] = (),
) -> PriorAttempt:
    """Turn a finished reconstruction into something the next one can use."""
    return PriorAttempt(
        target=str(getattr(plan, "target", "") or ""),
        summary=str(getattr(plan, "summary", "") or ""),
        entry_points=tuple(str(name) for name in (getattr(plan, "entry_points", ()) or ())),
        components=tuple(
            str(getattr(component, "name", "")) for component in (getattr(plan, "components", ()) or ())
        ),
        invariants=tuple(
            str(getattr(invariant, "description", ""))
            for invariant in (getattr(plan, "invariants", ()) or ())
        )
        if succeeded
        else (),
        corrections=tuple(str(finding) for finding in findings)[:6],
        succeeded=bool(succeeded),
        at=time.time(),
    )


__all__ = [
    "PriorAttempt",
    "TransferBrief",
    "attempt_from_outcome",
    "load_attempts",
    "recall_for",
    "remember_attempt",
]
