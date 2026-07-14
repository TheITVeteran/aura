"""Verifier Foundry — measured verifier reliability and the self-training gate.
================================================================================
P1 of the frontier-general arc (docs/FRONTIER_GENERAL_ARC.md).

The ceiling on a self-improving local mind is the boundary of verification:
she may only compound (train on) what she can check, or she amplifies her own
garbage. That boundary is movable — but only honestly if every verifier's
actual reliability is MEASURED against later ground truth, and admission to
the self-training loop is granted by evidence, not by assumption.

The foundry is that bookkeeper and gatekeeper:

  * every truth-engine verdict can be recorded (verifier, domain, hard pass,
    soft score) and later GRADED when reality arrives — an audit, a resolved
    prediction, an action outcome;
  * per (verifier, domain) reliability is tracked with pessimistic statistics
    (Wilson lower bound on accuracy; false-pass rate, the metric that poisons
    training data, bounded from above);
  * ``domain_admitted()`` answers the only question the training pipe may ask:
    has this domain's verification EARNED the right to mint training data?
  * ``weight_for()`` gives the registry reliability weights for soft-score
    folding (the hard gate is never softened — a provable failure is final);
  * classically deterministic domains (code, math, logic) are seed-admitted —
    their engines are correct by construction — but evidence still accumulates
    and can REVOKE an admission: the gate ratchets on measurement, not faith.

Ledger: event-sourced (events.jsonl fold) + AuditChain tamper evidence, the
same proven pattern as the Ulysses covenant — a mind must not be able to
quietly rewrite the record of how trustworthy its own checkers are.
"""
from __future__ import annotations

import json
import logging
import math
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from core.runtime.audit_chain import AuditChain
from core.runtime.errors import record_degradation
from core.runtime.flags import FlagKind, declare

logger = logging.getLogger("Aura.VerifierFoundry")

SCHEMA_VERSION = 1

_FOUNDRY_DIR_FLAG = declare(
    "AURA_FOUNDRY_DIR", kind=FlagKind.STRING, default="",
    description="Override directory for the verifier-foundry reliability ledger",
    owner="core.brain.verifiers.foundry",
)
_ADMIT_MIN_GRADED_FLAG = declare(
    "AURA_FOUNDRY_ADMIT_MIN_GRADED", kind=FlagKind.INT, default=50,
    description="Graded verdicts required before a domain can earn admission",
    owner="core.brain.verifiers.foundry",
)
_ADMIT_MIN_WILSON_FLAG = declare(
    "AURA_FOUNDRY_ADMIT_MIN_WILSON", kind=FlagKind.FLOAT, default=0.85,
    description="Wilson lower bound on accuracy required for admission",
    owner="core.brain.verifiers.foundry",
)
_ADMIT_MAX_FALSE_PASS_FLAG = declare(
    "AURA_FOUNDRY_ADMIT_MAX_FALSE_PASS", kind=FlagKind.FLOAT, default=0.05,
    description="Upper bound (Wilson) on false-pass rate tolerated for admission",
    owner="core.brain.verifiers.foundry",
)

# Domains whose engines are deterministic checkers (execution, proof, unit
# arithmetic): admitted from birth, revocable by evidence.
SEED_ADMITTED_DOMAINS = frozenset({"code", "math", "logic"})

_Z95 = 1.6449  # one-sided 95%


def wilson_lower_bound(successes: int, n: int, z: float = _Z95) -> float:
    """Pessimistic estimate of a success rate: the Wilson score interval's
    lower bound. With no evidence it is 0 — trust must be earned."""
    if n <= 0:
        return 0.0
    p = successes / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4 * n)) / n)
    return max(0.0, (centre - margin) / denom)


def wilson_upper_bound(successes: int, n: int, z: float = _Z95) -> float:
    """Pessimistic (i.e. large) estimate of a failure-mode rate."""
    if n <= 0:
        return 1.0
    p = successes / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4 * n)) / n)
    return min(1.0, (centre + margin) / denom)


@dataclass
class ReliabilityCell:
    """Running reliability of one (verifier, domain) pair."""

    verifier: str
    domain: str
    recorded: int = 0
    graded: int = 0
    correct: int = 0
    passes: int = 0          # verdicts that said pass, among graded
    false_passes: int = 0    # said pass, truth was fail — the poison metric
    false_fails: int = 0     # said fail, truth was pass — wasted compute
    brier_sum: float = 0.0

    def accuracy_lb(self) -> float:
        return wilson_lower_bound(self.correct, self.graded)

    def false_pass_ub(self) -> float:
        """Upper bound on P(truth=fail | verdict=pass) — bounded pessimistically
        over the verdicts that could have poisoned training data."""
        if self.passes <= 0:
            return 1.0
        return wilson_upper_bound(self.false_passes, self.passes)

    def brier(self) -> float:
        return self.brier_sum / self.graded if self.graded else 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "verifier": self.verifier,
            "domain": self.domain,
            "recorded": self.recorded,
            "graded": self.graded,
            "correct": self.correct,
            "false_passes": self.false_passes,
            "false_fails": self.false_fails,
            "accuracy_lb": round(self.accuracy_lb(), 4),
            "false_pass_ub": round(self.false_pass_ub(), 4),
            "brier": round(self.brier(), 4),
        }


@dataclass(frozen=True)
class AdmissionDecision:
    domain: str
    admitted: bool
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)


class VerifierFoundry:
    """Reliability ledger + admission gate. Thread-safe; durable off-loop."""

    def __init__(self, *, root: Path | None = None,
                 clock: Callable[[], float] = time.time):
        env_root = str(_FOUNDRY_DIR_FLAG.value() or "")
        self.root = Path(root) if root else (
            Path(env_root) if env_root
            else (Path.home() / ".aura" / "data" / "verifier_foundry")
        )
        self._ensure_root()
        self.events_path = self.root / "events.jsonl"
        self._chain = AuditChain(self.root / "chain")
        self._clock = clock
        self._lock = threading.RLock()

        self._cells: dict[tuple[str, str], ReliabilityCell] = {}
        self._pending: dict[str, dict[str, Any]] = {}   # verdict_id → verdict
        self._pending_order: list[str] = []
        self._revoked_seeds: set[str] = set()
        self._restore_errors = 0

        # durable writes on a dedicated thread — same no-on-loop-fsync
        # discipline as the covenant ledger
        self._queue: "queue.Queue[dict[str, Any] | None]" = queue.Queue()
        self._pending_writes = 0
        self._pending_writes_lock = threading.Lock()
        self._writer = threading.Thread(target=self._writer_loop,
                                        name="verifier-foundry-ledger", daemon=True)
        self._writer_started = False
        self._writer_running = True
        self._restore()

    # ── storage plumbing ─────────────────────────────────────────────────
    def _ensure_root(self) -> None:
        try:
            from core.governance_context import local_internal_governed_scope
            from core.runtime.file_write_gateway import get_file_write_gateway

            with local_internal_governed_scope("verifier_foundry",
                                               domain="state_mutation"):
                get_file_write_gateway().ensure_directory(
                    self.root, source="verifier_foundry")
        except (OSError, RuntimeError, ImportError, ValueError) as exc:
            record_degradation("verifier_foundry", exc, severity="critical",
                               action="foundry root could not be created")

    def _append_event(self, kind: str, body: dict[str, Any]) -> None:
        body = dict(body)
        body["event"] = kind
        body["event_id"] = body.get("event_id") or f"vf-{uuid.uuid4().hex[:12]}"
        body.setdefault("timestamp", self._clock())
        self._fold(body)
        if not self._writer_started:
            self._writer_started = True
            self._writer.start()
        with self._pending_writes_lock:
            self._pending_writes += 1
        self._queue.put(body)

    def _writer_loop(self) -> None:
        while self._writer_running:
            item = self._queue.get()
            if item is None:
                self._writer_running = False
                continue
            try:
                self._persist_event(item)
            finally:
                with self._pending_writes_lock:
                    self._pending_writes -= 1

    def _persist_event(self, body: dict[str, Any]) -> None:
        line = json.dumps(body, sort_keys=True, ensure_ascii=False,
                          default=str) + "\n"
        try:
            from core.governance_context import local_internal_governed_scope
            from core.runtime.file_write_gateway import get_file_write_gateway

            with local_internal_governed_scope("verifier_foundry",
                                               domain="state_mutation"):
                get_file_write_gateway().append_text(
                    self.events_path, line, source="verifier_foundry")
                self._chain.append(receipt_id=str(body["event_id"]),
                                   kind=f"foundry_{body['event']}",
                                   body=body,
                                   timestamp=float(body["timestamp"]))
        except (OSError, RuntimeError, ImportError, ValueError) as exc:
            record_degradation("verifier_foundry", exc, severity="critical",
                               action="foundry event held in memory only")

    def flush_ledger(self, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._pending_writes_lock:
                if self._pending_writes <= 0:
                    return True
            time.sleep(0.005)
        with self._pending_writes_lock:
            return self._pending_writes <= 0

    def close(self) -> None:
        self.flush_ledger()
        if self._writer_started and self._writer.is_alive():
            self._queue.put(None)
            self._writer.join(timeout=2.0)
        self._chain.close()

    def _restore(self) -> None:
        if not self.events_path.exists():
            return
        try:
            raw = self.events_path.read_text(encoding="utf-8")
        except OSError as exc:
            record_degradation("verifier_foundry", exc, severity="critical",
                               action="foundry ledger unreadable; starting empty")
            return
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                self._fold(json.loads(line))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                self._restore_errors += 1
                record_degradation("verifier_foundry", exc, severity="warning",
                                   action="skipped corrupt foundry event line")

    _MAX_PENDING = 5000

    def _fold(self, event: dict[str, Any]) -> None:
        kind = str(event.get("event", ""))
        if kind == "verdict":
            vid = str(event.get("verdict_id", ""))
            cell = self._cell(str(event.get("verifier", "?")),
                              str(event.get("domain", "?")))
            cell.recorded += 1
            if vid:
                self._pending[vid] = {
                    "verifier": cell.verifier,
                    "domain": cell.domain,
                    "hard_pass": bool(event.get("hard_pass")),
                    "score": float(event.get("score", 0.5)),
                }
                self._pending_order.append(vid)
                while len(self._pending_order) > self._MAX_PENDING:
                    old = self._pending_order.pop(0)
                    self._pending.pop(old, None)
        elif kind == "grade":
            vid = str(event.get("verdict_id", ""))
            verdict = self._pending.pop(vid, None)
            if verdict is None:
                return
            truth = bool(event.get("truth_pass"))
            cell = self._cell(verdict["verifier"], verdict["domain"])
            cell.graded += 1
            said_pass = verdict["hard_pass"]
            if said_pass:
                cell.passes += 1
            if said_pass == truth:
                cell.correct += 1
            elif said_pass and not truth:
                cell.false_passes += 1
            else:
                cell.false_fails += 1
            cell.brier_sum += (verdict["score"] - (1.0 if truth else 0.0)) ** 2
        elif kind == "revoke_seed":
            self._revoked_seeds.add(str(event.get("domain", "")))

    def _cell(self, verifier: str, domain: str) -> ReliabilityCell:
        key = (verifier, domain)
        cell = self._cells.get(key)
        if cell is None:
            cell = ReliabilityCell(verifier=verifier, domain=domain)
            self._cells[key] = cell
        return cell

    # ── recording and grading ────────────────────────────────────────────
    def record_verdict(self, *, verifier: str, domain: str, hard_pass: bool,
                       score: float, checked: bool, task_key: str = "",
                       meta: dict[str, Any] | None = None) -> str:
        """Record one engine verdict; returns the verdict_id used to grade it
        later. Unchecked verdicts (engine had nothing to verify) are not
        reliability evidence and are not recorded."""
        if not checked:
            return ""
        verdict_id = f"vd-{uuid.uuid4().hex[:12]}"
        with self._lock:
            self._append_event("verdict", {
                "verdict_id": verdict_id,
                "verifier": str(verifier or "?"),
                "domain": str(domain or "?"),
                "hard_pass": bool(hard_pass),
                "score": max(0.0, min(1.0, float(score))),
                "task_key": str(task_key or "")[:64],
                "meta": dict(meta or {}),
            })
        return verdict_id

    def grade_verdict(self, verdict_id: str, *, truth_pass: bool,
                      source: str) -> bool:
        """Reality arrived: grade a recorded verdict against ground truth.
        ``source`` names the ground-truth channel (audit, prediction
        resolution, action outcome, human)."""
        with self._lock:
            if verdict_id not in self._pending:
                return False
            self._append_event("grade", {
                "verdict_id": verdict_id,
                "truth_pass": bool(truth_pass),
                "source": str(source or "unknown")[:64],
            })
        return True

    def pending_verdicts(self, *, domain: str | None = None) -> list[str]:
        with self._lock:
            if domain is None:
                return list(self._pending_order)
            return [v for v in self._pending_order
                    if self._pending.get(v, {}).get("domain") == domain]

    # ── reliability and folding weights ──────────────────────────────────
    def reliability(self, verifier: str, domain: str) -> ReliabilityCell:
        with self._lock:
            return self._cell(verifier, domain)

    _WEIGHT_FLOOR = 0.25

    def weight_for(self, verifier: str, domain: str) -> float:
        """Soft-score folding weight. Unmeasured verifiers stay neutral (1.0)
        so behavior is unchanged until evidence exists; measured ones are
        weighted by their pessimistic accuracy, floored so a bad verifier is
        muted, never inverted. The HARD gate is never weighted."""
        with self._lock:
            cell = self._cells.get((verifier, domain))
        if cell is None or cell.graded < 10:
            return 1.0
        return max(self._WEIGHT_FLOOR, cell.accuracy_lb())

    # ── the admission gate ───────────────────────────────────────────────
    def _domain_evidence(self, domain: str) -> tuple[int, float, float]:
        """Aggregate graded evidence across every verifier in a domain,
        scored by the WEAKEST relevant false-pass bound (a chain of checkers
        is only as trustworthy as the leakiest one actually used)."""
        with self._lock:
            cells = [c for (v, d), c in self._cells.items()
                     if d == domain and c.graded > 0]
        if not cells:
            return 0, 0.0, 1.0
        graded = sum(c.graded for c in cells)
        correct = sum(c.correct for c in cells)
        worst_fp = max(c.false_pass_ub() for c in cells)
        return graded, wilson_lower_bound(correct, graded), worst_fp

    def domain_admitted(self, domain: str) -> AdmissionDecision:
        """May verifier-clean wins in this domain become training data?"""
        domain = str(domain or "").strip().lower()
        graded, acc_lb, fp_ub = self._domain_evidence(domain)

        if domain in SEED_ADMITTED_DOMAINS and domain not in self._revoked_seeds:
            # seeds are admitted by construction — but measured evidence can
            # revoke them (the ratchet works on measurement, not faith)
            if graded >= int(_ADMIT_MIN_GRADED_FLAG.value()) and (
                    fp_ub > float(_ADMIT_MAX_FALSE_PASS_FLAG.value()) * 2):
                with self._lock:
                    self._append_event("revoke_seed", {
                        "domain": domain,
                        "false_pass_ub": round(fp_ub, 4),
                        "graded": graded,
                    })
                logger.warning("Foundry: seed admission REVOKED for %r "
                               "(false-pass UB %.3f over %d graded)",
                               domain, fp_ub, graded)
                return AdmissionDecision(domain, False,
                                         "seed_revoked_by_evidence",
                                         {"false_pass_ub": fp_ub, "graded": graded})
            return AdmissionDecision(domain, True, "seed_admitted",
                                     {"graded": graded})

        min_graded = int(_ADMIT_MIN_GRADED_FLAG.value())
        if graded < min_graded:
            return AdmissionDecision(
                domain, False, "insufficient_evidence",
                {"graded": graded, "required": min_graded})
        if acc_lb < float(_ADMIT_MIN_WILSON_FLAG.value()):
            return AdmissionDecision(
                domain, False, "accuracy_below_threshold",
                {"accuracy_lb": round(acc_lb, 4)})
        if fp_ub > float(_ADMIT_MAX_FALSE_PASS_FLAG.value()):
            return AdmissionDecision(
                domain, False, "false_pass_rate_too_high",
                {"false_pass_ub": round(fp_ub, 4)})
        return AdmissionDecision(domain, True, "earned_by_evidence",
                                 {"graded": graded,
                                  "accuracy_lb": round(acc_lb, 4),
                                  "false_pass_ub": round(fp_ub, 4)})

    # ── observability ────────────────────────────────────────────────────
    def status(self) -> dict[str, Any]:
        with self._lock:
            cells = [c.to_dict() for c in self._cells.values()]
            pending = len(self._pending)
        domains = sorted({c["domain"] for c in cells}
                         | set(SEED_ADMITTED_DOMAINS))
        return {
            "schema_version": SCHEMA_VERSION,
            "cells": cells,
            "pending_verdicts": pending,
            "restore_errors": self._restore_errors,
            "admissions": {d: self.domain_admitted(d).admitted for d in domains},
            "chain_head": self._chain.head_hash(),
            "chain_length": self._chain.length(),
            "root": str(self.root),
        }

    def is_alive(self) -> bool:
        if not self.events_path.parent.is_dir():
            return False
        if not self._writer_started or self._writer.is_alive():
            return True
        with self._pending_writes_lock:
            return self._pending_writes <= 0

    def verify_ledger(self) -> tuple[bool, list[dict[str, Any]]]:
        self.flush_ledger()
        bodies: dict[str, dict[str, Any]] = {}
        problems: list[dict[str, Any]] = []
        if self.events_path.exists():
            for idx, line in enumerate(
                    self.events_path.read_text(encoding="utf-8").splitlines()):
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    bodies[str(event.get("event_id", f"line-{idx}"))] = event
                except json.JSONDecodeError:
                    problems.append({"seq": -1, "kind": "event_body",
                                     "receipt_id": f"line-{idx}",
                                     "reason": "event body is not valid JSON"})
        ok, chain_problems = self._chain.verify(
            body_loader=lambda rid, kind: bodies.get(rid))
        problems.extend(chain_problems)
        return (ok and not problems, problems)


_foundry: VerifierFoundry | None = None
_foundry_lock = threading.Lock()


def get_verifier_foundry() -> VerifierFoundry:
    global _foundry
    if _foundry is None:
        with _foundry_lock:
            if _foundry is None:
                _foundry = VerifierFoundry()
    return _foundry


def boot_verifier_foundry() -> VerifierFoundry:
    foundry = get_verifier_foundry()
    try:
        from core.container import ServiceContainer

        ServiceContainer.register_instance("verifier_foundry", foundry,
                                           required=False)
    except (ImportError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation("verifier_foundry", exc, severity="warning",
                           action="foundry built but not registered")
    return foundry
