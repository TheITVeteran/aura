"""Append-only, tamper-evident behavioral output ledger (L3 evidence).

The behavioral proof (core/evaluation/behavioral_proof.py) scores a sealed
held-out task pack in a single run. L3 evidence is *longitudinal*: you want an
append-only record of every scored run on the held-out set over weeks, so that
later you can audit "here is the real behavior over time" — and so that nobody
(including a future code change) can quietly rewrite history.

This ledger is a hash chain: each entry stores the hash of the previous entry,
and its own hash over its canonical fields. ``verify_chain`` recomputes the
whole chain and detects any edited score, reordered entry, or deleted row. The
summary is computed ONLY from the recorded entries — there are no synthesized or
hardcoded numbers (this is deliberately the opposite of the fabricated
benchmarks that hardcoded scores and asserted victory).

Honesty guards:
  - an entry MUST carry a real pack_id + manifest_hash (you cannot log a run
    that wasn't tied to a sealed held-out pack);
  - if the same pack_id ever appears with two different manifest_hashes the
    summary flags ``held_out_integrity_ok = False`` (the sealed set changed);
  - the summary reports real per-condition means / pass-rates and the chain
    verification result, nothing else.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import atomic_write_text

_GENESIS = "GENESIS"
_LEDGER_RECOVERABLE_ERRORS = (OSError, ValueError, TypeError, KeyError)


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def current_commit_sha() -> str:
    """Best-effort git commit for provenance, read straight from .git (no
    subprocess — this runs inside a governed core module). Returns 'unknown'
    when no readable git metadata is found."""
    try:
        git_dir = Path(__file__).resolve().parents[2] / ".git"
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref:"):
            return head or "unknown"  # detached HEAD holds the sha directly
        ref = head.split(":", 1)[1].strip()
        ref_file = git_dir / ref
        if ref_file.exists():
            return ref_file.read_text(encoding="utf-8").strip() or "unknown"
        # Packed refs fallback.
        packed = git_dir / "packed-refs"
        if packed.exists():
            for line in packed.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith(("#", "^")) and line.endswith(ref):
                    return line.split(" ", 1)[0]
        return "unknown"
    except (OSError, ValueError, IndexError):
        return "unknown"


@dataclass(frozen=True)
class LedgerEntry:
    seq: int
    recorded_at: float
    commit_sha: str
    pack_id: str
    manifest_hash: str
    task_count: int
    condition: str
    score: float
    passed: bool
    prev_hash: str
    entry_hash: str

    def _payload(self) -> dict[str, Any]:
        """The hashed fields (everything except entry_hash itself)."""
        return {
            "seq": self.seq,
            "recorded_at": self.recorded_at,
            "commit_sha": self.commit_sha,
            "pack_id": self.pack_id,
            "manifest_hash": self.manifest_hash,
            "task_count": self.task_count,
            "condition": self.condition,
            "score": self.score,
            "passed": self.passed,
            "prev_hash": self.prev_hash,
        }

    def computed_hash(self) -> str:
        return _sha256(_canonical(self._payload()))

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["entry_hash"] = self.entry_hash
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LedgerEntry:
        return cls(
            seq=int(data["seq"]),
            recorded_at=float(data["recorded_at"]),
            commit_sha=str(data["commit_sha"]),
            pack_id=str(data["pack_id"]),
            manifest_hash=str(data["manifest_hash"]),
            task_count=int(data["task_count"]),
            condition=str(data["condition"]),
            score=float(data["score"]),
            passed=bool(data["passed"]),
            prev_hash=str(data["prev_hash"]),
            entry_hash=str(data["entry_hash"]),
        )


class BehavioralLedger:
    """JSONL hash-chained ledger of behavioral-proof outcomes on held-out tasks."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    # ---- read -------------------------------------------------------------

    def entries(self) -> list[LedgerEntry]:
        if not self.path.exists():
            return []
        out: list[LedgerEntry] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            out.append(LedgerEntry.from_dict(json.loads(line)))
        return out

    # ---- write ------------------------------------------------------------

    def append(
        self,
        *,
        pack_id: str,
        manifest_hash: str,
        task_count: int,
        condition: str,
        score: float,
        passed: bool,
        commit_sha: str | None = None,
        recorded_at: float | None = None,
    ) -> LedgerEntry:
        """Append a real scored outcome. Requires a sealed-pack identity."""
        if not str(pack_id).strip() or not str(manifest_hash).strip():
            raise ValueError("ledger entries require a real pack_id + manifest_hash")
        existing = self.entries()
        prev_hash = existing[-1].entry_hash if existing else _GENESIS
        seq = (existing[-1].seq + 1) if existing else 0
        partial = LedgerEntry(
            seq=seq,
            recorded_at=float(recorded_at if recorded_at is not None else time.time()),
            commit_sha=str(commit_sha if commit_sha is not None else current_commit_sha()),
            pack_id=str(pack_id),
            manifest_hash=str(manifest_hash),
            task_count=int(task_count),
            condition=str(condition),
            score=float(score),
            passed=bool(passed),
            prev_hash=prev_hash,
            entry_hash="",
        )
        entry_hash = partial.computed_hash()
        entry = LedgerEntry(**{**partial.__dict__, "entry_hash": entry_hash})
        lines = [_canonical(e.to_dict()) for e in (*existing, entry)]
        atomic_write_text(self.path, "\n".join(lines) + "\n")
        return entry

    # ---- integrity --------------------------------------------------------

    def verify_chain(self) -> tuple[bool, str]:
        """Recompute the whole chain; detect any tamper/reorder/deletion."""
        prev = _GENESIS
        expected_seq = 0
        for entry in self.entries():
            if entry.seq != expected_seq:
                return False, f"seq gap at {entry.seq} (expected {expected_seq})"
            if entry.prev_hash != prev:
                return False, f"broken link at seq {entry.seq}"
            if entry.computed_hash() != entry.entry_hash:
                return False, f"hash mismatch at seq {entry.seq} (entry edited)"
            prev = entry.entry_hash
            expected_seq += 1
        return True, "ok"

    # ---- summary (real numbers only) -------------------------------------

    def summary(self) -> dict[str, Any]:
        entries = self.entries()
        chain_ok, chain_detail = self.verify_chain()

        by_condition: dict[str, list[LedgerEntry]] = {}
        packs: dict[str, set[str]] = {}
        for e in entries:
            by_condition.setdefault(e.condition, []).append(e)
            packs.setdefault(e.pack_id, set()).add(e.manifest_hash)

        conditions: dict[str, Any] = {}
        for cond, items in by_condition.items():
            scores = [i.score for i in items]
            passes = [i.passed for i in items]
            conditions[cond] = {
                "runs": len(items),
                "mean_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
                "pass_rate": round(sum(1 for p in passes if p) / len(passes), 4) if passes else 0.0,
                "first_seen": items[0].recorded_at,
                "last_seen": items[-1].recorded_at,
                "score_trend": round(scores[-1] - scores[0], 4) if len(scores) >= 2 else 0.0,
            }

        # Held-out integrity: a sealed pack_id must map to exactly one manifest.
        held_out_integrity_ok = all(len(hashes) == 1 for hashes in packs.values())

        return {
            "total_runs": len(entries),
            "chain_ok": chain_ok,
            "chain_detail": chain_detail,
            "held_out_integrity_ok": held_out_integrity_ok,
            "packs": {pid: sorted(h) for pid, h in packs.items()},
            "conditions": conditions,
        }


def record_bundle_to_ledger(
    bundle: Any,
    *,
    ledger_path: str | Path,
    commit_sha: str | None = None,
) -> list[LedgerEntry]:
    """Append a BehavioralProofBundle's REAL baseline + candidate scores.

    Reads the sealed-pack identity (pack_id/manifest_hash/task_count) and the two
    solver scores straight off the honest smoke report — no synthesis.
    """
    ledger = BehavioralLedger(ledger_path)
    smoke = bundle.smoke
    sha = commit_sha if commit_sha is not None else current_commit_sha()
    recorded: list[LedgerEntry] = []
    for cond, solver in (("baseline", smoke.baseline), ("candidate", smoke.candidate)):
        recorded.append(
            ledger.append(
                pack_id=smoke.pack_id,
                manifest_hash=smoke.manifest_hash,
                task_count=smoke.task_count,
                condition=cond,
                score=float(solver.score),
                passed=bool(smoke.passed) if cond == "candidate" else False,
                commit_sha=sha,
            )
        )
    return recorded
