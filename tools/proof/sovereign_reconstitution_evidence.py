"""Tamper-evident evidence custody for the sovereign reconstitution proof."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _now() -> float:
    return time.time()


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True, default=_json_default) + "\n")

@dataclass
class ChainReceipt:
    receipt_id: str
    kind: str
    sequence_id: int
    previous_hash: str
    hash: str


class ExternalReceiptChain:
    """Small tamper-evident receipt chain owned by the external harness."""

    def __init__(self, out_dir: Path, *, run_id: str) -> None:
        self.out_dir = out_dir
        self.run_id = run_id
        self.sequence_id = 0
        self.previous_hash = "GENESIS"
        self.records: list[dict[str, Any]] = []

    @staticmethod
    def receipt_hash(record: dict[str, Any]) -> str:
        body = dict(record)
        body.pop("hash", None)
        return _stable_hash(body)

    def emit(
        self,
        *,
        kind: str,
        task_id: str,
        action: str,
        payload: dict[str, Any],
        approved: bool = True,
        reason: str = "external_evaluator_authorized",
    ) -> ChainReceipt:
        self.sequence_id += 1
        payload_hash = _stable_hash(payload)
        base = {
            "schema": "aura.sovereignty.receipt.v1",
            "run_id": self.run_id,
            "sequence_id": self.sequence_id,
            "kind": kind,
            "task_id": task_id,
            "action": action,
            "payload_hash": payload_hash,
            "approved": bool(approved),
            "reason": reason,
            "previous_hash": self.previous_hash,
            "time_unix": _now(),
        }
        receipt_id = f"sov_{kind}_{_stable_hash(base)[:24]}"
        record = {**base, "receipt_id": receipt_id}
        record["hash"] = self.receipt_hash(record)
        self.previous_hash = str(record["hash"])
        self.records.append(record)

        _append_jsonl(self.out_dir / "receipt_chain.jsonl", record)
        by_kind = {
            "will": "will_receipts.jsonl",
            "tool": "tool_receipts.jsonl",
            "autonomy": "autonomy_receipts.jsonl",
            "memory": "memory_receipts.jsonl",
            "self_repair": "self_repair_receipts.jsonl",
        }
        name = by_kind.get(kind)
        if name:
            _append_jsonl(self.out_dir / name, record)
        return ChainReceipt(
            receipt_id=receipt_id,
            kind=kind,
            sequence_id=int(record["sequence_id"]),
            previous_hash=str(record["previous_hash"]),
            hash=str(record["hash"]),
        )

    @classmethod
    def verify(cls, records: list[dict[str, Any]]) -> tuple[bool, list[str]]:
        errors: list[str] = []
        previous = "GENESIS"
        for index, record in enumerate(records, start=1):
            if int(record.get("sequence_id") or -1) != index:
                errors.append(f"sequence_mismatch:{index}")
            if record.get("previous_hash") != previous:
                errors.append(f"previous_hash_mismatch:{index}")
            expected = cls.receipt_hash(record)
            if record.get("hash") != expected:
                errors.append(f"hash_mismatch:{index}")
            previous = str(record.get("hash") or "")
        return not errors, errors

__all__ = [
    "ChainReceipt",
    "ExternalReceiptChain",
]
