"""core/runtime/post_action_receipt.py — Post-Action Receipts.

Enforces proof that every governed action creates a corresponding post-action receipt.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.PostActionReceipt")
_RECEIPT_LOAD_ERRORS = (
    json.JSONDecodeError,
    TypeError,
    ValueError,
)


@dataclass
class PostActionReceipt:
    """Receipt proving the outcome and consequences of an executed action."""
    receipt_id: str
    will_receipt_id: str
    executor_name: str
    actual_outcome: str  # success / failure / partial / timeout
    output_hash: str
    error_status: str
    welfare_transaction_id: str
    body_delta: Dict[str, float] = field(default_factory=dict)
    memory_delta: Dict[str, Any] = field(default_factory=dict)
    rollback_target: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PostActionReceiptStore:
    """Thread-safe persistent store for post-action receipts."""

    _instance: Optional[PostActionReceiptStore] = None

    def __init__(self, persist_path: Optional[Path] = None) -> None:
        from core.config import config
        self.persist_path = Path(
            persist_path or config.paths.data_dir / "receipts" / "post_action_receipts.jsonl"
        )
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._receipts: Dict[str, PostActionReceipt] = {}
        self._lock = threading.RLock()
        self._load()

    @classmethod
    def get(cls) -> PostActionReceiptStore:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load(self) -> None:
        if not self.persist_path.exists():
            return
        try:
            with self.persist_path.open("r", encoding="utf-8") as fh:
                for line_no, line in enumerate(fh, start=1):
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        receipt = PostActionReceipt(**data)
                        self._receipts[receipt.receipt_id] = receipt
                    except _RECEIPT_LOAD_ERRORS as exc:
                        record_degradation(
                            "post_action_receipt",
                            exc,
                            action=f"skipped invalid receipt ledger line {line_no}",
                        )
                        logger.warning(
                            "Skipped invalid post-action receipt line %s: %s",
                            line_no,
                            exc,
                        )
        except OSError as exc:
            record_degradation("post_action_receipt", exc)
            logger.warning("Failed to read post-action receipts: %s", exc)

    def record(self, receipt: PostActionReceipt) -> None:
        """Record and persist a new post-action receipt."""
        if not receipt.receipt_id:
            raise ValueError("post-action receipt_id is required")
        line = json.dumps(receipt.to_dict(), sort_keys=True) + "\n"
        with self._lock:
            if receipt.receipt_id in self._receipts:
                raise ValueError(f"duplicate post-action receipt_id: {receipt.receipt_id}")
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with self.persist_path.open("a", encoding="utf-8") as fh:
                    fh.write(line)
                    fh.flush()
                    os.fsync(fh.fileno())
            except OSError as exc:
                record_degradation("post_action_receipt", exc)
                logger.warning("Failed to write post-action receipt: %s", exc)
                raise
            self._receipts[receipt.receipt_id] = receipt

    def get_receipt(self, receipt_id: str) -> Optional[PostActionReceipt]:
        with self._lock:
            return self._receipts.get(receipt_id)

    def get_by_will_id(self, will_receipt_id: str) -> List[PostActionReceipt]:
        with self._lock:
            return [r for r in self._receipts.values() if r.will_receipt_id == will_receipt_id]

    def list_receipts(self) -> List[PostActionReceipt]:
        with self._lock:
            return list(self._receipts.values())

    def clear(self) -> None:
        with self._lock:
            self._receipts.clear()
            self.persist_path.unlink(missing_ok=True)


def get_post_action_receipt_store() -> PostActionReceiptStore:
    return PostActionReceiptStore.get()


__all__ = [
    "PostActionReceipt",
    "PostActionReceiptStore",
    "get_post_action_receipt_store",
]
