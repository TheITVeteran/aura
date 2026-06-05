"""core/lab/research_memory.py — Research Lab Memory Store.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from typing import Dict, List

logger = logging.getLogger("Aura.ResearchMemory")


@dataclass
class ResearchMemo:
    memo_id: str
    topic: str
    hypothesis_statement: str
    validated: bool
    data_points: Dict[str, Any]
    summary_prose: str


class ResearchMemoryStore:
    """Manages files tracking completed research memos and datasets."""

    def __init__(self) -> None:
        self.memos: Dict[str, ResearchMemo] = {}

    def save_memo(self, memo: ResearchMemo) -> None:
        self.memos[memo.memo_id] = memo
        logger.info("🔬 Saved research memo [%s]: %s (validated: %s)", 
                    memo.memo_id, memo.topic, memo.validated)

    def list_memos(self) -> List[ResearchMemo]:
        return list(self.memos.values())
