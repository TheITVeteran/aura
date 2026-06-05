"""core/epistemics/source_ranker.py — Source Reliability Ranker."""
from __future__ import annotations

import logging
from typing import Dict

logger = logging.getLogger("Aura.SourceRanker")

# Domain credibility defaults
DEFAULT_RELIABILITY: Dict[str, float] = {
    "arxiv.org": 0.90,
    "pubmed.ncbi.nlm.nih.gov": 0.95,
    "nature.com": 0.98,
    "github.com": 0.85,
    "wikipedia.org": 0.80,
    "news.ycombinator.com": 0.70,
    "reddit.com": 0.35,
    "unknown": 0.50,
}


class SourceRanker:
    """Tracks and calibrates reliability scores for academic, web, and user sources."""

    def __init__(self) -> None:
        self.dynamic_scores: Dict[str, float] = dict(DEFAULT_RELIABILITY)
        self.evidence_count: Dict[str, int] = {}

    def get_reliability(self, source: str) -> float:
        """Looks up reliability, normalizing source name first."""
        # Find matching key
        for key, val in self.dynamic_scores.items():
            if key in source or source in key:
                return val
        return self.dynamic_scores.get("unknown", 0.50)

    def record_outcome(self, source: str, verified_true: bool) -> None:
        """Dynamically adjusts the source's score based on verification outcomes."""
        current = self.get_reliability(source)
        count = self.evidence_count.get(source, 0) + 1
        self.evidence_count[source] = count

        # Learning rate decays as we get more evidence
        lr = max(0.01, 0.1 / (count ** 0.5))
        delta = lr * (1.0 if verified_true else 0.0 - current)
        
        # Calculate new score clamped between 0.05 and 0.99
        new_score = max(0.05, min(0.99, current + delta))
        self.dynamic_scores[source] = new_score
        logger.info("Recalibrated source %s reliability: %.2f -> %.2f (count=%d)", source, current, new_score, count)


_ranker_instance: SourceRanker | None = None


def get_source_ranker() -> SourceRanker:
    global _ranker_instance
    if _ranker_instance is None:
        _ranker_instance = SourceRanker()
    return _ranker_instance
