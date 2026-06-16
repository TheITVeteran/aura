"""Hippocampal index — engram cues and pattern completion.

In the indexing theory of the hippocampus (Teyler & DiScenna; Goode et al. 2020;
Tanaka & McHugh 2018; Kolibius et al. 2025) the hippocampus does not store the
full content of an episode. It stores a *sparse index* — a blueprint that binds
together the distributed cortical patterns active during the experience. Later,
re-presenting *part* of that pattern (a smell, a word, an angry crow) lets the
hippocampus complete the rest and reinstate the whole assembly. This is
**pattern completion** (Grande et al. 2019, hippocampal subfield CA3).

This module gives Aura's episodic memory that associative index. Each episode is
bound to a small set of salient *cues* drawn from its content, the tools it
involved, and the dominant phenomenal dimension it was encoded in. A later,
partial cue set retrieves episodes whose engram overlaps it — an associative
recall path that complements vector similarity and keyword search.

The index lives in the same SQLite database as the episodes (one extra table)
so it shares their durability and pruning lifecycle.
"""
from __future__ import annotations

import logging
import re
import sqlite3
from collections.abc import Callable, Iterable
from typing import Any

logger = logging.getLogger("Memory.Hippocampus")

# Small, fast stoplist — enough to keep cues content-bearing without pulling in a
# tokenizer dependency. Pattern completion only needs salient hooks, not grammar.
_STOPWORDS = frozenset(
    """
    the a an and or but if then else for to of in on at by with from into over
    under again once is are was were be been being do does did doing have has had
    this that these those there here it its it's i you he she we they them his her
    our your their what which who whom when where why how all any both each few more
    most other some such no nor not only own same so than too very can will just
    about as up out off down get got use used user aura did done via per
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'\-]+")


class HippocampalIndex:
    """Associative cue index over episodes, supporting pattern completion."""

    MAX_CUES_PER_EPISODE = 24

    def __init__(self, conn_factory: Callable[[], sqlite3.Connection]):
        self._conn_factory = conn_factory
        self._ensure_schema()

    # ---- schema -------------------------------------------------------------

    def _ensure_schema(self) -> None:
        try:
            with self._conn_factory() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS engram_cues (
                        episode_id TEXT NOT NULL,
                        cue        TEXT NOT NULL,
                        weight     REAL DEFAULT 1.0,
                        PRIMARY KEY (episode_id, cue)
                    )
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_engram_cue ON engram_cues (cue)")
                conn.commit()
        except sqlite3.Error as exc:  # pragma: no cover - schema is best-effort
            logger.debug("Hippocampal index schema init skipped: %s", exc)

    # ---- cue extraction -----------------------------------------------------

    @classmethod
    def extract_cues(
        cls,
        context: str = "",
        action: str = "",
        outcome: str = "",
        *,
        tools: Iterable[str] | None = None,
        qualia_snapshot: dict[str, Any] | None = None,
        tags: Iterable[str] | None = None,
    ) -> list[str]:
        """Derive a compact set of salient cues for an episode.

        Content words become plain cues; tools and the dominant phenomenal
        dimension become *typed* cues (``tool:…``, ``dim:…``) so an associative
        query can hook on "what I was doing" or "how it felt", not just words.
        """
        cues: list[str] = []
        seen: set[str] = set()

        def _add(cue: str) -> None:
            cue = cue.strip().lower()
            if cue and cue not in seen:
                seen.add(cue)
                cues.append(cue)

        text = " ".join(p for p in (context, action, outcome) if p)
        for tok in _TOKEN_RE.findall(text.lower()):
            if len(tok) < 4 or tok in _STOPWORDS:
                continue
            _add(tok)
            if len(cues) >= cls.MAX_CUES_PER_EPISODE:
                break

        for tool in tools or []:
            if tool:
                _add(f"tool:{str(tool).lower()}")
        for tag in tags or []:
            if tag:
                _add(f"tag:{str(tag).lower()}")
        if qualia_snapshot:
            dim = qualia_snapshot.get("dominant_dim")
            if dim:
                _add(f"dim:{str(dim).lower()}")

        return cues[: cls.MAX_CUES_PER_EPISODE]

    # ---- binding ------------------------------------------------------------

    def bind(self, episode_id: str, cues: Iterable[str]) -> int:
        """Bind an episode's engram to its cues. Returns number of cues stored."""
        rows = [(episode_id, c) for c in dict.fromkeys(cues) if c]
        if not rows:
            return 0
        try:
            with self._conn_factory() as conn:
                conn.executemany(
                    "INSERT OR IGNORE INTO engram_cues (episode_id, cue) VALUES (?, ?)",
                    rows,
                )
                conn.commit()
            return len(rows)
        except sqlite3.Error as exc:
            logger.debug("Hippocampal bind failed for %s: %s", episode_id, exc)
            return 0

    def unbind(self, episode_ids: Iterable[str]) -> None:
        """Drop cues for pruned/forgotten episodes so the index stays in sync."""
        ids = [e for e in episode_ids if e]
        if not ids:
            return
        try:
            with self._conn_factory() as conn:
                placeholders = ",".join("?" for _ in ids)
                conn.execute(
                    f"DELETE FROM engram_cues WHERE episode_id IN ({placeholders})", ids
                )
                conn.commit()
        except sqlite3.Error as exc:
            logger.debug("Hippocampal unbind failed: %s", exc)

    # ---- retrieval ----------------------------------------------------------

    def pattern_complete(
        self,
        cues: Iterable[str],
        limit: int = 5,
        *,
        min_overlap: int = 1,
        exclude_ids: Iterable[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Complete an assembly from a partial cue set.

        Returns ``(episode_id, score)`` pairs where score is the fraction of the
        query cues that the episode's engram shares — ranked highest first.
        """
        cue_list = [c for c in dict.fromkeys(cues) if c]
        if not cue_list:
            return []
        exclude = set(exclude_ids or ())
        try:
            with self._conn_factory() as conn:
                placeholders = ",".join("?" for _ in cue_list)
                rows = conn.execute(
                    f"""
                    SELECT episode_id, COUNT(*) AS overlap
                    FROM engram_cues
                    WHERE cue IN ({placeholders})
                    GROUP BY episode_id
                    HAVING overlap >= ?
                    ORDER BY overlap DESC
                    LIMIT ?
                    """,
                    [*cue_list, max(1, min_overlap), limit * 4],
                ).fetchall()
        except sqlite3.Error as exc:
            logger.debug("Pattern completion failed: %s", exc)
            return []

        denom = float(len(cue_list))
        scored: list[tuple[str, float]] = []
        for row in rows:
            eid = row[0]
            if eid in exclude:
                continue
            scored.append((eid, row[1] / denom))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

    def stats(self) -> dict[str, int]:
        try:
            with self._conn_factory() as conn:
                cues = conn.execute("SELECT COUNT(*) FROM engram_cues").fetchone()[0]
                engrams = conn.execute(
                    "SELECT COUNT(DISTINCT episode_id) FROM engram_cues"
                ).fetchone()[0]
            return {"total_cues": int(cues), "indexed_engrams": int(engrams)}
        except sqlite3.Error:
            return {"total_cues": 0, "indexed_engrams": 0}
