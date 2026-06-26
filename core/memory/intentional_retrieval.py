"""Intentional retrieval — task-driven routing across a typed memory taxonomy.

The critique's largest remaining item: a 10M-item memory "cannot be one pile of embeddings."
Aura already *has* the stores (episodic, semantic, procedural, social, project, causal graph,
receipts/outcomes, autobiography, …); what was missing is the doc's actual ask — *intentional
retrieval*: "similarity search is not enough," retrieval should "depend on task, not just
similarity," answering —

    What am I trying to do?          → the task kind selects the primary stores
    What kind of memory matters?     → a per-kind store-weighting table
    What risks matter?               → risk-sensitive work boosts failure/value/receipt/causal
    Whose values matter?             → value/social retrieval keyed to the relevant agent
    What time horizon matters?       → "now" favors episodic/world-state; "long" favors
                                       semantic/project/autobiography
    What previous failures resemble this? → failure store, boosted for debug/irreversible work
    What tools have worked before?   → tool + procedural stores

This module is the router, not new stores. Heterogeneous existing stores plug in as adapters
(``register_store``); ``plan()`` turns an intent into a weighted set of stores with per-store
fetch allocations and a human-readable rationale; ``retrieve()`` runs that plan, fault-isolated
per store, and merges results by weighted score. It is deliberately reorganization over the
existing memory infrastructure — stated plainly — turning many similarity blobs into one
intentional surface.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional

from core.runtime.errors import record_degradation

logger = logging.getLogger("Memory.IntentionalRetrieval")


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


class MemoryStoreType(str, Enum):
    """The memory taxonomy from the critique — one role per store, distinct retrieval logic."""

    EPISODIC = "episodic"            # raw events, what happened
    SEMANTIC = "semantic"           # distilled facts
    PROCEDURAL = "procedural"       # how-to / skills
    SOCIAL = "social"               # people, relationships
    PROJECT = "project"             # ongoing work and goals
    VALUE = "value"                 # preferences, what matters, what to refuse
    FAILURE = "failure"             # past failures and scars
    TOOL = "tool"                   # tool schemas and what worked
    CAUSAL = "causal"               # cause→effect structure
    WORLD_STATE = "world_state"     # current beliefs about the world
    RECEIPT = "receipt"             # action receipts / outcomes
    AUTOBIOGRAPHY = "autobiography"  # compressed life narrative


# Adapter: query text + limit → an iterable of raw results (str / dict / object). The router
# normalizes whatever a store returns, so existing stores plug in with a thin lambda.
StoreAdapter = Callable[[str, int], Iterable[Any]]


@dataclass
class MemoryHit:
    content: str
    score: float
    store_type: str
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "score": round(self.score, 4),
            "store_type": self.store_type,
            "source": self.source,
            "metadata": self.metadata,
        }


@dataclass
class RetrievalIntent:
    """Why we're retrieving — the signal that makes retrieval intentional rather than blind."""

    task: str
    kind: str = "general"            # debug|plan|converse|decide|act_irreversible|recall_fact|learn|social|general
    query: str = ""                  # actual search text; defaults to the task description
    risk_sensitive: bool = False     # about to do something risky / irreversible
    time_horizon: str = "session"    # now | session | long
    whose_values: str = "bryan"      # whose value/social memory is relevant
    need_failures: bool = False      # explicitly want prior failures
    need_tools: bool = False         # explicitly want tools/procedures
    limit: int = 8

    def effective_query(self) -> str:
        return self.query or self.task


@dataclass
class RetrievalPlan:
    intent: RetrievalIntent
    weights: Dict[str, float]        # store_type value → weight, only those above threshold
    allocations: Dict[str, int]      # store_type value → per-store fetch budget
    rationale: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.intent.task,
            "kind": self.intent.kind,
            "weights": {k: round(v, 3) for k, v in self.weights.items()},
            "allocations": self.allocations,
            "rationale": self.rationale,
        }


@dataclass
class RetrievalResult:
    hits: List[MemoryHit]
    plan: RetrievalPlan
    stores_queried: List[str]
    stores_missing: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hits": [h.to_dict() for h in self.hits],
            "plan": self.plan.to_dict(),
            "stores_queried": self.stores_queried,
            "stores_missing": self.stores_missing,
        }


_T = MemoryStoreType

# Per-task-kind base store weights — "what kind of memory matters" for this kind of task.
_KIND_WEIGHTS: Dict[str, Dict[MemoryStoreType, float]] = {
    "debug": {_T.PROCEDURAL: 0.9, _T.FAILURE: 0.9, _T.TOOL: 0.8, _T.CAUSAL: 0.7,
              _T.EPISODIC: 0.6, _T.SEMANTIC: 0.4},
    "plan": {_T.PROJECT: 0.9, _T.SEMANTIC: 0.7, _T.CAUSAL: 0.7, _T.VALUE: 0.6,
             _T.EPISODIC: 0.6, _T.FAILURE: 0.5, _T.TOOL: 0.5},
    "converse": {_T.SOCIAL: 0.9, _T.EPISODIC: 0.7, _T.VALUE: 0.6, _T.SEMANTIC: 0.5,
                 _T.AUTOBIOGRAPHY: 0.4},
    "decide": {_T.VALUE: 0.9, _T.FAILURE: 0.8, _T.CAUSAL: 0.7, _T.RECEIPT: 0.7,
               _T.PROJECT: 0.6, _T.SEMANTIC: 0.5},
    "act_irreversible": {_T.VALUE: 1.0, _T.FAILURE: 0.9, _T.RECEIPT: 0.8, _T.CAUSAL: 0.7,
                         _T.WORLD_STATE: 0.6, _T.PROJECT: 0.5},
    "recall_fact": {_T.SEMANTIC: 1.0, _T.EPISODIC: 0.7, _T.AUTOBIOGRAPHY: 0.5,
                    _T.WORLD_STATE: 0.4},
    "learn": {_T.SEMANTIC: 0.8, _T.CAUSAL: 0.8, _T.FAILURE: 0.7, _T.PROCEDURAL: 0.6,
              _T.EPISODIC: 0.6},
    "social": {_T.SOCIAL: 1.0, _T.VALUE: 0.7, _T.EPISODIC: 0.6, _T.AUTOBIOGRAPHY: 0.3},
    "general": {_T.EPISODIC: 0.7, _T.SEMANTIC: 0.7, _T.SOCIAL: 0.5, _T.PROJECT: 0.5},
}

_SELECT_THRESHOLD = 0.25


class IntentionalRetriever:
    """Routes a retrieval intent across a registry of typed memory-store adapters."""

    def __init__(self, *, select_threshold: float = _SELECT_THRESHOLD) -> None:
        self._adapters: Dict[MemoryStoreType, StoreAdapter] = {}
        self._threshold = select_threshold

    # ── store registry ────────────────────────────────────────────────────

    def register_store(self, store_type: MemoryStoreType, adapter: StoreAdapter) -> None:
        """Plug an existing store in as ``(query, limit) -> iterable`` of results."""
        self._adapters[MemoryStoreType(store_type)] = adapter

    def registered_types(self) -> List[str]:
        return [t.value for t in self._adapters]

    # ── planning: intent → weighted store selection ───────────────────────

    def plan(self, intent: RetrievalIntent) -> RetrievalPlan:
        """Turn an intent into weighted stores + fetch allocations, with a rationale."""
        base = _KIND_WEIGHTS.get(intent.kind, _KIND_WEIGHTS["general"])
        weights: Dict[MemoryStoreType, float] = dict(base)
        rationale: List[str] = [f"task kind '{intent.kind}' → {self._fmt(base)}"]

        def boost(t: MemoryStoreType, amount: float) -> None:
            weights[t] = _clamp(weights.get(t, 0.0) + amount)

        if intent.risk_sensitive:
            for t in (_T.FAILURE, _T.VALUE, _T.RECEIPT, _T.CAUSAL):
                boost(t, 0.3)
            rationale.append("risk-sensitive → boost failure/value/receipt/causal")

        if intent.time_horizon == "now":
            boost(_T.EPISODIC, 0.2)
            boost(_T.WORLD_STATE, 0.3)
            rationale.append("horizon=now → boost episodic/world-state")
        elif intent.time_horizon == "long":
            for t in (_T.SEMANTIC, _T.PROJECT, _T.AUTOBIOGRAPHY):
                boost(t, 0.3)
            rationale.append("horizon=long → boost semantic/project/autobiography")

        if intent.need_failures or intent.kind in ("debug", "act_irreversible"):
            boost(_T.FAILURE, 0.25)
            rationale.append("failure-relevant → ensure failure store is consulted")
        if intent.need_tools or intent.kind in ("debug",):
            boost(_T.TOOL, 0.25)
            boost(_T.PROCEDURAL, 0.2)
            rationale.append("tool-relevant → ensure tool/procedural stores are consulted")
        if intent.whose_values:
            boost(_T.VALUE, 0.15)
            boost(_T.SOCIAL, 0.1)
            rationale.append(f"values of '{intent.whose_values}' matter → boost value/social")

        selected = {t.value: w for t, w in weights.items() if w >= self._threshold}
        selected = dict(sorted(selected.items(), key=lambda kv: kv[1], reverse=True))
        allocations = self._allocate(selected, intent.limit)
        return RetrievalPlan(intent=intent, weights=selected, allocations=allocations,
                             rationale=rationale)

    def _allocate(self, weights: Dict[str, float], limit: int) -> Dict[str, int]:
        # Over-fetch from each store in proportion to its weight, then the merge step trims to
        # the final limit — so a high-weight store can dominate the result if it's rich.
        total = sum(weights.values()) or 1.0
        out: Dict[str, int] = {}
        for store, w in weights.items():
            out[store] = max(2, min(limit, math.ceil(limit * (w / total) * 2)))
        return out

    # ── retrieval: run the plan, merge by weighted score ──────────────────

    def retrieve(self, intent: RetrievalIntent) -> RetrievalResult:
        """Execute a plan across registered stores; merge + rank; fault-isolated per store."""
        plan = self.plan(intent)
        query = intent.effective_query()
        hits: List[MemoryHit] = []
        queried: List[str] = []
        missing: List[str] = []

        for store, weight in plan.weights.items():
            adapter = self._adapters.get(MemoryStoreType(store))
            if adapter is None:
                missing.append(store)
                continue
            budget = plan.allocations.get(store, 2)
            try:
                raw = adapter(query, budget)
                store_hits = self._normalize(raw, store, weight)
                hits.extend(store_hits)
                queried.append(store)
            except Exception as exc:  # noqa: BLE001 - one store failing never sinks retrieval
                record_degradation("intentional_retrieval", exc, severity="debug",
                                   action=f"store '{store}' query failed")
                missing.append(store)

        merged = self._merge(hits, intent.limit)
        return RetrievalResult(hits=merged, plan=plan, stores_queried=queried,
                               stores_missing=missing)

    @staticmethod
    def _normalize(raw: Iterable[Any], store: str, weight: float) -> List[MemoryHit]:
        """Coerce a store's heterogeneous output into weighted MemoryHits.

        Scores: an explicit score/similarity/relevance/confidence if present, otherwise a
        rank-decayed base (stores that return a ranked list but no scores). Every score is
        multiplied by the store's plan weight so cross-store merging respects intent.
        """
        out: List[MemoryHit] = []
        items = list(raw or [])
        for rank, item in enumerate(items):
            base = 1.0 / (1.0 + rank)
            content, source, meta = "", store, {}
            if isinstance(item, str):
                content = item
            elif isinstance(item, dict):
                for key in ("content", "text", "summary", "description", "fact", "memory", "value"):
                    if item.get(key):
                        content = str(item[key])
                        break
                else:
                    content = str(item)
                for key in ("score", "similarity", "relevance", "confidence"):
                    if isinstance(item.get(key), (int, float)):
                        base = float(item[key])
                        break
                source = str(item.get("source") or store)
                meta = {k: v for k, v in item.items() if k not in ("content", "text")}
            else:
                content = getattr(item, "content", None) or getattr(item, "description", None) or str(item)
                base = float(getattr(item, "score", base) or base)
            if content:
                out.append(MemoryHit(content=content, score=_clamp(base) * weight,
                                     store_type=store, source=source, metadata=meta))
        return out

    @staticmethod
    def _merge(hits: List[MemoryHit], limit: int) -> List[MemoryHit]:
        # Highest weighted score wins; dedupe identical content, keeping the strongest.
        best: Dict[str, MemoryHit] = {}
        for h in hits:
            key = h.content.strip().lower()[:200]
            if key not in best or h.score > best[key].score:
                best[key] = h
        ranked = sorted(best.values(), key=lambda h: h.score, reverse=True)
        return ranked[:limit]

    @staticmethod
    def _fmt(weights: Dict[MemoryStoreType, float]) -> str:
        return ", ".join(f"{t.value}:{w:.1f}" for t, w in
                         sorted(weights.items(), key=lambda kv: kv[1], reverse=True))

    # ── opt-in default wiring over existing stores ────────────────────────

    def wire_default_stores(self) -> List[str]:
        """Best-effort registration of adapters for stores that exist and query synchronously.

        Opt-in (not called at import) so registration stays cheap and surprise-free. Returns the
        store-type values successfully wired. Other store types register via ``register_store``.
        """
        wired: List[str] = []

        # SEMANTIC — distilled facts via the memory synthesizer.
        try:
            from core.memory.memory_synthesizer import get_memory_synthesizer
            syn = get_memory_synthesizer()
            self.register_store(_T.SEMANTIC, lambda q, n: syn.get_relevant(q, limit=n))
            wired.append(_T.SEMANTIC.value)
        except Exception as exc:  # noqa: BLE001
            record_degradation("intentional_retrieval", exc, severity="debug",
                               action="semantic store not wired")

        # VALUE — learned preferences / regrets from the bounded value model.
        try:
            from core.values.value_model import get_value_model
            vm = get_value_model()
            self.register_store(_T.VALUE, lambda q, n: vm.retrieve(q, n))
            wired.append(_T.VALUE.value)
        except Exception as exc:  # noqa: BLE001
            record_degradation("intentional_retrieval", exc, severity="debug",
                               action="value store not wired")

        # SOCIAL — relationship milestones matched on token overlap with the query.
        try:
            from core.memory.social_memory import SocialMemory
            sm = SocialMemory()

            def _social(q: str, n: int) -> List[Dict[str, Any]]:
                toks = {t for t in q.lower().split() if len(t) > 2}
                scored = []
                for m in sm.milestones:
                    desc = m.description.lower()
                    overlap = sum(1 for t in toks if t in desc)
                    scored.append({"content": m.description, "score": 0.4 + 0.1 * overlap,
                                   "source": "social_memory"})
                scored.sort(key=lambda d: d["score"], reverse=True)
                return scored[:n]

            self.register_store(_T.SOCIAL, _social)
            wired.append(_T.SOCIAL.value)
        except Exception as exc:  # noqa: BLE001
            record_degradation("intentional_retrieval", exc, severity="debug",
                               action="social store not wired")

        return wired


_instance: Optional[IntentionalRetriever] = None


def get_intentional_retriever() -> IntentionalRetriever:
    global _instance
    if _instance is None:
        _instance = IntentionalRetriever()
    return _instance
