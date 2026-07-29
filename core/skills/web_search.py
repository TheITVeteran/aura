"""Enhanced web search and research skill for Aura."""


import logging
from typing import Any

from pydantic import BaseModel, Field

from core.container import ServiceContainer
from core.runtime.errors import record_degradation
from core.search import ResearchSearchPipeline
from core.search.research_pipeline import freshness_window_for_query, query_requires_source_reading
from core.skills.base_skill import BaseSkill
from core.skills.deep_research import run_deep_research

logger = logging.getLogger("Skills.WebSearch")


class _DeepResearchBrainAdapter:
    """Compat adapter for deep_research's ``brain.generate() -> {'response': ...}`` contract."""

    def __init__(self, engine: Any):
        self.engine = engine

    async def generate(self, prompt: str, **kwargs) -> dict[str, str]:
        # The caller's priority is HONOURED, not discarded.
        #
        # This accepted **kwargs and threw them away, hardcoding
        # is_background=True. So a synthesis the person is waiting for was
        # admitted as background work, queued behind foreground headroom, and
        # came back instantly empty — and the deep-research retry that asks for
        # foreground could not take effect because its request never left this
        # method. Measured live:
        #   "Deep research gathered 5 source(s) over 1 quer(ies) in 9.4s but
        #    could not synthesize them (the model returned no text)"
        # ...twice, including the retry.
        #
        # Background stays the default: ordinary research really is background.
        foreground = bool(kwargs.get("foreground_request", False))
        raw = await self.engine.generate(
            prompt,
            origin=str(kwargs.get("origin") or ("user" if foreground else "system")),
            purpose="research",
            use_strategies=False,
            is_background=not foreground,
            foreground_request=foreground,
            priority=float(kwargs.get("priority", 1.0 if foreground else 0.5)),
        )
        if isinstance(raw, dict):
            text = raw.get("response") or raw.get("content") or raw.get("result") or ""
        else:
            text = str(raw or "")
        return {"response": str(text or "")}


class WebSearchInput(BaseModel):
    query: str = Field(..., description="The search query to look up on the web.")
    deep: bool = Field(False, description="If True, fetch and synthesize multiple result pages.")
    num_results: int = Field(5, ge=1, le=20, description="Number of search hits to return.")
    retain: bool | None = Field(
        None,
        description="Whether Aura should retain what she learned from this search.",
    )
    force_refresh: bool = Field(False, description="If True, bypass cache and force a new live search.")


class EnhancedWebSearchSkill(BaseSkill):
    """Hybrid live web search with retrieval, synthesis, and retention."""

    name = "web_search"
    description = (
        "Search the internet for current information, research a topic across multiple pages, "
        "synthesize an evidence-grounded answer, and retain what was learned when appropriate."
    )
    input_model = WebSearchInput
    timeout_seconds = 60.0
    metabolic_cost = 2

    def __init__(self):
        super().__init__()
        self.pipeline = ResearchSearchPipeline()
        self.browser = _DormantBrowser()

    def _normalize_deep_research_result(self, query: str, result: dict[str, Any]) -> dict[str, Any]:
        sources = list(result.get("sources") or [])
        citations = []
        evidence = []
        for item in sources[:8]:
            url = str(item.get("url") or item.get("uri") or "").strip()
            title = str(item.get("title") or item.get("name") or url or "").strip()
            if not url:
                continue
            citations.append({"title": title, "url": url})
            evidence.append(
                {
                    "title": title,
                    "url": url,
                    "text": str(item.get("text") or item.get("snippet") or "").strip(),
                    "score": float(item.get("score", 0.0) or 0.0),
                }
            )

        answer = str(result.get("answer") or "").strip()
        summary = answer or str(result.get("summary") or "").strip()
        normalized = {
            "ok": True,
            "query": query,
            "answer": answer,
            "summary": summary,
            "facts": list(result.get("facts") or []),
            "confidence": float(result.get("confidence", 0.82) or 0.82),
            "sources": citations,
            "citations": citations,
            "source": citations[0]["url"] if citations else "",
            "mode": "deep",
            "count": len(citations),
            "chunks": evidence,
            "content": answer,
        }
        normalized["result"] = normalized["answer"] or normalized["content"] or ""
        normalized["message"] = self.pipeline._format_message(query, normalized)
        return normalized

    async def execute(self, params: Any, context: dict[str, Any]) -> dict[str, Any]:
        # Who asked? Curiosity researching on its own is a feature and stays
        # one — it simply must not escalate onto the foreground lane, because
        # the person at the keyboard is the one actually waiting.
        _ctx = dict(context or {})
        _origin = str(
            _ctx.get("authority_origin") or _ctx.get("origin") or _ctx.get("source") or ""
        ).strip().lower()
        _requested_by_user = bool(
            _ctx.get("foreground_request")
            or _ctx.get("user_facing")
            or _origin in {
                "user", "desktop_ui", "voice", "chat", "web_interlocutor",
                "desktop_chat", "admin",
            }
        )
        if isinstance(params, dict):
            query = params.get("query") or params.get("q", "")
            deep = bool(params.get("deep", False))
            num_results = int(params.get("num_results", 5))
            retain = params.get("retain")
            force_refresh = bool(params.get("force_refresh", False))
        elif isinstance(params, WebSearchInput):
            query = params.query
            deep = params.deep
            num_results = params.num_results
            retain = params.retain
            force_refresh = params.force_refresh
        else:
            query = str(params)
            deep = False
            num_results = 5
            retain = None
            force_refresh = False

        query = str(query or "").strip()
        if not query:
            return {"ok": False, "error": "No search query provided."}

        source_reading = query_requires_source_reading(query)
        effective_deep = bool(deep or source_reading)

        logger.info(
            "🔍 WebSearch: '%s' (deep=%s, effective_deep=%s, retain=%s, force_refresh=%s)",
            query[:80],
            deep,
            effective_deep,
            retain,
            force_refresh,
        )
        
        if deep and not source_reading:
            # v2.0: Deep Research LangGraph Pipeline implementation
            try:
                engine = (
                    ServiceContainer.get("cognitive_engine", default=None)
                    or ServiceContainer.get("brain", default=None)
                )
                if engine is None:
                    raise RuntimeError("No cognitive engine available for deep research")
                brain = _DeepResearchBrainAdapter(engine)
                
                # Adapting existing Search pipeline format to standard search_fn format
                async def _search_fn(q: str):
                    res = await self.pipeline.search(q, num_results=5, deep=False, force_refresh=force_refresh)
                    results = res.get("results", [])
                    # format sources
                    content = res.get("answer") or str([r.get("snippet", "") for r in results])
                    return {"ok": True, "content": content, "sources": results}
                
                # Curiosity may research all it likes; it just may not
                # take the foreground lane to do it. Only a person's request
                # earns that escalation.
                res = await run_deep_research(
                    query,
                    brain,
                    _search_fn,
                    requested_by_user=_requested_by_user,
                )
                answer = str(res.get("answer") or "").strip()
                if answer:
                    normalized = self._normalize_deep_research_result(query, res)
                    if self.pipeline._should_retain(
                        query,
                        deep=True,
                        retain=retain,
                        context=context or {},
                        result=normalized,
                    ):
                        artifact = self.pipeline._result_to_artifact(
                            normalized,
                            freshness_seconds=freshness_window_for_query(query),
                        )
                        await self.pipeline._retain_artifact(artifact, context or {})
                        normalized["retained"] = True
                        normalized["artifact_id"] = artifact.artifact_id
                    try:
                        from core.advanced_cognition import ExternalEvidenceDeliberator

                        normalized["deliberation_receipts"] = ExternalEvidenceDeliberator.deliberate_many(
                            normalized.get("chunks") or [],
                            source_type="web_search",
                            goal=query,
                        )
                    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                        record_degradation("web_search", exc, severity="warning", action="continued without deep evidence deliberation")
                    return normalized
                # "Empty answer" used to be the whole story, which read as
                # "the research found nothing". Usually it found plenty and
                # could not synthesize it — on 2026-07-25, because background
                # inference was queued behind foreground headroom. Those are
                # different failures and only one of them is about the web.
                logger.warning(
                    "Deep Research produced no answer for '%s' (%s; %d source(s) "
                    "gathered); falling back to retrieval pipeline.",
                    query,
                    res.get("synthesis_detail") or res.get("synthesis_status") or "no detail",
                    len(res.get("sources") or []),
                )
            except (ImportError, AttributeError, RuntimeError) as e:
                record_degradation(
                    "web_search",
                    e,
                    severity="warning",
                    action="fell back to retrieval pipeline after deep research failed",
                    extra={"query": query[:240]},
                )
                logger.error("Deep Research failed, falling back to legacy: %s", e)

        # Legacy direct search — a consequential network action, wrapped in
        # a welfare transaction (begin → execute → complete with outcome) so
        # the consequence bus sees real egress effects, not a decorative
        # import (the previous unused ActionExecutor import was exactly that
        # and rightly died to lint).
        from core.being.welfare_transaction import WelfareTransaction

        _tx = WelfareTransaction.begin(
            domain="network_research",
            action=f"web_search:{query[:80]}",
        )
        try:
            result = await self.pipeline.search(
                query,
                num_results=num_results,
                deep=effective_deep,
                retain=retain,
                context=context or {},
                force_refresh=force_refresh,
            )
        except (RuntimeError, OSError, ValueError, TypeError, AttributeError, ImportError) as exc:
            # A RAISING pipeline (missing backend, hard network failure) must
            # reach the local-corpus fallback exactly like a returned
            # failure — observed live: the exception path bypassed the
            # fallback and the curiosity loop logged web_search FAILED with
            # 6.5M offline documents sitting available.
            record_degradation(
                "web_search", exc, severity="warning",
                action="pipeline raised; degrading to local corpus",
            )
            offline = self._local_corpus_fallback(query, num_results)
            if offline is not None:
                offline["web_error"] = str(exc)[:200]
                _tx.complete(outcome="partial", error=str(exc)[:200])
                offline.setdefault("summary", offline.get("message") or "")
                return offline
            _tx.complete(outcome="failure", error=str(exc)[:200])
            raise
        _tx.complete(
            outcome="success" if result.get("ok") else "failure",
            error="" if result.get("ok") else str(result.get("error") or "")[:200],
        )
        if not result.get("ok") and force_refresh:
            logger.info(
                "WebSearch forced refresh failed for '%s'; retrying with retained-artifact fallback.",
                query[:80],
            )
            result = await self.pipeline.search(
                query,
                num_results=num_results,
                deep=effective_deep,
                retain=retain,
                context=context or {},
                force_refresh=False,
            )
        if not result.get("ok"):
            # Web unreachable/failed: answer from the local knowledge corpus
            # (6.5M offline reference docs) instead of returning empty-handed.
            # Provenance is explicit — a dated snapshot, never passed off as
            # live web results.
            offline = self._local_corpus_fallback(query, num_results)
            if offline is not None:
                offline["web_error"] = str(
                    result.get("error") or result.get("message") or "web search failed"
                )
                result = offline
        result.setdefault("summary", result.get("answer") or result.get("message") or "")
        if result.get("ok"):
            result.setdefault(
                "sources",
                result.get("citations")
                or result.get("chunks")
                or result.get("results")
                or ([] if not result.get("source") else [{"url": result.get("source")}]),
            )
            if result.get("sources"):
                criteria_results = result.get("criteria_results")
                if not isinstance(criteria_results, dict):
                    criteria_results = {}
                criteria_results["sources gathered"] = True
                result["criteria_results"] = criteria_results
        try:
            from core.advanced_cognition import ExternalEvidenceDeliberator

            artifacts = result.get("chunks") or result.get("results") or []
            if artifacts:
                result["deliberation_receipts"] = ExternalEvidenceDeliberator.deliberate_many(
                    artifacts,
                    source_type="web_search",
                    goal=query,
                )
            elif result.get("summary"):
                result["deliberation_receipts"] = [
                    ExternalEvidenceDeliberator()
                    .deliberate(
                        source_type="web_search",
                        source_ref=result.get("source") or query,
                        content=str(result.get("summary") or ""),
                        goal=query,
                        metadata=result,
                    )
                    .to_dict()
                ]
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("web_search", exc, severity="warning", action="continued without evidence deliberation receipts")
        return result

    @staticmethod
    def _local_corpus_fallback(query: str, num_results: int) -> dict[str, Any] | None:
        """Degrade to the local knowledge corpus when the web is unreachable.

        Returns None when the corpus is absent/empty or has no match, so the
        caller keeps the original web failure result.
        """
        try:
            from core.knowledge.local_corpus import get_local_corpus_store

            store = get_local_corpus_store()
            if store.document_count() <= 0:
                return None
            hits = store.search(query, limit=max(1, min(int(num_results), 10)))
            if not hits:
                return None
            logger.info(
                "WebSearch degraded to local corpus for '%s' (%d offline hits)",
                query[:80],
                len(hits),
            )
            return {
                "ok": True,
                "provenance": "local_corpus",
                "offline_fallback": True,
                "results": [
                    {
                        "title": hit.title,
                        "snippet": hit.snippet,
                        "source": hit.source,
                        "provenance": "local_corpus",
                    }
                    for hit in hits
                ],
                "summary": (
                    "Web search was unavailable; answered from the local "
                    f"offline reference corpus ({len(hits)} matches, dated snapshot)."
                ),
            }
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError, TypeError) as exc:
            record_degradation(
                "web_search",
                exc,
                severity="debug",
                action="local corpus fallback unavailable",
            )
            return None

    async def on_stop_async(self):
        """Lifecycle hook retained for skill manager shutdown symmetry."""
        return None


class _DormantBrowser:
    """Dormant browser adapter used until a governed browser session is opened."""

    is_active = False

    async def ensure_ready(self):
        return None

    async def browse(self, url: str):
        return False

    async def click(self, text_match: str = "", selector: str = "") -> bool:
        return False

    async def close(self):
        return None
