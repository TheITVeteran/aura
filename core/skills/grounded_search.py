"""
Grounded Search Skill — Ported from gemini-cli/web-search.ts

Uses google-genai SDK to perform real Google Search queries with
grounding metadata, providing inline citations and reducing hallucinations.
"""

from core.runtime.errors import record_degradation
from core.brain.llm.cloud_errors import cloud_call_error_types
import logging
import os
from typing import Any, Dict

from infrastructure import BaseSkill

logger = logging.getLogger("Skills.GroundedSearch")

class GroundedSearchSkill(BaseSkill):
    name = "grounded_search"
    description = "Searches the web using Google Search API with inline citation grounding."

    async def execute(self, goal: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        query = goal.get("params", {}).get("query", goal.get("objective", ""))
        
        if not query:
            return {"ok": False, "error": "No query provided"}

        # [FIX] Check config first — desktop/GUI mode may not inherit terminal env vars.
        try:
            from core.config import config as _gs_cfg
            api_key = getattr(getattr(_gs_cfg, "llm", None), "gemini_api_key", None)
        except (ImportError, AttributeError):
            api_key = None
        if not api_key:
            api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return {
                "ok": False, 
                "error": "GEMINI_API_KEY is not set. Cannot use Google Grounding.",
                "note": "Fallback to standard web_search if needed."
            }

        # The query is the user's own words and this SDK builds its own HTTP,
        # so NetworkGateway never sees it. Screen it here or send nothing.
        from core.security.egress_privacy import filter_model_prompt

        screened = filter_model_prompt(query, provider="gemini_grounded_search")
        if not screened.allowed:
            return {
                "ok": False,
                "error": f"Grounded search refused by egress privacy: {screened.reason}",
                "note": "Fallback to standard web_search if needed.",
            }
        query = screened.text or ""

        try:
            # We delay import until runtime to prevent strict dependencies
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)
            logger.info("Executing grounded search for: %s", query)

            response = client.models.generate_content(
                model='gemini-2.5-pro',
                contents=query,
                config=types.GenerateContentConfig(
                    tools=[{"google_search": {}}],
                    temperature=0.0
                )
            )

            # Format the text with inline citations if metadata is present
            answer = response.text
            sources = []
            
            # The python SDK typically exposes grounding_metadata if tools were used
            metadata = getattr(response.candidates[0], "grounding_metadata", None)
            if metadata and metadata.grounding_chunks:
                for chunk in metadata.grounding_chunks:
                    if hasattr(chunk, "web"):
                        sources.append({
                            "title": chunk.web.title,
                            "url": chunk.web.uri
                        })
            
            if sources:
                answer += "\n\n### Grounding Sources:\n"
                for i, src in enumerate(sources, 1):
                    answer += f"[{i}] [{src['title']}]({src['url']})\n"

            return {
                "ok": True,
                "answer": answer,
                "sources": sources,
                "note": "Grounded by Google Search"
            }
            
        except ImportError:
            return {"ok": False, "error": "google-genai package not installed (pip install google-genai)"}
        except (AttributeError, RuntimeError, *cloud_call_error_types()) as e:
            # A quota-exhausted (429 RESOURCE_EXHAUSTED), unauthorized, or
            # unreachable cloud provider must degrade to local search, never
            # crash the turn as an unhandled request exception (observed live).
            record_degradation(
                'grounded_search',
                e,
                action="fell back from Google grounding; caller uses local web search",
            )
            logger.warning(
                "Grounded search unavailable (%s: %s); falling back to local web_search.",
                type(e).__name__,
                str(e)[:200],
            )
            return {
                "ok": False,
                "error": str(e)[:240],
                "note": "Fallback to standard web_search if needed.",
            }
