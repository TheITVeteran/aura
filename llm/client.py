# llm/client.py - Sovereign Deployment
import logging
from typing import Any

# Sovereign Imports
from core.brain.local_llm import LocalBrain
from core.runtime.errors import record_degradation

logger = logging.getLogger("LLM.Sovereign")

_SOVEREIGN_CLIENT_ERRORS = (
    AttributeError,
    ConnectionError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


def _record_sovereign_client_degradation(exc: BaseException, *, action: str) -> None:
    record_degradation(
        "legacy_sovereign_llm_client",
        exc,
        severity="warning",
        action=action,
    )


class OpenAIClient:
    """
    Sovereign Replacement for Legacy OpenAI Client.
    Redirects all calls to the local Ollama brain.
    """

    def __init__(self, api_key: str | None = None) -> None:
        # Legacy compat: api_key is ignored in sovereign mode
        self.brain = LocalBrain()
        logger.info("Sovereign LLM Client initialized (Ollama bridge).")

    def call(self, prompt: str, system: str | None = None, **kwargs: Any) -> dict[str, Any]:
        """
        Calls the local sovereign brain.
        """
        try:
            # We map kwargs for compatibility if needed, but LocalBrain.generate handles core logic
            text = self.brain.generate(prompt, system_prompt=system)
            
            if "Error" in text:
                 return {"ok": False, "error": text}
                 
            return {
                "ok": True, 
                "text": text, 
                "raw": {"provider": "sovereign", "model": self.brain.model}
            }
        except _SOVEREIGN_CLIENT_ERRORS as e:
            _record_sovereign_client_degradation(e, action="failed during legacy sovereign LLM call")
            logger.exception("Sovereign brain call failed")
            return {"ok": False, "error": str(e)}
