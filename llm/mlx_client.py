import asyncio
import logging

from core.runtime.errors import record_degradation

logger = logging.getLogger("LLM.MLX")

_LEGACY_MLX_CLIENT_ERRORS = (
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _record_legacy_mlx_degradation(exc: BaseException, *, action: str) -> None:
    record_degradation(
        "legacy_mlx_client",
        exc,
        severity="warning",
        action=action,
    )


class MLXClient:
    """
    Client for running local LLMs on Apple Silicon via mlx_lm.
    """
    def __init__(self, model_path="mlx-community/Mistral-7B-Instruct-v0.3-4bit"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self._client = None
        self._load_model()

    def _load_model(self):
        try:
            from core.brain.llm.mlx_client import get_mlx_client

            self._client = get_mlx_client(self.model_path, origin="legacy_mlx_client")
            self.model = self._client
            logger.info("Legacy MLXClient attached to canonical model lane: %s", self.model_path)
        except _LEGACY_MLX_CLIENT_ERRORS as e:
            _record_legacy_mlx_degradation(
                e,
                action="failed to resolve canonical MLX client",
            )
            logger.error("Failed to resolve canonical MLX client: %s", e)

    @staticmethod
    def _extract_think_segments(text: str) -> tuple[str, str]:
        import re
        thoughts = []
        for m in re.finditer(r'<think>(.*?)</think>', text, flags=re.DOTALL):
            thought_text = m.group(1).strip()
            if thought_text:
                thoughts.append(thought_text)
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        cleaned = cleaned.replace('</think>', '').replace('<think>', '')
        return cleaned.strip(), "\n\n".join(thoughts)

    def call(self, prompt: str, system_prompt: str = None, max_tokens: int = 2048, **kwargs) -> dict:
        if self._client is None:
            return {"ok": False, "error": "Canonical MLX model lane unavailable"}

        try:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                response_text = asyncio.run(
                    self._client.generate_text_async(
                        prompt,
                        system_prompt=system_prompt or "",
                        max_tokens=max_tokens,
                        temperature=kwargs.get("temperature", 0.7),
                        owner_label="legacy_mlx_client",
                    )
                )
                if not response_text:
                    return {"ok": False, "error": "Canonical MLX generation failed"}
                cleaned, thought = self._extract_think_segments(response_text)
                return {"ok": True, "text": cleaned, "thought": thought}
            else:
                return {
                    "ok": False,
                    "error": "MLXClient.call cannot block an event loop; use call_stream",
                }
        except _LEGACY_MLX_CLIENT_ERRORS as e:
            _record_legacy_mlx_degradation(
                e,
                action="failed during canonical legacy-compatibility generation",
            )
            logger.error("MLX Generation Error: %s", e)
            return {"ok": False, "error": str(e)}

    # v15: Streaming support
    async def call_stream(self, prompt: str, system_prompt: str = None, max_tokens: int = 2048, **kwargs):
        if self._client is None:
            yield "Error: canonical MLX model lane unavailable"
            return
        try:
            response_text = await self._client.generate_text_async(
                prompt,
                system_prompt=system_prompt or "",
                max_tokens=max_tokens,
                temperature=kwargs.get("temperature", 0.7),
                owner_label="legacy_mlx_client_stream",
            )
            if not response_text:
                yield "Error: canonical MLX generation failed"
                return
            cleaned, _thought = self._extract_think_segments(response_text)
            if cleaned:
                yield cleaned
        except _LEGACY_MLX_CLIENT_ERRORS as e:
            _record_legacy_mlx_degradation(
                e,
                action="failed during canonical legacy-compatibility stream",
            )
            yield f"Error: {e}"
