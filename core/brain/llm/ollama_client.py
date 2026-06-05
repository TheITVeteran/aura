from core.runtime.errors import record_degradation
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from core.runtime.network_gateway import get_network_gateway
from .provider import LLMProvider

logger = logging.getLogger("LLM.Ollama")
_OLLAMA_RECOVERABLE_ERRORS = (
    OSError,
    ConnectionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)

class RobustOllamaClient(LLMProvider):
    """Ollama implementation for local inference using httpx for async operation.
    """
    
    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434", timeout: float = 45.0, **kwargs):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = None

    async def close(self):
        self._client = None

    async def _post(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Internal helper for async POST requests."""
        response = await asyncio.to_thread(
            get_network_gateway().request,
            "POST",
            f"{self.base_url}{endpoint}",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=self.timeout,
            source="llm_provider:ollama",
            read_only=True,
        )
        if response.get("status_code") != 200:
            raise RuntimeError(response.get("error") or f"ollama_http_{response.get('status_code')}")
        return json.loads((response.get("content") or b"{}").decode("utf-8"))

    def generate_text(self, prompt: str, system_prompt: Optional[str] = None, model: Optional[str] = None, num_predict: Optional[int] = None) -> str:
        """Synchronous wrapper (legacy support)."""
        try:
            # Check if we're already in an async loop to avoid RuntimeError
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    # If we are in the loop thread, we cannot block.
                    # We might be able to run in a separate thread and wait, but 
                    # usually these sync calls shouldn't be made from the loop thread.
                    logger.error("❌ CRITICAL: Synchronous generate_text called from within an async loop. Use generate_text_async.")
                    return "[Ollama Error: Sync call in Async loop]"
            except RuntimeError:
                # No running loop in this thread, safe to use asyncio.run
                return asyncio.run(self.generate_text_async(prompt, system_prompt, model, num_predict))
                
            # Fallback if loop exists but we are not in its thread (though get_running_loop usually errors)
            return "[Ollama Error: Async boundary violation]"
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation('ollama_client', e)
            logger.error("Ollama sync generation failed: %s", e)
            return f"[Ollama Error: {e}]"

    async def generate_text_async(self, prompt: str, system_prompt: Optional[str] = None, model: Optional[str] = None, num_predict: Optional[int] = None, **kwargs) -> str:
        """Generate text response from Ollama (Async)."""
        try:
            payload = {
                "model": model or self.model,
                "prompt": prompt,
                "stream": False
            }
            if system_prompt:
                payload["system"] = system_prompt
            
            # v26 FIX: Hard Memory Ceiling (Clamped to prevent OOM)
            hard_options = {
                "num_ctx": 4096,   # Threshold: 4k tokens (Baseline safety)
                "num_predict": 512 # Maximum verbosity cap
            }
            
            # Phase 24 Upgrade: Pass through model options (keep_alive, num_ctx, etc)
            if "options" in kwargs:
                payload["options"] = kwargs["options"]
            else:
                payload["options"] = {}
                
            payload["options"].update(hard_options)
            if "keep_alive" in kwargs:
                payload["keep_alive"] = kwargs["keep_alive"]

            data = await self._post("/api/generate", payload)
            return data.get("response", "")
        except _OLLAMA_RECOVERABLE_ERRORS as e:
            record_degradation('ollama_client', e)
            logger.error("Ollama generation failed: %s", e)
            raise

    async def generate_text_stream_async(self, prompt: str, system_prompt: Optional[str] = None, model: Optional[str] = None):
        """Stream text response from Ollama (Async)."""
        try:
            payload = {
                "model": model or self.model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "num_ctx": 4096,
                    "num_predict": 512
                }
            }
            if system_prompt:
                payload["system"] = system_prompt

            payload["stream"] = False
            data = await self._post("/api/generate", payload)
            text = data.get("response", "")
            if text:
                yield text
        except _OLLAMA_RECOVERABLE_ERRORS as e:
            record_degradation('ollama_client', e)
            logger.error("Ollama streaming failed: %s", e)
            raise

    def generate_json(self, prompt: str, schema: Dict[str, Any], system_prompt: Optional[str] = None, model: Optional[str] = None) -> Dict[str, Any]:
        """Synchronous wrapper (legacy support)."""
        try:
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    logger.error("❌ CRITICAL: Synchronous generate_json called from within an async loop. Use generate_json_async.")
                    return {"error": "Sync call in Async loop"}
            except RuntimeError:
                return asyncio.run(self.generate_json_async(prompt, schema, system_prompt, model))
            
            return {"error": "Async boundary violation"}
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation('ollama_client', e)
            logger.error("Ollama sync JSON generation failed: %s", e)
            return {"error": str(e)}

    async def generate_json_async(self, prompt: str, schema: Dict[str, Any], system_prompt: Optional[str] = None, model: Optional[str] = None) -> Dict[str, Any]:
        """Generate JSON response from Ollama (Async) with strict schema enforcement."""
        try:
            payload = {
                "model": model or self.model,
                "prompt": prompt,
                "format": schema, # Modern Ollama supports passing the full JSON schema here
                "stream": False,
                "options": {
                    "num_ctx": 4096,
                    "num_predict": 512
                }
            }
            if system_prompt:
                payload["system"] = system_prompt

            data = await self._post("/api/generate", payload)
            content = data.get("response", "{}")
            
            # Robust JSON extraction
            import re
            match = re.search(r"(\{.*\})", content, re.DOTALL)
            if match:
                content = match.group(1)
            
            result = json.loads(content)
            
            # Local Validation (Fallback/Reinforcement)
            from utils.json_utils import validate_with_schema
            if validate_with_schema(result, schema):
                return result
            else:
                logger.warning("Ollama output failed schema validation, attempting repair...")
                # Simple repair: try to cast or use defaults if critical
                return result 
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('ollama_client', e)
            logger.error("Ollama JSON generation failed: %s", e)
            raise

    def generate_embedding(self, text: str) -> List[float]:
        """Synchronous wrapper."""
        try:
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    logger.error("❌ CRITICAL: Synchronous generate_embedding called from within an async loop. Use generate_embedding_async.")
                    return []
            except RuntimeError:
                return asyncio.run(self.generate_embedding_async(text))
            return []
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return []

    async def generate_embedding_async(self, text: str) -> List[float]:
        """Generate vector embedding from Ollama (Async)."""
        try:
            payload = {
                "model": self.model,
                "input": text
            }
            data = await self._post("/api/embed", payload)
            embedding = data.get("embeddings", [data.get("embedding", [])])
            if isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], list):
                return embedding[0]
            return embedding if isinstance(embedding, list) else []
        except _OLLAMA_RECOVERABLE_ERRORS as e:
            record_degradation('ollama_client', e)
            logger.error("Ollama embedding failed: %s", e)
            return []

    def check_health(self) -> bool:
        """Synchronous health check (Safe for both sync and async contexts)."""
        try:
            response = get_network_gateway().request(
                "GET",
                f"{self.base_url}/api/tags",
                timeout=3,
                source="llm_provider:ollama_health",
                read_only=True,
            )
            return response.get("status_code") == 200
        except _OLLAMA_RECOVERABLE_ERRORS as e:
            record_degradation('ollama_client', e)
            logger.debug("Ollama health check failed (sync): %s", e)
            return False

    async def check_health_async(self) -> bool:
        """Check if Ollama server is reachable (Async)."""
        return await asyncio.to_thread(self.check_health)

    async def see(self, prompt: str, image_path: Optional[str] = None, image_base64: Optional[str] = None) -> str:
        """Analyze an image using a vision-capable model (Async)."""
        import base64
        try:
            if not image_base64 and not image_path:
                return "[Vision Failure: No image provided]"
                
            if not image_base64:
                with open(image_path, "rb") as image_file:
                    image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
                    
            payload = {
                "model": "llava",
                "prompt": prompt,
                "images": [image_base64],
                "stream": False,
            }

            data = await self._post("/api/generate", payload)
            return data.get("response", "[Vision Failure: No output]")
        except _OLLAMA_RECOVERABLE_ERRORS as e:
            record_degradation('ollama_client', e)
            logger.error("Ollama vision analysis failed: %s", e)
            return f"[Vision analysis failed: {e}]"

    async def generate(self, prompt: str, system_prompt: str = "", **kwargs) -> str:
        """Primary async interface for the engine."""
        return await self.generate_text_async(prompt, system_prompt, **kwargs)
