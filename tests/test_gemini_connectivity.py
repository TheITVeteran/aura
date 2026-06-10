import asyncio
import pytest
import os
import logging
import httpx
from core.brain.llm.gemini_adapter import GeminiAdapter

pytestmark = [pytest.mark.network, pytest.mark.external]  # requires Gemini API surface; default suite skips honestly

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_gemini")

@pytest.mark.asyncio
async def test_connectivity():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        pytest.fail("GEMINI_API_KEY not found; live Gemini connectivity cannot be verified")

    models_to_test = ["gemini-2.0-flash", "gemini-flash-latest", "gemini-pro-latest"]
    failures = []
    
    for model_id in models_to_test:
        logger.info(f"📡 Testing {model_id}...")
        adapter = GeminiAdapter(api_key=api_key, model=model_id)
        try:
            success, response, meta = await adapter.call(f"Say '{model_id} is online' if you can hear me.")
            if success:
                logger.info(f"✅ {model_id} Success: {response}")
            else:
                logger.error(f"❌ {model_id} Failed: {meta.get('error')}")
                failures.append((model_id, meta.get("error")))
        except (httpx.HTTPError, RuntimeError, TimeoutError, TypeError, ValueError) as e:
            logger.error(f"❌ Exception in {model_id}: {e}")
            failures.append((model_id, str(e)))
        finally:
            await adapter.close()
    assert not failures

if __name__ == "__main__":
    asyncio.run(test_connectivity())
