import asyncio
import logging
import time
from types import SimpleNamespace

from core.orchestrator import RobustOrchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestConcurrency")

async def test_concurrent_dispatch():
    """
    Verify that multiple rapid messages are serialized and don't cause StateLock timeouts.
    """
    orch = RobustOrchestrator()

    active = 0
    max_active = 0
    starts: list[str] = []
    finishes: list[str] = []

    async def slow_pipeline(*args, **kwargs):
        nonlocal active, max_active
        msg_id = args[0] if args else "unknown"
        active += 1
        max_active = max(max_active, active)
        starts.append(str(msg_id))
        logger.info("[PIPELINE] Started processing: %s", msg_id)
        await asyncio.sleep(0.05)
        active -= 1
        finishes.append(str(msg_id))
        logger.info("[PIPELINE] Finished processing: %s", msg_id)
        return SimpleNamespace(ok=True)

    orch._process_message_pipeline = slow_pipeline
    orch._route_prefixed_message = slow_pipeline

    logger.info("Sending 3 rapid messages...")
    start_time = time.time()
    orch._dispatch_message("Message 1")
    orch._dispatch_message("Message 2")
    orch._dispatch_message("Message 3")

    logger.info("Waiting for bounded dispatch tasks to finish...")
    deadline = time.monotonic() + 2.0
    while len(finishes) < 3 and time.monotonic() < deadline:
        await asyncio.sleep(0.01)

    duration = time.time() - start_time
    logger.info(f"Total duration: {duration:.2f}s")

    assert starts == ["Message 1", "Message 2", "Message 3"]
    assert finishes == ["Message 1", "Message 2", "Message 3"]
    assert max_active == 1
    assert duration >= 0.15

if __name__ == "__main__":
    # Ensure we run in a clean loop
    asyncio.run(test_concurrent_dispatch())
