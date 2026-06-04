import argparse
import asyncio
import logging
import os
import re
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

logger = logging.getLogger("Aura.NetHackChallenge")

DEFAULT_MAX_STEPS = int(os.getenv("AURA_NETHACK_MAX_STEPS", "900"))
DEFAULT_MAX_SECONDS = float(os.getenv("AURA_NETHACK_MAX_SECONDS", "1800"))
_VALID_MOVES = "hjklubny><."
_RECOVERABLE_CHALLENGE_ERRORS = (
    RuntimeError,
    TimeoutError,
    OSError,
    ValueError,
    AttributeError,
    TypeError,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Aura's bounded NetHack reflex challenge.")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS)
    return parser.parse_args()


def _extract_action(response: str) -> str | None:
    if "[SOMATIC:key=" in response:
        action_match = re.search(r"key=['\"](.*?)['\"]", response)
        return action_match.group(1) if action_match else None

    clean_response = response.strip()
    if len(clean_response) == 1 and clean_response.lower() in _VALID_MOVES:
        return clean_response.lower()
    return None


async def run(*, max_steps: int = DEFAULT_MAX_STEPS, max_seconds: float = DEFAULT_MAX_SECONDS) -> None:
    from core.adapters.nethack_adapter import NetHackAdapter
    from core.container import ServiceContainer
    from core.orchestrator.main import create_orchestrator

    orchestrator = create_orchestrator()
    adapter = NetHackAdapter()
    deadline = time.monotonic() + max(1.0, max_seconds)

    logger.info(
        "Starting bounded NetHack challenge pid=%s max_steps=%s max_seconds=%s",
        os.getpid(),
        max_steps,
        max_seconds,
    )
    try:
        await orchestrator.start()
        adapter.start(name="AuraSimple")
        ServiceContainer.register_instance("nethack_adapter", adapter)

        for step in range(max(1, max_steps)):
            if time.monotonic() >= deadline:
                logger.info("NetHack challenge deadline reached at step %s", step + 1)
                break
            if not adapter.is_alive():
                logger.info("NetHack process exited at step %s", step + 1)
                break

            try:
                obs = adapter.get_observation()
                obs_text = obs.get("text", "")
                logger.debug("NetHack prompt step=%s text=%r", step + 1, obs_text[:100])

                response = await orchestrator.process_user_input_priority(
                    f"{obs_text}\n\n[EMBODIED CONTROL CONTRACT] Somatic reflex matcher v3 ACTIVE.",
                    origin="embodied_motor_reflex",
                )
                if not response:
                    await asyncio.sleep(1)
                    continue

                action = _extract_action(str(response))
                if action:
                    logger.debug("Executing NetHack action step=%s action=%r", step + 1, action)
                    adapter.send_action(action)
                else:
                    logger.debug("Ignoring non-action NetHack response step=%s response=%r", step + 1, response)

                await asyncio.sleep(1)
            except _RECOVERABLE_CHALLENGE_ERRORS as exc:
                logger.warning(
                    "Recoverable NetHack loop error at step %s: %s\n%s",
                    step + 1,
                    exc,
                    traceback.format_exc(),
                )
                await asyncio.sleep(2)
    finally:
        adapter.stop()
        try:
            await orchestrator.stop()
        except _RECOVERABLE_CHALLENGE_ERRORS as exc:
            logger.warning("NetHack challenge orchestrator shutdown failed: %s", exc)


if __name__ == "__main__":
    logging.basicConfig(level=os.getenv("AURA_NETHACK_LOG_LEVEL", "INFO"))
    cli_args = parse_args()
    asyncio.run(run(max_steps=cli_args.max_steps, max_seconds=cli_args.max_seconds))
