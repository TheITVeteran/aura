import asyncio
import time
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.brain.inference_gate import InferenceGate
from core.state.aura_state import AuraState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [STRESS] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("StressSuite")


class DeterministicMLXSimulator:
    """Deterministic local inference stand-in for orchestration stress tests."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.is_warmed = False

    async def warmup(self):
        logger.info("Warming deterministic 32B-lane simulator: %s", self.model_path)
        await asyncio.sleep(0.05)
        self.is_warmed = True
        logger.info("Deterministic inference lane is ready.")

    async def generate_text_async(self, prompt, system_prompt="", messages=None, **kwargs):
        if not self.is_warmed:
            return False, "", {}

        await asyncio.sleep(0.15)

        response = (
            "Deterministic inference response for orchestration stress "
            "validation with stable token and latency characteristics."
        )
        tokens = len(response.split()) * 1.3
        await asyncio.sleep(tokens / 25.0)

        return True, response, {"tokens": int(tokens), "ttft": 0.15}

    def is_alive(self):
        return True


class EnterpriseStressSuite:
    def __init__(self):
        self.gate = InferenceGate()
        self.gate._mlx_client = DeterministicMLXSimulator(
            "Qwen2.5-32B-Instruct-8bit"
        )
        self.results = {
            "benchmarks": {},
            "reliability": {},
            "resource_security": {},
        }

    async def _generate_deterministic(self, prompt: str) -> str:
        ok, text, metadata = await self.gate._mlx_client.generate_text_async(prompt)
        if ok and isinstance(metadata, dict):
            self.results["benchmarks"].setdefault("ttft", metadata.get("ttft"))
            self.results["benchmarks"].setdefault("tokens", metadata.get("tokens"))
        return text if ok else ""

    async def test_performance_metrics(self):
        """Measure local orchestration latency against deterministic baselines."""
        logger.info("Benchmarking deterministic TTFT and TPS envelope.")

        start = time.monotonic()
        text = await self._generate_deterministic("What is the future of edge AI?")
        elapsed = time.monotonic() - start

        if text and len(text) > 20:
            tokens = len(text.split()) * 1.3
            tps = tokens / elapsed
            logger.info(
                "Inference path succeeded. Total latency %.2fs, estimated TPS %.1f",
                elapsed,
                tps,
            )
            self.results["benchmarks"]["total_latency"] = elapsed
            self.results["benchmarks"]["est_tps"] = tps
            self.results["reliability"]["primary"] = "PASS"
        else:
            self.results["reliability"]["primary"] = "FAIL"

    async def test_vault_concurrency(self):
        """Verify the State Vault async fix by running heavy commits during inference."""
        logger.info("Testing State Vault async resilience.")

        from core.state.vault import StateVaultActor
        vault = StateVaultActor()

        state = AuraState()
        state.cognition.working_memory = [
            {"role": "u", "content": "X" * 5000} for _ in range(1000)
        ]

        from core.state.state_repository import StateRepository
        repo = StateRepository()
        payload = {"state": repo._circular_safe_asdict(state), "cause": "stress"}

        logger.info("Starting 10 concurrent inferences during a large state commit.")
        start = time.monotonic()

        tasks = [self._generate_deterministic(f"Query {i}") for i in range(10)]
        tasks.append(vault._process_commit_inner(payload, "m-1"))

        responses = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.monotonic() - start

        success_count = sum(1 for r in responses if isinstance(r, str))
        logger.info(
            "Vault pressure test: %s/10 inferences succeeded during commit in %.2fs.",
            success_count,
            elapsed,
        )
        self.results["resource_security"]["vault_stall"] = (
            "NONE" if elapsed < 5.0 and success_count == 10 else "DETECTED"
        )

    async def test_background_tiering_lock(self):
        """Verify that background tasks are strictly locked to tertiary/fast tiers."""
        logger.info("Testing background tiering lock.")

        from core.brain.llm_health_router import HealthAwareLLMRouter
        router = HealthAwareLLMRouter()

        router.register("MLX-Cortex", "local", "32B", is_local=True, tier="primary")
        router.register("MLX-Solver", "local", "72B", is_local=True, tier="api_deep")
        router.register("MLX-Brainstem", "local", "7B", is_local=True, tier="local_fast")

        logger.info("Testing demotion of primary preference for background task.")
        tried_endpoints = []
        original_call = router._call_endpoint

        async def tracked_call(ep, *args, **kwargs):
            tried_endpoints.append(ep.name)
            return {"text": "simulated", "ok": True}

        router._call_endpoint = tracked_call
        try:
            await router.generate("Test", is_background=True, prefer_tier="primary")

            if any("Cortex" in name for name in tried_endpoints):
                logger.error("Tier Lock FAIL: background task used 32B Cortex.")
                self.results["resource_security"]["tier_lock"] = "FAIL"
            elif any("Brainstem" in name for name in tried_endpoints):
                logger.info("Tier Lock PASS: background task demoted to Brainstem.")
                self.results["resource_security"]["tier_lock"] = "PASS"
            else:
                logger.warning("Tier Lock UNKNOWN: tried %s", tried_endpoints)
                self.results["resource_security"]["tier_lock"] = "UNKNOWN"

            tried_endpoints.clear()
            await router.generate("Test", prefer_tier="tertiary")
            if tried_endpoints and "Brainstem" in tried_endpoints[0]:
                logger.info("Priority PASS: Brainstem prioritized for tertiary tasks.")
                self.results["resource_security"]["tertiary_priority"] = "PASS"
            else:
                first = tried_endpoints[0] if tried_endpoints else "none"
                logger.error("Priority FAIL: tertiary task tried %s first.", first)
                self.results["resource_security"]["tertiary_priority"] = "FAIL"
        finally:
            router._call_endpoint = original_call

    def report(self):
        print("\n" + "=" * 60)
        print("    AURA ENTERPRISE ORCHESTRATION STRESS TEST RESULTS")
        print("    (DETERMINISTIC INFERENCE LANE | REAL ORCHESTRATION)")
        print("=" * 60)
        print(json.dumps(self.results, indent=4))
        print("=" * 60 + "\n")

    def assert_passed(self) -> None:
        failures: list[str] = []
        for section, values in self.results.items():
            if section == "benchmarks":
                continue
            for key, value in values.items():
                if value != "PASS" and value != "NONE":
                    failures.append(f"{section}.{key}={value}")
        if failures:
            raise RuntimeError(
                "Enterprise stress suite failed: " + ", ".join(sorted(failures))
            )


async def main():
    suite = EnterpriseStressSuite()
    await suite.gate._mlx_client.warmup()
    await suite.test_performance_metrics()
    await suite.test_vault_concurrency()
    await suite.test_background_tiering_lock()
    suite.report()
    suite.assert_passed()

if __name__ == "__main__":
    asyncio.run(main())
