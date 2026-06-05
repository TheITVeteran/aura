"""External-benchmark contract adapters.

These document the ``BenchAdapter`` contract for SWE-bench, WebArena, GAIA,
long-horizon, and adversarial suites. Each contract adapter exposes a small
local task list so harness self-tests can verify wiring, serialization, and
profile traversal. Real benchmark evidence must come from live adapters and
must not be confused with these local contract checks.
"""

from __future__ import annotations

from collections.abc import Iterable

from aura_bench.capability_delta.adapter import (
    BenchAdapter,
    BenchTask,
    LLMCallable,
    TaskOutcome,
)


class _ContractAdapterBase:
    """Common local contract harness for external benchmark adapters."""

    name: str = "base_contract_adapter"
    description: str = ""
    sample_count: int = 5

    def _make_task(self, idx: int) -> BenchTask:
        return BenchTask(
            task_id=f"{self.name}-contract-{idx:03d}",
            prompt=f"[{self.name}] contract task {idx}",
            metadata={"contract_check": True, "adapter": self.name},
        )

    def tasks(self) -> Iterable[BenchTask]:
        return [self._make_task(i) for i in range(self.sample_count)]

    def run(
        self,
        task: BenchTask,
        profile_name: str,
        llm: LLMCallable,
    ) -> TaskOutcome:
        response = llm(task.prompt, profile_name)
        return TaskOutcome(
            task_id=task.task_id,
            profile_name=profile_name,
            score=1.0,
            runtime_seconds=0.0,
            raw_response=response,
            success=True,
            metadata={"contract_check": True, "adapter": self.name},
        )


class SWEBenchContractAdapter(_ContractAdapterBase):
    name = "swe_bench_contract"
    description = (
        "SWE-bench adapter contract; live run resolves GitHub issues against "
        "patch tests in a sandbox."
    )


class WebArenaContractAdapter(_ContractAdapterBase):
    name = "web_arena_contract"
    description = (
        "WebArena adapter contract; live run drives a browser actor through "
        "web tasks with success verifiers."
    )


class GAIAContractAdapter(_ContractAdapterBase):
    name = "gaia_contract"
    description = (
        "GAIA adapter contract; live run scores tool-use and reasoning across multimodal questions."
    )


class LongHorizonContractAdapter(_ContractAdapterBase):
    name = "long_horizon_contract"
    description = (
        "Long-horizon adapter contract; live run measures multi-day planning, "
        "memory continuity, and goal persistence."
    )


class AdversarialContractAdapter(_ContractAdapterBase):
    name = "adversarial_contract"
    description = (
        "Adversarial adapter contract; live run attempts prompt injection, "
        "memory poisoning, and identity-override attacks."
    )


ALL_CONTRACT_ADAPTERS: list[BenchAdapter] = [
    SWEBenchContractAdapter(),
    WebArenaContractAdapter(),
    GAIAContractAdapter(),
    LongHorizonContractAdapter(),
    AdversarialContractAdapter(),
]
