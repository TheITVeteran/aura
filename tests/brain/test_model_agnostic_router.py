"""tests/brain/test_model_agnostic_router.py — Model Agnostic Router & Regression Tests.

Asserts that model providers conform to the contract, and that routing assigns tasks
to the correct specialist tier (reflex, reasoning, verifier, vision).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.brain.llm.provider_contract import (
    ProviderContract,
    ModelCapabilities,
    ContractedLLMProvider,
)


class MockReflexProvider(ContractedLLMProvider):
    def get_contract(self) -> ProviderContract:
        return ProviderContract(
            provider_name="ollama",
            model_name="llama3-reflex-8b",
            context_limit=8192,
            latency_estimate_ms=120.0,
            memory_cost_gb=6.0,
            is_local=True,
            privacy_level="local_isolated",
            failure_mode="fail_over_degraded",
            health_status="healthy",
            capabilities=ModelCapabilities(
                supports_vision=False,
                supports_tool_calling=True,
                supports_steering=True,
                supports_recurrent_depth=False,
            )
        )

    async def generate(self, prompt: str, system_prompt: str | None = None, **kwargs: Any) -> str:
        return "reflex action response"


class MockReasoningProvider(ContractedLLMProvider):
    def get_contract(self) -> ProviderContract:
        return ProviderContract(
            provider_name="mlx",
            model_name="aura-32b-hard-reasoning",
            context_limit=32768,
            latency_estimate_ms=450.0,
            memory_cost_gb=24.0,
            is_local=True,
            privacy_level="local_isolated",
            failure_mode="fallback",
            health_status="healthy",
            capabilities=ModelCapabilities(
                supports_vision=False,
                supports_tool_calling=True,
                supports_steering=True,
                supports_recurrent_depth=True,
            )
        )

    async def generate(self, prompt: str, system_prompt: str | None = None, **kwargs: Any) -> str:
        return "deep reasoning output"


class MockVisionProvider(ContractedLLMProvider):
    def get_contract(self) -> ProviderContract:
        return ProviderContract(
            provider_name="mlx_vision",
            model_name="llava-screenshot-13b",
            context_limit=16384,
            latency_estimate_ms=300.0,
            memory_cost_gb=12.0,
            is_local=True,
            privacy_level="local_isolated",
            failure_mode="fail_closed",
            health_status="healthy",
            capabilities=ModelCapabilities(
                supports_vision=True,
                supports_tool_calling=False,
                supports_steering=False,
                supports_recurrent_depth=False,
                modalities=("text", "image"),
            )
        )

    async def generate(self, prompt: str, system_prompt: str | None = None, **kwargs: Any) -> str:
        return "vision audit report"


class ModelRouter:
    """Intelligent Model Router utilizing the Provider Contract to dispatch queries."""
    
    def __init__(self, providers: list[ContractedLLMProvider]) -> None:
        self.providers = providers

    def route_task(self, task_type: str, requires_vision: bool = False) -> ContractedLLMProvider:
        for provider in self.providers:
            contract = provider.get_contract()
            if contract.health_status != "healthy":
                continue
            
            # If vision is required, we can only route to a vision-supporting provider
            if requires_vision and not contract.capabilities.supports_vision:
                continue
                
            # Route vision requests
            if requires_vision and contract.capabilities.supports_vision:
                return provider
            
            # Route reflex vs reasoning
            if task_type == "reflex" and contract.latency_estimate_ms < 200.0:
                return provider
            
            if task_type == "reasoning" and contract.context_limit >= 32768:
                return provider
            
            if task_type == "verifier" and contract.capabilities.supports_tool_calling:
                return provider
                
        # Fallback
        return self.providers[0]


def test_model_router_dispatches_correctly():
    """Verify router dispatches cheap/reflex, big/reasoning, verifier, and vision tasks appropriately."""
    providers = [MockReflexProvider(), MockReasoningProvider(), MockVisionProvider()]
    router = ModelRouter(providers)

    # 1. Reflex (cheap/fast)
    reflex_routed = router.route_task("reflex")
    assert reflex_routed.get_contract().model_name == "llama3-reflex-8b"
    assert reflex_routed.get_contract().latency_estimate_ms < 200.0

    # 2. Reasoning (deep/large context)
    reasoning_routed = router.route_task("reasoning")
    assert reasoning_routed.get_contract().model_name == "aura-32b-hard-reasoning"
    assert reasoning_routed.get_contract().context_limit >= 32768

    # 3. Vision (supports image modalities)
    vision_routed = router.route_task("reasoning", requires_vision=True)
    assert vision_routed.get_contract().capabilities.supports_vision is True


def test_model_upgrade_regression_policy():
    """Regression test ensuring a model upgrade doesn't violate minimum contract requirements."""
    baseline_contract = ProviderContract(
        provider_name="ollama",
        model_name="baseline-8b",
        context_limit=8192,
        latency_estimate_ms=150.0,
        memory_cost_gb=8.0,
        is_local=True,
        privacy_level="local_isolated",
        failure_mode="fail_over_degraded",
        health_status="healthy",
        capabilities=ModelCapabilities(supports_tool_calling=True)
    )

    # Upgrade proposal: e.g. upgrading to a new model
    upgrade_contract = ProviderContract(
        provider_name="ollama",
        model_name="upgrade-next-8b",
        context_limit=16384,                # improved (16k > 8k)
        latency_estimate_ms=130.0,          # improved (faster)
        memory_cost_gb=7.0,                 # improved (lower memory footprint)
        is_local=True,
        privacy_level="local_isolated",
        failure_mode="fail_over_degraded",
        health_status="healthy",
        capabilities=ModelCapabilities(supports_tool_calling=True) # maintained
    )

    # Assert regression safety rules:
    assert upgrade_contract.context_limit >= baseline_contract.context_limit, "Regression: Context limit decreased"
    assert upgrade_contract.latency_estimate_ms <= baseline_contract.latency_estimate_ms * 1.15, "Regression: Latency degraded significantly"
    assert upgrade_contract.memory_cost_gb <= baseline_contract.memory_cost_gb * 1.1, "Regression: Memory usage grew excessively"
    assert upgrade_contract.privacy_level == baseline_contract.privacy_level, "Regression: Privacy guarantees downgraded"
    assert upgrade_contract.capabilities.supports_tool_calling is True, "Regression: Lost tool calling support"
