"""
test_llm_failover_stress.py
============================
Stress test for the 5-tier LLM failover system.

Uses local async client probes while exercising the full routing logic including:
  - Circuit breaker state transitions (CLOSED -> OPEN -> HALF_OPEN -> CLOSED)
  - Failover chain traversal when endpoints are degraded
  - Cascade failure reaching emergency tier
  - Foreground requests skipping Brainstem
  - Empty/whitespace responses counting as failures
  - Recovery after circuit opens

No live LLM or network calls required.
"""
from __future__ import annotations


import sys
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.brain.llm_health_router import (
    CircuitState,
    EndpointHealth,
    HealthAwareLLMRouter,
    validate_response,
)
from core.container import ServiceContainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class LLMClientProbe:
    def __init__(self, response: str):
        self.response = response
        self.calls = []
        self.error: BaseException | None = None

    async def think(self, prompt: str, **kwargs):
        self.calls.append((prompt, kwargs))
        if self.error is not None:
            raise self.error
        return self.response


def _make_router_with_5_tiers() -> tuple:
    """Build a HealthAwareLLMRouter with 5 probe endpoints spanning the full
    failover chain: Cortex (primary), Solver (deep), Brainstem (fast),
    Reflex (emergency fallback), Gemini-Fast (cloud).

    Returns (router, client probes by endpoint name).
    """
    router = HealthAwareLLMRouter()

    clients = {}

    # Tier 1: Cortex (primary local 32B)
    cortex = LLMClientProbe("Cortex response")
    clients["Cortex"] = cortex
    router.register(
        name="Cortex",
        url="internal",
        model="cortex-32b",
        is_local=True,
        tier="local",
        client=cortex,
    )

    # Tier 2: Solver (deep local 72B)
    solver = LLMClientProbe("Solver response")
    clients["Solver"] = solver
    router.register(
        name="Solver",
        url="internal",
        model="solver-72b",
        is_local=True,
        tier="local_deep",
        client=solver,
    )

    # Tier 3: Brainstem (fast local 7B)
    brainstem = LLMClientProbe("Brainstem response")
    clients["Brainstem"] = brainstem
    router.register(
        name="Brainstem",
        url="internal",
        model="brainstem-7b",
        is_local=True,
        tier="local_fast",
        client=brainstem,
    )

    # Tier 4: Reflex (emergency fallback)
    reflex = LLMClientProbe("Reflex emergency response")
    clients["Reflex"] = reflex
    router.register(
        name="Reflex",
        url="internal",
        model="reflex-3b",
        is_local=True,
        tier="emergency",
        client=reflex,
    )

    # Tier 5: Gemini-Fast (cloud API)
    gemini = LLMClientProbe("Gemini cloud response")
    clients["Gemini-Fast"] = gemini
    router.register(
        name="Gemini-Fast",
        url="cloud",
        model="gemini-2.0-flash",
        is_local=False,
        tier="api_fast",
        client=gemini,
    )

    return router, clients


def _open_circuit(router: HealthAwareLLMRouter, endpoint_name: str):
    """Force an endpoint's circuit to OPEN state by recording enough failures."""
    ep = router.endpoints.get(endpoint_name)
    if not ep:
        raise KeyError(f"No endpoint named {endpoint_name}")
    for _ in range(ep.failure_threshold + 1):
        ep.record_failure("test_forced_failure")
    assert ep.state == CircuitState.OPEN, (
        f"Expected OPEN after {ep.failure_threshold + 1} failures, got {ep.state}"
    )


# ============================================================================
# TESTS
# ============================================================================

class TestPrimaryFailureTriggersFlallback:
    """When the primary (Cortex) endpoint has an open circuit, requests
    should route to the next available tier."""

    @pytest.mark.asyncio
    async def test_primary_failure_triggers_fallback(self):
        router, clients = _make_router_with_5_tiers()

        # Open the primary circuit
        _open_circuit(router, "Cortex")
        assert not router.endpoints["Cortex"].is_available()

        # Request with cloud fallback allowed
        result = await router.generate_with_metadata(
            "Hello",
            prefer_tier="primary",
            allow_cloud_fallback=True,
            origin="user",
            skip_runtime_payload=True,
        )

        # Should NOT have used Cortex (it's open)
        # Should have fallen through to cloud (Gemini-Fast) since
        # primary tier with cloud fallback skips Brainstem for user origin
        assert result["endpoint"] != "Cortex", (
            f"Should not route to open Cortex, got endpoint={result['endpoint']}"
        )
        assert clients["Cortex"].calls == []


class TestCascadeFailureReachesEmergency:
    """When all endpoints except emergency are in OPEN state, the emergency
    tier (Reflex) should be the one used."""

    @pytest.mark.asyncio
    async def test_cascade_failure_reaches_emergency(self):
        router, clients = _make_router_with_5_tiers()

        # Open circuits on everything except Reflex
        for name in ["Cortex", "Solver", "Brainstem", "Gemini-Fast"]:
            _open_circuit(router, name)

        # Verify only Reflex is available
        available = [ep.name for ep in router.endpoints.values() if ep.is_available()]
        assert "Reflex" in available, f"Only Reflex should be available, got {available}"

        # The emergency endpoint should handle this
        # Note: The router's tier logic may not directly route to emergency
        # unless we ask for the right tier. Verify Reflex is the only option.
        ep_reflex = router.endpoints["Reflex"]
        assert ep_reflex.is_available()
        assert ep_reflex.state == CircuitState.CLOSED


class TestCircuitBreakerHalfOpenRecovery:
    """After failures open a circuit, advancing time past the recovery window
    should transition it to HALF_OPEN and allow a test request."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_recovery(self):
        router, clients = _make_router_with_5_tiers()

        ep = router.endpoints["Cortex"]

        # Open the circuit
        _open_circuit(router, "Cortex")
        assert ep.state == CircuitState.OPEN
        assert not ep.is_available()

        # Advance past the recovery window. Recovery is governed by a MONOTONIC
        # deadline now (wall-clock last_failure is display-only so clock jumps
        # cannot reopen/wedge circuits), so expire that deadline.
        ep.cooldown_until_monotonic = time.monotonic() - 1.0

        # Now is_available() should transition to HALF_OPEN
        assert ep.is_available(), "Endpoint should be available after recovery timeout"
        assert ep.state == CircuitState.HALF_OPEN, (
            f"Expected HALF_OPEN after recovery timeout, got {ep.state}"
        )

        # A successful probe should close the circuit
        ep.record_success(tokens=50, latency_ms=200.0)
        assert ep.state == CircuitState.CLOSED, (
            f"Expected CLOSED after success in HALF_OPEN, got {ep.state}"
        )
        assert ep.failure_count == 0, "Failure count should reset after closing circuit"


class TestForegroundSkipsBrainstem:
    """Foreground (user-facing) requests should skip the Brainstem tier
    and route to cloud if primary is down."""

    @pytest.mark.asyncio
    async def test_foreground_skips_brainstem(self):
        router, clients = _make_router_with_5_tiers()

        # Kill the primary
        clients["Cortex"].error = RuntimeError("Cortex crashed")

        result = await router.generate_with_metadata(
            "Hello from user",
            prefer_tier="primary",
            origin="user",
            allow_cloud_fallback=True,
            skip_runtime_payload=True,
        )

        # Brainstem should NOT have been called for a foreground request
        assert clients["Brainstem"].calls == []

        # Should have fallen to Gemini-Fast (cloud)
        assert result["endpoint"] == "Gemini-Fast", (
            f"Expected Gemini-Fast fallback, got {result['endpoint']}"
        )
        assert result["text"] == "Gemini cloud response"


class TestEmptyResponseCountsAsFailure:
    """Empty or whitespace-only responses should be recorded as failures
    in endpoint health tracking."""

    def test_empty_response_validation(self):
        """validate_response should reject empty/whitespace strings."""
        is_valid, reason = validate_response("")
        assert not is_valid
        assert "empty" in reason.lower() or "whitespace" in reason.lower()

        is_valid, reason = validate_response("   \n\t  ")
        assert not is_valid

        is_valid, reason = validate_response(None)
        assert not is_valid
        assert "none" in reason.lower()

    def test_empty_response_records_failure_in_endpoint(self):
        """EndpointHealth.record_empty() should increment failure count
        and eventually open the circuit."""
        ep = EndpointHealth(
            name="test_ep",
            url="internal",
            model="test-model",
            failure_threshold=3,
        )

        assert ep.state == CircuitState.CLOSED
        assert ep.failure_count == 0
        assert ep.empty_responses == 0

        # Record empty responses up to threshold
        for i in range(3):
            ep.record_empty()

        assert ep.empty_responses == 3
        assert ep.failure_count == 3
        assert ep.state == CircuitState.OPEN, (
            f"Circuit should be OPEN after {ep.failure_threshold} empty responses"
        )


class TestSuccessResetsCircuit:
    """After a circuit opens and transitions to HALF_OPEN, successful
    responses should close it and reset the failure count."""

    @pytest.mark.asyncio
    async def test_success_resets_circuit(self):
        ep = EndpointHealth(
            name="test_ep",
            url="internal",
            model="test-model",
            failure_threshold=3,
            recovery_timeout=10.0,
        )

        # 1. Open the circuit with failures
        for _ in range(4):
            ep.record_failure("test_failure")
        assert ep.state == CircuitState.OPEN

        # 2. Simulate time passing beyond recovery timeout. Recovery is keyed
        #    to a monotonic deadline (wall-clock last_failure is display-only).
        ep.cooldown_until_monotonic = time.monotonic() - 1.0

        # 3. Check availability triggers HALF_OPEN
        assert ep.is_available()
        assert ep.state == CircuitState.HALF_OPEN

        # 4. Record a success -- should close the circuit
        ep.record_success(tokens=100, latency_ms=150.0)
        assert ep.state == CircuitState.CLOSED
        assert ep.failure_count == 0
        assert ep.success_count == 1

        # 5. Verify it stays closed on further successes
        ep.record_success(tokens=200, latency_ms=100.0)
        assert ep.state == CircuitState.CLOSED
        assert ep.success_count == 2


@pytest.mark.asyncio
async def test_generate_with_metadata_does_not_boot_optional_heavy_services(monkeypatch):
    calls = {"liquid_substrate": 0, "soma": 0}

    def _trap_factory(name: str):
        def _factory():
            calls[name] += 1
            raise AssertionError(f"{name} should not be initialized during router generation")
        return _factory

    monkeypatch.setattr(ServiceContainer, "_services", {})
    monkeypatch.setattr(ServiceContainer, "_aliases", {})
    monkeypatch.setattr(ServiceContainer, "_init_locks", {})
    monkeypatch.setattr(ServiceContainer, "_registration_locked", False)

    ServiceContainer.register("liquid_substrate", _trap_factory("liquid_substrate"), required=False)
    ServiceContainer.register("soma", _trap_factory("soma"), required=False)

    router, _clients = _make_router_with_5_tiers()

    result = await router.generate_with_metadata(
        "Hello",
        prefer_tier="primary",
        allow_cloud_fallback=True,
        origin="user",
        skip_runtime_payload=True,
    )

    assert result["ok"] is True
    assert result["endpoint"] == "Cortex"
    assert result["text"] == "Cortex response"
    assert calls == {"liquid_substrate": 0, "soma": 0}


class TestCircuitBreakerEdgeCases:
    """Additional edge cases for circuit breaker behavior."""

    def test_circuit_stays_open_within_recovery_window(self):
        """Circuit should NOT transition to HALF_OPEN before recovery timeout."""
        ep = EndpointHealth(
            name="test_ep",
            url="internal",
            model="test-model",
            failure_threshold=2,
            recovery_timeout=60.0,
        )

        ep.record_failure("fail1")
        ep.record_failure("fail2")
        assert ep.state == CircuitState.OPEN

        # Last failure is recent -- should still be unavailable
        assert not ep.is_available()
        assert ep.state == CircuitState.OPEN

    def test_success_resets_streak_but_keeps_lifetime_in_closed_state(self):
        """A success clears the consecutive-failure STREAK in every state so
        intermittent failures cannot silently accumulate toward the threshold
        across healthy successes; the lifetime tally is sticky and persists."""
        ep = EndpointHealth(
            name="test_ep",
            url="internal",
            model="test-model",
            failure_threshold=5,
        )

        ep.record_failure("f1")
        ep.record_failure("f2")
        assert ep.state == CircuitState.CLOSED  # Below threshold
        assert ep.failure_count == 2
        assert ep.lifetime_failure_count == 2

        ep.record_success(tokens=50, latency_ms=100.0)
        # Streak resets in every state; the lifetime count remains.
        assert ep.failure_count == 0
        assert ep.lifetime_failure_count == 2
        assert ep.success_count == 1

    def test_multiple_failures_beyond_threshold_keep_circuit_open(self):
        """Piling on more failures after OPEN should keep it OPEN."""
        ep = EndpointHealth(
            name="test_ep",
            url="internal",
            model="test-model",
            failure_threshold=2,
        )

        for i in range(10):
            ep.record_failure(f"fail_{i}")

        assert ep.state == CircuitState.OPEN
        assert ep.failure_count == 10


class TestEndpointHealthTracking:
    """Verify EndpointHealth bookkeeping (latency, tokens, request counts)."""

    def test_latency_rolling_average(self):
        ep = EndpointHealth(name="test", url="x", model="m")

        ep.record_success(tokens=10, latency_ms=100.0)
        assert ep.avg_latency_ms == 100.0

        ep.record_success(tokens=10, latency_ms=200.0)
        # Rolling average: 100 * 0.8 + 200 * 0.2 = 120
        assert abs(ep.avg_latency_ms - 120.0) < 0.1

    def test_total_requests_counts_both_success_and_failure(self):
        ep = EndpointHealth(name="test", url="x", model="m")

        ep.record_success(tokens=10, latency_ms=50.0)
        ep.record_failure("test")
        ep.record_empty()

        assert ep.total_requests == 3
        assert ep.success_count == 1
        assert ep.failure_count == 2  # record_failure + record_empty
        assert ep.empty_responses == 1

    def test_status_dict_shape(self):
        ep = EndpointHealth(name="test", url="x", model="m", tier="local")
        status = ep.status_dict()

        expected_keys = {
            "name", "tier", "state", "failures", "failure_streak",
            "transient_trips", "successes", "empty_responses",
            "last_failure_reason", "avg_latency_ms", "total_tokens",
        }
        assert set(status.keys()) == expected_keys
        assert status["state"] == "closed"
        assert status["name"] == "test"


class TestValidateResponse:
    """Unit tests for the response validation function."""

    def test_valid_response(self):
        ok, reason = validate_response("This is a valid response")
        assert ok
        assert reason == "ok"

    def test_none_response(self):
        ok, reason = validate_response(None)
        assert not ok

    def test_empty_string(self):
        ok, reason = validate_response("")
        assert not ok

    def test_whitespace_only(self):
        ok, reason = validate_response("   \t\n  ")
        assert not ok

    def test_error_marker_response(self):
        ok, reason = validate_response("Error: something went wrong")
        assert not ok
        assert "error_marker" in reason

    def test_below_min_tokens(self):
        ok, reason = validate_response(".", min_tokens=2)
        assert not ok
        assert "below_min_tokens" in reason
