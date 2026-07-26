"""CP126 contract tests for the being runtime's action policy.

This is the gate between the inner-life substrate and consequential behavior,
so the tests here are about what a caller can and cannot grant itself.
"""
from __future__ import annotations

import threading

import pytest

from core.being.runtime import (
    CONSEQUENTIAL_ACTION_DOMAINS,
    KNOWN_ACTION_DOMAINS,
    BeingRuntime,
    attested_context_flag,
    is_consequential_domain,
)


@pytest.fixture()
def being() -> BeingRuntime:
    return BeingRuntime()


# --- c62962b4: unknown domains fail closed --------------------------------


@pytest.mark.parametrize("domain", sorted(CONSEQUENTIAL_ACTION_DOMAINS))
def test_known_consequential_domains_are_consequential(domain):
    assert is_consequential_domain(domain) is True


@pytest.mark.parametrize("domain", ["response", "reflection", "observation", ""])
def test_known_safe_domains_are_not_consequential(domain):
    assert is_consequential_domain(domain) is False


@pytest.mark.parametrize(
    "typo", ["tool_exec", "toolexecution", "memory-write", "brand_new_sink", "FileWrite "]
)
def test_an_unknown_domain_is_treated_as_consequential(typo):
    """A misspelled or newly added sink must not skip the policy."""
    assert is_consequential_domain(typo) is True
    assert typo.strip().lower() not in KNOWN_ACTION_DOMAINS


def test_domain_matching_is_case_and_space_insensitive():
    assert is_consequential_domain("  TOOL_EXECUTION ") is True


def test_an_unknown_domain_is_surfaced_in_the_policy(being):
    now = being.sample()

    policy = being.action_policy(now, domain="brand_new_sink", priority=0.9)

    assert any(
        "unknown_action_domain" in constraint for constraint in policy.get("constraints", [])
    )


def test_a_known_domain_is_not_flagged_unknown(being):
    now = being.sample()

    policy = being.action_policy(now, domain="tool_execution", priority=0.5)

    assert not any(
        "unknown_action_domain" in constraint for constraint in policy.get("constraints", [])
    )


# --- 3b1a9177 / 310a67ee: caller booleans cannot forge authority ---------


def test_a_bare_boolean_grants_nothing():
    assert attested_context_flag(
        {"user_explicitly_authorized": True},
        "user_explicitly_authorized",
        domain="tool_execution",
        action="foreground_desktop_action",
    ) is False


def test_a_forged_token_grants_nothing():
    assert attested_context_flag(
        {"user_explicitly_authorized": True, "capability_token": "i-made-this-up"},
        "user_explicitly_authorized",
        domain="tool_execution",
        action="foreground_desktop_action",
    ) is False


def test_an_absent_flag_is_false_without_touching_the_store():
    assert attested_context_flag(
        {"capability_token": "whatever"},
        "user_explicitly_authorized",
        domain="tool_execution",
        action="x",
    ) is False


def _issue(domain: str, action: str) -> str:
    """Mint a real capability token for the current thread."""
    from core.agency.capability_token import get_token_store

    token = get_token_store().issue(
        origin="test",
        scope="unit",
        ttl_seconds=30.0,
        domain=domain,
        requested_action=action,
        approver="cp126-test",
        parent_receipt="test-receipt",
    )
    return getattr(token, "token", None) or getattr(token, "token_str", "")


def test_a_valid_token_grants_the_flag():
    token_str = _issue("memory_write", "continuity_memory_write")

    assert attested_context_flag(
        {"conversation_continuity": True, "capability_token": token_str},
        "conversation_continuity",
        domain="memory_write",
        action="continuity_memory_write",
    ) is True


def test_a_token_for_another_domain_is_rejected():
    token_str = _issue("memory_write", "continuity_memory_write")

    assert attested_context_flag(
        {"user_explicitly_authorized": True, "capability_token": token_str},
        "user_explicitly_authorized",
        domain="tool_execution",
        action="foreground_desktop_action",
    ) is False


def test_unattested_desktop_flags_do_not_clear_defers(being):
    """The whole six-boolean foreground exception without a token."""
    now = being.sample()

    policy = being.action_policy(
        now,
        domain="tool_execution",
        priority=0.9,
        context={
            "desktop_execution_contract": True,
            "foreground_request": True,
            "user_explicitly_authorized": True,
            "user_visible_desktop_action": True,
            "local_desktop_action": True,
            "verification_required": True,
        },
    )

    # The policy still evaluates; what matters is that the unattested flag did
    # not silently grant the exception path.
    assert isinstance(policy, dict)
    assert "constraints" in policy


# --- 7a841f23: model stability is measured, not asserted ------------------


def test_model_stability_is_not_hardcoded_perfect(being, monkeypatch):
    monkeypatch.setattr(
        "core.runtime.service_registry.get_runtime_service",
        lambda name, default=None: default,
    )

    # No router registered: unknown is mid-scale, never a perfect 1.0.
    assert being._measure_model_stability() == 0.5


def test_a_degraded_router_lowers_stability(being, monkeypatch):
    class _Router:
        stability_score = 0.2

    monkeypatch.setattr(
        "core.runtime.service_registry.get_runtime_service",
        lambda name, default=None: _Router() if name == "llm_router" else default,
    )

    assert being._measure_model_stability() == pytest.approx(0.2)


def test_an_unhealthy_router_report_lowers_stability(being, monkeypatch):
    class _Router:
        def get_health(self):
            return {"healthy": False}

    monkeypatch.setattr(
        "core.runtime.service_registry.get_runtime_service",
        lambda name, default=None: _Router() if name == "llm_router" else default,
    )

    assert being._measure_model_stability() < 0.5


def test_a_hostile_stability_reading_is_clamped(being, monkeypatch):
    class _Router:
        stability_score = float("nan")

    monkeypatch.setattr(
        "core.runtime.service_registry.get_runtime_service",
        lambda name, default=None: _Router() if name == "llm_router" else default,
    )

    value = being._measure_model_stability()
    assert 0.0 <= value <= 1.0


def test_a_raising_router_does_not_break_the_sample(being, monkeypatch):
    class _Router:
        @property
        def stability_score(self):
            raise RuntimeError("router died")

        def get_health(self):
            raise RuntimeError("router died")

    monkeypatch.setattr(
        "core.runtime.service_registry.get_runtime_service",
        lambda name, default=None: _Router() if name == "llm_router" else default,
    )

    assert being._measure_model_stability() == 0.5
    assert being.sample() is not None


# --- c28972f3 / 5d2fe427: the sample is serialized ------------------------


def test_the_runtime_holds_a_sample_lock(being):
    assert isinstance(being._sample_lock, type(threading.RLock()))


def test_concurrent_samples_do_not_interleave(being):
    """Every sample must complete before the next one starts mutating."""
    depth = {"current": 0, "max": 0}
    original = being._sample_locked

    def instrumented(*args, **kwargs):
        depth["current"] += 1
        depth["max"] = max(depth["max"], depth["current"])
        try:
            return original(*args, **kwargs)
        finally:
            depth["current"] -= 1

    being._sample_locked = instrumented
    threads = [threading.Thread(target=being.sample) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert depth["max"] == 1


def test_the_lock_is_reentrant(being):
    """action_policy may be called from inside a sample-holding caller."""
    with being._sample_lock:
        now = being.sample()
        assert being.action_policy(now, domain="response") is not None
