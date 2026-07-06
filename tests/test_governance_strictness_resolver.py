"""Audit finding A pinned: the four historically-independent strictness
switches now cross-reference one resolver in core.runtime.mode, and the
"AURA_MODE=production but governance not hardened" trap is detected."""
from __future__ import annotations

import importlib

import pytest

import core.runtime.mode as mode


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (
        "AURA_MODE",
        "AURA_GOVERNANCE_MODE",
        "AURA_STRICT_WILL",
        "AURA_CONTRACTS_ENFORCE",
    ):
        monkeypatch.delenv(var, raising=False)
    yield


def test_production_default_is_flagged_unhardened(monkeypatch):
    # The exact trap from the audit: capability mode says production, but Will
    # default-deny is off, so the resolver reports inconsistency.
    s = mode.governance_strictness()
    assert s.mode_claims_production is True
    assert s.strict_will is False
    assert s.hardened is False
    assert s.consistent is False
    assert "not hardened" in s.advisory.lower() or "not hardened" in s.advisory.lower()


def test_governance_production_hardens_will(monkeypatch):
    monkeypatch.setenv("AURA_GOVERNANCE_MODE", "production")
    s = mode.governance_strictness()
    assert s.strict_will is True
    assert s.governance_production is True
    assert s.hardened is True
    assert s.consistent is True


def test_strict_will_env_hardens(monkeypatch):
    monkeypatch.setenv("AURA_STRICT_WILL", "1")
    s = mode.governance_strictness()
    assert s.strict_will is True
    assert s.consistent is True  # production + hardened


def test_non_production_mode_is_consistent_without_hardening(monkeypatch):
    monkeypatch.setenv("AURA_MODE", "research")
    s = mode.governance_strictness()
    assert s.mode_claims_production is False
    assert s.consistent is True  # research need not be fail-closed
    assert s.advisory == ""


def test_will_gate_reads_the_resolver(monkeypatch):
    # core.governance.will must agree with the canonical resolver, both ways.
    from core.governance import will

    monkeypatch.setenv("AURA_GOVERNANCE_MODE", "strict")
    assert will._strict_default_deny_enabled() is True
    assert will._strict_default_deny_enabled() == mode.governance_strictness().strict_will

    monkeypatch.delenv("AURA_GOVERNANCE_MODE", raising=False)
    assert will._strict_default_deny_enabled() is False
    assert will._strict_default_deny_enabled() == mode.governance_strictness().strict_will


def test_governance_context_reads_the_resolver(monkeypatch):
    import core.governance_context as gc

    monkeypatch.setenv("AURA_GOVERNANCE_MODE", "production")
    assert gc._governance_production_active() is True
    assert gc._governance_production_active() == mode.governance_strictness().governance_production

    monkeypatch.setenv("AURA_GOVERNANCE_MODE", "strict")  # strict != production here
    assert gc._governance_production_active() is False


def test_contracts_enforce_derives_from_resolver(monkeypatch):
    # contracts snapshots _ENFORCE at import; reload under the enforcing env
    # and confirm it tracks the resolver rather than a private env read.
    monkeypatch.setenv("AURA_CONTRACTS_ENFORCE", "1")
    import core.resilience.contracts as contracts

    contracts = importlib.reload(contracts)
    try:
        assert contracts._ENFORCE is True
        assert contracts._ENFORCE == mode.governance_strictness().enforce_contracts
    finally:
        monkeypatch.delenv("AURA_CONTRACTS_ENFORCE", raising=False)
        importlib.reload(contracts)
