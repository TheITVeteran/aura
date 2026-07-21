from __future__ import annotations

import pytest

from core.executive.authority_gateway import AuthorityGateway
from core.governance.recovery_authority import (
    build_internal_recovery_context,
    is_internal_recovery_context,
)


def test_allowlisted_internal_recovery_context_is_admitted():
    context = build_internal_recovery_context(
        "autopoiesis_engine",
        "heal",
        evidence={"component": "curiosity_explorer"},
    )

    assert is_internal_recovery_context("state_mutation", context)
    assert context["no_external_effects"] is True
    assert context["effect_scope"] == "internal_runtime_recovery"


def test_recovery_context_rejects_unknown_or_external_effects():
    with pytest.raises(ValueError, match="unrecognized internal recovery operation"):
        build_internal_recovery_context("unknown", "heal")

    with pytest.raises(ValueError, match="prohibited effect"):
        build_internal_recovery_context(
            "autopoiesis_engine",
            "heal",
            evidence={"network_call": True},
        )

    assert not is_internal_recovery_context(
        "state_mutation",
        {
            "source": "autopoiesis_engine",
            "recovery_operation": "heal",
            "internal_recovery_action": True,
            "effect_scope": "internal_runtime_recovery",
            "no_external_effects": True,
            "file_write": True,
        },
    )


def test_authority_gateway_derives_recovery_context_from_trusted_pair():
    recovery = AuthorityGateway._state_mutation_context(
        "adaptive_immune_system",
        "adaptive_immune_behavioral_rule",
    )
    generic = AuthorityGateway._state_mutation_context("unknown", "heal")

    assert is_internal_recovery_context("state_mutation", recovery)
    assert not is_internal_recovery_context("state_mutation", generic)
