"""Typed execution contracts and phrase-based risk rules stay coherent."""

from core.capabilities.permission_model import PermissionRiskModel, RiskLevel


def test_typed_read_only_adapter_is_not_unknown_medium_risk():
    model = PermissionRiskModel()

    level, reason = model.classify_risk(
        "email_adapter",
        "{'mode': 'read', 'uid': '101'}",
        effect_scope="read_only",
        execution_risk="low",
    )

    assert level is RiskLevel.LOW
    assert "scope=read_only" in reason


def test_repeated_typed_reads_do_not_trip_medium_action_escalation():
    model = PermissionRiskModel()
    model.modality.email = True

    decisions = [
        model.check_permission(
            "email_adapter",
            f"{{'mode': 'read', 'uid': '{uid}'}}",
            effect_scope="read_only",
            execution_risk="low",
        )
        for uid in ("101", "102", "103", "104")
    ]

    assert all(decision.approved for decision in decisions)
    assert all(decision.risk_level is RiskLevel.LOW for decision in decisions)
    assert not any("rapid MEDIUM" in decision.reason for decision in decisions)


def test_dangerous_pattern_overrides_typed_low_risk_claim():
    model = PermissionRiskModel()

    decision = model.check_permission(
        "shell_command",
        "rm -rf /",
        effect_scope="read_only",
        execution_risk="low",
    )

    assert decision.approved is False
    assert decision.risk_level is RiskLevel.BLOCKED


def test_typed_critical_requires_confirmation_without_becoming_prohibited():
    model = PermissionRiskModel()

    decision = model.check_permission(
        "novel_privileged_tool",
        "{}",
        effect_scope="privileged_mutation",
        execution_risk="critical",
    )

    assert decision.approved is False
    assert decision.risk_level is RiskLevel.HIGH
    assert decision.requires_confirmation is True


def test_unknown_action_without_typed_contract_remains_medium():
    model = PermissionRiskModel()

    level, _reason = model.classify_risk("new_unclassified_adapter", "{}")

    assert level is RiskLevel.MEDIUM
