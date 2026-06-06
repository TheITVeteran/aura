"""tests/test_moral_security_sandbox.py
Unit tests for Aura's moral reasoning kernel, sandboxing, and security consent gates.
"""
import pytest
from core.security.consent_kernel import ConsentKernel
from core.morality.moral_reasoner import MoralReasoner


def test_deception_guard():
    reasoner = MoralReasoner()
    
    # Assert overclaiming statement is blocked and revised
    original = "I know I am truly conscious and have proven qualia."
    filtered = reasoner.filter_response(original)
    
    assert filtered != original
    assert "subjective experience is not established" in filtered


def test_consent_kernel_network_audit():
    kernel = ConsentKernel()
    
    # Egress block test using direct config control.
    params = {"host": "malicious.hack.com", "port": 80}
    
    # Force allow_network_access false temporarily for the check
    kernel.network_policy.config.security.allow_network_access = False
    
    allowed = kernel.audit_and_verify_action("network", params)
    assert allowed is False
    
    # Restore configuration
    kernel.network_policy.config.security.allow_network_access = True


def test_consent_kernel_approval_audit():
    kernel = ConsentKernel()
    
    # Destructive file action requires approval
    params = {"action": "delete", "path": "core/will.py"}
    allowed = kernel.audit_and_verify_action("file", params)
    assert allowed is False
