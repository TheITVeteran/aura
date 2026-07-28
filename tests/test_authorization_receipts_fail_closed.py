"""A receipt that states no verdict has not granted permission.

CP126 fail-open class. The shape recurs, and it is always written the same
way::

    if not receipt.get("allowed", True):
        refuse()

which reads "refuse only if the receipt explicitly said no". A receipt that
said *nothing* — a partial dict, an early return, a stub, a validator that
did not understand the question — is therefore treated as approval. Absence
of a check reported as a passed check, and in
``core/runtime/network_gateway.py`` it sat on the outbound network boundary.

The distinction a boolean cannot hold is three-valued: allow, deny, and
unstated. ``.get(key, True)`` collapses unstated into allow. ``.get(key,
False)`` collapses it into deny — safe, but it makes a broken validator look
like a policy decision, so the operator learns nothing. Keeping the three
apart lets a gate fail closed AND say which of the two reasons it failed on.
"""
from __future__ import annotations

import pytest

from core.runtime.authorization_receipt import (
    ALLOW,
    DENY,
    UNSTATED,
    read_verdict,
    receipt_allows,
)


class TestOnlyAnExplicitYesAllows:
    def test_an_explicit_allow_allows(self):
        assert read_verdict({"allowed": True}).allows is True

    def test_an_explicit_deny_denies(self):
        verdict = read_verdict({"allowed": False, "reason": "blocked host"})
        assert verdict.state == DENY
        assert verdict.reason == "blocked host"

    @pytest.mark.parametrize(
        "receipt",
        [
            {},                                   # the live bug
            {"checked_at": 123, "source": "x"},   # partial receipt
            {"allowed": None},                    # explicitly undetermined
            None,
            "ok",
            [],
            42,
        ],
    )
    def test_anything_without_a_verdict_does_not_allow(self, receipt):
        verdict = read_verdict(receipt)
        assert verdict.allows is False
        assert verdict.state == UNSTATED

    def test_an_unstated_verdict_is_not_a_denial(self):
        """The reason the third state exists: a broken validator and a
        policy refusal need different responses."""
        verdict = read_verdict({})
        assert verdict.denies is False
        assert verdict.is_stated is False


class TestNegativeKeysAndContradictions:
    def test_a_denied_flag_denies(self):
        assert read_verdict({"denied": True, "reason": "exfil"}).state == DENY

    def test_a_contradictory_receipt_is_read_restrictively(self):
        """Both set is self-contradictory; at a gate the safe reading of a
        contradiction is the restrictive one."""
        assert read_verdict({"allowed": True, "denied": True}).state == DENY

    @pytest.mark.parametrize("key", ["approved", "permitted", "granted", "authorized"])
    def test_other_positive_verdict_keys_are_read(self, key):
        assert read_verdict({key: True}).state == ALLOW

    @pytest.mark.parametrize("value", ["yes", "true", "on", "allow", 1])
    def test_non_boolean_affirmatives_are_understood(self, value):
        assert read_verdict({"allowed": value}).allows is True

    @pytest.mark.parametrize("value", ["no", "false", "off", "", 0])
    def test_non_boolean_negatives_deny(self, value):
        assert read_verdict({"allowed": value}).state == DENY


class TestTheShorthand:
    def test_receipt_allows_matches_the_verdict(self):
        assert receipt_allows({"allowed": True}) is True
        assert receipt_allows({}) is False


class TestTheNetworkBoundaryUsesIt:
    def test_the_gateway_no_longer_defaults_to_permission(self):
        """Checked against EXECUTABLE code, not raw source.

        A comment explaining the removed pattern contains the pattern, so a
        substring search over the file finds it and reports the fix undone.
        """
        import ast
        import inspect

        from core.runtime import network_gateway

        tree = ast.parse(inspect.getsource(network_gateway))
        permissive = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and len(node.args) == 2
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value
            in {"allowed", "approved", "permitted", "granted", "authorized"}
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value is True
        ]
        assert not permissive, (
            "an authorization key still defaults to True at "
            + ", ".join(f"line {node.lineno}" for node in permissive)
        )

    def test_a_real_outbound_check_still_passes(self):
        """Over-refusal would be the worse bug: this must not block traffic
        the defensive runtime actually approves."""
        from core.runtime.authorization_receipt import read_verdict as rv
        from core.security.defensive_runtime import validate_outbound_network

        receipt = validate_outbound_network(
            method="GET", url="https://example.com/x", data_length=0, source="test",
        )
        assert rv(receipt).allows is True

    def test_an_unstated_preflight_blocks_the_request(self, monkeypatch):
        """The defect, driven through the gateway."""
        from core.runtime import network_gateway

        monkeypatch.setattr(
            "core.security.defensive_runtime.validate_outbound_network",
            lambda **_kwargs: {"host": "example.com"},  # no verdict at all
        )
        result = network_gateway.NetworkGateway().request(
            "GET", "https://example.com/x", source="test",
        )
        assert result["ok"] is False
        assert result["defensive_verdict"]["state"] == UNSTATED
