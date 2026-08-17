"""Unknown trust is its own answer, and it is not "yes".

`SafeSelfModification.validate_proposal` asked one question — "did this turn
read untrusted content?" — and accepted two answers where there are three.
A provenance lookup that raised returned `""`, exactly what a clean turn
returns, and the patch went through. The docstring said so plainly: it
failed open on purpose, with a degradation recorded.

The reasoning behind that was real. A broken lookup must not stop Aura
repairing herself on a turn that read nothing. But it converts "I could not
check" into "I checked and it was fine" on the one surface where being wrong
means she rewrites her own source at a stranger's suggestion.

So there are three answers now, and unknown DEFERS rather than refuses: the
path was allowed, the patch keeps its evidence, and what is missing can be
true on the next turn.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.self_modification.safe_modification import (
    SafeSelfModification,
    TurnTrust,
    _owner_approved,
)


class _Handler:
    def __init__(self, violates):
        self._violates = violates

    def violates_now(self):
        return self._violates


class _Registry:
    def __init__(self, handler=None):
        self._handler = handler

    def get(self, _name):
        return self._handler


def _pipeline():
    return SafeSelfModification.__new__(SafeSelfModification)


def _install(monkeypatch, registry, describe=lambda: "read a web page"):
    import core.security.content_provenance as provenance
    import core.security.rule_of_two as rule_of_two

    monkeypatch.setattr(rule_of_two, "get_rule_of_two_registry", lambda: registry)
    monkeypatch.setattr(rule_of_two, "install_known_handlers", lambda: [])
    monkeypatch.setattr(provenance, "describe_untrusted_context", describe)


# ── the three answers ───────────────────────────────────────────────────────


def test_a_clean_turn_is_trusted(monkeypatch):
    _install(monkeypatch, _Registry(_Handler(violates=False)))

    assert _pipeline()._turn_trust_verdict() == TurnTrust(state="trusted", reason="")


def test_a_turn_that_read_something_is_untrusted(monkeypatch):
    _install(monkeypatch, _Registry(_Handler(violates=True)))

    verdict = _pipeline()._turn_trust_verdict()

    assert verdict.state == "untrusted"
    assert "web page" in verdict.reason


def test_a_broken_check_is_unknown_not_trusted(monkeypatch):
    """The whole defect, in one test."""
    import core.security.rule_of_two as rule_of_two

    def _explode():
        raise RuntimeError("provenance store unreachable")

    monkeypatch.setattr(rule_of_two, "get_rule_of_two_registry", _explode)

    verdict = _pipeline()._turn_trust_verdict()

    assert verdict.state == "unknown"
    assert verdict.state != "trusted"
    assert "provenance store unreachable" in verdict.reason


def test_an_uninstalled_registry_is_installed_before_it_is_called_unknown(monkeypatch):
    """An empty registry is uninstalled, not unknowable.

    The declarations are a static property of the source. Calling an empty
    registry "unknown" would defer every self-modification in any process
    that had not booted the security module — which is a different failure
    from not knowing what this turn read.
    """
    import core.security.rule_of_two as rule_of_two

    installed = []
    registry = _Registry(None)

    def _install_known():
        installed.append(True)
        registry._handler = _Handler(violates=False)
        return ["self_modification_apply"]

    monkeypatch.setattr(rule_of_two, "get_rule_of_two_registry", lambda: registry)
    monkeypatch.setattr(rule_of_two, "install_known_handlers", _install_known)

    verdict = _pipeline()._turn_trust_verdict()

    assert installed, "an empty registry was not installed before being judged"
    assert verdict.state == "trusted"


def test_a_state_outside_the_three_is_refused():
    with pytest.raises(ValueError, match="turn-trust state"):
        TurnTrust(state="probably_fine", reason="")


# ── what the verdict does to a proposal ─────────────────────────────────────


def test_owner_approval_reads_every_field_the_pipeline_honours():
    for field in ("owner_approved", "human_approved", "explicit_owner_approval"):
        assert _owner_approved(SimpleNamespace(**{field: True}))
    assert not _owner_approved(SimpleNamespace())


def test_a_deferral_says_it_is_a_deferral():
    """The caller must be able to tell "not yet" from "never"."""
    import inspect

    from core.self_modification.safe_modification import SafeSelfModification

    source = inspect.getsource(SafeSelfModification.validate_proposal)

    assert "deferred" in source.lower()
    assert "deferred_unknown_trust" in source
    # And the permanent refusals are asked first: a deferral about a
    # constitutionally protected path tells the caller to try again at
    # something that will never be allowed.
    assert source.index("constitutionally protected") < source.index("deferred_unknown_trust")
