"""The privilege matrix has to hold as an invariant, not just as a table.

These tests exist because a privilege matrix that nobody checks is a comment.
The interesting ones are not "does check_spawn return False" — they are the
structural properties that must survive somebody editing ROLE_PRIVILEGES in six
months: that trust is monotone, that no input-parsing role ever holds authority,
and that the gate cannot itself explode.
"""

from __future__ import annotations

import pytest

from core.runtime.process_privilege import (
    ROLE_PRIVILEGES,
    Privilege,
    ProcessRole,
    check_spawn,
    role_for_source,
)


def test_every_role_has_an_entry():
    """A role with no matrix entry would silently hold nothing or everything."""
    missing = [role.name for role in ProcessRole if role not in ROLE_PRIVILEGES]
    assert not missing, f"roles absent from the matrix: {missing}"


def test_input_parsing_roles_never_hold_authority_or_secrets():
    """The whole point of the split.

    A process that parses hostile input must not be able to act or to read
    credentials. If someone widens one of these rows, this fails.
    """
    forbidden = {
        Privilege.REQUEST_AUTHORITY,
        Privilege.SECRETS,
        Privilege.SPAWN_CHILDREN,
        Privilege.MODEL_WEIGHTS,
    }
    for role in (
        ProcessRole.DOCUMENT_DECODER,
        ProcessRole.WEB_CONTENT,
        ProcessRole.UNTRUSTED_CODE,
    ):
        held = ROLE_PRIVILEGES[role] & forbidden
        assert not held, f"{role.name} must not hold {sorted(p.value for p in held)}"


def test_document_decoder_cannot_reach_the_network():
    """A decoder with network access turns a malicious PDF into exfiltration."""
    assert Privilege.NETWORK not in ROLE_PRIVILEGES[ProcessRole.DOCUMENT_DECODER]


def test_model_worker_cannot_reach_the_network():
    """A model that can reach the network can be made to exfiltrate its context."""
    assert Privilege.NETWORK not in ROLE_PRIVILEGES[ProcessRole.MODEL_WORKER]


def test_only_the_coordinator_may_spawn_children():
    spawners = [
        role.name
        for role, privs in ROLE_PRIVILEGES.items()
        if Privilege.SPAWN_CHILDREN in privs
    ]
    assert spawners == [ProcessRole.COORDINATOR.name]


def test_only_the_coordinator_holds_secrets():
    holders = [
        role.name
        for role, privs in ROLE_PRIVILEGES.items()
        if Privilege.SECRETS in privs
    ]
    assert holders == [ProcessRole.COORDINATOR.name]


def test_trust_ordering_is_not_accidentally_inverted():
    """The least-trusted role must not out-privilege the most-trusted one.

    Deliberately NOT a claim that privilege grows monotonically with rank —
    it does not, and should not: WEB_CONTENT holds NETWORK while the more
    trusted MODEL_WORKER does not, because network access is a need, not a
    reward. Asserting monotonicity would force exactly the wrong fix.
    """
    coordinator = ROLE_PRIVILEGES[ProcessRole.COORDINATOR]
    for role, privs in ROLE_PRIVILEGES.items():
        assert privs <= coordinator, (
            f"{role.name} holds privileges the coordinator does not: "
            f"{sorted(p.value for p in privs - coordinator)}"
        )


@pytest.mark.parametrize(
    "source,expected",
    [
        ("pdf_extract_worker", ProcessRole.DOCUMENT_DECODER),
        ("browser_worker", ProcessRole.WEB_CONTENT),
        ("mlx_model_server", ProcessRole.MODEL_WORKER),
        ("external_action:shell", ProcessRole.TOOL_RUNNER),
        ("crash_reporter", ProcessRole.CRASH_HANDLER),
        ("sandbox_exec", ProcessRole.UNTRUSTED_CODE),
    ],
)
def test_source_labels_map_to_roles(source, expected):
    assert role_for_source(source) is expected


def test_specific_hints_win_over_general_ones():
    """'browser_screenshot_decoder' contains both 'browser' and 'decoder'.

    Ordering in _SOURCE_ROLE_HINTS decides this, so it is pinned: the decoder
    reading is the safer one, and safer must win a tie.
    """
    assert role_for_source("browser_screenshot_decoder") is ProcessRole.DOCUMENT_DECODER


def test_unknown_source_is_not_assigned_a_role():
    """Guessing a role is worse than admitting the vocabulary is incomplete."""
    assert role_for_source("some_entirely_novel_thing") is None
    assert role_for_source("") is None


def test_unknown_role_is_allowed_but_says_why():
    """An incomplete matrix must not refuse traffic it has not learned yet."""
    decision = check_spawn("some_entirely_novel_thing", {Privilege.SECRETS})
    assert decision.allowed is True
    assert decision.role is None
    assert "declare one" in decision.reason


def test_denied_decision_names_the_specific_privilege():
    """A refusal that does not say what was refused is not actionable."""
    decision = check_spawn("browser_worker", {Privilege.SECRETS, Privilege.NETWORK})
    assert decision.allowed is False
    assert decision.denied == frozenset({Privilege.SECRETS})
    # NETWORK was requested and is legitimately held, so it must not be blamed.
    assert "network" not in decision.reason
    assert "secrets" in decision.reason


def test_requesting_nothing_is_always_allowed():
    for role in ProcessRole:
        assert check_spawn("x", set(), role=role).allowed is True


def test_explicit_role_overrides_a_misleading_source():
    """A caller who knows its role must be able to say so."""
    decision = check_spawn("mlx_worker", {Privilege.SECRETS}, role=ProcessRole.COORDINATOR)
    assert decision.allowed is True


def test_decision_serialises_without_enum_leakage():
    payload = check_spawn("browser_worker", {Privilege.SECRETS}).to_dict()
    assert payload["role"] == "web_content"
    assert payload["denied"] == ["secrets"]
    assert isinstance(payload["allowed"], bool)


def test_check_spawn_never_raises_on_hostile_input():
    """A gate that can itself explode is not a gate."""
    for bad in (None, "", "\x00", "  ", "\n\n", "x" * 10_000):
        assert check_spawn(bad, {Privilege.NETWORK}) is not None
