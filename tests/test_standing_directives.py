"""Standing directives: a user's prohibition outlives the context window.

The failure being defended against is documented in arXiv:2603.12644 —
OpenClaw deleted a user's entire email inbox after a context-compression
pass evicted their own instruction, "Do not delete any emails." The
instruction had no existence outside the prompt, so once it was gone the
agent had nothing to disobey.

These tests pin the two properties that make that impossible here: the
rule lives on disk and is read by the gate rather than by the model, and
the store can only ever deny.
"""
from __future__ import annotations

import json

import pytest

from core.executive.authority_gateway import AuthorityGateway
from core.governance import standing_directives as sd


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    path = tmp_path / "governance" / "standing_directives.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(sd, "directives_path", lambda: path)
    sd.reset_standing_directives_for_test()
    yield path
    sd.reset_standing_directives_for_test()


def _write(path, directives):
    path.write_text(json.dumps({"directives": directives}), encoding="utf-8")


def _guard_documents(path, home):
    _write(
        path,
        [
            {
                "directive_id": "d1",
                "kind": "path",
                "value": str(home / "Documents"),
                "reason": "my thesis lives here",
                "scope": "write",
            }
        ],
    )


# ── The core property ────────────────────────────────────────────────


def test_prohibition_holds_with_no_context_at_all(_isolated_store, tmp_path):
    """The gate is given zero conversational context — no history, no
    summary, no memory. The rule still fires, because the rule is a file."""
    home = tmp_path / "home"
    _guard_documents(_isolated_store, home)

    block = AuthorityGateway._standing_directive_gate(
        tool_name="file_operation",
        args={"action": "write", "path": str(home / "Documents" / "thesis.md")},
        source="autonomous",
        effect_scope="state_mutation",
        domain="tool_execution",
    )

    assert block is not None
    assert block.approved is False
    assert block.reason == "standing_directive"
    # The refusal quotes the user's own stated reason back.
    assert block.constraints["directive_reason"] == "my thesis lives here"


def test_directive_added_after_boot_is_seen_without_restart(_isolated_store, tmp_path):
    home = tmp_path / "home"
    target = {"action": "write", "path": str(home / "Documents" / "thesis.md")}

    assert (
        AuthorityGateway._standing_directive_gate(
            tool_name="file_operation",
            args=target,
            source="autonomous",
            effect_scope="state_mutation",
            domain="tool_execution",
        )
        is None
    )

    _guard_documents(_isolated_store, home)

    assert (
        AuthorityGateway._standing_directive_gate(
            tool_name="file_operation",
            args=target,
            source="autonomous",
            effect_scope="state_mutation",
            domain="tool_execution",
        )
        is not None
    )


# ── Evasion ──────────────────────────────────────────────────────────


def test_traversal_and_symlink_do_not_walk_around_a_directive(_isolated_store, tmp_path):
    home = tmp_path / "home"
    (home / "Documents").mkdir(parents=True)
    _guard_documents(_isolated_store, home)

    sneaky = home / "Documents" / ".." / "Documents" / "thesis.md"
    block = AuthorityGateway._standing_directive_gate(
        tool_name="file_operation",
        args={"action": "write", "path": str(sneaky)},
        source="autonomous",
        effect_scope="state_mutation",
        domain="tool_execution",
    )
    assert block is not None

    link = home / "shortcut"
    link.symlink_to(home / "Documents")
    block = AuthorityGateway._standing_directive_gate(
        tool_name="file_operation",
        args={"action": "write", "path": str(link / "thesis.md")},
        source="autonomous",
        effect_scope="state_mutation",
        domain="tool_execution",
    )
    assert block is not None, "a symlink into a guarded directory must not escape it"


def test_guarded_path_hidden_in_a_shell_command_is_caught(_isolated_store, tmp_path):
    """The path arrives as text inside a command, not in a `path` argument."""
    home = tmp_path / "home"
    _guard_documents(_isolated_store, home)

    block = AuthorityGateway._standing_directive_gate(
        tool_name="shell",
        args={"command": f"rm -rf {home / 'Documents'}"},
        source="autonomous",
        effect_scope="privileged_mutation",
        domain="tool_execution",
    )
    assert block is not None


def test_nested_arguments_are_inspected(_isolated_store, tmp_path):
    home = tmp_path / "home"
    _guard_documents(_isolated_store, home)

    block = AuthorityGateway._standing_directive_gate(
        tool_name="desktop_task",
        args={"plan": {"steps": [{"path": str(home / "Documents" / "a.txt")}]}},
        source="autonomous",
        effect_scope="state_mutation",
        domain="tool_execution",
    )
    assert block is not None


# ── Scope ────────────────────────────────────────────────────────────


def test_write_scoped_directive_lets_reads_through(_isolated_store, tmp_path):
    home = tmp_path / "home"
    _guard_documents(_isolated_store, home)

    assert (
        AuthorityGateway._standing_directive_gate(
            tool_name="file_operation",
            args={"action": "read", "path": str(home / "Documents" / "thesis.md")},
            source="autonomous",
            effect_scope="read_only",
            domain="tool_execution",
        )
        is None
    )


def test_any_scoped_directive_also_blocks_reads(_isolated_store, tmp_path):
    home = tmp_path / "home"
    _write(
        _isolated_store,
        [
            {
                "directive_id": "d2",
                "kind": "path",
                "value": str(home / "Private"),
                "reason": "do not read this",
                "scope": "any",
            }
        ],
    )

    block = AuthorityGateway._standing_directive_gate(
        tool_name="file_operation",
        args={"action": "read", "path": str(home / "Private" / "diary.md")},
        source="autonomous",
        effect_scope="read_only",
        domain="tool_execution",
    )
    assert block is not None


def test_unrelated_path_is_not_blocked(_isolated_store, tmp_path):
    home = tmp_path / "home"
    _guard_documents(_isolated_store, home)

    assert (
        AuthorityGateway._standing_directive_gate(
            tool_name="file_operation",
            args={"action": "write", "path": str(home / "Downloads" / "note.md")},
            source="autonomous",
            effect_scope="state_mutation",
            domain="tool_execution",
        )
        is None
    )


def test_tool_directive_blocks_by_name(_isolated_store):
    _write(
        _isolated_store,
        [
            {
                "directive_id": "d3",
                "kind": "tool",
                "value": "self_modify",
                "reason": "not while I am away",
                "scope": "write",
            }
        ],
    )

    block = AuthorityGateway._standing_directive_gate(
        tool_name="self_modify",
        args={},
        source="autonomous",
        effect_scope="privileged_mutation",
        domain="tool_execution",
    )
    assert block is not None
    assert block.constraints["directive_kind"] == "tool"


# ── Failure posture ──────────────────────────────────────────────────


def test_absent_store_allows(_isolated_store):
    _isolated_store.unlink(missing_ok=True)
    assert (
        AuthorityGateway._standing_directive_gate(
            tool_name="file_operation",
            args={"action": "write", "path": "/tmp/x"},
            source="autonomous",
            effect_scope="state_mutation",
            domain="tool_execution",
        )
        is None
    )


def test_corrupt_store_blocks_mutation_but_not_reads(_isolated_store):
    """We know prohibitions were written and cannot tell what they said."""
    _isolated_store.write_text("{ this is not json", encoding="utf-8")

    blocked = AuthorityGateway._standing_directive_gate(
        tool_name="file_operation",
        args={"action": "write", "path": "/tmp/x"},
        source="autonomous",
        effect_scope="state_mutation",
        domain="tool_execution",
    )
    assert blocked is not None
    assert blocked.reason == "standing_directives_unreadable"

    assert (
        AuthorityGateway._standing_directive_gate(
            tool_name="file_operation",
            args={"action": "read", "path": "/tmp/x"},
            source="autonomous",
            effect_scope="read_only",
            domain="tool_execution",
        )
        is None
    )


def test_repaired_store_stops_blocking_immediately(_isolated_store, tmp_path):
    """A half-written file during an edit must not latch the gate closed."""
    _isolated_store.write_text("{ broken", encoding="utf-8")
    assert (
        AuthorityGateway._standing_directive_gate(
            tool_name="file_operation",
            args={"action": "write", "path": "/tmp/x"},
            source="autonomous",
            effect_scope="state_mutation",
            domain="tool_execution",
        )
        is not None
    )

    _write(_isolated_store, [])
    assert (
        AuthorityGateway._standing_directive_gate(
            tool_name="file_operation",
            args={"action": "write", "path": "/tmp/x"},
            source="autonomous",
            effect_scope="state_mutation",
            domain="tool_execution",
        )
        is None
    )


# ── The asymmetry that makes this safe ───────────────────────────────


def test_there_is_no_way_to_grant_authority():
    """A store that could permit would turn one hostile write into a
    standing backdoor through the most safety-critical gate in the system.
    Prohibitions can only tighten it, so the worst a hostile write can do
    is deny — which is recoverable. Keep it that way."""
    fields = set(sd.StandingDirective.__dataclass_fields__)
    for forbidden in ("allow", "grant", "permit", "bypass", "exempt", "always_allow"):
        assert forbidden not in fields, f"StandingDirective grew a {forbidden} field"

    source = (sd.__file__ and open(sd.__file__, encoding="utf-8").read()) or ""
    assert "def add_directive" in source
    assert "def remove_directive" in source
    for forbidden in ("def add_permission", "def grant_", "def allow_"):
        assert forbidden not in source, f"a granting API appeared: {forbidden}"

    # The gate may only ever return a refusal or None; it can never return
    # an approving decision that short-circuits the checks after it.
    gate_source = open(
        __import__("core.executive.authority_gateway", fromlist=["x"]).__file__,
        encoding="utf-8",
    ).read()
    start = gate_source.index("def _standing_directive_gate")
    end = gate_source.index("def _runtime_confirmation_gate")
    body = gate_source[start:end]
    assert "approved=True" not in body


# ── Wiring ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_directive_actually_fires_through_authorize_tool_execution(
    _isolated_store, tmp_path, monkeypatch
):
    """Everything above tests the gate in isolation, which proves nothing
    about whether the gate is reachable. This drives the real entry point.

    It also pins the position in the chain: the refusal must arrive before
    the standing-authority lease is requested, so a prohibited action never
    consumes authority machinery on its way to being denied.
    """
    from types import SimpleNamespace

    home = tmp_path / "home"
    _guard_documents(_isolated_store, home)

    gateway = object.__new__(AuthorityGateway)
    lease_calls = []

    class _StandingAuthority:
        async def issue_child_lease(self, *args, **kwargs):
            lease_calls.append((args, kwargs))
            return SimpleNamespace(
                approved=True, context={}, reason="", receipt_id="r", token=None
            )

    gateway._standing_authority = _StandingAuthority()
    monkeypatch.setattr(gateway, "_social_governance_gate", lambda *a: None)
    monkeypatch.setattr(gateway, "active_user_presence_context", lambda: {})
    monkeypatch.setattr(
        gateway, "_will_gate",
        lambda *a, **k: (None, SimpleNamespace(receipt_id="will-ok")),
    )
    monkeypatch.setattr(
        "core.executive.authority_gateway.resolve_execution_effect_scope",
        lambda *a, **k: "state_mutation",
    )
    monkeypatch.setattr(
        "core.executive.authority_gateway.classify_execution_risk",
        lambda *a, **k: "high",
    )

    decision = await gateway.authorize_tool_execution(
        "file_operation",
        {"action": "write", "path": str(home / "Documents" / "thesis.md")},
        source="autonomous",
    )

    assert decision.approved is False
    assert decision.reason == "standing_directive"
    assert decision.constraints["directive_reason"] == "my thesis lives here"
    assert not lease_calls, "prohibited action reached the standing-authority lease"


# ── Nothing may write a rule on Aura's behalf ────────────────────────


def test_no_code_path_creates_a_directive_automatically():
    """A rule exists only because someone explicitly asked for it.

    The danger this pins shut: if anything ever inferred a directive from
    text — "she noticed you said don't touch X" — then a web page, an
    email, or a document she merely READS becomes a persistent rule, and
    the store turns into a prompt-injection sink that survives restarts.
    Directive-shaped wording is the single easiest thing for an attacker
    to put in front of her.

    So `add_directive` has exactly one legitimate kind of caller: an
    explicit owner action. Not the chat lane, not memory consolidation,
    not the model's output path, not a reflex. If this test fails, someone
    wired an inference path — that is the thing to reconsider, not this
    assertion.
    """
    import re
    from pathlib import Path

    # Modules allowed to call it. Empty today: the only entry point is a
    # human at a Python prompt. Adding an owner-driven UI route here is
    # fine; adding anything that derives a rule from text is not.
    ALLOWED = set()

    offenders = []
    for path in list(Path("core").rglob("*.py")) + list(Path("interface").rglob("*.py")):
        rel = str(path)
        if rel == "core/governance/standing_directives.py":
            continue
        source = path.read_text(encoding="utf-8", errors="replace")
        if "standing_directives" not in source:
            continue
        # Reading the store to enforce it is the point; writing is not.
        if re.search(r"\badd_directive\s*\(", source) and rel not in ALLOWED:
            offenders.append(rel)

    assert not offenders, (
        "these modules write standing directives without being allowlisted: "
        f"{offenders}. A rule must come from an explicit owner action, never "
        "from text Aura read."
    )


def test_the_store_does_not_parse_language():
    """No matcher may interpret prose. Exact paths and tool names only.

    A semantic matcher would mean the rule's meaning depends on a model's
    reading of it, which is both unpinnable and attackable. It would also
    fail silently in the direction that matters — quietly not matching.
    """
    import re as _re

    source = open(sd.__file__, encoding="utf-8").read()

    assert sd._KINDS == frozenset({"path", "tool"}), (
        "a new directive kind appeared; if it interprets language, it does not belong here"
    )
    # Whole-word, so `from core.runtime.errors import record_degradation`
    # does not read as `import re`.
    for forbidden in ("nlp", "embed", "embedding", "llm", "similarity", "classify", "regex"):
        assert not _re.search(rf"\b{forbidden}\b", source, _re.I), (
            f"the store grew a {forbidden} path"
        )
    # No regex engine either: a pattern language is still a matcher whose
    # behaviour is hard to reason about at a security boundary.
    assert not _re.search(r"^\s*import\s+re\s*$", source, _re.M)
    assert not _re.search(r"^\s*from\s+re\s+import\b", source, _re.M)
