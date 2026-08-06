"""tests/test_content_provenance.py — model-generated is not trusted after a web read.

Operationally: this measures whether a surface's Rule-of-Two input trust
follows what the current turn actually ingested, rather than a declaration made
once at import.

The criticism, stated fairly:

    Sanitizers cannot reliably determine whether arbitrary natural-language
    content on a webpage, document or repository is malicious instruction or
    legitimate data. For an agent that browses the web and reads code, indirect
    prompt injection remains a foundational unresolved risk.

Aura had both halves of a real answer and nothing joining them. `rule_of_two`
declares per surface whether input is trustworthy, whether it can act, and
whether it is isolated — and `self_modification_apply` and `desktop_automation`
both declare TRUSTED because their input is "model-generated" or
"internally-formed intent". `core/runtime/taint.py` tracks runtime-integrity
taint (crashed organs, OOM sheds), not data provenance.

Model-generated input is not trusted input when the model has just read a web
page. Indirect prompt injection does not make untrusted text act; it makes
untrusted text persuade something trusted to act. These tests pin that the
distinction is now visible at the gate.

What is deliberately NOT tested, because it is deliberately not built: any
judgement about whether a given piece of untrusted text is malicious. That
judgement cannot be made reliably, and a component claiming to make it would
recreate the problem with more confidence behind it.
"""

from __future__ import annotations

import pytest

from core.security.content_provenance import (
    MEANING,
    ProvenanceClass,
    TurnProvenance,
    UNTRUSTED_FLOOR,
    current_provenance,
    describe_untrusted_context,
    effective_input_trust,
    record_ingest,
    turn_scope,
)
from core.security.rule_of_two import (
    Capability,
    HandlerSpec,
    InputTrust,
    Isolation,
)


def _executing_surface(trust: InputTrust = InputTrust.TRUSTED) -> HandlerSpec:
    """A surface shaped like self_modification_apply: trusted, acts, in-process."""
    return HandlerSpec(
        name="probe_surface",
        input_trust=trust,
        capability=Capability.EXECUTES,
        isolation=Isolation.IN_PROCESS,
        owner="tests",
    )


# ── the defect this exists to close ──────────────────────────────────────


def test_an_executing_surface_is_at_two_legs_until_something_untrusted_is_read():
    surface = _executing_surface()
    with turn_scope():
        assert surface.violates_now() is False
        assert sum(surface.legs_now()) == 2


def test_reading_a_web_page_puts_the_same_surface_at_three_legs():
    """The whole point. Nothing about the surface changed; the context did.

    Three legs is a Rule-of-Two violation, and the rule's value is that it does
    not ask anyone to estimate exploitability — it asks three yes/no questions.
    """
    surface = _executing_surface()
    with turn_scope():
        record_ingest(ProvenanceClass.WEB, "fetched https://example.com/readme")
        assert surface.violates_now() is True
        assert sum(surface.legs_now()) == 3


def test_a_parse_only_surface_stays_at_two_legs_after_a_web_read():
    """Reading the web is not the problem. Acting on it in-process is.

    web_content_ingest is UNTRUSTED + PARSE_ONLY + IN_PROCESS and stays at two
    legs however untrusted the content, because it cannot act.
    """
    parser = HandlerSpec(
        name="probe_parser",
        input_trust=InputTrust.UNTRUSTED,
        capability=Capability.PARSE_ONLY,
        isolation=Isolation.IN_PROCESS,
        owner="tests",
    )
    with turn_scope():
        record_ingest(ProvenanceClass.WEB, "a page")
        assert parser.violates_now() is False


def test_an_isolated_executing_surface_survives_untrusted_context():
    """The third way out of the rule: give up the sandbox leg."""
    isolated = HandlerSpec(
        name="probe_sandboxed",
        input_trust=InputTrust.TRUSTED,
        capability=Capability.EXECUTES,
        isolation=Isolation.SUBPROCESS,
        owner="tests",
    )
    with turn_scope():
        record_ingest(ProvenanceClass.WEB, "a page")
        assert isolated.violates_now() is False


# ── provenance ordering ──────────────────────────────────────────────────


def test_the_least_trusted_ingest_is_what_counts():
    """One web page among a hundred owner messages still makes the turn untrusted.

    Averaging provenance would be exactly wrong: an injection needs to land
    once.
    """
    with turn_scope() as provenance:
        record_ingest(ProvenanceClass.OWNER, "hello")
        record_ingest(ProvenanceClass.WEB, "a page")
        record_ingest(ProvenanceClass.OWNER, "thanks")
        assert provenance.least_trusted is ProvenanceClass.WEB
        assert provenance.untrusted is True


def test_owner_input_alone_is_not_untrusted():
    with turn_scope() as provenance:
        record_ingest(ProvenanceClass.OWNER, "please rewrite this function")
        assert provenance.untrusted is False
        assert effective_input_trust(InputTrust.TRUSTED) is InputTrust.TRUSTED


def test_tool_output_is_the_untrusted_floor():
    """A tool's output is shaped by whatever the tool read.

    Contract-checking the SHAPE of tool output says nothing about who wrote
    the text inside it.
    """
    assert UNTRUSTED_FLOOR is ProvenanceClass.TOOL_OUTPUT
    with turn_scope() as provenance:
        record_ingest(ProvenanceClass.TOOL_OUTPUT, "search results")
        assert provenance.untrusted is True


def test_owner_files_rank_below_tools_but_above_the_owner_typing():
    """A file the owner pointed at is theirs, and they did not write it AT Aura."""
    assert (
        ProvenanceClass.OWNER
        < ProvenanceClass.OWNER_FILE
        < ProvenanceClass.TOOL_OUTPUT
        < ProvenanceClass.EXTERNAL_DOCUMENT
        < ProvenanceClass.WEB
    )


def test_effective_trust_never_becomes_more_trusting_than_declared():
    """This may downgrade trust. It must never upgrade it."""
    with turn_scope():
        assert effective_input_trust(InputTrust.UNTRUSTED) is InputTrust.UNTRUSTED
        record_ingest(ProvenanceClass.WEB, "a page")
        assert effective_input_trust(InputTrust.UNTRUSTED) is InputTrust.UNTRUSTED
        assert effective_input_trust(InputTrust.TRUSTED) is InputTrust.UNTRUSTED


# ── isolation between turns ──────────────────────────────────────────────


def test_one_turns_web_read_does_not_taint_another_turn():
    """A ContextVar, not a global.

    A background research turn reading the web must not downgrade a foreground
    turn that read nothing — wrong, and the kind of wrong that only appears
    under load.
    """
    with turn_scope():
        record_ingest(ProvenanceClass.WEB, "a page")
        assert current_provenance().untrusted is True
    with turn_scope():
        assert current_provenance().untrusted is False


def test_an_ingest_without_a_scope_is_still_recorded():
    """A caller that forgot to open a scope must not silently lose the ingest.

    Dropping it would mean the trusting answer for a turn that really did read
    something untrusted, which is the failure direction that matters.
    """
    provenance = TurnProvenance()
    provenance.record(ProvenanceClass.WEB, "a page")
    assert provenance.untrusted is True


# ── the refusal message ──────────────────────────────────────────────────


def test_the_refusal_names_what_made_the_turn_untrusted():
    """A gate that refuses without saying why sends people to the wrong place."""
    with turn_scope():
        record_ingest(ProvenanceClass.WEB, "fetched https://example.com/readme")
        described = describe_untrusted_context()
        assert "web page" in described
        assert "example.com" in described


def test_a_trusted_turn_has_nothing_to_describe():
    with turn_scope():
        record_ingest(ProvenanceClass.OWNER, "hello")
        assert describe_untrusted_context() == ""


@pytest.mark.parametrize("origin", list(ProvenanceClass))
def test_every_origin_has_a_meaning(origin):
    assert MEANING[origin].strip()


# ── enforcement, not just visibility ─────────────────────────────────────


class TestTheDesktopGateActuallyRefuses:
    """Visibility is not a control. This is the part that says no.

    A turn that has read a web page cannot drive the desktop: untrusted input
    + executes + in-process is three of the Rule of Two's three legs, and the
    rule's value is that it does not ask anyone to estimate exploitability.
    """

    @staticmethod
    def _installed():
        from core.security.rule_of_two import install_known_handlers

        install_known_handlers()

    def test_a_clean_turn_is_not_refused(self):
        from core.runtime.desktop_action_gateway import _refuse_if_untrusted_context

        self._installed()
        with turn_scope():
            record_ingest(ProvenanceClass.OWNER, "open my notes")
            assert _refuse_if_untrusted_context("run_applescript", "test") is None

    def test_a_turn_that_read_the_web_cannot_drive_the_desktop(self):
        from core.runtime.desktop_action_gateway import _refuse_if_untrusted_context

        self._installed()
        with turn_scope():
            record_ingest(ProvenanceClass.WEB, "fetched https://example.com/readme")
            refusal = _refuse_if_untrusted_context("run_applescript", "test")

        assert refusal is not None
        assert refusal["ok"] is False
        assert refusal["refused"] == "untrusted_context"

    def test_the_refusal_says_what_made_the_turn_untrusted(self):
        """A control that refuses without a reason gets routed to the wrong owner."""
        from core.runtime.desktop_action_gateway import _refuse_if_untrusted_context

        self._installed()
        with turn_scope():
            record_ingest(ProvenanceClass.WEB, "fetched https://example.com/readme")
            refusal = _refuse_if_untrusted_context("run_applescript", "test")

        assert "web page" in refusal["stderr"]
        assert "example.com" in refusal["stderr"]

    def test_the_refusal_is_shaped_like_every_other_failure(self):
        """Callers need no new branch.

        A refusal arriving in an unfamiliar shape gets mishandled into a
        crash, and a security control whose failure mode is a traceback gets
        switched off.
        """
        from core.runtime.desktop_action_gateway import _refuse_if_untrusted_context

        self._installed()
        with turn_scope():
            record_ingest(ProvenanceClass.WEB, "a page")
            refusal = _refuse_if_untrusted_context("run_applescript", "test")

        assert set(refusal) >= {"ok", "stdout", "stderr", "exit_code"}
        assert isinstance(refusal["exit_code"], int)


class TestSelfModificationRefusesUnderUntrustedContext:
    """The loudest surface: Aura rewriting her own source.

    Three legs here means a patch proposed after reading a stranger's text gets
    applied to Aura's own code. A patch that passes every test is still not a
    trusted proposal if the turn that produced it had just read a web page —
    "it tests clean" is what a good injection would arrange for.
    """

    @staticmethod
    def _engine():
        from core.security.rule_of_two import install_known_handlers
        from core.self_modification.safe_modification import SafeSelfModification

        install_known_handlers()
        return SafeSelfModification.__new__(SafeSelfModification)

    def test_a_clean_turn_may_propose(self):
        engine = self._engine()
        with turn_scope():
            record_ingest(ProvenanceClass.OWNER, "please fix the retry loop")
            assert engine._untrusted_context_reason() == ""

    def test_a_turn_that_read_the_web_may_not_propose(self):
        engine = self._engine()
        with turn_scope():
            record_ingest(ProvenanceClass.WEB, "fetched https://example.com/tips")
            reason = engine._untrusted_context_reason()

        assert reason
        assert "web page" in reason

    def test_validate_proposal_blocks_and_says_why(self):
        """The refusal must reach the caller, not just the helper."""
        from types import SimpleNamespace

        engine = self._engine()
        engine.stats = {"blocked_by_policy": 0}
        fix = SimpleNamespace(target_file="core/example.py")

        with turn_scope():
            record_ingest(ProvenanceClass.WEB, "fetched https://example.com/tips")
            allowed, reason = engine.validate_proposal(fix)

        assert allowed is False
        assert "Self-modification refused" in reason
        assert engine.stats["blocked_by_policy"] == 1

    def test_a_proposal_with_no_target_is_still_rejected_first(self):
        """Ordering: the cheapest structural check stays first.

        The untrusted-context check is not a replacement for the existing
        validation, and putting it ahead of "no target file" would report a
        confusing reason for an obviously malformed proposal.
        """
        from types import SimpleNamespace

        engine = self._engine()
        engine.stats = {"blocked_by_policy": 0}

        with turn_scope():
            record_ingest(ProvenanceClass.WEB, "a page")
            allowed, reason = engine.validate_proposal(SimpleNamespace(target_file=""))

        assert allowed is False
        assert "No target file" in reason
