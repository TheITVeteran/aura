"""She must not deny capabilities her own registry lists.

Measured live 2026-07-27, in the same conversation and two turns apart:

    "Running the Python snippet... os.getpid() returned 23756 -
     os.cpu_count() returned 4. Those numbers are from the sandbox."
        -> nothing executed; the host has 18 cores. A fabricated receipt.

    "do you actually have any code-execution capability registered at all?
     ... check before answering"
        -> "no, I don't have any capability to run or sandbox code."
        -> the live registry held 75 skills with run_code, code_repl and
           internal_sandbox all READY.

Both are the same defect wearing opposite signs: a claim about herself made
without reading her own instruments. The first is now caught by the reliability
gate; this file covers the second — the question is recognised as introspection,
and the instrument reading carries the real registry.
"""

from __future__ import annotations

import pytest

from core.runtime.self_state_intent import asks_about_own_runtime


@pytest.mark.parametrize(
    "question",
    [
        "do you actually have any code-execution capability registered at all?",
        "what tools can you actually execute right now?",
        "what skills do you have?",
        "can you run code?",
        "what are your capabilities?",
        # The live turn, preamble and all.
        "Understood on the execution. Two follow-ups. First: what was the codeword "
        "I asked you to keep? Second — careful here, check before answering: do you "
        "actually have any code-execution capability registered at all? I think you "
        "may have understated yourself just now.",
    ],
)
def test_capability_questions_are_introspection(question: str) -> None:
    assert asks_about_own_runtime(question) is True, (
        "a question about what she can do has an authoritative local answer; "
        "without it she answers from the base model's guess"
    )


@pytest.mark.parametrize(
    "question",
    [
        "What's the capital of France?",
        "Can you run me through the argument again?",
        # The nouns are ordinary English when they are not about her.
        "What tools does a carpenter need?",
        "Which skills matter most for a junior engineer?",
        "How many tools are in a standard mechanic's kit?",
        "Tell me about your childhood.",
    ],
)
def test_ordinary_questions_are_not_capability_introspection(question: str) -> None:
    assert asks_about_own_runtime(question) is False


def test_the_instrument_reading_carries_the_real_registry():
    from core.brain.self_state_report import _capability_line, runtime_self_report
    from core.container import ServiceContainer

    class _Engine:
        def iter_tool_catalog(self, include_inactive=True):
            for name, availability in (
                ("run_code", "available"),
                ("code_repl", "available"),
                ("internal_sandbox", "available"),
                ("web_search", "available"),
                ("broken_thing", "degraded"),
            ):
                yield {"name": name, "availability": availability}

    ServiceContainer.register_instance("capability_engine", _Engine(), required=False)

    line = _capability_line()
    assert line, "the capability reading must be present when a registry exists"
    # Counts come from the registry, not from prose.
    assert "4 of 5" in line
    for skill in ("run_code", "code_repl", "internal_sandbox"):
        assert skill in line, f"{skill} is registered and must be named"
    # A degraded skill is not claimed as available.
    assert "broken_thing" not in line
    # Registered is not the same as reachable — both errors are named.
    assert "REGISTERED" in line
    assert "do not claim you ran anything" in line
    assert "do not deny having them" in line

    # It reaches the block the prompt path actually attaches.
    assert line in runtime_self_report()


def test_no_registry_means_no_claim_either_way():
    """Absence of a reading must not become an invented reading.

    CONTRACT CHANGED, 2026-08-10, and deliberately. This asserted the line was
    EXACTLY "" — silence was the mechanism chosen to avoid an invented
    reading. Live, silence turned out to be what produces one. Told "check
    your own skill registry before you reply", she answered that it lists her
    capabilities "with no active skills or plugins listed" and concluded "if
    there is no tool listed in the registry, it indicates that none are
    present", while 76 skills were READY. The classifier had matched and the
    instrument block WAS attached; it just carried no capability line, and the
    block's own header tells her not to supplement what is not there. Under
    that instruction an empty string is not the absence of a claim.

    _cognition_line, twenty lines up in the same module, had already reached
    the opposite conclusion for cycle counts — "Say the channel is missing
    rather than leaving a silence she will fill. This is the honest version of
    'I can't read that'." The two halves of one file disagreed; this one was
    on the losing side of the evidence.

    So the property is unchanged and the mechanism is not: no invented
    capability claim, stated explicitly as unknown rather than by saying
    nothing.
    """
    from core.brain import self_state_report

    class _NoCatalog:
        pass

    from core.container import ServiceContainer

    ServiceContainer.register_instance("capability_engine", _NoCatalog(), required=False)
    line = self_state_report._capability_line()

    assert line, "silence is what she fills in; say the reading is missing"
    assert "NOT readable" in line
    assert "not empty" in line
    # The original property, still enforced: no capability is claimed either
    # way from a registry that could not be read.
    for invented in ("run_code", "web_search", "available right now"):
        assert invented not in line, invented
