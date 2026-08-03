"""A hardwired root is a routing proposal, and it fires on an instruction.

Two CP126 findings that are the same mistake at two levels.

39f4805f: the module described its direct roots as "unblockable" and as
bypassing the reasoning loop "entirely". The runtime never granted the first —
every effect-producing consumer does run an authority gate — but a file that
tells the next contributor the roots are unblockable is one ungated caller away
from making its own description true.

961f7fae: the two shipped action routes were unanchored regexes searched
against user text, so "system check" dispatched self-repair from "what does a
system check actually do?" and "google" dispatched a web search from "my google
account password". An action taken from a substring of a sentence that was not
a request for it is the same failure as an action taken past its gate: the
route claimed an authority the utterance never gave it.

So the description is now the contract, this holds every consumer to it, and
the negative controls the routes never had live alongside.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from core.mycelium import MycelialNetwork

pytestmark = pytest.mark.unit

#: Every module that turns a hardwired match into an effect, and the gate it
#: must pass the match through first.
_EFFECT_CONSUMERS = {
    "core/orchestrator/mixins/incoming_logic.py": "allow_direct_user_shortcut",
    "core/orchestrator/mixins/response_processing.py": "approve_response",
}

#: Consumers that only read a routing hint and produce no effect.
_ADVISORY_CONSUMERS = {"core/brain/llm_health_router.py"}


def _callers_of_match_hardwired() -> set[str]:
    """Modules that CALL it. A docstring mentioning the name is not a caller,
    so this parses rather than greps."""
    found: set[str] = set()
    for path in pathlib.Path(".").rglob("*.py"):
        text = str(path)
        if ".venv" in text or ".claude/worktrees" in text or text.startswith("tests/"):
            continue
        if path.name == "mycelium.py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "match_hardwired"
            ):
                found.add(text)
                break
    return found


def test_every_hardwired_consumer_is_accounted_for():
    """A new caller must be classified as gated or advisory, deliberately."""
    known = set(_EFFECT_CONSUMERS) | _ADVISORY_CONSUMERS

    unaccounted = _callers_of_match_hardwired() - known

    assert not unaccounted, (
        "New consumer(s) of match_hardwired must either run an authority gate "
        f"(add to _EFFECT_CONSUMERS) or produce no effect: {sorted(unaccounted)}"
    )


@pytest.mark.parametrize("path,gate", sorted(_EFFECT_CONSUMERS.items()))
def test_an_effect_producing_consumer_runs_its_authority_gate(path, gate):
    source = pathlib.Path(path).read_text(encoding="utf-8")

    assert "match_hardwired(" in source
    assert gate in source


@pytest.mark.parametrize("path", sorted(_ADVISORY_CONSUMERS))
def test_an_advisory_consumer_does_not_execute_the_pathway(path):
    """It may read the match for routing preference; it may not act on it."""
    source = pathlib.Path(path).read_text(encoding="utf-8")
    tree = ast.parse(source)

    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert "execute_skill" not in called
    assert "execute_tool" not in called


def test_the_module_no_longer_claims_its_roots_are_unblockable():
    """Behaviour-adjacent, so scan the docstrings the claim lived in rather
    than the whole file — the explanatory comment necessarily quotes the word."""
    tree = ast.parse(pathlib.Path("core/mycelium.py").read_text(encoding="utf-8"))

    module_doc = ast.get_docstring(tree) or ""
    header = module_doc.split("This file used to", 1)[0]

    assert "unblockable" not in header.lower()
    assert "bypass the LLM reasoning loop entirely" not in header


def test_the_pathway_class_says_what_direct_means():
    from core.mycelium import HardwiredPathway

    doc = (HardwiredPathway.__doc__ or "").lower()

    assert "routing proposal" in doc
    assert "governance" in doc or "authority" in doc


@pytest.fixture()
def network():
    MycelialNetwork._instance = None
    MycelialNetwork._initialized = False
    net = MycelialNetwork()
    yield net
    MycelialNetwork._instance = None
    MycelialNetwork._initialized = False


def _route(network, text):
    match = network.match_hardwired(text)
    return match[0].pathway_id if match else None


# --- it still routes what it is for ------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "search the web for tide tables",
        "look up the ferry timetable",
        "google the weather in Astoria",
        "find info on orca pods",
        "hey aura, search for tide tables",
        "Aura: look up the ferry timetable",
        "can you please look up the ferry timetable",
        "just google the weather",
        "go ahead and search the web for tide tables",
    ],
)
def test_a_search_instruction_still_routes(network, text):
    assert _route(network, text) == "direct_web_search"


@pytest.mark.parametrize(
    "text",
    [
        "run a system check",
        "diagnose yourself",
        "repair yourself",
        "hey aura, run a self-diagnostic",
        "please do a system check",
        "system check",
    ],
)
def test_a_repair_instruction_still_routes(network, text):
    assert _route(network, text) == "direct_self_repair"


# --- and no longer fires on a mention ----------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "what does a system check actually do?",
        "don't run a system check yet",
        "IT already did a system check on the laptop",
        "I'm worried a system check would wipe my settings",
        "why would repair yourself even be a command?",
        "the manual says to diagnose yourself first, but I disagree",
        "before you do anything, tell me what a system check involves",
    ],
)
def test_talking_about_repair_does_not_trigger_it(network, text):
    assert _route(network, text) != "direct_self_repair"


@pytest.mark.parametrize(
    "text",
    [
        "my google account password needs changing",
        "I don't want you to google that",
        "she said to look up the answer, but I already know it",
        "remind me why people google things instead of thinking",
        "the search the web for X pattern is a bad idea",
    ],
)
def test_talking_about_search_does_not_trigger_it(network, text):
    assert _route(network, text) != "direct_web_search"


def test_the_search_route_still_captures_its_query(network):
    """Anchoring must not eat the capture group the skill routes on."""
    pattern = network.pathways["direct_web_search"].pattern
    found = pattern.search("hey aura, please search the web for tide tables")

    assert found is not None
    assert found.group(1) == "tide tables"


def test_the_shipped_patterns_are_anchored(network):
    """The property, not one example: an unanchored action route dispatches
    from any sentence that happens to contain the verb."""
    for pathway_id in ("direct_web_search", "direct_self_repair"):
        pattern = network.pathways[pathway_id].pattern
        assert pattern.pattern.startswith("^"), pathway_id
