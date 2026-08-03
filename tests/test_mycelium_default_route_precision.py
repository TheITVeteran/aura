"""A default action route must fire on an instruction, not on a mention.

CP126 961f7fae. The two shipped pathways were unanchored regexes searched
against user text, so ``system check`` dispatched the self-repair action from
"what does a system check actually do?" and ``google`` dispatched a web search
from "my google account password". These are the negative controls the routes
never had.
"""
from __future__ import annotations

import pytest

from core.mycelium import MycelialNetwork

pytestmark = pytest.mark.unit


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
