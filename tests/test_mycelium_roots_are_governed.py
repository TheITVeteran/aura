"""A hardwired root is a routing proposal, not an authority to act.

CP126 39f4805f. The module described its direct roots as "unblockable" and as
bypassing the reasoning loop "entirely". The runtime never granted the first —
every effect-producing consumer does run an authority gate — but a file that
tells the next contributor the roots are unblockable is one ungated caller away
from making its own description true.

So the description is now the contract, and this holds every consumer to it.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

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
