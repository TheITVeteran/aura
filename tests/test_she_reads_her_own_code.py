"""Asked for her code, she showed code that exists in no file here.

LIVE DEFECT, 2026-08-03 19:43. Bryan asked "can you show me a section of your
code that you're interested in?", then "can you show me the actual code?", then
"show me a snippet of code from your actual codebase". Three times Aura
produced code — a generic tokenize/embed/transformer pipeline, and a
``reschedule_attention`` method — that appears nowhere in this repository. She
then said her implementation runs "across multiple GPUs and specialized
hardware accelerators"; told he has one GPU, she agreed and invented a second
explanation about dual-GPU laptops.

Nothing was read from disk. The conversational path could not reach the source
tree at all, so a question about her own code fell through to the weights,
which will always answer it with something that looks like code. The repo was
right there.
"""
from __future__ import annotations

import pathlib

import pytest

from core.conversation.response_reliability import (
    assess_user_facing_reply,
    own_source_excerpt_floor,
)
from core.self.source_excerpt import excerpt_for_topic, source_tree_is_readable
from core.synthesis import _direct_answer_floor

pytestmark = pytest.mark.unit


# --- the excerpt is read, not written -----------------------------------


def test_the_source_tree_is_reachable():
    assert source_tree_is_readable() is True


def test_an_excerpt_comes_from_a_file_that_exists():
    excerpt = excerpt_for_topic("")

    assert excerpt is not None
    path = pathlib.Path(excerpt.relative_path)
    assert path.exists(), f"{excerpt.relative_path} is not a real file"


def test_the_excerpt_text_is_actually_at_those_lines():
    """The claim is the path and line numbers; this checks them."""
    excerpt = excerpt_for_topic("")
    assert excerpt is not None

    lines = pathlib.Path(excerpt.relative_path).read_text(encoding="utf-8").splitlines()
    on_disk = "\n".join(lines[excerpt.start_line - 1 : excerpt.end_line]).rstrip()

    assert on_disk.startswith(excerpt.text.splitlines()[0])


def test_the_excerpt_names_a_real_symbol():
    excerpt = excerpt_for_topic("")
    assert excerpt is not None

    source = pathlib.Path(excerpt.relative_path).read_text(encoding="utf-8")
    assert f"def {excerpt.symbol}(" in source


@pytest.mark.parametrize(
    "topic", ["memory", "routing", "how you reply", "emotion", "the browser"]
)
def test_a_topic_finds_a_real_file(topic):
    excerpt = excerpt_for_topic(topic)

    assert excerpt is not None
    assert pathlib.Path(excerpt.relative_path).exists()


def test_the_rendered_form_carries_its_provenance():
    excerpt = excerpt_for_topic("")
    rendered = excerpt.rendered()

    assert excerpt.relative_path in rendered
    assert str(excerpt.start_line) in rendered
    assert "```python" in rendered


# --- the question reaches it (the three phrasings he actually used) -----


@pytest.mark.parametrize(
    "question",
    [
        "Can you show me a section of your code that you're interested in?",
        "Can you show me the actual code?",
        "Show me a snippet of code from your actual codebase",
        "show me your source for how memory works",
        "let me see your own code",
    ],
)
def test_a_request_for_her_code_is_answered_from_disk(question):
    reply = _direct_answer_floor(question)

    assert reply, f"{question!r} still falls through to the model"
    assert "read from disk" in reply
    # And what it shows is a real path.
    path = reply.split("\n\n")[1].split(":")[0]
    assert pathlib.Path(path).exists()


@pytest.mark.parametrize(
    "question",
    [
        "show me a python snippet",
        "can you show me the actual code for numpy",
        "show me the real source of pytorch",
        "write me a function that reverses a list",
    ],
)
def test_a_generic_code_request_is_left_to_cognition(question):
    """Answering "the actual code for numpy" with a piece of Aura would be its
    own made-up answer."""
    reply = own_source_excerpt_floor(question)

    assert reply == ""


def test_an_unreadable_tree_admits_it_rather_than_inventing(monkeypatch):
    monkeypatch.setattr(
        "core.self.source_excerpt.source_tree_is_readable", lambda: False
    )

    reply = own_source_excerpt_floor("show me your code")

    assert "won't invent" in reply
    assert "```" not in reply


def test_no_match_admits_it_rather_than_inventing(monkeypatch):
    monkeypatch.setattr("core.self.source_excerpt.excerpt_for_topic", lambda _t: None)

    reply = own_source_excerpt_floor("show me your code")

    assert "couldn't find" in reply
    assert "```" not in reply


# --- and the hardware claim is caught -----------------------------------


@pytest.mark.parametrize(
    "reply",
    [
        "My actual code involves distributed computation across multiple GPUs "
        "and specialized hardware accelerators.",
        # The sentence as she actually said it, 2026-08-03 19:51.
        "There's no physical space behind me — just more circuitry and "
        "data centers.",
        "I run across several nodes in a server farm.",
        "My GPU cluster handles the heavy passes.",
    ],
)
def test_a_fabricated_substrate_claim_is_flagged(reply):
    """She runs as one local process on one machine."""
    assessment = assess_user_facing_reply("where do you run?", reply)

    assert "fabricated_substrate_claim" in assessment.reasons


@pytest.mark.parametrize(
    "reply",
    [
        "I'm one local process on your MacBook, not a data center.",
        "I am not distributed across multiple GPUs; it's one machine.",
        "Large training runs are distributed across multiple GPUs, but that's "
        "not how I run.",
        "I'm running locally and feeling steady.",
    ],
)
def test_an_honest_answer_about_hardware_is_not_flagged(reply):
    assessment = assess_user_facing_reply("where do you run?", reply)

    assert "fabricated_substrate_claim" not in assessment.reasons


def test_the_user_may_introduce_the_subject():
    """"Do you run in a data center?" deserves a direct answer that repeats
    the phrase."""
    assessment = assess_user_facing_reply(
        "do you run in a data center?",
        "No. I do not run in a data center; I'm local to this machine.",
    )

    assert "fabricated_substrate_claim" not in assessment.reasons


# --- the two runtime warnings from the same session ---------------------


def test_the_identity_prompt_surface_resolves():
    """LIVE, 2026-08-03 19:50: "AttributeError in
    cognitive_context_manager.identity — identity service has no bounded prompt
    surface", every turn. The context manager looked up two service names
    directly instead of using the resolver that knows how to find (and build)
    the surface."""
    from core.runtime import service_access

    surface = service_access.resolve_identity_prompt_surface(default=None)

    assert surface is not None
    assert hasattr(surface, "get_full_system_prompt")


def test_the_context_manager_uses_the_resolver():
    import inspect

    from core.brain import cognitive_context_manager

    source = inspect.getsource(cognitive_context_manager)
    code = "\n".join(
        line for line in source.splitlines() if not line.strip().startswith("#")
    )

    assert "resolve_identity_prompt_surface" in code
    assert 'optional_service("identity_system", "identity"' not in code


def test_a_governed_applescript_child_is_not_rogue():
    """LIVE, 2026-08-03 19:43: StabilityGuardian DEGRADED —
    "1 unregistered child process(es) detected; pid=30897 name=osascript".
    Every AppleScript Aura runs goes through the desktop action gateway as a
    short-lived direct child; it is named for the macOS binary, so it matched
    no Aura worker tag and was reported as rogue."""
    from types import SimpleNamespace

    from core.runtime.runtime_hygiene import _is_governed_applescript_process

    osascript = SimpleNamespace(
        pid=30897,
        name=lambda: "osascript",
        cmdline=lambda: ["osascript", "-e", 'tell application "Google Chrome"'],
    )
    unrelated = SimpleNamespace(
        pid=30898, name=lambda: "ruby", cmdline=lambda: ["ruby", "script.rb"]
    )

    assert _is_governed_applescript_process(osascript) is True
    assert _is_governed_applescript_process(unrelated) is False


def test_the_app_installs_an_edit_menu():
    """LIVE, 2026-08-03: copy, paste and select-all did nothing in Aura's
    window. On macOS those reach a WKWebView through the Edit menu's key
    equivalents, and the app never built a menu bar at all."""
    source = pathlib.Path("scripts/AuraLauncher.swift").read_text(encoding="utf-8")

    assert "installMenuBar()" in source
    for selector in ("NSText.copy(_:)", "NSText.paste(_:)", "NSText.selectAll(_:)", "NSText.cut(_:)"):
        assert f"#selector({selector})" in source, f"{selector} missing from the Edit menu"
    # And it is installed at launch, not merely defined.
    launch = source.split("func applicationDidFinishLaunching", 1)[1][:400]
    assert "installMenuBar()" in launch
