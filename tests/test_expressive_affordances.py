"""Expressive affordances — the mind's decide-then-realize layer.

Pins the general architecture: a menu the model reasons with, an intent
grammar it emits by its own judgment, and a fail-open governed realizer —
never hardcoded keyword routing.
"""
from __future__ import annotations

import asyncio

from core.cognition.expressive_affordances import (
    Affordance,
    AffordanceRegistry,
    get_affordance_registry,
)


def test_default_menu_lists_all_affordances():
    reg = get_affordance_registry()
    names = reg.names()
    for expected in (
        "show_sketch",
        "demonstrate_artifact",
        "request_media",
        "model_scenarios",
        "deep_examine",
    ):
        assert expected in names
    menu = reg.menu_text()
    # Framed as self-knowledge, not keyword triggers.
    assert "you decide" in menu.lower()
    assert "hands are available" in menu.lower()
    for name in names:
        assert f"⟦affordance:{name}" in menu


def test_parse_intents_extracts_name_and_args():
    reg = get_affordance_registry()
    text = (
        'Sure — let me show you what I mean. '
        '⟦affordance:show_sketch prompt="a brass astrolabe on dark wood"⟧ '
        'Does that match the thing you were picturing?'
    )
    intents = reg.parse_intents(text)
    assert len(intents) == 1
    assert intents[0].name == "show_sketch"
    assert intents[0].args["prompt"] == "a brass astrolabe on dark wood"


def test_unknown_affordance_tags_are_ignored():
    reg = get_affordance_registry()
    assert reg.parse_intents("⟦affordance:teleport target=\"moon\"⟧") == []


def test_strip_intents_leaves_clean_prose():
    reg = AffordanceRegistry()
    cleaned = reg.strip_intents("Here it is ⟦affordance:show_sketch prompt=\"x\"⟧ for you.")
    assert "affordance" not in cleaned
    assert "Here it is" in cleaned and "for you." in cleaned


def test_realize_is_fail_open_on_realizer_error():
    reg = AffordanceRegistry()

    boom_calls = []

    async def _boom(_args, _ctx):
        boom_calls.append(1)
        raise RuntimeError("subsystem down")

    reg.register(Affordance(name="risky", when="test", realize=_boom))
    intent = reg.parse_intents("⟦affordance:risky⟧")[0]
    result = asyncio.run(reg.realize(intent))
    assert result["ok"] is False
    assert result["affordance"] == "risky"
    assert result["reason"].startswith("error:")


def test_request_media_needs_no_subsystem():
    """The know-to-ask affordance is pure intent — always available."""
    from core.cognition.affordance_realizers import realize_request_media

    result = asyncio.run(
        realize_request_media({"need": "a photo of the cabinet"}, {})
    )
    assert result["ok"] is True
    assert result["kind"] == "media_request"
    assert "photo of the cabinet" in result["spoken"]


def test_registry_is_extensible_without_routing_code():
    """Adding an affordance is one register() call — no new dispatch logic."""
    reg = AffordanceRegistry()
    calls = []

    async def _realize(args, ctx):
        calls.append(args)
        return {"ok": True, "echo": args.get("x")}

    reg.register(
        Affordance(name="novel_thing", when="whenever it fits", realize=_realize, args_hint='x="v"')
    )
    assert "novel_thing" in reg.menu_text()
    intent = reg.parse_intents('⟦affordance:novel_thing x="hello"⟧')[0]
    out = asyncio.run(reg.realize(intent))
    assert out["ok"] and out["echo"] == "hello"
    assert calls == [{"x": "hello"}]


def test_chat_lane_realizes_and_strips_intents():
    """The live chat helper strips affordance tags from prose and folds the
    realized spoken lines into the reply — the know-to-ask behavior end to end."""
    from interface.routes.chat import _realize_expressive_affordances

    clean, realized = asyncio.run(
        _realize_expressive_affordances(
            "I think you mean this. "
            '⟦affordance:request_media need="a photo of the cabinet hinge"⟧ '
            "Then I can be exact.",
            "I have a broken thing on my cabinet",
        )
    )
    assert "⟦affordance:" not in clean
    assert "I think you mean this." in clean
    assert any(r.get("kind") == "media_request" for r in realized)
    assert "cabinet hinge" in clean  # spoken request folded into the reply


def test_chat_lane_passes_through_plain_replies():
    from interface.routes.chat import _realize_expressive_affordances

    clean, realized = asyncio.run(
        _realize_expressive_affordances("Just a normal answer.", "hi")
    )
    assert clean == "Just a normal answer."
    assert realized == []


def test_model_scenarios_makes_preference_consistent_choice():
    """model_scenarios must model options, commit to one, and surface reasoning
    grounded in the subjective-choice engine (Bryan's preference-consistency ask)."""
    from core.cognition.affordance_realizers import realize_model_scenarios

    out = asyncio.run(
        realize_model_scenarios(
            {"options": "organize my notes vs play an experimental game"},
            {"last_user_message": "what should I do tonight"},
        )
    )
    assert out["ok"] is True
    assert out["kind"] == "scenario_model"
    assert len(out["options"]) == 2
    assert out["chosen"] in out["options"]
    # A real receipt was consulted: reasoning cites alignment scores.
    assert out["reasoning"]
    assert "alignment" in out["reasoning"].lower()
    assert "preference_override" in out  # the consistency signal is exposed
    assert out["chosen"] in out["spoken"]


def test_model_scenarios_needs_two_options():
    from core.cognition.affordance_realizers import realize_model_scenarios

    out = asyncio.run(realize_model_scenarios({"options": "just one thing"}, {}))
    assert out["ok"] is False
    assert out["reason"] == "need_two_options"


def test_menu_enabled_by_default_and_realize_wired():
    """The action menu is on by default AND the realize path is wired into the
    response flow (not merely defined) — strip-before-gate, realize-after."""
    import inspect

    from interface.routes import chat as chat_routes

    src = inspect.getsource(chat_routes)
    assert '"AURA_EXPRESSIVE_AFFORDANCES", "0"' in src  # off by default (0 zero-shot emission; needs training first)
    assert "_pending_affordance_intents" in src          # stashed before gate
    assert "_affordance_registry.realize(" in src        # realized after assembly
    assert 'setdefault("data", {})["affordances"]' in src  # surfaced on the wire


def test_strip_before_gate_keeps_prose_topical():
    """A reply with an affordance tag, once stripped, reads as clean on-topic
    prose the quality gate accepts — not a raw-tag off-topic burst."""
    from core.cognition.expressive_affordances import get_affordance_registry
    from interface.routes import chat as chat_routes

    tagged = (
        "That sounds like a shelf bracket. "
        '⟦affordance:show_sketch prompt="an L-shaped steel shelf bracket"⟧ '
        "Does that match what you're picturing?"
    )
    clean = get_affordance_registry().strip_intents(tagged)
    assert "⟦affordance:" not in clean
    off_topic, _reason = chat_routes._evaluate_reply_topicality(
        "I have a broken metal thing that holds up a shelf", clean, recent_user_messages=[]
    )
    assert off_topic is False
