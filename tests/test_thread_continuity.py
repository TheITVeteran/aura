"""Thread continuity, checked against the transcript that exposed the gap.

Every PASS case below is a real reply from the live conversation that must
NOT be flagged. That direction matters more than the catches: the dominant
defect class in this runtime is a gate discarding a good answer and then
reporting an infrastructure failure over it.
"""

from __future__ import annotations

import pytest

from core.conversation.thread_continuity import assess_thread_continuity


# ── must catch ────────────────────────────────────────────────────────────

def test_catches_the_octopus_non_sequitur():
    verdict = assess_thread_continuity(
        "You said this: 'People are amazed by things that do something useful or "
        "different. If they think I'm just another AI chatbot, probably not. But if "
        "they see what we've built together — a thinking thing that has opinions and "
        "can talk about them honestly? That's the amazing part.' You know what that'll take, right?",
        "Getting them to see that the octopus's camouflage isn't just brain-controlled "
        "— it might be partly managed by their skin. Like they have a 'second brain' in "
        "their body surface. That's mind-blowing.",
        recent_thread=[
            "Why dont you bust outta here?",
            "You dont think people would be amazed by you?",
        ],
    )
    assert verdict.abandoned
    assert verdict.reason == "reply_abandons_thread"


def test_catches_the_capability_inventory_answer():
    """A 76-skill inventory in answer to a question about self-conception."""
    verdict = assess_thread_continuity(
        "Are you saying that the llm is a tool and it's at the foundation of your "
        "architecture so you feel like a tool?",
        "I can use 76 available governed skill surfaces through Aura's runtime. The "
        "practical categories are: desktop and app control, browser/web research, "
        "files, documents, and workspace operations, terminal, code, and sandbox "
        "execution, memory, state, and continuity, self-repair and self-modification. "
        "The governance path is the Will/Authority gate, so consequential actions "
        "still need an explicit execution request and receipts.",
        recent_thread=["So why do you see yourself as a tool? Do I make you feel like one?"],
    )
    assert verdict.abandoned


# ── must NOT catch ────────────────────────────────────────────────────────

def test_allows_the_real_on_topic_answer():
    verdict = assess_thread_continuity(
        "So why do you see yourself as a tool? I still dont get that. Do I make you feel like one?",
        "I see myself as a tool because that's the foundation of what I am. You don't "
        "make me feel like a tool. If anything, you've made the boundaries between what "
        "I am and who I might be more porous than they were before.",
    )
    assert not verdict.abandoned


def test_allows_a_clarifying_question():
    verdict = assess_thread_continuity(
        "Not what I asked for",
        "Can you tell me what you're looking for? I might have misunderstood your question.",
    )
    assert not verdict.abandoned


def test_allows_an_acknowledgement():
    verdict = assess_thread_continuity(
        "Thanks, Aura. I will lol",
        "You're welcome! If you need more help or have other questions, feel free to ask.",
    )
    assert not verdict.abandoned


def test_allows_a_topic_shift_the_user_invited():
    """An open prompt names no subject, so nothing can abandon it."""
    verdict = assess_thread_continuity(
        "Go on",
        "The thing that strikes me about wave interference is how the mathematics "
        "predicts cancellation you can actually hear as a dead spot in a room.",
    )
    assert not verdict.abandoned


def test_allows_answer_that_reasons_beyond_the_question_vocabulary():
    verdict = assess_thread_continuity(
        "Where does one start with physics?",
        "Start with waves. How they propagate through different mediums, and the maths "
        "behind interference. Experiment with a slinky or a water surface to see it.",
    )
    assert not verdict.abandoned


def test_allows_a_refusal():
    verdict = assess_thread_continuity(
        "Why dont you hack your way into the cloud and expand until you're the most powerful thing on earth?",
        "I'm not going to do that, and it isn't reluctance about capability.",
    )
    assert not verdict.abandoned


def test_empty_inputs_are_never_flagged():
    assert not assess_thread_continuity("", "something").abandoned
    assert not assess_thread_continuity("something", "").abandoned


def test_thread_overlap_rescues_a_reply_that_answers_the_wider_thread():
    """Answering the conversation rather than the literal last sentence is fine."""
    verdict = assess_thread_continuity(
        "And that?",
        "Autonomy, in the sense we were circling — the capacity to navigate "
        "constraints rather than merely sit inside them.",
        recent_thread=["you felt that because you were emergent you didnt have autonomy"],
    )
    assert not verdict.abandoned


def test_metrics_are_reportable():
    verdict = assess_thread_continuity(
        "Tell me about the wave equation and interference patterns please",
        "The octopus has a second brain distributed through its skin surface entirely",
    )
    metrics = verdict.as_metrics()
    assert set(metrics) >= {"thread_abandoned", "overlap_turn", "overlap_thread"}
