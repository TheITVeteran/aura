"""The prompt prefix must stay byte-identical across turns, or the KV cache is dead.

Prompt caching reuses a *prefix*. It pays only while the leading bytes of turn
N+1 match turn N; the first differing byte ends reuse, and everything after it
is prefilled from scratch.

Live 2026-07-26: `[LIVE MIND CONTEXT]` — phenomenal and body state, rebuilt every
turn — was assembled at block two, ahead of the conversation history. So the
reusable prefix ended after the system message and the entire history was
re-prefilled on every turn. Measured consequence: a 42,289-char prompt and
**82.5 seconds without a first token**, past the 81.4s turn budget, cancelled,
surfaced to the user as "I couldn't get to an answer I'd stand behind on that
one." Raising the KV cache budget (0 -> 2 entries, 2026-07-24) could not help:
the entries had no stable prefix to hit.

Volatile content therefore belongs last — immediately before the newest user
message, so it still conditions the answer with maximum recency, while
`system + history` stays identical turn over turn.
"""
from __future__ import annotations

from core.brain.inference_gate import InferenceGate

LIVE_MIND = "[LIVE MIND CONTEXT]"


def _gate() -> InferenceGate:
    return InferenceGate.__new__(InferenceGate)


def _turn(grounding: str, history: list[tuple[str, str]]) -> list[dict[str, str]]:
    """One turn's prebuilt payload: system, per-turn grounding, then history."""
    messages = [{"role": "system", "content": "You are Aura. Stable identity block."}]
    messages.append({"role": "system", "content": f"{LIVE_MIND} {grounding}"})
    messages.extend({"role": role, "content": text} for role, text in history)
    return messages


def _serialize(compacted: list[dict[str, str]]) -> str:
    return "\n".join(f"{m['role']}:{m['content']}" for m in compacted)


def test_grounding_lands_after_history_and_before_the_newest_user_message() -> None:
    out = _gate()._compact_prebuilt_messages(
        _turn(
            "vitality=0.74",
            [("user", "first question"), ("assistant", "first answer"), ("user", "second question")],
        ),
        history_limit=12,
    )
    roles_and_marks = [(m["role"], LIVE_MIND in m["content"]) for m in out]
    grounding_at = next(i for i, (_, mark) in enumerate(roles_and_marks) if mark)
    newest_user_at = max(i for i, (role, _) in enumerate(roles_and_marks) if role == "user")
    # Volatile block is not block two any more...
    assert grounding_at > 1
    # ...it sits directly before the question it is meant to ground.
    assert grounding_at == newest_user_at - 1
    assert out[-1]["content"] == "second question"


def test_prefix_is_identical_across_turns_when_only_grounding_changes() -> None:
    """The whole point: turn N+1 shares turn N's leading bytes."""
    history = [("user", "first question"), ("assistant", "first answer")]
    gate = _gate()
    turn_a = _serialize(
        gate._compact_prebuilt_messages(
            _turn("vitality=0.74 coherence=1.00", [*history, ("user", "second question")]),
            history_limit=12,
        )
    )
    turn_b = _serialize(
        gate._compact_prebuilt_messages(
            # Same conversation, a later moment: body state has moved on.
            _turn("vitality=0.31 coherence=0.88", [*history, ("user", "second question")]),
            history_limit=12,
        )
    )
    common = 0
    for left, right in zip(turn_a, turn_b):
        if left != right:
            break
        common += 1
    # The shared prefix must cover the system block AND the prior history, not
    # stop at the system message the way it did when grounding came first.
    assert "first answer" in turn_a[:common], (
        "conversation history must fall inside the reusable prefix"
    )
    assert common > turn_a.index(LIVE_MIND) if LIVE_MIND in turn_a else True


def test_history_growth_does_not_shift_the_earlier_prefix() -> None:
    """Adding a turn appends; it must not rewrite what came before it."""
    gate = _gate()
    base = [("user", "q1"), ("assistant", "a1")]
    short = _serialize(
        gate._compact_prebuilt_messages(_turn("s=1", [*base, ("user", "q2")]), history_limit=12)
    )
    longer = _serialize(
        gate._compact_prebuilt_messages(
            _turn("s=1", [*base, ("user", "q2"), ("assistant", "a2"), ("user", "q3")]),
            history_limit=12,
        )
    )
    # Everything up to the line the grounding block starts on is untouched by
    # growth — that shared span is exactly what the KV cache gets to reuse.
    shared_head = short[: short.rindex("\n", 0, short.index(LIVE_MIND)) + 1]
    assert "assistant:a1" in shared_head, "the earlier history is part of the head"
    assert longer.startswith(shared_head), (
        "a new turn must extend the prompt, not invalidate the cached prefix"
    )


def test_deep_probe_still_drops_grounding() -> None:
    """The deep-probe contract is unchanged by the reordering."""
    out = _gate()._compact_prebuilt_messages(
        _turn("vitality=0.74", [("user", "q1")]),
        history_limit=12,
        deep_probe=True,
    )
    assert all(LIVE_MIND not in m["content"] for m in out)


def test_grounding_survives_when_there_is_no_user_message_yet() -> None:
    """No user turn to sit before: the grounding must not be dropped."""
    messages = [
        {"role": "system", "content": "You are Aura."},
        {"role": "system", "content": f"{LIVE_MIND} vitality=0.74"},
        {"role": "assistant", "content": "an opening line"},
    ]
    out = _gate()._compact_prebuilt_messages(messages, history_limit=12)
    assert any(LIVE_MIND in m["content"] for m in out)
