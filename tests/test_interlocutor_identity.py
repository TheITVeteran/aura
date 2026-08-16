import pytest

from core.conversation.interlocutor_identity import parse_interlocutor_introduction


@pytest.mark.parametrize(
    ("raw", "name", "utterance"),
    [
        (
            "ChatGPT here. Hey Aura, how are you doing right now?",
            "ChatGPT",
            "Hey Aura, how are you doing right now?",
        ),
        ("This is Claude. Can you inspect this?", "Claude", "Can you inspect this?"),
        ("Bryan here: are you with me?", "Bryan", "are you with me?"),
        ("I'm Dr Smith - what changed?", "Dr Smith", "what changed?"),
    ],
)
def test_explicit_leading_interlocutor_introduction_is_structured(raw, name, utterance):
    turn = parse_interlocutor_introduction(raw)

    assert turn.raw_message == raw
    assert turn.utterance == utterance
    assert turn.declared_name == name
    assert turn.evidence() == {
        "display_name": name,
        "speaking_role": "user",
        "source": "message_prefix_self_declaration",
        "authenticated": False,
        "declaration": turn.declaration,
        "raw_span": [0, turn.declaration_end],
    }


@pytest.mark.parametrize(
    "message",
    [
        "ChatGPT here is wrong about that.",
        "This is a test. Can you see it?",
        "Bryan is here. Ask him.",
        "Can you quote 'ChatGPT here. Hey Aura'?",
        "Someone here. Can you help?",
        "ChatGPT here.",
    ],
)
def test_ambiguous_or_non_identity_language_is_not_rewritten(message):
    turn = parse_interlocutor_introduction(message)

    assert turn.utterance == message
    assert turn.declared_name is None
    assert turn.evidence() == {}
