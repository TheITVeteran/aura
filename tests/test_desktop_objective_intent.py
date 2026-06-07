from __future__ import annotations


def test_shared_desktop_objective_detector_covers_general_document_tasks() -> None:
    from core.runtime.desktop_objective_intent import looks_like_desktop_objective

    assert looks_like_desktop_objective(
        "Open a tab for Google Docs and start typing a coherent essay about climate adaptation."
    )
    assert looks_like_desktop_objective(
        "Could you open my notes app, write a timestamped summary, and save it as a PDF?"
    )
    assert looks_like_desktop_objective(
        "Open a browser window, search for climate news, and show me the articles."
    )
    assert looks_like_desktop_objective(
        "Create a local file with the draft and save it on my desktop."
    )


def test_shared_desktop_objective_detector_rejects_explanation_only_requests() -> None:
    from core.runtime.desktop_objective_intent import looks_like_desktop_objective

    assert not looks_like_desktop_objective("Can you explain Docker Compose documentation?")
    assert not looks_like_desktop_objective("What is a browser tab?")
    assert not looks_like_desktop_objective("How would you open Notes if you had to?")


def test_chat_and_voice_use_the_shared_desktop_objective_detector() -> None:
    from core.voice.voice_bridge import VoiceConversationBridge
    from interface.routes.chat import _looks_like_desktop_objective

    prompts = [
        "Open a tab for Google Docs and start typing a coherent essay about climate adaptation.",
        "Could you open my notes app and save a PDF?",
        "Can you explain Docker Compose documentation?",
    ]

    for prompt in prompts:
        assert _looks_like_desktop_objective(prompt) is VoiceConversationBridge._looks_like_desktop_objective(prompt)
