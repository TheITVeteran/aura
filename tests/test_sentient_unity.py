################################################################################

import asyncio
from typing import Any
import logging
from core.container import ServiceContainer
from core.service_registration import register_all_services

class MockOrchestrator:
    def __init__(self):
        self.enqueued = []
    
    def enqueue_from_thread(self, message: Any, origin: str = "user"):
        if isinstance(message, dict) and "origin" not in message:
            message["origin"] = origin
        self.enqueued.append(message)

async def main():
    orc = MockOrchestrator()
    register_all_services()
    
    print("\n--- Test 1: Dict Queue Injection ---")
    orc.enqueue_from_thread({
        "content": "A sudden rush of insight regarding quantum computing!",
        "context": {"urgency": "HIGH", "emotion": "EXCITED"}
    }, origin="impulse")
    print("Enqueued:", orc.enqueued)
    assert orc.enqueued[0]["origin"] == "impulse"
    
    print("\n--- Test 2: AffectEngine Context Extraction ---")
    affect = ServiceContainer.get("affect_engine")
    if affect:
        print(f"Current Affect State: {affect.state.dominant_emotion}")
        await affect.modify(0.8, 0.8, 0.8, source="integration_test")
        print(f"New Affect State after trigger: {affect.state.dominant_emotion}")
    
    print("\n--- Test 3: Archiver ---")
    archiver = ServiceContainer.get("archive_engine")
    if archiver:
        res = await archiver.archive_vital_logs()
        print(f"Archive Result: {res}")

if __name__ == "__main__":
    asyncio.run(main())


##


# `main()` above carries three real assertions and was never collected. The
# origin-tagging one is the load-bearing case: a thought that loses its
# origin becomes indistinguishable from something the user said, which is
# how an inner monologue reaches a person as conversation.

import pytest


def test_dict_message_keeps_the_origin_it_was_enqueued_with():
    orc = MockOrchestrator()
    orc.enqueue_from_thread(
        {"content": "A sudden rush of insight regarding quantum computing!",
         "context": {"urgency": "HIGH", "emotion": "EXCITED"}},
        origin="impulse",
    )
    assert orc.enqueued[0]["origin"] == "impulse"


def test_an_explicit_origin_is_not_overwritten():
    orc = MockOrchestrator()
    orc.enqueue_from_thread({"content": "x", "origin": "dream"}, origin="impulse")
    assert orc.enqueued[0]["origin"] == "dream"


def test_origin_defaults_to_user_when_unspecified():
    orc = MockOrchestrator()
    orc.enqueue_from_thread({"content": "x"})
    assert orc.enqueued[0]["origin"] == "user"


def test_non_dict_messages_pass_through_unchanged():
    orc = MockOrchestrator()
    orc.enqueue_from_thread("a bare string", origin="impulse")
    assert orc.enqueued[0] == "a bare string"


def test_every_enqueued_thought_is_recorded():
    orc = MockOrchestrator()
    for i in range(5):
        orc.enqueue_from_thread({"content": f"thought {i}"}, origin="impulse")
    assert len(orc.enqueued) == 5
    assert all(m["origin"] == "impulse" for m in orc.enqueued)
