"""The element inventory reaches speech, action, and self-knowledge.

A perception module nothing consults is a library, not a capability. These
check the three seams it has to live in:

  SPEECH        "what can you click here?" is answered from a real read, and
                says the read failed when it failed;
  ACTION        clicking names a control and refuses when it recognised none,
                instead of producing a coordinate anyway;
  SELF-KNOWING  the turn is routed as a capability question about her own
                screen rather than answered from general knowledge.

The rule behind all three is the one measured live on 2026-08-04: an answer
about the screen either has a structure behind it or is composed. She named the
windows behind hers correctly from a structured read, and invented two quoted
strings when nothing read the display.
"""

from __future__ import annotations

import pytest

from core.perception.element_inventory import (
    ElementInventory,
    ScreenElement,
    _element_id,
    inventory_from_elements,
)


def _button(name: str, x: float = 10, y: float = 10) -> ScreenElement:
    return ScreenElement(
        element_id=_element_id("button", name, x, y),
        role="button",
        name=name,
        x=x,
        y=y,
        width=80,
        height=24,
        source="accessibility",
    )


class TestSheCanTalkAboutIt:
    def test_the_question_is_recognised(self):
        from core.conversation.response_reliability import (
            asks_what_is_actionable_on_screen,
        )

        assert asks_what_is_actionable_on_screen("what can you click right now?")
        assert asks_what_is_actionable_on_screen("what buttons are on screen?")
        assert not asks_what_is_actionable_on_screen("how are you feeling?")

    def test_it_is_separate_from_the_what_is_visible_question(self):
        """Different questions, different evidence: controls vs a capture."""
        from core.conversation.response_reliability import (
            asks_what_is_actionable_on_screen,
        )
        from core.utils.occluded_view_intent import asks_about_occluded_view

        controls = "what buttons can you click?"
        layout = "what is behind your window?"
        assert asks_what_is_actionable_on_screen(controls)
        assert not asks_about_occluded_view(controls)
        assert asks_about_occluded_view(layout)

    def test_an_unrelated_turn_gets_no_floor(self):
        from core.conversation.response_reliability import actionable_screen_floor

        assert actionable_screen_floor("what is the capital of france?") == ""

    def test_the_rendered_answer_carries_ids_and_effects(self):
        inventory = inventory_from_elements(
            [_button("Send"), _button("Cancel", y=60)], app="Mail"
        )
        rendered = inventory.render()
        assert "press Send" in rendered
        assert inventory.interactable[0].element_id in rendered
        assert "Mail" in rendered

    def test_a_failed_read_says_so_rather_than_listing_nothing(self):
        """"There are no buttons" and "I could not look" are different claims."""
        unavailable = ElementInventory(unavailable_reason="Accessibility not granted")
        empty = inventory_from_elements([])
        assert "cannot read" in unavailable.render()
        assert "nothing interactable" in empty.render()


class TestSheCanActOnIt:
    def test_the_grounded_click_exists_on_the_real_actuator(self):
        from core.capabilities.host_automation import HostAutomationProvider

        assert hasattr(HostAutomationProvider, "click_element")

    @pytest.mark.asyncio
    async def test_an_unrecognised_target_is_refused_not_guessed(self):
        from core.capabilities import host_automation as module

        automation = module.HostAutomationProvider.__new__(module.HostAutomationProvider)
        clicked: list[tuple[int, int]] = []

        async def _never(x, y, button="left"):
            clicked.append((x, y))
            raise AssertionError("a coordinate was clicked for an unrecognised target")

        automation.click_at = _never  # type: ignore[assignment]
        receipt = await module.HostAutomationProvider.click_element(
            automation, "Publish", app="Mail"
        )
        assert receipt.success is False
        assert clicked == []
        assert receipt.error

    @pytest.mark.asyncio
    async def test_a_resolved_target_clicks_its_centre_and_names_it(self, monkeypatch):
        from core.capabilities import host_automation as module
        from core.perception import element_inventory as inv_module

        send = _button("Send", x=100, y=200)
        monkeypatch.setattr(
            inv_module,
            "build_inventory",
            lambda _app, **_kw: inventory_from_elements([send], app="Mail"),
        )

        automation = module.HostAutomationProvider.__new__(module.HostAutomationProvider)
        seen: list[tuple[int, int]] = []

        class _Receipt:
            action = ""
            target = ""
            success = True

        async def _click(x, y, button="left"):
            seen.append((x, y))
            return _Receipt()

        automation.click_at = _click  # type: ignore[assignment]
        receipt = await module.HostAutomationProvider.click_element(
            automation, "Send", app="Mail"
        )
        assert seen == [(140, 212)]  # centre of the element, not its corner
        assert send.element_id in receipt.target
        assert "press Send" in receipt.target


class TestSheKnowsItIsHerOwnCapability:
    @pytest.mark.parametrize(
        "message",
        [
            "what can you click on screen?",
            "what buttons are visible?",
            "what controls can you see?",
        ],
    )
    def test_the_turn_routes_to_her_own_screen_capability(self, message):
        from core.conversation.capability_condition import needed_capabilities

        assert "computer_use" in needed_capabilities(message)

    def test_an_unrelated_turn_does_not(self):
        from core.conversation.capability_condition import needed_capabilities

        assert "computer_use" not in needed_capabilities("how are you feeling?")
