"""The screen as citable elements, and an action that refuses to guess.

Clean-room adoption of structured screen parsing (the OmniParser line of work):
give the model an inventory of interactable elements with stable ids rather
than a picture to aim at, and require an action to cite an id that exists in a
FRESH parse.

The rule this enforces is the one measured live on 2026-08-04. Asked what was
behind her window, she was exactly right — that answer came from a structured
window read. Asked to quote the visible text, she produced two plausible strings
while the display was showing nothing; an independent capture came back all
black. The difference was whether the answer had a structure behind it.

Applied to acting instead of speaking: an agent that always produces a target
will click something when it recognised nothing, and a wrong click is not a
smaller version of the right one.
"""

from __future__ import annotations

import time

import pytest

from core.perception.element_inventory import (
    INVENTORY_FRESHNESS_S,
    ElementInventory,
    ScreenElement,
    build_inventory,
    elements_from_accessibility,
    inventory_from_elements,
    merge_overlapping,
    resolve_action_target,
)


def _element(role: str, name: str, x: float, y: float, w: float = 80, h: float = 24, source: str = "accessibility") -> ScreenElement:
    from core.perception.element_inventory import _element_id

    return ScreenElement(
        element_id=_element_id(role, name, x, y),
        role=role,
        name=name,
        x=x,
        y=y,
        width=w,
        height=h,
        source=source,
    )


class TestAnElementSaysWhatActingOnItWouldDo:
    @pytest.mark.parametrize(
        ("role", "name", "expected"),
        [
            ("button", "Send", "press Send"),
            ("text field", "Search", "type into Search"),
            ("checkbox", "Remember me", "toggle Remember me"),
            ("link", "Terms", "open Terms"),
            ("slider", "Volume", "adjust Volume"),
            ("static text", "Welcome", "read Welcome"),
        ],
    )
    def test_the_description_is_the_effect_not_the_appearance(self, role, name, expected):
        assert _element(role, name, 0, 0).function() == expected

    def test_an_unlabelled_control_says_so(self):
        assert "unlabelled" in _element("button", "", 0, 0).function()

    def test_static_text_is_not_a_click_target(self):
        assert _element("static text", "Welcome", 0, 0).interactable is False
        assert _element("button", "Send", 0, 0).interactable is True

    def test_every_element_carries_its_source(self):
        assert _element("button", "Send", 0, 0).to_dict()["source"] == "accessibility"

    def test_the_rendered_line_carries_the_id_the_model_must_cite(self):
        element = _element("button", "Send", 10, 20)
        assert element.element_id in element.as_line()
        assert "press Send" in element.as_line()


class TestIdsAreStableAgainstTheScreenChanging:
    def test_the_same_control_gets_the_same_id(self):
        assert _element("button", "Send", 10, 20).element_id == _element("button", "Send", 10, 20).element_id

    def test_a_control_appearing_above_does_not_renumber_the_others(self):
        """A positional counter would make 'press e7' mean a different button."""
        send = _element("button", "Send", 10, 200)
        before = inventory_from_elements([send])
        after = inventory_from_elements([_element("button", "New", 10, 40), send])
        assert before.elements[0].element_id == after.by_id(send.element_id).element_id

    def test_different_controls_get_different_ids(self):
        assert _element("button", "Send", 0, 0).element_id != _element("button", "Cancel", 0, 0).element_id


class TestTwoSourcesSeeingOneControl:
    def test_the_duplicate_is_collapsed(self):
        ax = _element("button", "Send", 100, 100, source="accessibility")
        ocr = _element("static text", "Send", 102, 101, source="ocr")
        assert len(merge_overlapping([ax, ocr])) == 1

    def test_accessibility_wins_over_ocr(self):
        """One reports what a control IS, the other what it looks like."""
        ax = _element("button", "Send", 100, 100, source="accessibility")
        ocr = _element("static text", "Send", 102, 101, w=200, h=60, source="ocr")
        kept = merge_overlapping([ocr, ax])
        assert kept[0].source == "accessibility"
        assert kept[0].interactable is True

    def test_separate_controls_are_both_kept(self):
        a = _element("button", "Send", 0, 0)
        b = _element("button", "Cancel", 400, 400)
        assert len(merge_overlapping([a, b])) == 2


class TestResolvingATargetRefusesRatherThanGuessing:
    def _inventory(self) -> ElementInventory:
        return inventory_from_elements(
            [
                _element("button", "Send", 10, 10),
                _element("button", "Save draft", 10, 60),
                _element("text field", "Search", 10, 110),
                _element("static text", "Inbox", 10, 160),
            ],
            app="Mail",
        )

    def test_an_id_resolves(self):
        inv = self._inventory()
        target = inv.interactable[0]
        result = resolve_action_target(inv, target.element_id)
        assert result.resolved and result.element.element_id == target.element_id

    def test_an_exact_name_resolves(self):
        result = resolve_action_target(self._inventory(), "Send")
        assert result.resolved and result.element.name == "Send"

    def test_a_unique_word_resolves(self):
        result = resolve_action_target(self._inventory(), "search")
        assert result.resolved and result.element.name == "Search"

    def test_nothing_matching_is_refused(self):
        result = resolve_action_target(self._inventory(), "Publish")
        assert result.resolved is False
        assert "nothing on screen matches" in result.reason

    def test_an_ambiguous_reference_is_refused_with_the_candidates(self):
        inv = inventory_from_elements(
            [_element("button", "Save now", 10, 10), _element("button", "Save later", 10, 60)]
        )
        result = resolve_action_target(inv, "save")
        assert result.resolved is False
        assert "cite an id" in result.reason

    def test_a_non_interactable_label_is_not_a_target(self):
        result = resolve_action_target(self._inventory(), "Inbox")
        assert result.resolved is False

    def test_an_empty_reference_is_refused(self):
        assert resolve_action_target(self._inventory(), "  ").resolved is False


class TestAnInventoryGoesStale:
    def test_a_stale_inventory_refuses_every_action(self):
        """A screen changes. An old parse is a memory of one, not a reading."""
        inv = ElementInventory(
            elements=(_element("button", "Send", 10, 10),),
            captured_at=time.time() - (INVENTORY_FRESHNESS_S + 5.0),
        )
        result = resolve_action_target(inv, "Send")
        assert result.resolved is False
        assert "re-read the screen" in result.reason

    def test_a_fresh_inventory_does_not(self):
        inv = inventory_from_elements([_element("button", "Send", 10, 10)])
        assert inv.is_fresh()
        assert resolve_action_target(inv, "Send").resolved is True

    def test_an_unavailable_read_refuses_and_says_why(self):
        inv = ElementInventory(unavailable_reason="macOS has not granted Accessibility access")
        assert inv.available is False
        result = resolve_action_target(inv, "Send")
        assert result.resolved is False
        assert "Accessibility access" in result.reason
        assert "cannot read" in inv.render()


class TestReadingTheAccessibilityPayload:
    def test_a_real_payload_becomes_typed_elements(self):
        payload = {
            "ok": True,
            "app": "Mail",
            "window": "Inbox",
            "elements": [
                {"role": "button", "name": "Send", "x": 10, "y": 20, "w": 80, "h": 24},
                {"role": "text field", "name": "To", "x": 10, "y": 60, "w": 300, "h": 24},
            ],
        }
        elements = elements_from_accessibility(payload)
        assert len(elements) == 2
        assert {e.app for e in elements} == {"Mail"}
        assert all(e.source == "accessibility" for e in elements)

    @pytest.mark.parametrize(
        "payload",
        [{}, {"ok": False, "error": "no windows"}, {"ok": True, "elements": "nonsense"}, None],
    )
    def test_a_failed_read_yields_nothing_rather_than_something(self, payload):
        assert elements_from_accessibility(payload) == []

    def test_a_failed_read_produces_an_unavailable_inventory_not_an_empty_one(self):
        inv = build_inventory("Mail", reader=lambda _app: {"ok": False, "error": "no windows"})
        assert inv.available is False
        assert inv.unavailable_reason == "no windows"

    def test_a_reader_that_raises_is_recorded_not_swallowed(self):
        def boom(_app):
            raise RuntimeError("accessibility exploded")

        inv = build_inventory("Mail", reader=boom)
        assert inv.available is False
        assert "accessibility exploded" in inv.unavailable_reason

    def test_a_real_read_renders_ids_a_model_can_cite(self):
        inv = build_inventory(
            "Mail",
            reader=lambda _app: {
                "ok": True,
                "app": "Mail",
                "elements": [{"role": "button", "name": "Send", "x": 1, "y": 2, "w": 9, "h": 9}],
            },
        )
        rendered = inv.render()
        assert "press Send" in rendered
        assert inv.interactable[0].element_id in rendered
