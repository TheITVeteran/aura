"""An owned selector's measured contents, distinct from arbitrary screen motion.

Only a confirmed reveal creates a view. Later captures can validate that same
view, but cannot create a different one. Its retained signature excludes capture
epochs, times and the positional names used to dispatch anonymous sources.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from core.cognition.procedure_binding import _object_witnesses
from core.perception.drawn_objects import MAX_OBJECTS, candidate_witness, measured_objects
from core.perception.observed_transfer import freeze_screen
from core.runtime.skill_contract import PredicateState

Box = tuple[float, float, float, float]
Source = tuple[Box, str]


class UnknownSelectorView(Enum):
    UNKNOWN = "unknown_selector_view"


UNKNOWN_SELECTOR_VIEW = UnknownSelectorView.UNKNOWN


def _box(value: Any) -> Box | None:
    if (not isinstance(value, (list, tuple)) or len(value) != 4
            or any(type(item) not in (int, float) or not math.isfinite(item) for item in value)):
        return None
    x, y, width, height = value
    if min(x, y) < 0 or min(width, height) <= 0 or x + width > 1.000001 or y + height > 1.000001:
        return None
    return tuple(value)


def _within(box: Box, container: Box) -> bool:
    x, y, width, height = box
    cx, cy, cw, ch = container
    return cx <= x and cy <= y and x + width <= cx + cw and y + height <= cy + ch


@dataclass(frozen=True)
class SelectorSignature:
    """The normalized container and sorted measured extents with local witnesses."""

    container: Box
    sources: tuple[Source, ...]

    def as_memory(self) -> dict[str, Any]:
        return {"version": 1, "container": list(self.container),
                "sources": [[list(box), witness] for box, witness in self.sources]}

    @classmethod
    def from_memory(cls, value: Any) -> SelectorSignature | None:
        if (not isinstance(value, Mapping) or type(value.get("version")) is not int or value.get("version") != 1
                or (container := _box(value.get("container"))) is None):
            return None
        raw = value.get("sources")
        if not isinstance(raw, (list, tuple)) or not 1 <= len(raw) <= MAX_OBJECTS:
            return None
        sources = []
        for source in raw:
            if (not isinstance(source, (list, tuple)) or len(source) != 2
                    or (box := _box(source[0])) is None or not _within(box, container)
                    or not isinstance(source[1], str) or len(source[1]) != 64
                    or any(letter not in "0123456789abcdef" for letter in source[1])):
                return None
            sources.append((box, source[1]))
        if len(set(sources)) != len(sources):
            return None
        return cls(container, tuple(sorted(sources)))


@dataclass(frozen=True)
class SelectorView:
    """A confirmed selector effect whose contents must still match this capture."""

    surface: str
    bounds: tuple
    epoch: str
    capture_at: float
    signature: SelectorSignature

    @classmethod
    def from_effect(cls, receipt: Any, choices: Any) -> SelectorView | None:
        if choices is None or receipt.state is not PredicateState.SATISFIED or receipt.changed is not True:
            return None
        after = receipt.after.evidence
        if (choices.opened_by != receipt.step_key or not choices.for_step
                or choices.surface != after.get("surface_id") or choices.bounds != tuple(after.get("bounds") or ())
                or choices.epoch != after.get("capture_epoch") or choices.capture_at != after.get("capture_at")):
            return None
        objects = _object_witnesses(after)
        held = dict(choices.object_witnesses)
        if not held:
            return None
        from core.agency.what_i_can_do_here import what_is_clicked

        owned = [objects.get(what_is_clicked(move)) for move in held]
        if any(witness is None or witness != held[move] for move, witness in zip(held, owned, strict=True)):
            return None
        containers = {witness[2] for witness in owned}
        if len(containers) != 1:
            return None
        container = containers.pop()
        sources = tuple(sorted((box, identity) for identity, box, enclosing in objects.values() if enclosing == container))
        signature = SelectorSignature.from_memory({"version": 1, "container": container, "sources": sources})
        return cls(choices.surface, choices.bounds, choices.epoch, choices.capture_at, signature) if signature else None

    def current(self, observation: Mapping[str, Any], clickable: Sequence[str] = ()) -> SelectorView | UnknownSelectorView:
        """Validate an existing view; changed or missing contents remain unknown."""
        screen = freeze_screen(observation)
        surface = str(observation.get("surface_id") or observation.get("scoped_to") or "")
        if (surface != self.surface or screen.bounds != self.bounds or screen.picture is None
                or not screen.settled or screen.occluded or not screen.epoch or screen.capture_at is None
                or screen.capture_at < self.capture_at
                or (screen.epoch == self.epoch and screen.capture_at != self.capture_at)):
            return UNKNOWN_SELECTOR_VIEW
        objects = measured_objects(observation)
        sources = []
        for region in objects.values():
            if _box(region.get("container_bbox")) != self.signature.container:
                continue
            box = _box(tuple(region.get(key) for key in ("x", "y", "width", "height")))
            if box is None or not _within(box, self.signature.container):
                return UNKNOWN_SELECTOR_VIEW
            sources.append((box, region["candidate_id"]))
        if tuple(sorted(sources)) != self.signature.sources:
            return UNKNOWN_SELECTOR_VIEW
        return SelectorView(self.surface, self.bounds, screen.epoch, screen.capture_at, self.signature)

    def closed_by(self, move: str, before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
        """Release a view after a named dismissal removes its measured contents.

        Missing contents alone have no authority to retire a view. The caller
        must also establish that this exact delivered input was answered.
        """
        from core.agency.what_i_can_do_here import what_is_clicked

        label = what_is_clicked(move)
        if (not label or not re.fullmatch(r"\W*(?:close|hide|dismiss|cancel|back)(?:\s+(?:panel|window|dialog|library|menu))?\W*",
                                         label, re.I)
                or not isinstance(self.current(before), SelectorView)):
            return False
        regions = [region for region in before.get("layout") or ()
                   if isinstance(region, Mapping) and str(region.get("text") or "").casefold() == label.casefold()]
        if len(regions) != 1:
            return False
        region = regions[0]
        x, y = region.get("center_x"), region.get("center_y")
        cx, cy, cw, ch = self.signature.container
        if (type(x) not in (int, float) or type(y) not in (int, float)
                or not cx <= x <= cx + cw or not cy <= y <= cy + ch):
            return False
        screen = freeze_screen(after)
        if (str(after.get("surface_id") or after.get("scoped_to") or "") != self.surface
                or screen.bounds != self.bounds or screen.picture is None or not screen.settled or screen.occluded
                or not screen.epoch or screen.epoch == before.get("_capture_epoch")
                or screen.capture_at is None or screen.capture_at <= before.get("capture_at", math.inf)):
            return False
        # The dismissed control disappears too. An emptied but still-open
        # palette retains its Close control and remains a selector.
        if any(isinstance(region, Mapping) and str(region.get("text") or "").casefold() == label.casefold()
               for region in after.get("layout") or ()):
            return False
        if any(candidate_witness(screen.picture, box) == witness for box, witness in self.signature.sources):
            return False
        return not any(_box(region.get("container_bbox")) == self.signature.container
                       for region in measured_objects(after).values())
