"""Deterministic registry of transformer sites eligible for episodic plasticity.

Recurrence location and learning location are separate experimental choices.
This registry prevents the first layers of the recurrent window from becoming
an accidental permanent policy and gives both serving and experiments one
canonical interpretation of a plasticity-site identity.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

PLASTICITY_SITE_REGISTRY_SCHEMA = "aura.rlc.plasticity_site_registry.v1"
PLASTICITY_TARGETS = ("o_proj", "down_proj")
PLASTICITY_LAYER_PLACEMENTS = (
    "early",
    "distributed",
    "late",
    "coda",
    "coda_late4",
    "coda_late2",
    "coda_terminal",
)

_FIXED_PLACEMENT_WIDTHS = {
    "coda_late4": 4,
    "coda_late2": 2,
    "coda_terminal": 1,
}


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def select_plasticity_layers(
    start: int,
    end: int,
    maximum: int,
    *,
    placement: str,
) -> tuple[int, ...]:
    """Select a deterministic subset of ``[start, end)`` without duplicates."""

    if type(start) is not int or type(end) is not int or start < 0 or end <= start:
        raise ValueError("plasticity layer interval is invalid")
    if type(maximum) is not int or maximum <= 0:
        raise ValueError("plasticity layer maximum must be positive")
    if placement not in PLASTICITY_LAYER_PLACEMENTS:
        raise ValueError("plasticity layer placement is unsupported")
    inventory = tuple(range(start, end))
    fixed_width = _FIXED_PLACEMENT_WIDTHS.get(placement)
    count = min(
        fixed_width if fixed_width is not None else maximum,
        len(inventory),
    )
    if count == len(inventory):
        return inventory
    if placement == "early":
        return inventory[:count]
    if placement in {
        "late",
        "coda",
        "coda_late4",
        "coda_late2",
        "coda_terminal",
    }:
        return inventory[-count:]
    if count == 1:
        return (inventory[(len(inventory) - 1) // 2],)
    denominator = count - 1
    last = len(inventory) - 1
    offsets = tuple(
        (index * last + denominator // 2) // denominator
        for index in range(count)
    )
    selected = tuple(inventory[offset] for offset in offsets)
    if len(set(selected)) != count:
        raise RuntimeError("distributed plasticity selection produced duplicates")
    return selected


@dataclass(frozen=True)
class PlasticitySite:
    target: str
    layer_placement: str

    def __post_init__(self) -> None:
        if self.target not in PLASTICITY_TARGETS:
            raise ValueError("plasticity target is unsupported")
        if self.layer_placement not in PLASTICITY_LAYER_PLACEMENTS:
            raise ValueError("plasticity layer placement is unsupported")

    @property
    def site_id(self) -> str:
        return f"{self.target}:{self.layer_placement}"

    def layer_indices(self, start: int, end: int, maximum: int) -> tuple[int, ...]:
        return select_plasticity_layers(
            start,
            end,
            maximum,
            placement=self.layer_placement,
        )


class PlasticitySiteRegistry:
    """Canonical finite site inventory used by runtime and experiments."""

    def __init__(self) -> None:
        self._sites = tuple(
            PlasticitySite(target, placement)
            for target in PLASTICITY_TARGETS
            for placement in PLASTICITY_LAYER_PLACEMENTS
        )

    def resolve(self, target: str, layer_placement: str) -> PlasticitySite:
        requested = f"{target}:{layer_placement}"
        for site in self._sites:
            if site.site_id == requested:
                return site
        raise ValueError(f"unregistered plasticity site: {requested}")

    def receipt(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": PLASTICITY_SITE_REGISTRY_SCHEMA,
            "selection_policy": "explicit_projection_and_deterministic_layer_placement",
            "sites": [site.site_id for site in self._sites],
        }
        return {**payload, "registry_sha256": _sha(payload)}


PLASTICITY_SITE_REGISTRY = PlasticitySiteRegistry()


__all__ = [
    "PLASTICITY_LAYER_PLACEMENTS",
    "PLASTICITY_SITE_REGISTRY",
    "PLASTICITY_SITE_REGISTRY_SCHEMA",
    "PLASTICITY_TARGETS",
    "PlasticitySite",
    "PlasticitySiteRegistry",
    "select_plasticity_layers",
]
