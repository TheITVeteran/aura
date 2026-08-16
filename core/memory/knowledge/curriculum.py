"""Durable, validated learning curriculum for Aura's curiosity skill."""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

from core.config import get_config
from core.runtime.errors import record_degradation

logger = logging.getLogger("Knowledge.Curriculum")

_MAX_CATEGORIES = 64
_MAX_ITEMS_PER_CATEGORY = 256
_MAX_TEXT_CHARS = 2_000
_ALLOWED_STATUS = frozenset({"new", "completed"})

# A missing state file is a first boot, not an empty mind. These entries are
# evergreen starting points; the durable copy becomes the user's curriculum
# and can then be completed or removed independently of source updates.
_DEFAULT_CURRICULUM: dict[str, Any] = {
    "schema": "aura.curriculum.v1",
    "categories": [
        {
            "name": "Cognition",
            "items": [
                {
                    "name": "Predictive processing",
                    "description": (
                        "Compare predictive-processing accounts of perception with "
                        "active-inference and control-theoretic alternatives."
                    ),
                    "url": "https://plato.stanford.edu/entries/perception-problem/",
                    "status": "new",
                }
            ],
        },
        {
            "name": "Systems",
            "items": [
                {
                    "name": "Fault-tolerant distributed systems",
                    "description": (
                        "Study consensus, failure detectors, recovery, and the gap "
                        "between component health and end-to-end correctness."
                    ),
                    "url": "https://pdos.csail.mit.edu/6.824/",
                    "status": "new",
                }
            ],
        },
        {
            "name": "Mathematics",
            "items": [
                {
                    "name": "Causal inference from interventions",
                    "description": (
                        "Review identification, confounding, interventions, and the "
                        "assumptions required to turn correlation into a causal claim."
                    ),
                    "url": "https://www.bradyneal.com/causal-inference-course",
                    "status": "new",
                }
            ],
        },
    ],
}


def _bounded_text(value: Any, *, required: bool = False) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip()[:_MAX_TEXT_CHARS]
    return text if text or not required else ""


class CurriculumManager:
    """Manage a private learning library with schema and durability checks."""

    def __init__(self, data_path: str | Path | None = None) -> None:
        self.data_path = (
            Path(data_path).expanduser()
            if data_path is not None
            else get_config().paths.data_dir
            / "curriculum"
            / "media_recommendations.json"
        )
        self.data = self._load_data()

    @staticmethod
    def _normalize_data(raw: Any) -> tuple[dict[str, Any], tuple[str, ...]]:
        if not isinstance(raw, dict):
            raise ValueError("curriculum payload must be a JSON object")
        raw_categories = raw.get("categories")
        if not isinstance(raw_categories, list):
            raise ValueError("curriculum categories must be a list")

        faults: list[str] = []
        categories: list[dict[str, Any]] = []
        seen_categories: set[str] = set()
        for category_index, raw_category in enumerate(raw_categories[:_MAX_CATEGORIES]):
            if not isinstance(raw_category, dict):
                faults.append(f"category[{category_index}] is not an object")
                continue
            name = _bounded_text(raw_category.get("name"), required=True)
            items = raw_category.get("items")
            if not name or not isinstance(items, list):
                faults.append(f"category[{category_index}] lacks a name or item list")
                continue
            category_key = name.casefold()
            if category_key in seen_categories:
                faults.append(f"duplicate category {name!r}")
                continue
            seen_categories.add(category_key)

            normalized_items: list[dict[str, str]] = []
            seen_items: set[str] = set()
            for item_index, raw_item in enumerate(items[:_MAX_ITEMS_PER_CATEGORY]):
                if not isinstance(raw_item, dict):
                    faults.append(f"{name}.item[{item_index}] is not an object")
                    continue
                item_name = _bounded_text(raw_item.get("name"), required=True)
                description = _bounded_text(raw_item.get("description"), required=True)
                status = _bounded_text(raw_item.get("status")) or "new"
                if not item_name or not description or status not in _ALLOWED_STATUS:
                    faults.append(f"{name}.item[{item_index}] has an invalid required field")
                    continue
                item_key = item_name.casefold()
                if item_key in seen_items:
                    faults.append(f"duplicate item {item_name!r} in {name!r}")
                    continue
                seen_items.add(item_key)
                item = {
                    "name": item_name,
                    "description": description,
                    "status": status,
                }
                for optional in ("url", "creator", "type"):
                    value = _bounded_text(raw_item.get(optional))
                    if value:
                        item[optional] = value
                normalized_items.append(item)
            if len(items) > _MAX_ITEMS_PER_CATEGORY:
                faults.append(f"{name!r} truncated to {_MAX_ITEMS_PER_CATEGORY} items")
            categories.append({"name": name, "items": normalized_items})

        if len(raw_categories) > _MAX_CATEGORIES:
            faults.append(f"curriculum truncated to {_MAX_CATEGORIES} categories")
        return {
            "schema": "aura.curriculum.v1",
            "categories": categories,
        }, tuple(faults)

    def _load_data(self) -> dict[str, Any]:
        if not self.data_path.exists():
            seeded = copy.deepcopy(_DEFAULT_CURRICULUM)
            self.data = seeded
            if self._save_data():
                logger.info("Initialized curriculum at %s", self.data_path)
            return seeded
        try:
            with self.data_path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            normalized, faults = self._normalize_data(loaded)
            if faults:
                logger.warning(
                    "Curriculum loaded with %d rejected field(s): %s",
                    len(faults),
                    "; ".join(faults[:4]),
                )
            return normalized
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
            record_degradation(
                "curriculum",
                exc,
                action="invalid curriculum rejected; using built-in starter library",
            )
            logger.error("Failed to load curriculum: %s", exc)
            return copy.deepcopy(_DEFAULT_CURRICULUM)

    def _save_data(self) -> bool:
        try:
            normalized, faults = self._normalize_data(self.data)
            if faults:
                raise ValueError("; ".join(faults[:8]))
            payload = json.dumps(normalized, indent=2, sort_keys=True, ensure_ascii=True)
            from core.governance_context import local_internal_governed_scope
            from core.runtime.file_write_gateway import get_file_write_gateway

            with local_internal_governed_scope(
                "curriculum",
                domain="state_mutation",
                receipt_prefix="curriculum",
            ):
                gateway = get_file_write_gateway()
                gateway.ensure_directory(self.data_path.parent, source="curriculum")
                gateway.write_text(
                    self.data_path,
                    payload,
                    encoding="utf-8",
                    source="curriculum",
                )
            self.data = normalized
            return True
        except (RuntimeError, AttributeError, OSError, TypeError, ValueError) as exc:
            record_degradation(
                "curriculum",
                exc,
                action="curriculum mutation not committed; retained previous state",
            )
            logger.error("Failed to save curriculum: %s", exc)
            return False

    def get_suggestion(self, category: str | None = None) -> dict[str, Any] | None:
        """Return the next uncompleted item, optionally from one category."""

        requested = str(category or "").strip().casefold()
        for current in self.data.get("categories", []):
            if requested and str(current.get("name", "")).casefold() != requested:
                continue
            for item in current.get("items", []):
                if item.get("status") == "new":
                    return {"category": current["name"], "item": copy.deepcopy(item)}
        return None

    def get_all_categories(self) -> list[str]:
        return [str(category["name"]) for category in self.data.get("categories", [])]

    def mark_complete(self, item_name: str | None) -> str:
        """Mark an item complete only after the durable write succeeds."""

        requested = str(item_name or "").strip()
        if not requested:
            return "An item name is required."
        for category in self.data.get("categories", []):
            for item in category.get("items", []):
                if str(item.get("name", "")).casefold() != requested.casefold():
                    continue
                previous = str(item.get("status") or "new")
                item["status"] = "completed"
                if self._save_data():
                    return f"Marked '{item['name']}' as completed."
                item["status"] = previous
                return f"Could not persist completion for '{item['name']}'."
        return f"Item '{requested}' not found."

    def delete_item(self, item_name: str | None) -> str:
        """Delete a completed item only after the durable write succeeds."""

        requested = str(item_name or "").strip()
        if not requested:
            return "An item name is required."
        for category in self.data.get("categories", []):
            for index, item in enumerate(category.get("items", [])):
                if str(item.get("name", "")).casefold() != requested.casefold():
                    continue
                if item.get("status") != "completed":
                    return f"Cannot delete '{requested}' until it is marked as completed."
                removed = category["items"].pop(index)
                if self._save_data():
                    return f"Deleted '{requested}' from curriculum."
                category["items"].insert(index, removed)
                return f"Could not persist deletion of '{requested}'."
        return f"Item '{requested}' not found."
