"""Request-scoped provenance carried with generated text.

``AttributedText`` remains string-compatible for existing reasoning organs while
keeping the exact generation receipt attached to the candidate that produced it.
String transformations must explicitly re-wrap derived text with
``attributed_text`` so provenance cannot silently jump between concurrent calls.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any


def _copy_metadata(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        return {}
    try:
        return copy.deepcopy(dict(metadata))
    except (TypeError, ValueError, RuntimeError):
        return dict(metadata)


class AttributedText(str):
    """A string whose generation metadata belongs to these exact text bytes."""

    generation_metadata: dict[str, Any]

    def __new__(cls, value: Any, metadata: Any = None) -> AttributedText:
        instance = super().__new__(cls, "" if value is None else str(value))
        instance.generation_metadata = _copy_metadata(metadata)
        return instance


def generation_metadata_of(value: Any) -> dict[str, Any]:
    """Return a defensive copy of metadata attached to ``value``."""

    return _copy_metadata(getattr(value, "generation_metadata", None))


def attributed_text(value: Any, metadata: Any = None) -> AttributedText:
    """Attach explicit metadata, or preserve metadata already carried by value."""

    resolved_metadata = (
        metadata if isinstance(metadata, Mapping) else generation_metadata_of(value)
    )
    return AttributedText(value, resolved_metadata)
