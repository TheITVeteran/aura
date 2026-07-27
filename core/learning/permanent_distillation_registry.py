"""Durable, append-only home for the permanent-distillation lineage.

`permanent_distillation` is the contract; this is where a lineage lives between
campaigns. Two properties matter and both are enforced on every write:

- **Append-only.** A write may extend the lineage and nothing else. The records
  already on disk must survive byte-identically into the new document, so a
  promotion cannot quietly rewrite the generation it is replacing, and a
  rollback cannot erase the promotion it is reverting. The failed promotion
  stays in the history; that is the point of keeping one.
- **Valid before durable.** The whole chain is replayed *including* the new
  record before anything is written. A registry file that exists is a registry
  file that validated.

Writes go through the file-write gateway inside a governed scope, like every
other consequential write in this runtime.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from core.learning.permanent_distillation import (
    PermanentDistillationError,
    validate_lineage,
)

logger = logging.getLogger("Aura.PermanentDistillationRegistry")

REGISTRY_SCHEMA: Final = "aura.rlc.permanent_distillation.registry.v1"
_MAX_REGISTRY_BYTES: Final = 32 * 1024 * 1024


def _document(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema": REGISTRY_SCHEMA,
        "generations": [dict(row) for row in records],
        "head_generation_sha256": records[-1]["generation_sha256"],
        "generation_count": len(records),
    }


def _serialize(document: Mapping[str, Any]) -> str:
    return json.dumps(document, indent=1, sort_keys=True) + "\n"


def load_lineage(path: Path | str) -> list[dict[str, Any]]:
    """Read and fully replay a stored lineage."""

    target = Path(path).expanduser()
    try:
        raw = target.read_bytes()
    except OSError as exc:
        raise PermanentDistillationError(
            "permanent_distillation_registry_unreadable"
        ) from exc
    if len(raw) > _MAX_REGISTRY_BYTES:
        raise PermanentDistillationError("permanent_distillation_registry_oversized")
    try:
        document = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise PermanentDistillationError(
            "permanent_distillation_registry_malformed"
        ) from exc
    if (
        not isinstance(document, Mapping)
        or document.get("schema") != REGISTRY_SCHEMA
        or not isinstance(document.get("generations"), list)
    ):
        raise PermanentDistillationError("permanent_distillation_registry_invalid")

    records = validate_lineage(document["generations"])
    if (
        document.get("head_generation_sha256") != records[-1]["generation_sha256"]
        or document.get("generation_count") != len(records)
    ):
        raise PermanentDistillationError("permanent_distillation_registry_head_differs")
    return records


def write_lineage(path: Path | str, records: Sequence[Mapping[str, Any]]) -> str:
    """Validate then durably write a lineage, refusing any non-append change.

    Returns the head generation digest that is now on disk.
    """

    replayed = validate_lineage(records)
    target = Path(path).expanduser()

    if target.exists():
        existing = load_lineage(target)
        if len(existing) > len(replayed):
            raise PermanentDistillationError(
                "permanent_distillation_registry_truncated"
            )
        for stored, incoming in zip(existing, replayed[: len(existing)], strict=True):
            if stored != incoming:
                raise PermanentDistillationError(
                    "permanent_distillation_registry_rewrites_history"
                )

    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    gateway = get_file_write_gateway()
    with local_internal_governed_scope("permanent_distillation_registry"):
        gateway.ensure_directory(target.parent, source="permanent_distillation")
        gateway.write_text(
            target,
            _serialize(_document(replayed)),
            source="permanent_distillation",
        )
    logger.info(
        "permanent distillation lineage now at generation %d (%s)",
        replayed[-1]["generation_index"],
        replayed[-1]["kind"],
    )
    return replayed[-1]["generation_sha256"]


def append_generation(path: Path | str, record: Mapping[str, Any]) -> str:
    """Extend the stored lineage by exactly one validated generation."""

    target = Path(path).expanduser()
    existing = load_lineage(target) if target.exists() else []
    return write_lineage(target, [*existing, dict(record)])


__all__ = [
    "REGISTRY_SCHEMA",
    "append_generation",
    "load_lineage",
    "write_lineage",
]
