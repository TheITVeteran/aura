"""What is known about recurrence on the model actually loaded.

The engine admits any mlx_lm decoder that exposes ``model.layers``, then
reapplies a middle-layer window several times. Structural access proves the
call succeeds. It does not establish that repeating those layers is meaningful
for this architecture, that the positional contract survives the repetition, or
that the frozen checkpoint was ever trained under anything resembling it.

Nothing here refuses a model. It records which of the three claims are backed
by evidence, so an episode's receipt says "experimental" when that is the true
answer instead of leaving the reader to infer support from the absence of a
complaint.

Certification is a registry entry, not a code change: an architecture appears
in ``config/recurrence_certified_architectures.json`` only when paired evidence
for it exists, and the file names the evidence. An empty or missing registry
means nothing is certified, which is the honest default.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

RECURRENCE_SUPPORT_SCHEMA = "aura.latent_cortex.recurrence_support.v1"

_REGISTRY_FILENAME = "recurrence_certified_architectures.json"

# Support levels, weakest first.
EXPERIMENTAL = "experimental"
STRUCTURAL = "structural_only"
CERTIFIED = "certified"


def _registry_path() -> Path:
    return Path(__file__).resolve().parents[4] / "config" / _REGISTRY_FILENAME


def load_certified_architectures(path: Path | None = None) -> dict[str, Any]:
    """Architectures with registered paired evidence for recurrent depth.

    A malformed or absent registry certifies nothing. It never certifies
    everything: an unreadable file is missing evidence, not a waiver.
    """

    target = path if path is not None else _registry_path()
    try:
        payload = json.loads(target.read_text("utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, Mapping):
        return {}
    entries = payload.get("architectures")
    if not isinstance(entries, Mapping):
        return {}
    certified: dict[str, Any] = {}
    for name, evidence in entries.items():
        if not isinstance(name, str) or not name.strip():
            continue
        if not isinstance(evidence, Mapping):
            continue
        # A registry row without evidence is a claim, and a claim is what the
        # registry exists to replace.
        if not str(evidence.get("evidence_path") or "").strip():
            continue
        certified[name.strip()] = dict(evidence)
    return certified


def model_architecture(model: Any) -> str:
    """The loaded checkpoint's own architecture name, or "" if it has none."""

    args = getattr(model, "args", None)
    if isinstance(args, Mapping):
        raw = args.get("model_type")
    else:
        raw = getattr(args, "model_type", None)
    return str(raw).strip() if isinstance(raw, str) else ""


def positional_contract(model: Any) -> dict[str, Any]:
    """Whether the model states a position limit the window can be checked against.

    A window reapplied N times keeps its cache offsets fixed, so the positional
    contract is that the recurrent region never asks for a position the model
    was not built for. A model that does not state its limit cannot be checked,
    and that is a gap rather than a pass.
    """

    args = getattr(model, "args", None)
    if isinstance(args, Mapping):
        raw = args.get("max_position_embeddings")
    else:
        raw = getattr(args, "max_position_embeddings", None)
    limit = int(raw) if type(raw) is int and raw > 0 else 0
    return {
        "declared": bool(limit),
        "max_position_embeddings": limit,
        "source": "model_config" if limit else "undeclared",
    }


def classify_recurrence_support(
    model: Any,
    *,
    layer_count: int,
    window: tuple[int, int],
    registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One receipt-shaped verdict on what backs this episode's recurrence."""

    certified = (
        dict(registry)
        if registry is not None
        else load_certified_architectures()
    )
    architecture = model_architecture(model)
    positions = positional_contract(model)
    start, end = int(window[0]), int(window[1])
    structural_ok = bool(layer_count) and 0 <= start < end <= int(layer_count)
    entry = certified.get(architecture) if architecture else None

    if not structural_ok:
        level = EXPERIMENTAL
    elif entry is not None and positions["declared"]:
        level = CERTIFIED
    elif positions["declared"]:
        level = STRUCTURAL
    else:
        level = EXPERIMENTAL

    reasons: list[str] = []
    if not architecture:
        reasons.append("architecture_undeclared")
    elif entry is None:
        reasons.append("architecture_not_registered")
    if not positions["declared"]:
        reasons.append("position_limit_undeclared")
    if not structural_ok:
        reasons.append("recurrent_window_invalid")

    return {
        "schema": RECURRENCE_SUPPORT_SCHEMA,
        "level": level,
        "architecture": architecture,
        "window": [start, end],
        "layer_count": int(layer_count),
        "positional_contract": positions,
        "evidence": dict(entry) if isinstance(entry, Mapping) else {},
        "reasons": reasons,
    }


__all__ = [
    "CERTIFIED",
    "EXPERIMENTAL",
    "RECURRENCE_SUPPORT_SCHEMA",
    "STRUCTURAL",
    "classify_recurrence_support",
    "load_certified_architectures",
    "model_architecture",
    "positional_contract",
]
