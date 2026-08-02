"""Project live Reality Reach channels into Aura's canonical body schema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.reality_reach.contracts import ChannelDeclaration, ChannelKind
from core.runtime.audit_chain import canonical_json, sha256_hex
from core.somatic.body_schema import Limb, LimbType, get_body_schema


@dataclass(frozen=True, slots=True)
class PhysicalBodyProjection:
    adapter_id: str
    limb_names: tuple[str, ...]


def _digest(value: Any) -> str:
    return str(sha256_hex(canonical_json(value)))


def project_adapter_to_body(
    adapter: Any,
    *,
    device_id: str,
    display_name: str,
    transport: str,
    privacy_sensitive: bool = False,
    persistent_identity: bool = False,
    manifest_sha256: str = "",
) -> PhysicalBodyProjection:
    """Expose every declared physical channel as a sensor or actuator limb."""

    adapter_id = str(getattr(adapter, "adapter_id", "") or "")
    declarations_fn = getattr(adapter, "declarations", None)
    if not adapter_id or not callable(declarations_fn):
        raise TypeError("attached adapter must expose an identity and declarations")
    declarations = tuple(declarations_fn())
    if not declarations or any(
        not isinstance(item, ChannelDeclaration) for item in declarations
    ):
        raise TypeError("attached adapter declarations are invalid")

    body = get_body_schema()
    names: list[str] = []
    try:
        for declaration in declarations:
            limb_type = (
                LimbType.SENSOR
                if declaration.kind == ChannelKind.SENSOR
                else LimbType.ACTUATOR
            )
            suffix = _digest(
                {
                    "adapter_id": adapter_id,
                    "channel_id": declaration.channel_id,
                    "kind": declaration.kind.value,
                }
            ).removeprefix("sha256:")[:24]
            name = f"reality_{declaration.kind.value}_{suffix}"
            body.add_limb(
                Limb(
                    name=name,
                    limb_type=limb_type,
                    description=(
                        f"Physical {declaration.observable} channel on "
                        f"{display_name or device_id} via {transport}."
                    ),
                    source=f"reality:{adapter_id}",
                    metadata={
                        "adapter_id": adapter_id,
                        "channel_id": declaration.channel_id,
                        "observable": declaration.observable,
                        "unit": declaration.unit,
                        "device_id": str(device_id),
                        "transport": str(transport),
                        "privacy_sensitive": bool(privacy_sensitive),
                        "persistent_identity": bool(persistent_identity),
                        "manifest_sha256": str(manifest_sha256),
                    },
                )
            )
            names.append(name)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        for name in names:
            body.remove_limb(name)
        raise
    return PhysicalBodyProjection(adapter_id=adapter_id, limb_names=tuple(names))


def remove_body_projection(projection: PhysicalBodyProjection) -> None:
    body = get_body_schema()
    for name in projection.limb_names:
        body.remove_limb(name)


__all__ = [
    "PhysicalBodyProjection",
    "project_adapter_to_body",
    "remove_body_projection",
]
