"""Boot catalog for optional concrete Reality Reach connector families."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any

_OPENHAB_CONNECTOR_ID = "openhab.local"


@dataclass(frozen=True, slots=True)
class ConnectorBootStatus:
    connector_id: str
    configured: bool
    registered: bool
    state: str
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "connector_id": self.connector_id,
            "configured": self.configured,
            "registered": self.registered,
            "state": self.state,
            "error": self.error,
        }


class RealityConnectorCatalog:
    """Own boot-time connector construction without retaining credentials."""

    def __init__(
        self,
        connectors: tuple[Any, ...],
        statuses: tuple[ConnectorBootStatus, ...],
    ) -> None:
        self._connectors = connectors
        self._statuses = statuses

    @property
    def connectors(self) -> tuple[Any, ...]:
        return self._connectors

    def register_with(self, broker: Any) -> None:
        register = getattr(broker, "register_connector", None)
        if not callable(register):
            raise TypeError("broker must expose register_connector")
        registered: set[str] = set()
        for connector in self._connectors:
            register(connector)
            registered.add(connector.connector_id)
        self._statuses = tuple(
            replace(
                status,
                registered=status.connector_id in registered,
                state=(
                    "registered"
                    if status.connector_id in registered
                    else status.state
                ),
            )
            for status in self._statuses
        )

    def status(self) -> dict[str, Any]:
        entries = [status.to_dict() for status in self._statuses]
        return {
            "alive": True,
            "ready": not any(item["state"] == "invalid" for item in entries),
            "configured": sum(bool(item["configured"]) for item in entries),
            "registered": sum(bool(item["registered"]) for item in entries),
            "connectors": entries,
        }

    def get_status(self) -> dict[str, Any]:
        return self.status()

    def is_alive(self) -> bool:
        return True

    def is_ready(self) -> bool:
        return bool(self.status()["ready"])


def build_configured_reality_connector_catalog() -> RealityConnectorCatalog:
    """Build configured connectors; absence is valid, partial config is explicit."""

    connectors: list[Any] = []
    statuses: list[ConnectorBootStatus] = []
    url = str(os.getenv("AURA_OPENHAB_URL") or "").strip()
    token = str(os.getenv("AURA_OPENHAB_TOKEN") or "").strip()
    if not url and not token:
        statuses.append(
            ConnectorBootStatus(
                connector_id=_OPENHAB_CONNECTOR_ID,
                configured=False,
                registered=False,
                state="not_configured",
            )
        )
    elif not url or not token:
        statuses.append(
            ConnectorBootStatus(
                connector_id=_OPENHAB_CONNECTOR_ID,
                configured=True,
                registered=False,
                state="invalid",
                error=(
                    "AURA_OPENHAB_URL is missing"
                    if not url
                    else "AURA_OPENHAB_TOKEN is missing"
                ),
            )
        )
    else:
        try:
            from core.embodiment.openhab_connector import (
                OpenHABConnector,
                OpenHABTransport,
            )

            connector = OpenHABConnector(OpenHABTransport())
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            statuses.append(
                ConnectorBootStatus(
                    connector_id=_OPENHAB_CONNECTOR_ID,
                    configured=True,
                    registered=False,
                    state="invalid",
                    error=f"{type(exc).__name__}:{exc}"[:240],
                )
            )
        else:
            connectors.append(connector)
            statuses.append(
                ConnectorBootStatus(
                    connector_id=connector.connector_id,
                    configured=True,
                    registered=False,
                    state="ready",
                )
            )
    return RealityConnectorCatalog(tuple(connectors), tuple(statuses))


__all__ = [
    "ConnectorBootStatus",
    "RealityConnectorCatalog",
    "build_configured_reality_connector_catalog",
]
