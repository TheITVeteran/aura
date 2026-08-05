from __future__ import annotations

from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.routing import APIRoute

from core.reality_reach.acceptance import AcceptanceError, AcceptanceEvidenceClass
from interface.routes import reality_reach as routes


class _Service:
    def __init__(self, *, error: BaseException | None = None) -> None:
        self.error = error
        self.requests: list[Any] = []

    def status(self) -> dict[str, Any]:
        return {"alive": True, "ready": True}

    async def run(self, request: Any) -> dict[str, Any]:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return {"campaign_id": request.campaign_id, "certificate_sha256": "sha256:" + "a" * 64}


def _payload() -> routes.ScalarAcceptancePayload:
    return routes.ScalarAcceptancePayload(
        campaign_id="cp810.route.live",
        connector_id="fixture.connector",
        adapter_id="fixture.adapter",
        target=7.0,
        expected_source_commit_sha256="sha256:" + "b" * 64,
        evidence_class=AcceptanceEvidenceClass.LIVE,
    )


def test_acceptance_routes_require_both_internal_and_token_guards() -> None:
    acceptance_routes = {
        route.path: route
        for route in routes.router.routes
        if isinstance(route, APIRoute)
    }
    assert set(acceptance_routes) == {
        "/reality-reach/acceptance/status",
        "/reality-reach/acceptance/run",
    }
    for route in acceptance_routes.values():
        dependencies = {dependency.call for dependency in route.dependant.dependencies}
        assert routes._require_internal in dependencies
        assert routes._verify_token in dependencies


@pytest.mark.asyncio
async def test_acceptance_route_constructs_typed_runtime_request(monkeypatch) -> None:
    service = _Service()
    monkeypatch.setattr(routes.ServiceContainer, "get", lambda *_args, **_kwargs: service)

    response = await routes.run_acceptance(_payload(), None, None)

    assert response.status_code == 201
    assert len(service.requests) == 1
    assert service.requests[0].evidence_class is AcceptanceEvidenceClass.LIVE
    assert service.requests[0].expected_source_commit_sha256 == "sha256:" + "b" * 64


@pytest.mark.asyncio
async def test_acceptance_route_maps_controlled_refusal_to_conflict(monkeypatch) -> None:
    service = _Service(error=AcceptanceError("acceptance_runtime_source_dirty"))
    monkeypatch.setattr(routes.ServiceContainer, "get", lambda *_args, **_kwargs: service)

    with pytest.raises(HTTPException) as raised:
        await routes.run_acceptance(_payload(), None, None)

    assert raised.value.status_code == 409
    assert raised.value.detail == "acceptance_runtime_source_dirty"


@pytest.mark.asyncio
async def test_acceptance_status_fails_closed_when_service_is_missing(monkeypatch) -> None:
    monkeypatch.setattr(routes.ServiceContainer, "get", lambda *_args, **_kwargs: None)

    with pytest.raises(HTTPException) as raised:
        await routes.acceptance_status(None, None)

    assert raised.value.status_code == 503
