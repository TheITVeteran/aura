"""Bounded, idempotent request/response lane for managed physical adapters."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Mapping
from typing import Any

from core.reality_reach.middleware_contracts import (
    RealityMiddlewareError,
    ServiceEndpoint,
    ServiceReceipt,
    bounded_payload,
    bounded_seconds,
    canonical_identifier,
)
from core.runtime.audit_chain import canonical_json, sha256_hex
from core.runtime.lifecycle import State
from core.utils.task_tracker import get_task_tracker


class RealityServiceLane:
    """Singleflight service execution shared by the middleware runtime."""

    async def call_service(
        self,
        endpoint_id: str,
        request: Mapping[str, Any],
        *,
        request_id: str | None = None,
        timeout_s: float | None = None,
    ) -> ServiceReceipt:
        node, endpoint = self._service_endpoint(endpoint_id)
        if node.organ.state is not State.ACTIVE:
            raise RealityMiddlewareError("service node is not active")
        if not endpoint.read_only:
            raise RealityMiddlewareError(
                "state-changing physical services must be declared as actions"
            )
        request_id = canonical_identifier(request_id or f"srv-{uuid.uuid4().hex}", name="request_id")
        body = bounded_payload(request, name="service request", maximum=endpoint.request_bytes)
        request_sha = str(sha256_hex(canonical_json(body)))
        budget = endpoint.timeout_s
        if timeout_s is not None:
            budget = min(
                budget,
                bounded_seconds(timeout_s, name="timeout_s", minimum=0.01, maximum=300.0),
            )
        async with self._lock:
            prior = self._service_receipts.get(request_id)
            if prior is not None:
                if prior.endpoint_id != endpoint.endpoint_id or prior.request_sha256 != request_sha:
                    raise RealityMiddlewareError("service request id was reused with different content")
                return prior
            inflight = self._service_inflight.get(request_id)
            if inflight is not None:
                inflight_endpoint, inflight_sha, task = inflight
                if inflight_endpoint != endpoint.endpoint_id or inflight_sha != request_sha:
                    raise RealityMiddlewareError(
                        "service request id was reused with different content"
                    )
            else:
                task = get_task_tracker().create_task(
                    self._execute_service(
                        node,
                        endpoint,
                        body,
                        request_id=request_id,
                        request_sha=request_sha,
                        budget=budget,
                    ),
                    name=f"RealityService:{endpoint.endpoint_id}:{request_id}",
                )
                self._service_inflight[request_id] = (
                    endpoint.endpoint_id,
                    request_sha,
                    task,
                )
        return await asyncio.shield(task)

    async def _execute_service(
        self,
        node: Any,
        endpoint: ServiceEndpoint,
        body: dict[str, Any],
        *,
        request_id: str,
        request_sha: str,
        budget: float,
    ) -> ServiceReceipt:
        try:
            started_at_ns = int(self._wall_clock_ns())
            ok = False
            response: dict[str, Any] = {}
            error = ""
            try:
                async with node.service_limits[endpoint.endpoint_id]:
                    raw = await asyncio.wait_for(
                        node.adapter.handle_service(endpoint.endpoint_id, body),
                        timeout=budget,
                    )
                response = bounded_payload(
                    raw, name="service response", maximum=endpoint.response_bytes
                )
                ok = True
            except TimeoutError:
                error = f"service_deadline_exceeded:{budget:.3f}s"
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - adapter boundary becomes a receipt
                error = f"{type(exc).__name__}:{exc}"[:1024]
            receipt = ServiceReceipt(
                request_id=request_id,
                endpoint_id=endpoint.endpoint_id,
                request_sha256=request_sha,
                ok=ok,
                response=response,
                error=error,
                started_at_ns=started_at_ns,
                completed_at_ns=int(self._wall_clock_ns()),
                adapter_identity_sha256=node.declaration.adapter_identity_sha256,
            )
            async with self._lock:
                self._service_receipts[request_id] = receipt
                while len(self._service_receipts) > self._max_service_receipts:
                    self._service_receipts.pop(next(iter(self._service_receipts)))
            await self._persist()
            return receipt
        finally:
            async with self._lock:
                current = self._service_inflight.get(request_id)
                if current is not None and current[2] is asyncio.current_task():
                    self._service_inflight.pop(request_id, None)



__all__ = ["RealityServiceLane"]
