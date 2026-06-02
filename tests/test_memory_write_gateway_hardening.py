from __future__ import annotations

import pytest

from core.memory.memory_write_gateway import ConcreteMemoryWriteGateway
from core.runtime.gateways import MemoryWriteRequest


@pytest.mark.asyncio
async def test_memory_write_gateway_denies_async_governance_outage(tmp_path):
    attempted = {"governance": False}

    async def _governance_down(**_kwargs):
        attempted["governance"] = True
        raise RuntimeError("will unavailable")

    gateway = ConcreteMemoryWriteGateway(root=tmp_path, governance_decide=_governance_down)

    with pytest.raises(PermissionError):
        await gateway.write(
            MemoryWriteRequest(
                content={"note": "must not persist"},
                cause="unit-test",
                metadata={"family": "episodic", "record_id": "blocked"},
            )
        )

    assert not (tmp_path / "episodic" / "blocked.json").exists()
    assert attempted["governance"] is True
