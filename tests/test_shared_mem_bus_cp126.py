"""Shared-memory transport: corruption published as success, and stolen segments."""
from __future__ import annotations

import asyncio

import pytest

from core.bus.shared_mem_bus import SharedMemoryTransport

pytestmark = pytest.mark.unit


@pytest.fixture()
def segment():
    import uuid
    shm = SharedMemoryTransport(f"cp126-{uuid.uuid4().hex[:12]}", size=4096)
    asyncio.run(shm.create())
    yield shm
    try:
        shm.close()
    except Exception:  # noqa: BLE001 - teardown best effort
        pass


def test_oversized_write_is_refused_not_truncated(segment):
    """Truncating UTF-8 at a byte boundary yields unparseable bytes, which were
    then published under a COMPLETED version: the writer was told it succeeded
    while every reader decoded None."""
    segment.write({"ok": "small"})
    assert asyncio.run(segment.read())["ok"] == "small"

    huge = {"blob": "x" * (segment.payload_capacity + 500)}
    with pytest.raises(ValueError, match="too large"):
        segment.write(huge)

    # The previous valid payload must survive a refused write.
    recovered = asyncio.run(segment.read())
    assert recovered is not None and recovered["ok"] == "small"


def test_write_that_fits_still_round_trips(segment):
    payload = {"n": list(range(50)), "s": "unicode ✓ ünïcødé"}
    segment.write(payload)

    got = asyncio.run(segment.read())
    assert got["n"] == payload["n"]
    assert got["s"] == payload["s"]


def test_out_of_bounds_length_prefix_is_rejected(segment):
    """The length prefix is bytes another process wrote; a forged or torn value
    used to be sliced directly and surfaced as an ambiguous None."""
    segment.write({"ok": True})
    buf = segment.shm.buf
    buf[8:12] = (segment.payload_capacity + 10_000).to_bytes(4, byteorder="big")

    assert asyncio.run(segment.read()) is None


def test_attaching_does_not_confer_ownership(segment):
    """close() left _is_owner set, so an object that later attached to a
    replacement segment would unlink data it did not own."""
    import uuid

    other = SharedMemoryTransport(f"cp126-{uuid.uuid4().hex[:12]}", size=4096)
    asyncio.run(other.create())
    assert other._is_owner is True
    other.close()
    assert other._is_owner is False, "close must relinquish ownership"

    # Re-attaching to someone else's live segment must not claim it.
    other.name = segment.name
    asyncio.run(other.attach())
    assert other._is_owner is False
    other.close()

    # The original owner's segment is still readable.
    segment.write({"alive": True})
    assert asyncio.run(segment.read())["alive"] is True
