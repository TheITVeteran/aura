from __future__ import annotations

from dataclasses import dataclass

from core.agency.agency_bus import AgencyBus


@dataclass
class _Decision:
    receipt_id: str = "receipt-ok"
    reason: str = "ok"
    approved: bool = True

    def is_approved(self) -> bool:
        return self.approved


class _Will:
    def __init__(self, *, approved: bool = True, valid_receipt: bool = True):
        self.approved = approved
        self.valid_receipt = valid_receipt

    def decide(self, **_kwargs):
        return _Decision(approved=self.approved)

    def verify_receipt(self, receipt_id: str) -> bool:
        return receipt_id == "receipt-ok" and self.valid_receipt


def test_agency_bus_auto_acquires_and_verifies_will_receipt(monkeypatch):
    monkeypatch.setattr("core.governance.will.get_will", lambda: _Will())
    bus = AgencyBus()

    proposal = {"origin": "test", "text": "hello", "priority_class": "duty"}

    assert bus.submit(proposal) is True
    assert proposal["will_receipt"] == "receipt-ok"
    assert bus.stats["recent_audit"]


def test_agency_bus_fails_closed_when_will_unavailable(monkeypatch):
    def unavailable():
        raise RuntimeError("will offline")

    monkeypatch.setattr("core.governance.will.get_will", unavailable)
    bus = AgencyBus()

    assert bus.submit({"origin": "test", "text": "hello", "priority_class": "duty"}) is False
    assert bus.stats["recent_audit"] == []


def test_agency_bus_rejects_invalid_receipt(monkeypatch):
    monkeypatch.setattr("core.governance.will.get_will", lambda: _Will(valid_receipt=False))
    bus = AgencyBus()

    assert (
        bus.submit(
            {
                "origin": "test",
                "text": "hello",
                "priority_class": "duty",
                "will_receipt": "receipt-ok",
            }
        )
        is False
    )
    assert bus.stats["recent_audit"] == []
