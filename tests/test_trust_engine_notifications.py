import core.event_bus as event_bus
from types import SimpleNamespace

from core.security import cheat_codes as cheat_code_module
from core.security import trust_engine as trust_engine_module
from core.security.trust_engine import TrustEngine, TrustLevel


class _Bus:
    def __init__(self, published):
        self._published = published

    def publish_threadsafe(self, topic, payload):
        self._published.append((topic, payload))


def test_establish_sovereign_session_can_be_silent(monkeypatch):
    published = []
    monkeypatch.setattr(event_bus, "get_event_bus", lambda: _Bus(published))

    engine = TrustEngine()
    level = engine.establish_sovereign_session(reason="test", announce=False)

    assert level == TrustLevel.SOVEREIGN
    assert published == []


def test_sovereign_cheat_code_emits_single_message(monkeypatch):
    published = []
    monkeypatch.setattr(event_bus, "get_event_bus", lambda: _Bus(published))
    monkeypatch.setattr(cheat_code_module, "_matches_sovereign_code", lambda _code: True)

    trust_engine_module._engine = None
    try:
        result = cheat_code_module.activate_cheat_code("owner", silent=False, source="test")
    finally:
        trust_engine_module._engine = None

    assert result["ok"] is True
    assert len(published) == 1
    assert published[0][0] == "telemetry"
    assert published[0][1]["message"] == result["message"]
    assert published[0][1]["metadata"]["system"] is True
    assert published[0][1]["metadata"]["cheat_code"] == "owner_sovereign"


def test_trust_event_log_uses_file_write_governance(monkeypatch):
    from core.governance_context import require_governance

    calls = []

    class Gateway:
        def append_text(self, path, text, *, source):
            token = require_governance(
                f"file_write_gateway.append_text:{source}",
                strict=True,
                allowed_domains=("file_write",),
            )
            calls.append((source, token.domain, "trust_elevated" in text))

        # Async lane delegators: production code now calls *_async; fakes
        # must mirror the gateway surface or every governed write breaks.
        async def append_text_async(self, *args, **kwargs):
            return self.append_text(*args, **kwargs)

    monkeypatch.setattr(trust_engine_module, "get_file_write_gateway", lambda: Gateway())

    engine = TrustEngine()
    engine._log_event("trust_elevated", {"from": "guest", "to": "trusted"})

    assert calls == [("security.trust_engine.event", "file_write", True)]


def test_security_event_logs_use_file_write_governance(monkeypatch, tmp_path):
    from core.governance_context import require_governance
    from core.security import audit_log as audit_module
    from core.security import emergency_protocol as emergency_module

    calls = []

    class Gateway:
        def append_text(self, path, text, *, source):
            token = require_governance(
                f"file_write_gateway.append_text:{source}",
                strict=True,
                allowed_domains=("file_write",),
            )
            calls.append((source, token.domain))

        # Async lane delegators: production code now calls *_async; fakes
        # must mirror the gateway surface or every governed write breaks.
        async def append_text_async(self, *args, **kwargs):
            return self.append_text(*args, **kwargs)

    monkeypatch.setattr(audit_module, "get_file_write_gateway", lambda: Gateway())
    monkeypatch.setattr(emergency_module, "get_file_write_gateway", lambda: Gateway())

    audit = audit_module.SecurityAuditLogger.__new__(audit_module.SecurityAuditLogger)
    audit.log_path = tmp_path / "security_audit.jsonl"
    audit.log_event("probe", {"ok": True})

    protocol = emergency_module.EmergencyProtocol.__new__(emergency_module.EmergencyProtocol)
    protocol._threat_score = 0.25
    protocol._log_threat(
        SimpleNamespace(
            timestamp=1.0,
            source="test",
            description="probe",
            severity=0.1,
        )
    )

    assert calls == [
        ("security.audit_log", "file_write"),
        ("security.emergency_protocol.threat", "file_write"),
    ]


def test_a_partial_threat_signal_is_still_logged(monkeypatch):
    """A threat log must not drop what it cannot fully parse.

    Attribute access here used to be direct, so a signal missing one field
    raised AttributeError, the handler swallowed it as a degradation, and the
    threat was never recorded at all. The absence of that record is
    indistinguishable from "nothing happened", which is the worst possible
    failure for a security log.
    """
    import json
    import types

    from core.security import emergency_protocol as emergency_module

    written = []

    class Gateway:
        def append_text(self, path, text, *, source):
            written.append(json.loads(text))

        async def append_text_async(self, *args, **kwargs):
            return self.append_text(*args, **kwargs)

    monkeypatch.setattr(emergency_module, "get_file_write_gateway", lambda: Gateway())

    protocol = emergency_module.EmergencyProtocol.__new__(
        emergency_module.EmergencyProtocol
    )
    protocol._threat_score = 0.5
    # Deliberately missing threat_class and evidence.
    protocol._log_threat(
        types.SimpleNamespace(
            timestamp=1.0, source="probe", description="partial", severity="high"
        )
    )

    assert written, "a partial threat signal produced no log entry at all"
    assert written[0]["source"] == "probe"
    assert written[0]["severity"] == "high"
    assert written[0]["threat_class"] == ""
