"""CP126 contract tests for the reflex path.

A reflex bypasses cognition, so anything it says is said without having
checked. These pin that it only says what it can verify.
"""
from __future__ import annotations

import re
import time

import pytest

from core.brain import reflex as module
from core.brain.reflex import ReflexiveCore


@pytest.fixture()
def reflex() -> ReflexiveCore:
    return ReflexiveCore()


@pytest.fixture()
def healthy(monkeypatch):
    monkeypatch.setattr(
        module.ReflexiveCore, "_health_report",
        staticmethod(lambda: {"status": "healthy", "degraded_subsystems": []}),
    )


# --- ab407fb9: triggers must not hijack unrelated messages ---------------


@pytest.mark.parametrize(
    "text",
    [
        "what is the status of the migration you described",
        "I identity-mapped the columns yesterday",
        "can you tell me about the time we fixed the router",
        "the clock on the wall is broken",
        "ping the server and tell me what you find",
        "who are you going to ask about this?",
        "status codes in the 400 range mean what exactly",
        "remember that time you said the vault was hardened?",
    ],
)
def test_longer_messages_are_not_hijacked(reflex, text):
    """Every one of these contained a trigger as a substring."""
    assert reflex.process(text) is None


@pytest.mark.parametrize(
    "text", ["ping", "Ping", "  ping  ", "ping?", "hey aura, ping", "aura ping!"]
)
def test_a_bare_trigger_still_fires(reflex, text):
    assert reflex.process(text) == "Reflex path active."


def test_empty_input_is_ignored(reflex):
    assert reflex.process("") is None
    assert reflex.process("   ") is None
    assert reflex.process(None) is None


# --- 67d526bf: status reflects the real health contract -----------------


def test_status_declines_when_no_health_report_exists(reflex, monkeypatch):
    monkeypatch.setattr(module.ReflexiveCore, "_health_report", staticmethod(lambda: None))

    assert reflex.process("status") is None


def test_a_healthy_runtime_reports_healthy(reflex, healthy):
    reply = reflex.process("status")

    assert "Healthy" in reply
    assert "health contract" in reply


def test_a_degraded_runtime_is_not_reported_as_operational(reflex, monkeypatch):
    monkeypatch.setattr(
        module.ReflexiveCore, "_health_report",
        staticmethod(lambda: {"status": "degraded", "degraded_subsystems": ["memory", "router"]}),
    )

    reply = reflex.process("status")

    assert "degraded" in reply.lower()
    assert "memory" in reply
    assert "operational" not in reply.lower()


def test_a_critical_runtime_says_so(reflex, monkeypatch):
    monkeypatch.setattr(
        module.ReflexiveCore, "_health_report",
        staticmethod(lambda: {"status": "critical", "degraded_subsystems": ["cortex"]}),
    )

    reply = reflex.process("status")

    assert "Not good" in reply
    assert "critical" in reply


def test_degradations_alone_prevent_a_healthy_claim(reflex, monkeypatch):
    monkeypatch.setattr(
        module.ReflexiveCore, "_health_report",
        staticmethod(lambda: {"status": "healthy", "degraded_subsystems": ["memory"]}),
    )

    assert "degraded" in reflex.process("status").lower()


def test_an_unrecognized_health_shape_declines(reflex, monkeypatch):
    monkeypatch.setattr(
        module.ReflexiveCore, "_health_report",
        staticmethod(lambda: {"status": "whatever"}),
    )

    assert reflex.process("status") is None


def test_a_raising_health_probe_declines(reflex, monkeypatch):
    def boom():
        raise RuntimeError("health contract down")

    monkeypatch.setattr("core.runtime.health_contract.runtime_health_report", boom)

    assert reflex.process("status") is None


def test_no_hardcoded_operational_claim_can_be_returned(reflex, monkeypatch):
    """Assert on BEHAVIOUR: the phrases survive only in explanatory comments."""
    for report in (
        None,
        {"status": "healthy", "degraded_subsystems": []},
        {"status": "degraded", "degraded_subsystems": ["memory"]},
        {"status": "critical", "degraded_subsystems": []},
        {"status": "whatever"},
    ):
        monkeypatch.setattr(
            module.ReflexiveCore, "_health_report", staticmethod(lambda r=report: r)
        )
        reply = reflex.process("status") or ""
        assert "All actors supervised" not in reply
        assert "state-vault hardened" not in reply


# --- d8ac7a5e: identity comes from the live self-model ------------------


def test_identity_declines_without_a_live_self_model(reflex, monkeypatch):
    monkeypatch.setattr(module.ReflexiveCore, "_live_identity", staticmethod(lambda: ""))

    assert reflex.process("who are you") is None
    assert reflex.process("what are you") is None


def test_identity_uses_the_live_name(reflex, monkeypatch):
    monkeypatch.setattr(
        module.ReflexiveCore, "_live_identity", staticmethod(lambda: "Aura-abcdefgh")
    )

    reply = reflex.process("who are you")

    assert "Aura-abcdefgh" in reply


def test_no_hardcoded_identity_claim_can_be_returned(reflex, monkeypatch):
    for identity in ("", "Aura-abcdefgh", "Something Else"):
        monkeypatch.setattr(
            module.ReflexiveCore, "_live_identity", staticmethod(lambda i=identity: i)
        )
        reply = reflex.process("who are you") or ""
        assert "hardened digital intelligence" not in reply
        assert "Aura Zenith" not in reply


def test_a_raising_identity_probe_declines(reflex, monkeypatch):
    def boom(name, default=None):
        raise RuntimeError("registry down")

    monkeypatch.setattr("core.runtime.service_registry.get_runtime_service", boom)

    assert reflex.process("identity") is None


# --- 1eb0f395: the clock is not mislabeled ------------------------------


def test_time_is_not_labeled_utc_when_it_is_local(reflex):
    reply = reflex.process("time")

    local_hour = time.strftime("%H", time.localtime())
    utc_hour = time.strftime("%H", time.gmtime())
    stamp = re.search(r"awareness: (\d{2}):", reply).group(1)

    assert stamp == local_hour
    if local_hour != utc_hour:
        # The local reading must not be presented AS the UTC reading.
        assert not re.search(rf"{stamp}:\d{{2}}:\d{{2}} UTC\b", reply)


def test_time_names_its_zone(reflex):
    reply = reflex.process("time")

    assert re.search(r"\d{2}:\d{2}:\d{2} \S+", reply)


def test_time_offers_the_real_utc_reading_when_offset(reflex):
    reply = reflex.process("time")

    if time.strftime("%H", time.localtime()) != time.strftime("%H", time.gmtime()):
        assert "UTC" in reply
        utc_hour = time.strftime("%H", time.gmtime())
        assert f"UTC {utc_hour}" in reply


def test_the_time_reflex_calls_a_real_timezone_function():
    """The old code formatted localtime and appended the literal "UTC"."""
    import inspect

    body = inspect.getsource(module.ReflexiveCore._handle_time)
    code_lines = [
        line for line in body.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    code = "\n".join(code_lines)

    assert "time.localtime()" in code
    assert "time.gmtime()" in code


def test_ping_is_the_only_unconditional_claim(reflex, monkeypatch):
    """Ping asserts only about the reflex path itself, which it can know."""
    monkeypatch.setattr(module.ReflexiveCore, "_health_report", staticmethod(lambda: None))
    monkeypatch.setattr(module.ReflexiveCore, "_live_identity", staticmethod(lambda: ""))

    assert reflex.process("ping") == "Reflex path active."
    assert reflex.process("status") is None
    assert reflex.process("who are you") is None
