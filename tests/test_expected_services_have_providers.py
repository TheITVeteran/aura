"""Every service a turn expects must have something that registers it.

LIVE DEFECT, 2026-07-27. Bryan's log carried this on every conversation:

    FAULT RUNTIME-CHAT-TURN_ENGAGEMENT [CRITICAL] in chat.turn_engagement:
    RuntimeError: soul absent for 12 consecutive turns: identity continuity
    across turns and restarts

The report was accurate and the wait it implied was infinite. The
orchestrator constructed a ``Soul`` at boot (boot_identity.py) and never
published it to the service spine, so ``ServiceContainer.peek("soul")``
answered None for the life of the process.

The second-order damage was worse than the noise. ``get_panzer_soul()``
looks that service up and substitutes a metadata proxy carrying
``logic = None`` when it is missing — so the personality engine ran on a
shell while the real Soul sat one attribute away on the orchestrator. An
absence presented as a presence, the bug class this codebase keeps finding.

A runtime probe cannot catch this, because the honest answer at any moment
is "not registered *yet*". What separates a warm-up from a wiring gap is
whether a registration site exists in the source at all — a static question,
asked here.
"""
from __future__ import annotations

import pytest

from core.runtime.service_wiring_audit import (
    audit_expected_services,
    audit_turn_organs,
    expected_turn_organ_names,
)


class TestTheRosterIsReadNotDuplicated:
    def test_the_organ_roster_is_recovered_from_the_route(self):
        """A second copy of this list would drift from the one that raises
        the fault, which is the exact failure being guarded against."""
        organs = expected_turn_organ_names()
        assert "soul" in organs
        assert "personality_engine" in organs
        assert len(organs) >= 8

    def test_the_roster_matches_the_routes_own_tuple(self):
        from interface.routes.chat import _EXPECTED_TURN_ORGANS

        assert set(expected_turn_organ_names()) == {
            name for name, _why in _EXPECTED_TURN_ORGANS
        }


class TestEveryExpectedOrganIsWired:
    def test_no_expected_organ_is_unwired(self):
        """The gate. A name here cannot become available by waiting."""
        report = audit_turn_organs()
        assert report.ok, report.explain()

    def test_soul_has_a_registration_site(self):
        """The specific regression."""
        report = audit_turn_organs()
        assert report.sites.get("soul"), (
            "soul is constructed at boot but never published to the spine"
        )

    def test_the_orchestrator_publishes_the_soul_it_builds(self):
        """Registering a DIFFERENT soul would satisfy the audit and not the
        defect, so pin that the constructed object is the one published."""
        import inspect

        from core.orchestrator.mixins.boot import boot_identity

        source = inspect.getsource(boot_identity)
        built = source.index("self.soul = Soul(self)")
        following = source[built : built + 1600]
        assert "register_instance" in following
        assert "self.soul" in following.split("register_instance", 1)[1][:120]


class TestConstantBasedRegistrationsAreSeen:
    """A scanner that only understands string literals reports a wired
    service as missing. data_honesty_governor is registered exclusively via
    ServiceNames.DATA."""

    def test_a_servicenames_constant_registration_is_found(self):
        report = audit_turn_organs()
        assert report.sites.get("data_honesty_governor")

    def test_the_constant_resolves_to_the_expected_string(self):
        from core.service_names import ServiceNames

        assert str(ServiceNames.DATA) == "data_honesty_governor"


class TestTheAuditActuallyDetectsAGap:
    """Proof the gate can fail. An audit that always passes is not a gate."""

    def test_an_invented_service_is_reported_unwired(self):
        report = audit_expected_services(["definitely_not_registered_xyz"])
        assert report.ok is False
        assert report.unwired == ["definitely_not_registered_xyz"]

    def test_the_explanation_names_the_missing_service(self):
        report = audit_expected_services(["definitely_not_registered_xyz"])
        assert "definitely_not_registered_xyz" in report.explain()
        assert "wiring gap" in report.explain()

    def test_a_mixed_roster_separates_the_two(self):
        report = audit_expected_services(["soul", "definitely_not_registered_xyz"])
        assert report.wired == ["soul"]
        assert report.unwired == ["definitely_not_registered_xyz"]

    def test_the_report_is_serializable(self):
        payload = audit_turn_organs().to_dict()
        assert payload["schema"] == "aura.service_wiring_audit.v1"
        assert payload["ok"] is True
        assert "soul" in payload["sites"]


class TestTheProxyNoLongerHidesTheGap:
    """The fallback stays — a personality engine that raises because an
    optional organ is warming is worse than one that runs flat — but it may
    not be silent, and it must be identifiable."""

    def test_the_proxy_declares_itself(self, monkeypatch):
        import core.being.panzer_soul as mod

        monkeypatch.setattr(mod, "get_runtime_service", lambda *a, **k: None)
        soul = mod.get_panzer_soul()
        assert getattr(soul, "is_proxy", False) is True
        assert soul.logic is None

    def test_falling_back_records_a_degradation(self, monkeypatch):
        import core.being.panzer_soul as mod

        recorded: list = []
        monkeypatch.setattr(mod, "get_runtime_service", lambda *a, **k: None)
        monkeypatch.setattr(
            "core.runtime.errors.record_degradation",
            lambda *a, **k: recorded.append(a),
        )
        mod.get_panzer_soul()
        assert recorded, "the proxy substitution was silent"

    def test_a_real_soul_passes_straight_through(self, monkeypatch):
        import core.being.panzer_soul as mod

        class _RealSoul:
            drives = {"curiosity": object()}

        real = _RealSoul()
        monkeypatch.setattr(mod, "get_runtime_service", lambda *a, **k: real)
        soul = mod.get_panzer_soul()
        assert soul is real
        assert getattr(soul, "is_proxy", False) is False
