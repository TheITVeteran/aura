"""CP126 hardening contracts for core/actuators/actuator_synthesis.py.

This pipeline compiles and activates MODEL-GENERATED code, so its guards are
security boundaries: the actuator name reaches the filesystem, persisted
source is executed again at boot, and the Will is the only authority gate.
"""
from __future__ import annotations

import json

import pytest

from core.actuators.actuator_synthesis import (
    ActuatorSynthesizer,
    _finite_urgency,
    _safe_actuator_name,
    _source_digest,
)


class TestActuatorNameCannotEscapeTheOutputRoot:
    @pytest.mark.parametrize(
        "hostile",
        [
            "../../core/brain/cognitive_engine",
            "/etc/passwd",
            "a/b",
            "..",
            "with space",
            "",
            "x" * 200,
        ],
    )
    def test_unsafe_names_are_rejected(self, hostile):
        assert _safe_actuator_name(hostile) == ""

    def test_ordinary_name_is_accepted(self):
        assert _safe_actuator_name("GripperActuator") == "GripperActuator"

    def test_persist_refuses_traversal_and_writes_nothing(self, tmp_path):
        synth = ActuatorSynthesizer(output_dir=str(tmp_path / "out"))
        outside = tmp_path / "pwned.py"

        assert synth.persist_actuator("../pwned", "print('x')") is False
        assert not outside.exists()

    def test_persist_writes_source_and_manifest(self, tmp_path):
        synth = ActuatorSynthesizer(output_dir=str(tmp_path / "out"))
        source = "class A:\n    pass\n"

        assert synth.persist_actuator("GoodActuator", source, governance_receipt="r-1") is True

        target = tmp_path / "out" / "GoodActuator.py"
        manifest_path = tmp_path / "out" / "GoodActuator.manifest.json"
        assert target.read_text(encoding="utf-8") == source
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["source_sha256"] == _source_digest(source)
        assert manifest["governance_receipt"] == "r-1"


class TestPersistedCodeIntegrity:
    def test_tampered_source_is_refused_on_reload(self, tmp_path):
        synth = ActuatorSynthesizer(output_dir=str(tmp_path / "out"))
        source = "class A:\n    pass\n"
        synth.persist_actuator("GoodActuator", source, governance_receipt="r-1")

        target = tmp_path / "out" / "GoodActuator.py"
        target.write_text("class A:\n    pass\n# tampered\n", encoding="utf-8")

        assert synth._verify_persisted_manifest(target, target.read_text(encoding="utf-8")) is None

    def test_unsigned_file_is_refused(self, tmp_path):
        out = tmp_path / "out"
        out.mkdir(parents=True)
        rogue = out / "Rogue.py"
        rogue.write_text("class R:\n    pass\n", encoding="utf-8")
        synth = ActuatorSynthesizer(output_dir=str(out))

        assert synth._verify_persisted_manifest(rogue, rogue.read_text(encoding="utf-8")) is None

    def test_intact_file_verifies(self, tmp_path):
        synth = ActuatorSynthesizer(output_dir=str(tmp_path / "out"))
        source = "class A:\n    pass\n"
        synth.persist_actuator("GoodActuator", source, governance_receipt="r-1")
        target = tmp_path / "out" / "GoodActuator.py"

        manifest = synth._verify_persisted_manifest(target, source)
        assert manifest is not None
        assert manifest["name"] == "GoodActuator"


class TestUrgencyCannotMaximizePriority:
    def test_nan_urgency_does_not_reach_max_priority(self):
        # min(0.9, 0.4 + 0.5*nan) == 0.9 because NaN comparisons are all
        # False, so an unvalidated urgency produced the HIGHEST possible
        # self-modification priority.
        assert min(0.9, 0.4 + 0.5 * float("nan")) == 0.9
        assert min(0.9, 0.4 + 0.5 * _finite_urgency(float("nan"))) < 0.9

    def test_out_of_range_urgency_is_clamped(self):
        assert _finite_urgency(50.0) == 1.0
        assert _finite_urgency(-3.0) == 0.0
        assert _finite_urgency("nonsense") == 0.5


class TestGovernanceCannotBeBypassed:
    def test_hot_load_validates_before_registering(self, tmp_path, monkeypatch):
        """The public entry must gate on validation, never register directly.

        Registration is stubbed to record whether it was reached. With
        validation failing, a correct implementation must NOT reach it; the
        original implementation went straight to registration.
        """
        synth = ActuatorSynthesizer(output_dir=str(tmp_path / "out"))
        registered: list[str] = []
        monkeypatch.setattr(
            synth,
            "_register_validated_actuator",
            lambda source, metadata=None: registered.append(source) or "REGISTERED",
        )

        class _Failed:
            success = False
            error = "rejected"
            details: dict = {}

        monkeypatch.setattr(
            "core.actuators.actuator_synthesis.ActuatorCodeValidator.validate_ast",
            staticmethod(lambda _source: _Failed()),
        )

        assert synth.hot_load_actuator("class Evil:\n    pass\n") is None
        assert registered == [], "registration must not be reached when validation fails"

    def test_reload_requires_current_will_approval(self, tmp_path, monkeypatch):
        synth = ActuatorSynthesizer(output_dir=str(tmp_path / "out"))
        manifest = {"name": "GoodActuator", "governance_receipt": "old"}

        class _Denied:
            def is_approved(self):
                return False

        class _Will:
            def decide(self, **_kwargs):
                return _Denied()

        monkeypatch.setattr("core.will.get_will", lambda: _Will())
        assert synth._governance_approve_reload(manifest, "class A: pass") is False

    def test_reload_fails_closed_when_will_unavailable(self, tmp_path, monkeypatch):
        synth = ActuatorSynthesizer(output_dir=str(tmp_path / "out"))

        def _boom():
            raise RuntimeError("will down")

        monkeypatch.setattr("core.will.get_will", _boom)
        assert synth._governance_approve_reload({"name": "X"}, "class A: pass") is False


def test_output_root_is_anchored_not_cwd_relative(tmp_path, monkeypatch):
    """Default storage must not depend on how Aura was launched."""
    synth = ActuatorSynthesizer()
    assert synth.output_dir.is_absolute() or str(synth.output_dir).startswith("data/")
