"""Silent decay of optional integrations must become a loud, testable fact.

The failure has a signature: a subsystem imports, its tests stay green, and
it does nothing. Aura's core stays honest because it is exercised
constantly; optional dependencies rot quietly because nothing runs them —
the XTTS/Coqui voice path was dead from an import error and nothing noticed.

The mechanism that hides it is `importlib.util.find_spec` used as an
availability check. find_spec answers "is this module on disk", which is not
the question: a package that is present but raises on import is reported
available, the readiness surface reports healthy, and the feature is dead.
"""
from __future__ import annotations

import sys

import pytest

from core.runtime import integration_liveness as il


class TestProbeDistinguishesTheThreeStates:
    def test_a_healthy_module_is_live(self):
        result = il.probe(il.Integration("json", "json", "stdlib control"))
        assert result.state == il.LIVE
        assert result.is_defect is False

    def test_a_missing_module_is_absent_not_broken(self):
        """An optional dependency that is not installed is not a defect."""
        result = il.probe(
            il.Integration("ghost", "definitely_not_installed_xyz", "nothing"),
        )
        assert result.state == il.ABSENT
        assert result.is_defect is False

    def test_an_unimportable_module_is_broken(self, tmp_path, monkeypatch):
        """The silent-decay case: present on disk, raises on import.

        find_spec returns a spec for this module — it exists. Importing it
        raises. That is precisely the state the old check reported as
        available.
        """
        package = tmp_path / "decayed_pkg"
        package.mkdir()
        (package / "__init__.py").write_text(
            "raise ImportError('cannot import name moved_api')", encoding="utf-8",
        )
        monkeypatch.setenv("PYTHONPATH", str(tmp_path))

        integration = il.Integration("decayed", "decayed_pkg", "a dead feature")
        result = il.probe(integration)
        assert result.state == il.BROKEN
        assert result.is_defect is True
        assert "ImportError" in result.detail

    def test_find_spec_would_have_called_it_available(self, tmp_path, monkeypatch):
        """Pins WHY the probe exists, by showing the old check disagreeing."""
        import importlib.util

        package = tmp_path / "decayed_two"
        package.mkdir()
        (package / "__init__.py").write_text("raise RuntimeError('boom')", encoding="utf-8")
        monkeypatch.syspath_prepend(str(tmp_path))
        importlib.invalidate_caches()

        # The old-style check: present on disk, so "available".
        assert importlib.util.find_spec("decayed_two") is not None

        monkeypatch.setenv("PYTHONPATH", str(tmp_path))
        result = il.probe(il.Integration("decayed_two", "decayed_two", "x"))
        assert result.state == il.BROKEN


class TestApiDriftIsCaught:
    def test_a_missing_required_attribute_is_broken(self):
        """An import that succeeds against a moved API is still dead."""
        result = il.probe(
            il.Integration(
                "json_drift", "json", "stdlib", requires_attrs=("no_such_function",),
            ),
        )
        assert result.state == il.BROKEN
        assert "missing attributes" in result.detail

    def test_a_present_attribute_passes(self):
        result = il.probe(
            il.Integration("json_ok", "json", "stdlib", requires_attrs=("loads", "dumps")),
        )
        assert result.state == il.LIVE


class TestPreflightIsPartOfTheIntegration:
    def test_a_preflight_runs_before_the_import(self, tmp_path, monkeypatch):
        """Probing a raw import in a vacuum answers the wrong question.

        Coqui TTS does not import against current transformers; Aura installs
        a shim first. A probe without the preflight reports a defect the
        running system does not have.
        """
        (tmp_path / "shim_mod.py").write_text(
            "import sys, types\n"
            "def install():\n"
            "    sys.modules['repaired_pkg'] = types.ModuleType('repaired_pkg')\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("PYTHONPATH", str(tmp_path))

        without = il.probe(il.Integration("r", "repaired_pkg", "x"))
        assert without.state == il.ABSENT

        with_shim = il.probe(
            il.Integration("r", "repaired_pkg", "x", preflight="shim_mod:install"),
        )
        assert with_shim.state == il.LIVE

    def test_a_failing_preflight_is_itself_broken(self, tmp_path, monkeypatch):
        (tmp_path / "bad_shim.py").write_text(
            "def install():\n    raise RuntimeError('shim is broken')\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("PYTHONPATH", str(tmp_path))
        result = il.probe(
            il.Integration("j", "json", "stdlib", preflight="bad_shim:install"),
        )
        assert result.state == il.BROKEN
        assert "preflight" in result.detail


class TestProbeCannotWedgeTheGate:
    def test_a_hanging_import_times_out_as_broken(self, tmp_path, monkeypatch):
        package = tmp_path / "sleepy_pkg"
        package.mkdir()
        (package / "__init__.py").write_text("import time; time.sleep(30)", encoding="utf-8")
        monkeypatch.setenv("PYTHONPATH", str(tmp_path))

        result = il.probe(
            il.Integration("sleepy", "sleepy_pkg", "x"), timeout_s=2.0,
        )
        assert result.state == il.BROKEN
        assert "did not complete" in result.detail

    def test_a_child_that_dies_is_not_evidence_of_health(self, tmp_path, monkeypatch):
        package = tmp_path / "crashy_pkg"
        package.mkdir()
        (package / "__init__.py").write_text("import os; os._exit(3)", encoding="utf-8")
        monkeypatch.setenv("PYTHONPATH", str(tmp_path))

        result = il.probe(il.Integration("crashy", "crashy_pkg", "x"))
        assert result.state == il.BROKEN


class TestReportSemantics:
    def _report(self, states):
        results = []
        for index, state in enumerate(states):
            results.append(
                il.ProbeResult(il.Integration(f"i{index}", f"m{index}", "x"), state),
            )
        return il.LivenessReport(results)

    def test_only_broken_fails_the_gate(self):
        assert self._report([il.LIVE, il.ABSENT]).ok is True
        assert self._report([il.LIVE, il.BROKEN]).ok is False

    def test_absent_optional_dependencies_are_reported_not_failed(self):
        report = self._report([il.ABSENT, il.ABSENT])
        assert report.ok is True
        assert len(report.absent) == 2

    def test_the_report_is_serializable(self):
        payload = self._report([il.LIVE, il.BROKEN, il.ABSENT]).to_dict()
        assert payload["schema"] == "aura.integration_liveness.v1"
        assert payload["broken"] == 1
        assert payload["ok"] is False
        assert len(payload["results"]) == 3


class TestDeclaredIntegrationsAreCoherent:
    def test_every_integration_says_what_it_powers(self):
        for integration in il.INTEGRATIONS:
            assert integration.powers.strip(), integration.name
            assert integration.module.strip(), integration.name

    def test_names_are_unique(self):
        names = [i.name for i in il.INTEGRATIONS]
        assert len(names) == len(set(names))

    def test_the_voice_stack_is_declared(self):
        names = {i.name for i in il.INTEGRATIONS}
        # These are the paths that decayed silently; they must stay covered.
        assert {"coqui_tts", "faster_whisper", "sounddevice"} <= names


class TestVoiceAvailabilityHonoursItsOwnFailures:
    """The concrete defect: a status surface reporting a backend available
    after having already watched its import fail."""

    def test_tts_availability_respects_a_failed_import(self, monkeypatch):
        import core.senses.voice_engine as ve

        monkeypatch.setattr(ve, "TTS", None, raising=False)
        monkeypatch.setattr(ve, "_tts_api_import_attempted", True, raising=False)
        assert ve._tts_dependency_available() is False

    def test_tts_availability_is_true_after_a_real_import(self, monkeypatch):
        import core.senses.voice_engine as ve

        monkeypatch.setattr(ve, "TTS", object(), raising=False)
        monkeypatch.setattr(ve, "_tts_api_import_attempted", True, raising=False)
        assert ve._tts_dependency_available() is True

    def test_stt_availability_respects_a_failed_import(self, monkeypatch):
        import core.senses.voice_engine as ve

        monkeypatch.setattr(ve, "_WhisperModel", None, raising=False)
        monkeypatch.setattr(ve, "_whisper_import_attempted", True, raising=False)
        assert ve._stt_dependency_available() is False

    def test_presence_only_stands_in_before_an_attempt(self, monkeypatch):
        import core.senses.voice_engine as ve

        monkeypatch.setattr(ve, "TTS", None, raising=False)
        monkeypatch.setattr(ve, "_tts_api_import_attempted", False, raising=False)
        # Not yet attempted: presence on disk is allowed to stand in.
        assert ve._tts_dependency_available() is True


@pytest.mark.skipif(
    sys.platform != "darwin", reason="the declared stack is the macOS voice stack",
)
class TestTheRealStackIsAlive:
    def test_no_declared_integration_is_broken(self):
        """The gate itself: every declared integration must import for real.

        An absent optional dependency is fine. A present-but-unimportable
        one is the defect, and this is the assertion that would have caught
        the dead XTTS path the day it broke.
        """
        report = il.probe_all()
        broken = [
            f"{r.integration.name}: {r.detail}" for r in report.broken
        ]
        assert not broken, "broken integrations: " + "; ".join(broken)
