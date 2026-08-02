"""A skill that declares its dependencies must get an answer, not a crash.

``ServiceContainer.check_package`` had three callers and no definition:
per-skill requirement validation, the capability engine's async proxy, and
the boot dependency check. Every one raised AttributeError.
"""
from __future__ import annotations

import pytest

from core.container import ServiceContainer

_ABSENT = "definitely_not_a_real_package_zzq"


@pytest.fixture(autouse=True)
def _clear_cache():
    ServiceContainer._package_availability.clear()
    yield
    ServiceContainer._package_availability.clear()


def test_the_method_exists():
    """The whole finding: three call sites, no method."""
    assert callable(getattr(ServiceContainer, "check_package", None))


def test_an_installed_package_is_available():
    assert ServiceContainer.check_package("json") is True


def test_a_missing_package_is_not_available():
    assert ServiceContainer.check_package(_ABSENT) is False


def test_a_blank_name_is_not_available():
    assert ServiceContainer.check_package("") is False
    assert ServiceContainer.check_package("   ") is False
    assert ServiceContainer.check_package(None) is False


def test_an_installed_but_unimportable_package_reports_unavailable(monkeypatch):
    """The state find_spec would have called 'available'.

    A package whose __init__ raises is present on disk and dead in practice.
    Reporting it available is how a feature stays broken while the readiness
    surface stays green.
    """
    import importlib

    def _explode(name):
        raise RuntimeError("moved API in a transitive dependency")

    monkeypatch.setattr(importlib, "import_module", _explode)
    assert ServiceContainer.check_package("anything") is False


def test_results_are_cached():
    calls = []
    import importlib

    real = importlib.import_module

    def _counting(name, *args, **kwargs):
        calls.append(name)
        return real(name, *args, **kwargs)

    import unittest.mock

    with unittest.mock.patch.object(importlib, "import_module", _counting):
        ServiceContainer.check_package("json")
        ServiceContainer.check_package("json")
        ServiceContainer.check_package("json")
    assert calls.count("json") == 1


def test_auto_install_is_refused_not_performed(caplog):
    """Runtime installs mutate a shared venv; the refusal must be visible."""
    with caplog.at_level("WARNING"):
        result = ServiceContainer.check_package("json", auto_install=True)
    assert result is True
    assert any("refusing to install" in record.message for record in caplog.records)


def test_auto_install_does_not_make_a_missing_package_appear():
    assert ServiceContainer.check_package(_ABSENT, auto_install=True) is False


def test_skill_requirements_get_a_truthful_verdict():
    """The call site the finding named."""
    from core.capability_engine import SkillRequirements

    ok, errors = SkillRequirements(packages=["json"]).check()
    assert ok is True and errors == []

    ok, errors = SkillRequirements(packages=[_ABSENT]).check()
    assert ok is False
    assert any(_ABSENT in error for error in errors)
