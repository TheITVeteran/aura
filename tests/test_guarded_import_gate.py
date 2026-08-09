"""The gate that stops a renamed symbol from silently disabling a feature.

Every case here is a shape that actually occurred in this repository. The
gate is only worth having if it fires on those and stays quiet on the four
legitimate patterns the codebase relies on, so both directions are tested.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools import lint_guarded_imports as gate  # noqa: E402


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A throwaway source tree the gate treats as the repository root."""
    (tmp_path / "core").mkdir()
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    for cached in (gate._module_path, gate._is_package_dir, gate._exported_names):
        cached.cache_clear()
    yield tmp_path
    for cached in (gate._module_path, gate._is_package_dir, gate._exported_names):
        cached.cache_clear()


def _write(root: Path, relative: str, body: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def test_a_renamed_symbol_behind_import_error_is_caught(tree):
    """OutcomeSimulator -> OutcomeSimulationEngine, the real case."""
    _write(tree, "core/sim.py", "class OutcomeSimulationEngine:\n    pass\n")
    _write(
        tree,
        "core/caller.py",
        "def go():\n"
        "    try:\n"
        "        from core.sim import OutcomeSimulator\n"
        "        return OutcomeSimulator()\n"
        "    except ImportError:\n"
        "        return None\n",
    )
    findings, _ = gate._findings()
    assert len(findings) == 1
    assert "OutcomeSimulator" in findings[0]


def test_a_symbol_that_never_existed_is_caught(tree):
    """PRIME_DIRECTIVES: the module was real, the name never was."""
    _write(tree, "core/values.py", "class PrimeDirectives:\n    pass\n")
    _write(
        tree,
        "core/authority.py",
        "def load():\n"
        "    try:\n"
        "        from core.values import PRIME_DIRECTIVES\n"
        "        return PRIME_DIRECTIVES\n"
        "    except ImportError:\n"
        "        return {}\n",
    )
    findings, _ = gate._findings()
    assert len(findings) == 1
    assert "PRIME_DIRECTIVES" in findings[0]


def test_a_bare_except_exception_hides_it_just_as_well(tree):
    """`except Exception` catches ImportError; the gate must treat it the same."""
    _write(tree, "core/sim.py", "class Real:\n    pass\n")
    _write(
        tree,
        "core/caller.py",
        "def go():\n"
        "    try:\n"
        "        from core.sim import Gone\n"
        "        return Gone\n"
        "    except Exception:\n"
        "        return None\n",
    )
    findings, _ = gate._findings()
    assert len(findings) == 1


def test_a_deleted_module_is_caught(tree):
    _write(
        tree,
        "core/caller.py",
        "def go():\n"
        "    try:\n"
        "        from core.memory_store import Strategy\n"
        "        return Strategy\n"
        "    except ImportError:\n"
        "        return None\n",
    )
    findings, _ = gate._findings()
    assert len(findings) == 1
    assert "core.memory_store" in findings[0]


class TestPatternsThatMustNotFire:
    """Four things this codebase does on purpose. A gate that flags these
    gets switched off, and then it protects nothing."""

    def test_pep_562_module_getattr_exports_are_accepted(self, tree):
        # core/consciousness and core/world_model/belief_graph both keep
        # construction off the import path this way. Measured: the eager
        # version cost ~17 minutes of boot.
        _write(
            tree,
            "core/lazy/__init__.py",
            '__all__ = ["ConsciousnessSystem"]\n\n'
            "def __getattr__(name):\n"
            "    if name == 'ConsciousnessSystem':\n"
            "        from core.lazy.system import ConsciousnessSystem\n"
            "        return ConsciousnessSystem\n"
            "    raise AttributeError(name)\n",
        )
        _write(tree, "core/lazy/system.py", "class ConsciousnessSystem:\n    pass\n")
        _write(
            tree,
            "core/caller.py",
            "def go():\n"
            "    try:\n"
            "        from core.lazy import ConsciousnessSystem\n"
            "        return ConsciousnessSystem\n"
            "    except ImportError:\n"
            "        return None\n",
        )
        findings, _ = gate._findings()
        assert findings == []

    def test_a_namespace_package_submodule_is_accepted(self, tree):
        # tools/ has no __init__.py and is imported all the same.
        _write(tree, "tools/thing.py", "VALUE = 1\n")
        _write(
            tree,
            "core/caller.py",
            "def go():\n"
            "    try:\n"
            "        from tools import thing\n"
            "        return thing\n"
            "    except ImportError:\n"
            "        return None\n",
        )
        findings, _ = gate._findings()
        assert findings == []

    def test_a_conditionally_bound_name_is_accepted(self, tree):
        _write(
            tree,
            "core/optional.py",
            "try:\n"
            "    import numpy\n"
            "    HAVE_NUMPY = True\n"
            "except ImportError:\n"
            "    HAVE_NUMPY = False\n",
        )
        _write(
            tree,
            "core/caller.py",
            "def go():\n"
            "    try:\n"
            "        from core.optional import HAVE_NUMPY\n"
            "        return HAVE_NUMPY\n"
            "    except ImportError:\n"
            "        return False\n",
        )
        findings, _ = gate._findings()
        assert findings == []

    def test_a_third_party_optional_dependency_is_not_this_gates_business(self, tree):
        _write(
            tree,
            "core/caller.py",
            "def go():\n"
            "    try:\n"
            "        from mlx_lm import load\n"
            "        return load\n"
            "    except ImportError:\n"
            "        return None\n",
        )
        findings, _ = gate._findings()
        assert findings == []


def test_the_live_repository_has_no_dead_guarded_imports():
    """The gate, against the real tree. This is the ratchet."""
    findings, checked = gate._findings()
    assert checked > 3000, "the scanner stopped seeing the codebase"
    assert findings == [], "\n".join(findings)
