"""There is exactly one affect engine, and it is the one that runs.

Aura shipped two. ``core/affect/damasio_v2.py::AffectEngineV2`` is registered as
the required `affect_engine` service and is read by the metabolic coordinator,
vector memory, self_object, runtime tools and spiking active inference.
``core/affect/__init__.py::AffectEngine`` was a PAD engine documented as the
"lightweight fallback when DamasioV2 is unavailable" whose only constructor
anywhere was ``core/affect/emotion_engine.py`` — a module with zero importers.
The fallback had no path to it and could never have run, while two tests spent
their coverage on it.

It was removed rather than wired up. Wiring it would break CP126's rule that
"no engine must never look like a calm engine": a silent fallback to a
different affect model answers with confident values from a model nothing else
reads, and no caller can tell which engine produced them. Unavailability is the
honest output.

This mirrors tests/test_one_canonical_workspace.py, which pins the same
property for the broadcast workspace.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
AFFECT_DIR = ROOT / "core" / "affect"

#: The one engine. Anything else defining an affect-engine class is a second.
CANONICAL_MODULE = "damasio_v2.py"
CANONICAL_CLASS = "AffectEngineV2"


def _engine_classes() -> dict[str, list[str]]:
    """Every class under core/affect whose name looks like an affect engine."""
    found: dict[str, list[str]] = {}
    for path in sorted(AFFECT_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name.endswith("AffectEngine")
            or isinstance(node, ast.ClassDef)
            and node.name.startswith("AffectEngine")
        ]
        if names:
            found[path.name] = sorted(set(names))
    return found


def test_only_one_module_defines_an_affect_engine():
    found = _engine_classes()
    assert list(found) == [CANONICAL_MODULE], (
        f"more than one module under core/affect defines an affect engine: {found}"
    )
    assert found[CANONICAL_MODULE] == [CANONICAL_CLASS]


def test_the_retired_pad_engine_is_gone():
    import core.affect as affect

    assert not hasattr(affect, "AffectEngine")
    assert hasattr(affect, "AffectState"), "AffectState is canonical and must stay"


def test_the_retired_shim_no_longer_builds_an_engine_on_import():
    """It used to construct one at module scope, so importing it forked state."""
    source = (AFFECT_DIR / "emotion_engine.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    module_level_calls = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
    ]
    assert not module_level_calls, (
        f"emotion_engine.py constructs at import time again: {module_level_calls}"
    )


def test_the_shim_still_imports_and_points_at_the_canonical_engine():
    from core.affect.damasio_v2 import AffectEngineV2
    from core.affect.emotion_engine import AffectEngineV2 as ReExported

    assert ReExported is AffectEngineV2


def test_the_shim_accessor_returns_the_registered_service_not_a_new_engine():
    """Handing back a private engine is how the second one stayed alive."""
    from core.affect.emotion_engine import get_emotion_engine

    # With nothing registered the honest answer is None, not a fresh engine.
    assert get_emotion_engine() is None or hasattr(get_emotion_engine(), "markers")


def test_the_registered_service_is_the_canonical_engine():
    """Whatever the provider builds must be the one class that survives."""
    provider = (ROOT / "core" / "providers" / "consciousness_provider.py").read_text(
        encoding="utf-8"
    )
    assert "from core.affect.damasio_v2 import AffectEngineV2" in provider
    assert "container.register('affect_engine'" in provider


@pytest.mark.parametrize("dead_name", ["EmotionEngine", "LegacyEmotionState"])
def test_the_legacy_compatibility_classes_are_not_resurrected(dead_name: str):
    source = (AFFECT_DIR / "emotion_engine.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    assert dead_name not in classes
