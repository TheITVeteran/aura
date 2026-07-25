"""A daemon's log is not a terminal.

One live hour on 2026-07-25 wrote 4,650 stdout lines. 3,377 of them — 73% —
were sentence-transformers tqdm progress bars reading
``Batches: 100%|...| 1/1``. Every real event in that hour, including the
welfare defer storm and 192 silent immune failures, was buried under them.

Progress bars are for a human watching a script. Aura runs for days. Every
encode call has to opt out explicitly, so this is a ratchet, not a one-time
sweep: a new embedding call site without ``show_progress_bar=False`` fails
here rather than quietly re-flooding the operator's only window on the system.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[1]
_SEARCH_DIRS = ("core", "interface")

# Attribute names that denote a sentence-transformers style encoder.
_ENCODER_NAMES = {"encoder", "_encoder", "model", "_model", "embedder", "_embedder"}


def _encode_calls_missing_optout(root: Path = _ROOT) -> list[str]:
    offenders: list[str] = []
    for directory in _SEARCH_DIRS:
        for path in (root / directory).rglob("*.py"):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if not isinstance(func, ast.Attribute) or func.attr != "encode":
                    continue
                target = func.value
                name = getattr(target, "attr", None) or getattr(target, "id", None)
                if name not in _ENCODER_NAMES:
                    continue
                if any(kw.arg == "show_progress_bar" for kw in node.keywords):
                    continue
                offenders.append(
                    f"{path.relative_to(root)}:{node.lineno}"
                )
    return offenders


def test_no_embedding_call_may_print_progress_bars():
    offenders = _encode_calls_missing_optout()
    assert not offenders, (
        "these embedding calls will print tqdm progress bars into the live log; "
        "pass show_progress_bar=False: " + ", ".join(offenders)
    )


def test_the_detector_actually_detects(tmp_path):
    """A guard that cannot fail is not a guard."""
    probe = tmp_path / "core"
    probe.mkdir()
    (probe / "noisy.py").write_text("self._model.encode(texts)\n", encoding="utf-8")
    (probe / "quiet.py").write_text(
        "self._model.encode(texts, show_progress_bar=False)\n", encoding="utf-8"
    )

    assert _encode_calls_missing_optout(tmp_path) == ["core/noisy.py:1"]
