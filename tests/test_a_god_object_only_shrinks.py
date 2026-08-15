"""Module size, as a ratchet.

`interface/routes/chat.py` is 29,481 lines with 457 module-level functions.
`core/brain/llm/mlx_client.py` is 15,423 with a 165-method class.
`core/brain/inference_gate.py` is 13,416 with a 193-method class handling worker
processes, cloud fallback, health probing, warm-up, desktop resource guards,
background deferral, PII scrubbing, PBKDF2 offloading, RAM diagnostics and UI
prompt strings. Thirty-two files are over three thousand lines.

None of that is fixable in one commit. What is fixable in one commit is the
direction of travel: nothing stopped chat.py reaching forty thousand lines, and
nothing stopped the next God object being created from scratch.
"""
from __future__ import annotations

import json
from pathlib import Path

from tools.lint_module_size import (
    MAX_NEW_CLASS_METHODS,
    MAX_NEW_MODULE_LINES,
    Measurement,
    check,
    load_baseline,
    measure_tree,
)

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "config" / "module_size_baseline.json"


def _live() -> tuple[dict[str, Measurement], dict[str, dict[str, int]]]:
    return measure_tree(), load_baseline(BASELINE)


def test_the_tree_is_within_its_baseline():
    measurements, baseline = _live()
    failures, stale = check(measurements, baseline)

    assert failures == [], "\n".join(failures)
    assert stale == [], "\n".join(stale)


def test_the_baseline_records_the_known_offenders():
    """A baseline that omitted them would pass while they grew."""
    baseline = load_baseline(BASELINE)

    for path in (
        "interface/routes/chat.py",
        "core/brain/inference_gate.py",
        "core/brain/llm/mlx_client.py",
    ):
        assert path in baseline, path


def test_growth_in_a_baselined_file_fails():
    measurements, baseline = _live()
    tightened = dict(baseline)
    tightened["core/brain/inference_gate.py"] = {"lines": 100, "max_class_methods": 5}

    failures, _ = check(measurements, tightened)

    assert any("inference_gate" in f and "grew to" in f for f in failures), failures


def test_a_new_oversized_module_is_never_grandfathered():
    measurements, baseline = _live()
    measurements = dict(measurements)
    measurements["core/a_brand_new_god.py"] = Measurement(
        path="core/a_brand_new_god.py",
        lines=MAX_NEW_MODULE_LINES + 1,
        max_class_methods=2,
        largest_class="Small",
    )

    failures, _ = check(measurements, baseline)

    assert any("a_brand_new_god" in f for f in failures), failures


def test_a_new_oversized_class_is_never_grandfathered():
    measurements, baseline = _live()
    measurements = dict(measurements)
    measurements["core/a_brand_new_class.py"] = Measurement(
        path="core/a_brand_new_class.py",
        lines=50,
        max_class_methods=MAX_NEW_CLASS_METHODS + 1,
        largest_class="Everything",
    )

    failures, _ = check(measurements, baseline)

    assert any("Everything" in f for f in failures), failures


def test_a_file_that_shrank_must_be_re_recorded():
    """A stale entry is headroom nobody earned, and it is how a ratchet quietly
    stops ratcheting."""
    measurements, baseline = _live()
    measurements = dict(measurements)
    real = measurements["core/brain/inference_gate.py"]
    measurements["core/brain/inference_gate.py"] = Measurement(
        path=real.path,
        lines=real.lines - 500,
        max_class_methods=real.max_class_methods,
        largest_class=real.largest_class,
    )

    _, stale = check(measurements, baseline)

    assert any("inference_gate" in s for s in stale), stale


def test_a_deleted_file_leaves_no_entry_behind():
    measurements, baseline = _live()
    measurements = dict(measurements)
    measurements.pop("core/brain/inference_gate.py")

    _, stale = check(measurements, baseline)

    assert any("no longer exists" in s for s in stale), stale


def test_an_ordinary_file_needs_no_entry():
    measurements, baseline = _live()
    measurements = dict(measurements)
    measurements["core/a_normal_module.py"] = Measurement(
        path="core/a_normal_module.py", lines=200, max_class_methods=6, largest_class="Ok"
    )

    failures, stale = check(measurements, baseline)

    assert not any("a_normal_module" in f for f in failures)
    assert not any("a_normal_module" in s for s in stale)


def test_the_thresholds_come_from_the_distribution_not_from_taste():
    """p98 of file length in this tree is 2,115 lines and p98 of class size is
    26 methods. A ceiling far from those would be an opinion."""
    import ast

    measurements, _ = _live()
    lines = sorted(m.lines for m in measurements.values())
    p98_lines = lines[int(len(lines) * 0.98)]
    assert 0.5 * p98_lines <= MAX_NEW_MODULE_LINES <= 1.5 * p98_lines

    # Measured over CLASSES, which is what the threshold governs and what the
    # tool's docstring cites. A per-file maximum is a different distribution
    # and would justify a different number, so checking that one would be
    # checking a claim nobody made.
    per_class: list[int] = []
    for root in ("core", "interface"):
        for path in (ROOT / root).rglob("*.py"):
            if "__pycache__" in str(path):
                continue
            try:
                tree = ast.parse(path.read_text("utf-8"))
            except (OSError, SyntaxError, UnicodeError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    per_class.append(
                        sum(
                            1
                            for item in node.body
                            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                        )
                    )
    per_class.sort()
    p98_methods = per_class[int(len(per_class) * 0.98)]

    assert MAX_NEW_CLASS_METHODS >= p98_methods
    assert MAX_NEW_CLASS_METHODS <= 2 * p98_methods


def test_the_baseline_is_a_record_of_offenders_not_of_everything():
    payload = json.loads(BASELINE.read_text("utf-8"))

    assert payload["schema"] == "aura.module_size_baseline.v1"
    assert 0 < len(payload["modules"]) < 200, "a baseline of everything is noise"


def test_the_gate_is_wired_into_the_makefile():
    makefile = (ROOT / "Makefile").read_text("utf-8")

    assert "module-size:" in makefile
    assert "module-size-baseline:" in makefile
    assert "tools/lint_module_size.py" in makefile
