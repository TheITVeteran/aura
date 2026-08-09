"""The RLC has to be on the phase that actually answers.

There are two substantial response architectures — ``ResponseGenerationPhase``
(legacy pipeline) and ``UnitaryResponsePhase`` (the sovereign kernel path) —
and for a while the complete foreground RLC route lived only on the legacy
one while the kernel path jumped straight to ``llm.think``. An external audit
read the legacy integration, found it beautiful, and concluded the RLC served
ordinary chat. It did not.

CP070 fixed it. These tests exist so it cannot quietly revert: a capability
on a phase that does not run is the same defect as a capability with no
caller, and it is harder to see because the code is right there.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNITARY = ROOT / "core" / "phases" / "response_generation_unitary.py"


def _calls_in(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


def test_the_kernel_installs_the_unitary_phase_as_its_response_phase():
    """Whichever phase this is, it is the one that has to carry the cortex."""
    from core.kernel import aura_kernel

    source = inspect.getsource(aura_kernel)
    assert "self.response_phase = UnitaryResponsePhase(self)" in source
    assert "self.response_phase = ResponseGenerationPhase(self)" not in source


def test_the_canonical_response_phase_runs_a_foreground_latent_episode():
    assert "run_foreground_latent_episode" in _calls_in(UNITARY), (
        "the sovereign response phase no longer routes a foreground RLC "
        "episode. If the cortex moved, it moved off the path that answers."
    )


def test_the_canonical_response_phase_composes_the_reasoning_amplifier():
    """CP073's point: compose the cortex WITH the amplifier, not pick one."""
    calls = _calls_in(UNITARY)
    assert "amplify_turn" in calls
    assert "reasoning_amplifier_v2_enabled" in calls


def test_the_composition_is_recorded_so_it_can_be_checked_in_a_receipt():
    source = UNITARY.read_text(encoding="utf-8")
    assert "latent_cortex_amplifier_composed" in source, (
        "nothing marks that the cortex and the amplifier both contributed; "
        "without it, 'the complete engine ran' is unfalsifiable from outside"
    )


def test_the_legacy_phase_is_not_silently_the_only_one_with_the_cortex():
    """The shape of the original defect, stated as a test.

    If the legacy phase gains an RLC route the canonical one lacks, that is
    the bug returning — good work landing on the pipeline that is not
    serving traffic.
    """
    legacy = ROOT / "core" / "phases" / "response_generation.py"
    legacy_calls = _calls_in(legacy)
    canonical_calls = _calls_in(UNITARY)

    if "run_foreground_latent_episode" in legacy_calls:
        assert "run_foreground_latent_episode" in canonical_calls, (
            "the legacy ResponseGenerationPhase routes a foreground latent "
            "episode and the canonical UnitaryResponsePhase does not. That is "
            "exactly the split CP070 closed."
        )
