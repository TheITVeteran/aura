"""A refused critic must say which invariant it failed.

LIVE DEFECT, 2026-08-10. Every foreground turn on the live desktop logged

    Latent critic authority rejected: critic function identity is not
    independently proven

and the episode receipt carried ``disjoint_critic_authority_unproven``. Eleven
distinct conditions in validate_critic_identity shared that one sentence, so
there was no way to learn WHICH invariant was unmet. The recurrent cortex ran
without critic authority on every turn — the verifier attaches as None — and
the reason stayed illegible.

The real cause: the critic's declared source closure imports four internal
modules it does not declare, and two of them (neural_transition_tissue,
systematic_neural_alu) import mlx.core and mlx.nn at module scope. Enlarging
the closure to cover them makes the audit report forbidden_imports=['mlx.core',
'mlx.nn'] — so the closure is not the "deterministic_symbolic_parameterless"
thing its own receipt claims. That architectural conflict is real, and the
audit is right to refuse; it was simply never legible.
"""

from __future__ import annotations

import pytest


def _identity():
    from core.brain.llm.latent_cortex.critic_identity import build_critic_identity
    from core.brain.llm.latent_cortex.task_verifiers import EpisodeTaskVerifier
    from tests.fixtures.rlc_runtime_integrity import complete_worker_identity

    worker_identity = complete_worker_identity()
    verifier = EpisodeTaskVerifier(
        "answer with an integer", facet_reliability={}, response_contract=""
    )
    return build_critic_identity(verifier, worker_identity=worker_identity), worker_identity


def test_refusal_names_the_failing_invariant() -> None:
    from core.brain.llm.latent_cortex.critic_identity import validate_critic_identity

    identity, worker_identity = _identity()

    try:
        validate_critic_identity(identity, worker_identity=worker_identity)
    except ValueError as exc:
        message = str(exc)
    else:  # pragma: no cover - passes only once the closure conflict is resolved
        pytest.skip("critic identity now validates; the diagnosis path is unused")

    # The generic sentence must still open it, for anything matching on it.
    assert "critic function identity is not independently proven" in message
    # And it must now carry a cause.
    assert ":" in message
    assert message.rstrip().endswith(".") is False
    assert "no individual condition reported a cause" not in message


def test_refusal_names_the_undeclared_modules() -> None:
    """An operator must learn which imports broke the audit."""
    from core.brain.llm.latent_cortex.critic_identity import validate_critic_identity

    identity, worker_identity = _identity()

    try:
        validate_critic_identity(identity, worker_identity=worker_identity)
    except ValueError as exc:
        message = str(exc)
    else:  # pragma: no cover
        pytest.skip("critic identity now validates")

    assert "dependency_audit_failed" in message
    assert "undeclared_internal_imports" in message

    # The PROPERTY, not one module's name. This used to assert the literal
    # "neural_transition_tissue", which pinned whichever module happened to
    # be undeclared the day it was written — the offender is now
    # `episodic_neural_readout_contract`, and the test failed for a rename
    # while the guarantee it exists to protect was completely intact.
    #
    # What an operator needs is a concrete import path to go and look at, so
    # that is what is checked: at least one dotted module under core.
    import re

    named = re.findall(r"core(?:\.[a-z_][a-z0-9_]*)+", message)
    assert named, (
        f"the refusal names no concrete module, so an operator cannot tell "
        f"which imports broke the audit: {message}"
    )


def test_the_neural_closure_conflict_is_real_not_bookkeeping() -> None:
    """Declaring the modules would surface a forbidden framework, not fix it.

    This is the evidence that the audit's refusal is substantive: widening the
    closure to include the four undeclared modules makes mlx.core and mlx.nn
    visible as forbidden imports. Making this audit pass therefore requires
    removing the neural dependency from the critic, not extending an allowlist.
    """
    from pathlib import Path

    from core.brain.llm.latent_cortex.critic_identity import (
        _CRITIC_SOURCE_FILES,
        audit_python_dependencies,
    )

    root = Path(__file__).resolve().parents[2]
    extra = [
        "core/brain/llm/latent_cortex/neural_transition_tissue.py",
        "core/brain/llm/latent_cortex/systematic_neural_alu.py",
        "core/brain/llm/latent_cortex/typed_action_compiler.py",
        "core/brain/llm/latent_cortex/typed_program_executor.py",
    ]
    sources = {}
    for relative in list(_CRITIC_SOURCE_FILES) + extra:
        path = root / relative
        if path.exists():
            sources[relative] = path.read_text(encoding="utf-8")

    audit = audit_python_dependencies(sources)

    assert audit["passed"] is False
    assert "mlx.core" in audit["forbidden_imports"]


def test_a_clean_closure_still_passes_its_audit() -> None:
    """The audit must not be broken in the permissive direction."""
    from core.brain.llm.latent_cortex.critic_identity import audit_python_dependencies

    audit = audit_python_dependencies(
        {"clean.py": "import json\nfrom core.brain.canonical_json import dumps\n"}
    )

    assert audit["passed"] is True
    assert audit["forbidden_imports"] == []


def test_a_forbidden_framework_is_still_caught() -> None:
    from core.brain.llm.latent_cortex.critic_identity import audit_python_dependencies

    audit = audit_python_dependencies({"bad.py": "import mlx.core as mx\n"})

    assert audit["passed"] is False
    assert audit["forbidden_imports"] == ["mlx.core"]
