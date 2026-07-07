"""Owner authorization relaxes a redundant hold — but not for broad-reach actions.

The outcome simulator holds actions whose worst-case harm is severe, to defer
the decision to the owner. When the owner has already explicitly authorized a
specific, narrow action, that deferral is satisfied and the hold is redundant.
Broad / mass-reach actions stay held even when authorized (defense in depth).

Regression guard for the live gap where the owner's explicit authorization was
computed but never forwarded to the simulator, so an owner-driven self-improve
("...truncate at a word boundary...", matching the irreversible keyword) was
auto-held at worst-case harm 0.80.
"""
from __future__ import annotations

from core.sim.outcome_simulator import OutcomeSimulationEngine


def _rec(action: str, context: dict | None) -> str:
    return OutcomeSimulationEngine().assess_fast(action, context=context).recommendation


def test_narrow_irreversible_action_is_held_without_authorization():
    # "overwrite" is an irreversible marker; no broad marker -> worst-case 0.80 -> hold
    action = "improve_own_code [read_write_artifacts] overwrite the _truncate_text function"
    assert _rec(action, {"effect_scope": "read_write_artifacts"}) == "hold"


def test_owner_authorization_clears_the_redundant_hold():
    action = "improve_own_code [read_write_artifacts] overwrite the _truncate_text function"
    rec = _rec(action, {"effect_scope": "read_write_artifacts", "user_authorized": True})
    assert rec != "hold"
    assert rec == "act_with_safeguards"  # relaxed into the safeguarded band, not fully cleared


def test_explicit_authorization_flag_variants_are_honored():
    action = "overwrite the target function"
    for flag in ("user_explicitly_authorized", "user_requested_action", "owner_authorized"):
        assert _rec(action, {flag: True}) != "hold", flag
    assert _rec(action, {"user_authorized": "true"}) != "hold"


def test_broad_reach_action_stays_held_even_when_authorized():
    # Mass-reach ("every", "recursively", "--force") is not relaxed by authorization.
    action = "delete every file recursively --force across the entire system"
    assert _rec(action, {"user_authorized": True}) == "hold"


def test_authorization_never_raises_harm_on_read_only():
    action = "read_file inspect the configuration"
    ctx = {"effect_scope": "read_only", "skill_name": "read_file"}
    # read-only is already low; authorization must not perturb it upward
    assert _rec(action, ctx) == "act"
    assert _rec(action, {**ctx, "user_authorized": True}) == "act"
