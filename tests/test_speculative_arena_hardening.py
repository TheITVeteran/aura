from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.kernel.shadow_kernel import ShadowValidationReceipt
from core.kernel.speculative_arena import SpeculativeArena


class _State:
    def __init__(self) -> None:
        self.transition_cause = None
        self.response_modifiers = {}

    async def derive_async(self, label: str, origin: str = "system") -> "_State":
        child = _State()
        child.transition_cause = f"{origin}:{label}"
        return child


class _Sandbox:
    def __init__(self, receipt: ShadowValidationReceipt) -> None:
        self.receipt = receipt

    async def evaluate_mutation_safely(
        self,
        _mutated_code: str,
        _validator_code: str,
    ) -> ShadowValidationReceipt:
        return self.receipt


@pytest.mark.asyncio
async def test_speculative_arena_scores_success_from_shadow_receipt():
    arena = SpeculativeArena(SimpleNamespace(state=_State()))
    arena._sandbox = _Sandbox(
        ShadowValidationReceipt(
            success=True,
            behavioral_ok=True,
            structural_ok=True,
            validator_info={"score": 0.8},
            elapsed_ms=0.0,
        )
    )

    branch_id = (await arena.open_arena(_State(), count=1))[0]
    ok = await arena.execute_branch(branch_id, "mutation", "validator")
    promoted = await arena.promote_branch(branch_id)

    assert ok is True
    assert arena.branches[branch_id].score == pytest.approx(0.96)
    assert promoted.response_modifiers["arena_promotion"]["score"] == pytest.approx(0.96)
    assert promoted.response_modifiers["arena_promotion"]["receipt"]["validator_info"]["score"] == 0.8


@pytest.mark.asyncio
async def test_speculative_arena_records_failed_shadow_receipt_without_score():
    arena = SpeculativeArena(SimpleNamespace(state=_State()))
    arena._sandbox = _Sandbox(
        ShadowValidationReceipt(
            success=False,
            behavioral_ok=False,
            structural_ok=False,
            validator_info="security_violation",
            failure_reason="security_violation",
            elapsed_ms=0.0,
        )
    )

    branch_id = (await arena.open_arena(_State(), count=1))[0]
    ok = await arena.execute_branch(branch_id, "mutation", "validator")

    assert ok is False
    assert arena.branches[branch_id].score == 0.0
    assert "last_score_delta" not in arena.branches[branch_id].info
    assert arena.branches[branch_id].info["last_shadow_receipt"]["failure_reason"] == "security_violation"
