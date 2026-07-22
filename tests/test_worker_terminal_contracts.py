"""CP126 mlx_worker: cancellation, salvage honesty, and proof shape.

* ``269ff364`` — soft cancellation broke out with the partial response as-is,
  ahead of proof completeness, operator-evidence merit and capability
  grounding, so cancelling was a way around the terminal contracts.
* ``fa3d2a13`` — the self-claim salvage suffix asserted that the description
  "comes from my own state and self-model", a provenance claim nothing in
  the runtime supported, and the amended text was then re-evaluated — so a
  fabricated sentence turned a rejected self-claim into accepted text.
* ``007c5cd3`` — a proof prompt not matched by a narrow format regex was
  rewritten to demand 3-6 sentences and avoid numbered lists, contradicting
  contracts that had stated their own shape.
"""
from __future__ import annotations

import inspect

from core.brain.llm.mlx_worker import (
    _SELF_CLAIM_BOUNDARY_SUFFIX,
    _proof_prompt_declares_format,
    _terminal_contract_refusal,
)


class TestCancellationHonoursTerminalContracts:
    def test_empty_text_has_nothing_to_refuse(self):
        assert _terminal_contract_refusal({}, "") == ""
        assert _terminal_contract_refusal({}, "   ") == ""

    def test_an_incomplete_proof_fragment_is_refused(self):
        broken = "```python\ndef f(:\n"
        assert (
            _terminal_contract_refusal({}, broken, proof_evaluation_contract=True)
            == "proof_fragment_incomplete"
        )

    def test_a_complete_proof_passes(self):
        ok = "```python\ndef f(x):\n    return x * 2\n```"
        assert _terminal_contract_refusal({}, ok, proof_evaluation_contract=True) == ""

    def test_thin_operator_evidence_is_refused(self):
        assert _terminal_contract_refusal(
            {}, "Sure.", operator_evidence_contract=True,
        ).startswith("operator_evidence")

    def test_an_ungrounded_capability_inventory_is_refused(self):
        assert _terminal_contract_refusal(
            {"capability_inventory_contract": True},
            "I can do many things for you.",
        ) == "capability_inventory_ungrounded"

    def test_a_grounded_capability_inventory_passes(self):
        grounded = (
            "I can use browser/web research and file/desktop actions, all "
            "governed by Will/authority checks, and I am not executing any of "
            "them in this turn."
        )
        assert _terminal_contract_refusal(
            {"capability_inventory_contract": True}, grounded,
        ) == ""

    def test_contracts_not_selected_are_not_applied(self):
        # A plain conversational partial is not held to the proof contract.
        assert _terminal_contract_refusal({}, "Half a thought about the ") == ""

    def test_the_cancel_path_applies_the_refusals(self):
        from core.brain.llm import mlx_worker

        source = inspect.getsource(mlx_worker)
        block = source.split("Soft-cancel honored for job seq", 1)[0][-2500:]
        assert "_terminal_contract_refusal(" in block
        assert 'response_text = ""' in block

    def test_a_refused_cancelled_proof_marks_the_contract_failed(self):
        from core.brain.llm import mlx_worker

        source = inspect.getsource(mlx_worker)
        block = source.split("Soft-cancel honored for job seq", 1)[0][-2500:]
        assert "proof_contract_incomplete = True" in block

    def test_the_helper_is_side_effect_free(self):
        source = inspect.getsource(_terminal_contract_refusal)
        # Safe to call on the cancellation path: no retries, no IPC, no state.
        for forbidden in ("ipc_writer", "continue", "kwargs[", "logger.warning"):
            assert forbidden not in source


class TestSalvageDoesNotInventProvenance:
    def test_the_suffix_claims_no_provenance(self):
        lowered = _SELF_CLAIM_BOUNDARY_SUFFIX.lower()
        assert "comes from my own state" not in lowered
        assert "self-model" not in lowered

    def test_the_suffix_states_a_limit(self):
        lowered = _SELF_CLAIM_BOUNDARY_SUFFIX.lower()
        assert "not" in lowered and "proof" in lowered
        assert "phenomenal" in lowered

    def test_the_suffix_still_satisfies_the_boundary_guard(self):
        from core.conversation.response_reliability import (
            _SELF_CLAIM_EVIDENCE_BOUNDARY_RE,
        )

        assert _SELF_CLAIM_EVIDENCE_BOUNDARY_RE.search(_SELF_CLAIM_BOUNDARY_SUFFIX)

    def test_the_repair_is_still_disclosed(self):
        from core.brain.llm import mlx_worker

        source = inspect.getsource(mlx_worker._salvage_exhausted_user_surface)
        assert 'applied_repairs.append("self_claim_boundary_suffix")' in source


class TestProofShapeRespectsDeclaredFormat:
    def test_bounded_counts_are_recognised(self):
        for text in (
            "Answer in at most 3 sentences.",
            "Reply in no more than 2 lines.",
            "Use up to five bullets.",
            "Summarize in 2-4 sentences.",
        ):
            assert _proof_prompt_declares_format(text) is True, text

    def test_structured_serialisations_are_recognised(self):
        for text in (
            "Return valid JSON with keys a and b.",
            "Output CSV with a header row.",
            "Emit YAML.",
            "Reply with a fenced code block.",
            "```",
        ):
            assert _proof_prompt_declares_format(text) is True, text

    def test_brevity_requests_are_recognised(self):
        for text in ("Be brief.", "Give a concise answer.", "Short answer only."):
            assert _proof_prompt_declares_format(text) is True, text

    def test_schema_directives_are_recognised(self):
        for text in (
            "Output format: name,count",
            "schema: {a: int}",
            "Follow this template exactly.",
        ):
            assert _proof_prompt_declares_format(text) is True, text

    def test_the_original_patterns_still_match(self):
        for text in (
            "Answer in one word.",
            "Give exactly 3 bullets.",
            "Present it as a markdown table.",
            "Respond with only the number.",
        ):
            assert _proof_prompt_declares_format(text) is True, text

    def test_an_open_ended_task_still_gets_the_default_shape(self):
        for text in (
            "Explain how the allocator handles fragmentation.",
            "Describe the tradeoffs of this approach.",
            "Why did the lane recycle?",
        ):
            assert _proof_prompt_declares_format(text) is False, text
