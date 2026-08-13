from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from core.brain.llm.latent_cortex.resource_accounting import (
    ModelComputeProfile,
    ResourceLedger,
    validate_resource_receipt,
)
from tools.rlc_complete_system_closed_book import build_integrated_candidate
from tools.rlc_integrated_recurrent_producer import (
    INITIAL_CONTROL_SOURCE,
    PRODUCER_SCHEMA,
    RESOURCE_ESTIMATOR,
    build_general_recurrent_resource_receipt,
    produce_integrated_recurrent_candidate,
)


def _profile() -> ModelComputeProfile:
    return ModelComputeProfile(
        model_type="fixture",
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=6,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=32,
        head_dim=4,
    )


def test_general_recurrent_resource_receipt_counts_incremental_work():
    receipt = build_general_recurrent_resource_receipt(
        profile=_profile(),
        prompt_tokens=3,
        generated_tokens=2,
        prelude_end=1,
        coda_start=5,
        recurrence_depth=4,
        correction_rank=2,
        depth_basis_size=3,
        renormalize=True,
    )
    validated = validate_resource_receipt(receipt)
    effective_layers = 1 + 4 * 4 + 1
    assert validated["accounting_complete"] is True
    assert validated["totals"]["transformer_layer_apps"] == 4 * effective_layers
    assert validated["totals"]["attention_query_key_pairs"] == 13 * effective_layers
    assert validated["totals"]["output_head_tokens"] == 4
    assert validated["totals"]["tensor_scalar_ops"] > 0
    assert validated["estimated_flops"] > 0
    assert f"{RESOURCE_ESTIMATOR}:controller" in validated["operations"]


def test_general_recurrent_resource_receipt_rejects_impossible_topology():
    with pytest.raises(ValueError, match="topology is invalid"):
        build_general_recurrent_resource_receipt(
            profile=_profile(),
            prompt_tokens=3,
            generated_tokens=2,
            prelude_end=5,
            coda_start=4,
            recurrence_depth=4,
            correction_rank=2,
            depth_basis_size=3,
            renormalize=True,
        )


class _Tokenizer:
    eos_token_id = 99

    @staticmethod
    def decode(token_ids, *, skip_special_tokens=True):
        assert skip_special_tokens is True
        if list(token_ids) == [7, 8]:
            return 'FINAL_ANSWER: {"value":42}'
        return "prompt"


class _Loaded:
    def __init__(self) -> None:
        self.receipt = {
            "recurrence_depth": 4,
            "package_id": "fixture-package",
            "manifest_sha256": "a" * 64,
            "controller_sha256": "b" * 64,
        }
        self.controller = _Controller("b" * 64)
        self.spec = SimpleNamespace(
            plan_at=lambda depth: SimpleNamespace(
                prelude_end=1,
                coda_start=5,
                iterations=depth,
                renormalize=True,
            )
        )

    @staticmethod
    def decode_general_recurrent_tokens(
        model,
        public_tokens,
        *,
        max_tokens,
        recurrence_depth,
        controller,
        completion_check,
        activity,
    ):
        assert model is not None
        assert tuple(public_tokens) == (1, 2, 3)
        assert max_tokens == 32
        assert recurrence_depth == 4
        assert controller.parameter_sha256() in {"b" * 64, "c" * 64}
        assert completion_check((7, 8)) is True
        if activity is not None:
            activity()
        return (7, 8), True, 0


class _Controller:
    def __init__(self, digest: str) -> None:
        self.config = SimpleNamespace(correction_rank=2, depth_basis_size=3)
        self._digest = digest

    def parameter_sha256(self):
        return self._digest


def _model():
    args = SimpleNamespace(
        model_type="fixture",
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=32,
        head_dim=4,
    )
    return SimpleNamespace(model=SimpleNamespace(args=args, layers=[object()] * 6))


def test_integrated_producer_binds_terminal_text_package_task_and_compute():
    task = SimpleNamespace(
        task_id="task-fixture",
        public=SimpleNamespace(prompt="What is 6 * 7?"),
    )
    candidate, producer = produce_integrated_recurrent_candidate(
        model=_model(),
        tokenizer=_Tokenizer(),
        task=task,
        loaded=_Loaded(),
        public_tokens=(1, 2, 3),
        max_tokens=32,
    )
    assert producer["schema"] == PRODUCER_SCHEMA
    assert producer["task_id"] == task.task_id
    assert producer["score_observed"] is False
    assert producer["answer_key_used"] is False
    assert candidate["text"] == 'FINAL_ANSWER: {"value":42}'
    assert candidate["source_receipt_sha256"] == producer["receipt_sha256"]
    assert candidate["resource_accounting"]["accounting_complete"] is True


def test_integrated_producer_binds_an_initialization_matched_controller_control():
    task = SimpleNamespace(
        task_id="task-fixture",
        public=SimpleNamespace(prompt="What is 6 * 7?"),
    )
    candidate, producer = produce_integrated_recurrent_candidate(
        model=_model(),
        tokenizer=_Tokenizer(),
        task=task,
        loaded=_Loaded(),
        public_tokens=(1, 2, 3),
        max_tokens=32,
        controller=_Controller("c" * 64),
        source=INITIAL_CONTROL_SOURCE,
    )
    assert candidate["source"] == INITIAL_CONTROL_SOURCE
    assert producer["package_controller_sha256"] == "b" * 64
    assert producer["controller_sha256"] == "c" * 64


def test_integrated_candidate_refuses_incomplete_resource_accounting():
    ledger = ResourceLedger(_profile())
    ledger.charge("known", transformer_layer_apps=1)
    ledger.mark_unknown("unmeasured_controller")
    text = 'FINAL_ANSWER: {"value":42}'
    source_body = {
        "schema": PRODUCER_SCHEMA,
        "source": "unified_recurrent_controller",
        "task_id": "task-fixture",
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "resource_accounting_sha256": ledger.to_receipt()["receipt_sha256"],
        "same_public_information": True,
        "answer_key_used": False,
    }
    source_receipt = {
        **source_body,
        "receipt_sha256": hashlib.sha256(
            json.dumps(
                source_body,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest(),
    }
    with pytest.raises(ValueError, match="resource accounting is incomplete"):
        build_integrated_candidate(
            source="unified_recurrent_controller",
            task_id="task-fixture",
            text=text,
            resource_accounting=ledger.to_receipt(),
            source_receipt=source_receipt,
            source_receipt_sha256=source_receipt["receipt_sha256"],
        )
