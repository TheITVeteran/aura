from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

from core.brain.llm import unified_recurrent_shadow as shadow
from core.brain.llm.unified_recurrent_shadow_probe_contract import (
    seal_shadow_probe_request,
    shadow_probe_receipt_errors,
)
from core.learning.recurrent_answer_emission import RecurrentAnswerEmissionContract


class _Model:
    def __call__(self, tokens):
        token_id = 14 if int(tokens.shape[-1]) == 3 else 999
        return _select(token_id)


def _loaded() -> shadow.LoadedUnifiedRecurrentShadow:
    answer = RecurrentAnswerEmissionContract(
        digit_token_ids=tuple(range(10, 20)),
        eos_token_id=999,
        family_markers=(
            ("khop", (201,)),
            ("modular", (204,)),
            ("register_trace", (208,)),
        ),
        syntax=(
            ("khop", (301,)),
            ("modular", (302,)),
            ("register_head", (303,)),
            ("register_mid_r1", (304,)),
            ("register_mid_r2", (305,)),
            ("close", (306,)),
        ),
    )
    return shadow.LoadedUnifiedRecurrentShadow(
        controller=object(),
        spec=SimpleNamespace(plan_at=lambda depth: SimpleNamespace(iterations=depth)),
        answer_contract=answer,
        receipt={
            "package_id": "fixture",
            "controller_sha256": "c" * 64,
            "families": ["khop", "modular", "register_trace"],
            "recurrence_depth": 4,
        },
    )


def _select(token_id: int):
    vocabulary = mx.arange(1000)
    return mx.where(vocabulary == token_id, 0.0, -1e9)[None, None, :]


def test_probe_measures_matched_decodes_without_exposing_tokens(monkeypatch) -> None:
    loaded = _loaded()

    def recurrent(_model, tokens, _plan, _controller, **_kwargs):
        return (_select(12 if int(tokens.shape[-1]) == 3 else 999), object())

    monkeypatch.setattr(shadow, "unified_recurrent_logits", recurrent)
    request = seal_shadow_probe_request([0, 201, 0], [12, 999], max_tokens=2)

    receipt = loaded.probe(_Model(), request)

    assert shadow_probe_receipt_errors(receipt) == []
    assert receipt["base_exact_match"] is False
    assert receipt["shadow_exact_match"] is True
    assert receipt["outputs_equal"] is False
    assert receipt["base_output_sha256"] != receipt["shadow_output_sha256"]
    assert "text" not in receipt
    assert "tokens" not in receipt


def test_probe_abstains_before_model_execution_on_an_unsupported_family() -> None:
    loaded = _loaded()
    request = seal_shadow_probe_request([0, 1, 2], [12, 999], max_tokens=2)

    receipt = loaded.probe(_Model(), request)

    assert shadow_probe_receipt_errors(receipt) == []
    assert receipt["status"] == "abstained"
    assert receipt["reason"] == "unsupported_public_token_family"
    assert receipt["base_token_count"] == 0
    assert receipt["shadow_token_count"] == 0
