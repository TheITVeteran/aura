from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import mlx.core as mx
import pytest

from core.brain.llm import unified_recurrent_qualified_decode as qualified
from core.brain.llm.unified_recurrent_shadow import LoadedUnifiedRecurrentShadow
from core.learning.recurrent_action_schema import (
    OP_ADD_MOD,
    OP_BOOL_AND,
    OP_BOOL_NOT,
    OP_BOOL_OR,
    OP_BOOL_XOR,
    OP_COPY_VALUE,
    OP_MUL_MOD,
    OP_SUB_MOD,
)
from core.learning.recurrent_answer_emission import RecurrentAnswerEmissionContract
from core.learning.recurrent_literal_grounding import LiteralObservationContract
from core.learning.recurrent_opcode_grounding import OpcodeObservationContract


def _contracts():
    literal = LiteralObservationContract(tuple(range(10, 20)))
    opcode = OpcodeObservationContract(
        patterns=(
            (OP_COPY_VALUE, (300,)),
            (OP_ADD_MOD, (301,)),
            (OP_MUL_MOD, (302,)),
            (OP_SUB_MOD, (303,)),
            (OP_BOOL_NOT, (304,)),
            (OP_BOOL_AND, (305,)),
            (OP_BOOL_OR, (306,)),
            (OP_BOOL_XOR, (307,)),
        ),
        contexts=(
            ("graph", (201,)),
            ("graph_edges_start", (202,)),
            ("graph_edges_end", (203,)),
            ("modular_start", (210,)),
            ("modular_end", (211,)),
            ("boolean_start", (212,)),
            ("boolean_end", (213,)),
            ("register", (220,)),
            ("register_ops_start", (221,)),
            ("register_ops_end", (222,)),
        ),
    )
    answer = RecurrentAnswerEmissionContract(
        digit_token_ids=tuple(range(10, 20)),
        eos_token_id=999,
        family_markers=(
            ("khop", (201,)),
            ("modular", (210,)),
            ("register_trace", (220,)),
        ),
        syntax=(
            ("khop", (401,)),
            ("modular", (402,)),
            ("register_head", (403,)),
            ("register_mid_r1", (404,)),
            ("register_mid_r2", (405,)),
            ("close", (409,)),
        ),
    )
    return literal, opcode, answer


def _loaded() -> LoadedUnifiedRecurrentShadow:
    literal, opcode, answer = _contracts()
    return LoadedUnifiedRecurrentShadow(
        controller=object(),
        spec=SimpleNamespace(plan_at=lambda depth: SimpleNamespace(iterations=depth)),
        answer_contract=answer,
        receipt={
            "package_id": "qualified-fixture",
            "controller_sha256": "c" * 64,
            "families": ["khop", "modular", "register_trace"],
            "task_depths": [1, 2, 4],
            "recurrence_depth": 4,
        },
        literal_contract=literal,
        opcode_contract=opcode,
    )


def _model():
    return SimpleNamespace(
        model=SimpleNamespace(
            embed_tokens=SimpleNamespace(weight=mx.zeros((1000, 1)))
        )
    )


def _activation() -> dict:
    body = {
        "schema": "aura.unified_intrinsic.qualified_activation.v1",
        "package_id": "qualified-fixture",
        "manifest_sha256": "b" * 64,
        "checkpoint_sha256": "d" * 64,
        "controller_sha256": "c" * 64,
        "pointer_sha256": "e" * 64,
        "lifecycle_result_sha256": "f" * 64,
        "canary_plan_sha256": "1" * 64,
        "families": ["khop", "modular", "register_trace"],
        "task_depths": [1, 2, 4],
        "recurrence_depth": 4,
        "mode": "qualified_typed_only",
        "ordinary_chat_authorized": False,
        "arbitrary_reasoning_authorized": False,
        "serving_authority": True,
    }
    digest = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    return {**body, "activation_sha256": digest}


def _select(token_id: int):
    vocabulary = mx.arange(1000)
    return mx.where(vocabulary == token_id, 0.0, -1e9)[None, None, :]


def test_classifies_each_public_program_from_grammar_not_caller_claims() -> None:
    literal, opcode, answer = _contracts()
    common = {
        "answer_contract": answer,
        "literal_contract": literal,
        "opcode_contract": opcode,
    }

    assert qualified.classify_public_program(
        [201, 202, 10, 800, 11, 203, 13, 800, 12], **common
    ) == ("khop", 2)
    assert qualified.classify_public_program(
        [19, 13, 210, 301, 12, 303, 11, 211], **common
    ) == ("modular", 2)
    assert qualified.classify_public_program(
        [220, 221, 10, 800, 11, 800, 12, 800, 13, 800, 14, 800, 15, 222],
        **common,
    ) == ("register_trace", 1)


def test_validates_canonical_typed_answers_without_expected_answer() -> None:
    _literal, _opcode, answer = _contracts()

    assert qualified.validate_qualified_answer(
        [201], [401, 12, 409, 999], answer_contract=answer
    ) == {"node": 2}
    assert qualified.validate_qualified_answer(
        [220], [403, 11, 404, 12, 405, 13, 409, 999], answer_contract=answer
    ) == {"r0": 1, "r1": 2, "r2": 3}
    with pytest.raises(
        qualified.UnifiedRecurrentQualifiedDecodeError,
        match="trailing tokens",
    ):
        qualified.validate_qualified_answer(
            [201], [401, 12, 409, 10, 999], answer_contract=answer
        )


def test_runs_answer_blind_recurrent_decode_and_returns_bound_tokens(monkeypatch) -> None:
    loaded = _loaded()
    public_tokens = [201, 202, 10, 800, 11, 203, 13, 800, 12]
    expected = [401, 12, 409, 999]

    def recurrent(_model, tokens, _plan, _controller, **_kwargs):
        generated = int(tokens.shape[-1]) - len(public_tokens)
        return _select(expected[generated]), object()

    monkeypatch.setattr(qualified, "unified_recurrent_logits", recurrent)
    request = qualified.seal_qualified_decode_request(
        public_tokens,
        package_id="qualified-fixture",
        controller_sha256="c" * 64,
        family="khop",
        task_depth=2,
        max_tokens=4,
    )

    result = qualified.run_qualified_decode(loaded, _model(), request)

    assert result["generated_token_ids"] == expected
    assert result["parsed_values"] == {"node": 2}
    assert result["output_exposed"] is True
    assert result["serving_authority"] is False
    assert result["request_sha256"] == request["request_sha256"]
    assert qualified.qualified_decode_result_errors(result) == []


def test_only_matching_activation_can_authorize_a_typed_result(monkeypatch) -> None:
    loaded = _loaded()
    public_tokens = [201, 202, 10, 800, 11, 203, 13, 800, 12]
    expected = [401, 12, 409, 999]

    def recurrent(_model, tokens, _plan, _controller, **_kwargs):
        generated = int(tokens.shape[-1]) - len(public_tokens)
        return _select(expected[generated]), object()

    monkeypatch.setattr(qualified, "unified_recurrent_logits", recurrent)
    request = qualified.seal_qualified_decode_request(
        public_tokens,
        package_id="qualified-fixture",
        controller_sha256="c" * 64,
        family="khop",
        task_depth=2,
        max_tokens=4,
    )
    result = qualified.run_qualified_decode(loaded, _model(), request)
    activation = _activation()

    authorized = qualified.authorize_qualified_decode_result(result, activation)

    assert authorized["serving_authority"] is True
    assert authorized["qualified_activation_sha256"] == activation["activation_sha256"]
    assert qualified.qualified_decode_result_errors(
        authorized,
        expected_request_sha256=request["request_sha256"],
        expected_activation_sha256=activation["activation_sha256"],
    ) == []

    activation["controller_sha256"] = "d" * 64
    activation_body = {
        key: value for key, value in activation.items() if key != "activation_sha256"
    }
    activation["activation_sha256"] = qualified._sha(activation_body)
    with pytest.raises(
        qualified.UnifiedRecurrentQualifiedDecodeError,
        match="activation identity differs",
    ):
        qualified.authorize_qualified_decode_result(result, activation)


def test_declared_depth_cannot_expand_the_certified_domain(monkeypatch) -> None:
    loaded = _loaded()
    request = qualified.seal_qualified_decode_request(
        [201, 202, 10, 800, 11, 203, 13, 800, 12],
        package_id="qualified-fixture",
        controller_sha256="c" * 64,
        family="khop",
        task_depth=4,
        max_tokens=4,
    )
    executed = False

    def recurrent(*_args, **_kwargs):
        nonlocal executed
        executed = True
        return _select(999), object()

    monkeypatch.setattr(qualified, "unified_recurrent_logits", recurrent)

    with pytest.raises(
        qualified.UnifiedRecurrentQualifiedDecodeError,
        match="domain differs",
    ):
        qualified.run_qualified_decode(loaded, _model(), request)
    assert executed is False


def test_out_of_vocabulary_program_fails_before_recurrent_execution(monkeypatch) -> None:
    loaded = _loaded()
    request = qualified.seal_qualified_decode_request(
        [201, 202, 10, 1000, 11, 203, 13, 800, 12],
        package_id="qualified-fixture",
        controller_sha256="c" * 64,
        family="khop",
        task_depth=2,
        max_tokens=4,
    )
    executed = False

    def recurrent(*_args, **_kwargs):
        nonlocal executed
        executed = True
        return _select(999), object()

    monkeypatch.setattr(qualified, "unified_recurrent_logits", recurrent)

    with pytest.raises(
        qualified.UnifiedRecurrentQualifiedDecodeError,
        match="exceeds model vocabulary",
    ):
        qualified.run_qualified_decode(loaded, _model(), request)
    assert executed is False


def test_result_identity_and_parsed_shape_are_not_ornamental(monkeypatch) -> None:
    loaded = _loaded()
    public_tokens = [201, 202, 10, 800, 11, 203, 13, 800, 12]
    expected = [401, 12, 409, 999]

    def recurrent(_model, tokens, _plan, _controller, **_kwargs):
        return _select(expected[int(tokens.shape[-1]) - len(public_tokens)]), object()

    monkeypatch.setattr(qualified, "unified_recurrent_logits", recurrent)
    request = qualified.seal_qualified_decode_request(
        public_tokens,
        package_id="qualified-fixture",
        controller_sha256="c" * 64,
        family="khop",
        task_depth=2,
        max_tokens=4,
    )
    result = qualified.run_qualified_decode(loaded, _model(), request)
    result["request_sha256"] = "not-a-digest"
    body = {key: value for key, value in result.items() if key != "result_sha256"}
    result["result_sha256"] = qualified._sha(body)
    assert "qualified_decode_result_invalid" in qualified.qualified_decode_result_errors(result)

    result = qualified.run_qualified_decode(loaded, _model(), request)
    result["parsed_values"] = {"residue": 2}
    body = {key: value for key, value in result.items() if key != "result_sha256"}
    result["result_sha256"] = qualified._sha(body)
    assert "qualified_decode_result_invalid" in qualified.qualified_decode_result_errors(result)


def test_parent_expectations_bind_every_result_domain_field(monkeypatch) -> None:
    loaded = _loaded()
    public_tokens = [201, 202, 10, 800, 11, 203, 13, 800, 12]
    expected = [401, 12, 409, 999]

    def recurrent(_model, tokens, _plan, _controller, **_kwargs):
        return _select(expected[int(tokens.shape[-1]) - len(public_tokens)]), object()

    monkeypatch.setattr(qualified, "unified_recurrent_logits", recurrent)
    request = qualified.seal_qualified_decode_request(
        public_tokens,
        package_id="qualified-fixture",
        controller_sha256="c" * 64,
        family="khop",
        task_depth=2,
        max_tokens=4,
    )
    result = qualified.authorize_qualified_decode_result(
        qualified.run_qualified_decode(loaded, _model(), request),
        _activation(),
    )

    assert qualified.qualified_decode_result_errors(
        result,
        expected_request_sha256=request["request_sha256"],
        expected_activation_sha256=_activation()["activation_sha256"],
        expected_package_id="other-package",
        expected_controller_sha256="d" * 64,
        expected_family="modular",
        expected_task_depth=4,
    ) == ["qualified_decode_result_domain_differs"]
