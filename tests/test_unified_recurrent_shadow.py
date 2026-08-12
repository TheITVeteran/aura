from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from core.brain.llm import unified_recurrent_shadow as shadow
from core.learning.recurrent_answer_emission import RecurrentAnswerEmissionContract
from core.learning.recurrent_literal_grounding import LiteralObservationContract
from core.learning.recurrent_opcode_grounding import OpcodeObservationContract
from core.learning.unified_intrinsic_recurrence import (
    UnifiedRecurrenceConfig,
    UnifiedRecurrentController,
)


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()


def _contracts() -> tuple[
    LiteralObservationContract,
    OpcodeObservationContract,
    RecurrentAnswerEmissionContract,
]:
    literal = LiteralObservationContract(tuple(range(10, 20)), max_value=32)
    opcode = OpcodeObservationContract(
        tuple((index, (100 + index,)) for index in range(8)),
        (
            ("graph", (201,)),
            ("graph_edges_start", (202,)),
            ("graph_edges_end", (203,)),
            ("modular_start", (204,)),
            ("modular_end", (205,)),
            ("boolean_start", (206,)),
            ("boolean_end", (207,)),
            ("register", (208,)),
            ("register_ops_start", (209,)),
            ("register_ops_end", (210,)),
        ),
    )
    answer = RecurrentAnswerEmissionContract(
        digit_token_ids=literal.digit_token_ids,
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
    return literal, opcode, answer


def _identity() -> tuple[dict[str, object], UnifiedRecurrentController]:
    literal, opcode, answer = _contracts()
    controller = UnifiedRecurrentController(
        UnifiedRecurrenceConfig(
            hidden_size=8,
            correction_rank=2,
            depth_basis_size=2,
            minimum_iterations=1,
            initialization_seed=17,
            literal_digit_token_ids=literal.digit_token_ids,
            opcode_token_patterns=opcode.patterns,
            opcode_context_patterns=opcode.contexts,
        )
    )
    spec = {
        "prelude_end": 1,
        "coda_start": 2,
        "train_depths": [1, 2, 4],
        "heldout_depths": [8, 16],
        "answer_weight": 1.0,
        "anchor_weight": 1.0,
        "trajectory_weight": 0.25,
        "progression_margin": 0.01,
        "halt_weight": 0.1,
        "state_weight": 6.0,
        "stutter_weight": 0.1,
        "anchor_injection": 0.0,
        "renormalize": True,
        "schema": "aura.unified_intrinsic_objective.v1",
    }
    body: dict[str, object] = {
        "window_tissue_mode": "controller_only",
        "wiring": {
            "window_tissue_mode": "controller_only",
            "window": [1, 2],
            "adapted_sites": [],
            "adapted_projection_count": 0,
            "continuous_depth_operator_count": 0,
            "continuous_depth_basis_size": 0,
            "coda_adapted": False,
            "readout_adapted": False,
            "ordinary_inference_requires_scope": False,
            "recurrence_phase_trains_shared_state_bridge": False,
            "state_bridge": "typed_recurrent_controller_only",
        },
        "spec": spec,
        "controller_rank": 2,
        "depth_basis_size": 2,
        "init_seed": 17,
        "literal_observation_contract": {
            **literal.to_dict(),
            "contract_sha256": literal.contract_sha256,
        },
        "opcode_observation_contract": {
            **opcode.to_dict(),
            "contract_sha256": opcode.contract_sha256,
        },
        "answer_emission_contract": {
            **answer.to_dict(),
            "contract_sha256": answer.contract_sha256,
        },
        "source_sha256s": {"core/learning/example.py": "a" * 64},
        "model": {"canonical_path": "/fixture/model"},
    }
    return {**body, "identity_sha256": _sha(body)}, controller


def _loaded_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    extra_tensor: bool = False,
) -> tuple[Path, object, object]:
    identity, controller = _identity()
    controller_path = tmp_path / "controller.safetensors"
    tensors = {
        f"bundle.controller.{name}": value
        for name, value in tree_flatten(controller.trainable_parameters())
    }
    if extra_tensor:
        tensors["unexpected"] = mx.zeros((1,))
    mx.save_safetensors(str(controller_path), tensors)
    controller_path.chmod(0o400)
    checkpoint_sha256 = hashlib.sha256(controller_path.read_bytes()).hexdigest()
    manifest = {
        "package_id": "fixture-shadow",
        "manifest_sha256": "b" * 64,
        "checkpoint_sha256": checkpoint_sha256,
        "domain_contract": {
            "families": ["khop", "modular", "register_trace"],
            "task_depths": [1, 2, 4],
            "recurrence_depth": 4,
        },
    }
    monkeypatch.setattr(
        shadow,
        "inspect_shadow_package",
        lambda _path: {
            "manifest": manifest,
            "checkpoint": {"identity": identity},
            "controller_path": controller_path,
            "controller_binding": {
                "path": controller_path.name,
                "sha256": checkpoint_sha256,
                "size_bytes": controller_path.stat().st_size,
            },
        },
    )
    monkeypatch.setattr(shadow, "_source_mechanics_match", lambda _identity: True)
    monkeypatch.setattr(
        shadow,
        "_model_extent_matches",
        lambda _path, _identity: True,
    )
    _literal, opcode, answer = _contracts()
    monkeypatch.setattr(shadow, "tokenizer_opcode_contract", lambda _tokenizer: opcode)
    monkeypatch.setattr(
        shadow,
        "tokenizer_answer_emission_contract",
        lambda _tokenizer, _opcode: answer,
    )
    norm = SimpleNamespace(weight=mx.ones((8,)))
    model = SimpleNamespace(
        model=SimpleNamespace(layers=[SimpleNamespace(input_layernorm=norm) for _ in range(3)])
    )
    return tmp_path, model, object()


def test_loads_controller_as_shadow_without_mutating_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package, model, tokenizer = _loaded_fixture(tmp_path, monkeypatch)
    layers_before = tuple(model.model.layers)

    loaded = shadow.load_unified_recurrent_shadow(
        package,
        model=model,
        tokenizer=tokenizer,
        model_path=tmp_path,
    )

    assert tuple(model.model.layers) == layers_before
    assert loaded.receipt["mode"] == "shadow_only"
    assert loaded.receipt["serving_authority"] is False
    assert loaded.receipt["recurrence_depth"] == 4
    assert loaded.supports([0, 201, 0]) is True
    assert loaded.supports([0, 1, 2]) is False


def test_refuses_extra_controller_tensor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package, model, tokenizer = _loaded_fixture(
        tmp_path,
        monkeypatch,
        extra_tensor=True,
    )

    with pytest.raises(
        shadow.UnifiedRecurrentShadowError,
        match="tensor inventory differs",
    ):
        shadow.load_unified_recurrent_shadow(
            package,
            model=model,
            tokenizer=tokenizer,
            model_path=tmp_path,
        )


def test_refuses_mechanics_source_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package, model, tokenizer = _loaded_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(shadow, "_source_mechanics_match", lambda _identity: False)

    with pytest.raises(
        shadow.UnifiedRecurrentShadowError,
        match="mechanics or resident model binding differs",
    ):
        shadow.load_unified_recurrent_shadow(
            package,
            model=model,
            tokenizer=tokenizer,
            model_path=tmp_path,
        )
