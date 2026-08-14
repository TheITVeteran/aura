from __future__ import annotations

import argparse
import copy
from types import SimpleNamespace

import pytest

from tools import evaluate_unified_intrinsic_checkpoint as evaluator
from tools.evaluate_unified_intrinsic_checkpoint import (
    _evaluation_layout,
    _evaluation_preload_evidence,
    _sign_test_p_value,
)


def test_sign_test_is_exact_and_refuses_ties() -> None:
    assert _sign_test_p_value([0.0, 0.0]) is None
    assert _sign_test_p_value([1.0] * 8) == 0.0078125
    assert _sign_test_p_value([1.0] * 4 + [-1.0] * 4) == 1.0


def test_runtime_semantic_identity_preserves_interpreter_and_validates_commitment(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        evaluator,
        "_dependency_runtime_semantic_identity",
        lambda value: {"semantic": value["semantic"]},
    )
    body = {
        "environment": {"semantic": "same", "cache": "old"},
        "interpreter": {"sha256": "a" * 64},
    }
    runtime = {
        **body,
        "identity_sha256": evaluator._canonical_sha256(body),  # noqa: SLF001
    }
    changed_cache_body = copy.deepcopy(body)
    changed_cache_body["environment"]["cache"] = "new"
    changed_cache = {
        **changed_cache_body,
        "identity_sha256": evaluator._canonical_sha256(  # noqa: SLF001
            changed_cache_body
        ),
    }

    assert evaluator._runtime_semantic_identity(runtime) == (  # noqa: SLF001
        evaluator._runtime_semantic_identity(changed_cache)  # noqa: SLF001
    )

    changed_interpreter = copy.deepcopy(runtime)
    changed_interpreter["interpreter"]["sha256"] = "b" * 64
    changed_interpreter["identity_sha256"] = evaluator._canonical_sha256(  # noqa: SLF001
        {
            "environment": changed_interpreter["environment"],
            "interpreter": changed_interpreter["interpreter"],
        }
    )
    assert evaluator._runtime_semantic_identity(runtime) != (  # noqa: SLF001
        evaluator._runtime_semantic_identity(changed_interpreter)  # noqa: SLF001
    )

    tampered = copy.deepcopy(runtime)
    tampered["identity_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="runtime commitment differs"):
        evaluator._runtime_semantic_identity(tampered)  # noqa: SLF001


def test_evaluation_layout_supports_legacy_colocation(tmp_path) -> None:
    root = tmp_path.resolve()
    layout = _evaluation_layout(root)
    assert layout.checkpoint_dir == root
    assert layout.dataset_path == root / "dataset.json"
    assert layout.tokenized_dataset_path == root / "tokenized_dataset.json"
    assert layout.bootstrap_output_dir is None


def test_evaluation_layout_accepts_hash_verified_legacy_bootstrap_transport(tmp_path) -> None:
    root = tmp_path / "campaign"
    bootstrap = tmp_path / "bootstrap"
    root.mkdir()
    bootstrap.mkdir()

    layout = _evaluation_layout(root, bootstrap_output_dir=bootstrap)

    assert layout.bootstrap_output_dir == bootstrap.resolve()


def test_evaluation_layout_uses_resident_frozen_paths(tmp_path, monkeypatch) -> None:
    root = tmp_path.resolve()
    inputs = root / "inputs"
    output = root / "training-output"
    inputs.mkdir()
    output.mkdir()
    dataset = inputs / "dataset.json"
    tokenized = inputs / "tokenized_dataset.json"
    dataset.write_text("{}", encoding="ascii")
    tokenized.write_text("{}", encoding="ascii")
    bootstrap = inputs / "bootstrap-output"
    bootstrap.mkdir()
    (root / "campaign.json").write_text("{}", encoding="ascii")
    monkeypatch.setattr(
        evaluator,
        "_load_resident_campaign_config",
        lambda _path: {
            "paths": {
                "campaign_root": str(root),
                "training_output": str(output),
                "dataset": str(dataset),
                "tokenized_dataset": str(tokenized),
                "bootstrap_output": str(bootstrap),
            }
        },
    )
    layout = _evaluation_layout(root)
    assert layout.checkpoint_dir == output
    assert layout.dataset_path == dataset
    assert layout.tokenized_dataset_path == tokenized
    assert layout.bootstrap_output_dir == bootstrap


def test_evaluation_layout_refuses_resident_bootstrap_transport_substitution(
    tmp_path,
    monkeypatch,
) -> None:
    root = tmp_path / "campaign"
    inputs = root / "inputs"
    output = root / "training-output"
    bootstrap = inputs / "bootstrap-output"
    substitute = tmp_path / "substitute"
    bootstrap.mkdir(parents=True)
    output.mkdir()
    substitute.mkdir()
    dataset = inputs / "dataset.json"
    tokenized = inputs / "tokenized_dataset.json"
    dataset.write_text("{}", encoding="ascii")
    tokenized.write_text("{}", encoding="ascii")
    (root / "campaign.json").write_text("{}", encoding="ascii")
    monkeypatch.setattr(
        evaluator,
        "_load_resident_campaign_config",
        lambda _path: {
            "paths": {
                "campaign_root": str(root),
                "training_output": str(output),
                "dataset": str(dataset),
                "tokenized_dataset": str(tokenized),
                "bootstrap_output": str(bootstrap),
            }
        },
    )

    with pytest.raises(RuntimeError, match="differs from resident campaign"):
        _evaluation_layout(root, bootstrap_output_dir=substitute)


def test_random_initial_controller_refuses_bootstrapped_identity() -> None:
    with pytest.raises(RuntimeError, match="requires its committed parent"):
        evaluator._initial_controller(  # noqa: SLF001
            object(),
            object(),
            object(),
            {"bootstrap": {}},
            object(),
            object(),
        )


def test_bootstrap_initial_controller_loads_exact_committed_parent(
    tmp_path,
    monkeypatch,
) -> None:
    bootstrap_output = tmp_path / "bootstrap"
    bootstrap_output.mkdir()
    weights = bootstrap_output / "parent.safetensors"
    weights.write_bytes(b"parent-controller")
    compatibility = {
        "model": {
            "canonical_path": "/old/alias",
            "config_sha256": "config-sha",
            "weights": [
                {
                    "name": "model.safetensors",
                    "sha256": "weights-sha",
                    "size": 10,
                }
            ],
        },
        "spec": {"prelude_end": 7, "coda_start": 21, "state_weight": 2.0},
        "bridge": "assistant_answer",
        "lora_rank": 8,
        "controller_rank": 64,
        "state_codebook_sha256": "state-sha",
        "literal_observation_contract": {"contract_sha256": "literal-sha"},
        "opcode_observation_contract": {"contract_sha256": "opcode-sha"},
        "answer_emission_contract": {"contract_sha256": "answer-sha"},
        "depth_basis_size": 4,
        "lora_targets": ["o_proj", "v_proj"],
        "readout_sha256": "readout-sha",
    }
    compatibility["window_tissue_mode"] = "controller_only"
    parent_identity_body = dict(compatibility)
    parent_identity = {
        **parent_identity_body,
        "identity_sha256": evaluator._canonical_sha256(parent_identity_body),  # noqa: SLF001
    }
    receipt_body = {
        "step": 34,
        "checkpoint_sha256": evaluator._file_sha256(weights),  # noqa: SLF001
        "identity": parent_identity,
    }
    receipt = {
        **receipt_body,
        "receipt_sha256": evaluator._canonical_sha256(receipt_body),  # noqa: SLF001
    }
    identity = {
        **compatibility,
        "model": {**compatibility["model"], "canonical_path": "/new/alias"},
        "spec": {**compatibility["spec"], "state_weight": 9.0},
        "families": ["frontier_mathematics"],
        "task_depths": [7],
        "initial_controller_sha256": "controller-sha",
        "bootstrap": {
            "schema": "aura.unified_intrinsic.bootstrap_tissue.v1",
            "stem": "checkpoint_latest",
            "parent_step": 34,
            "parent_checkpoint_sha256": receipt["checkpoint_sha256"],
            "parent_receipt_sha256": receipt["receipt_sha256"],
            "parent_identity_sha256": parent_identity["identity_sha256"],
        },
    }
    parent_value = SimpleNamespace(shape=(1,), dtype="float32")
    child_value = SimpleNamespace(shape=(1,), dtype="float32")

    class FakeController:
        def __init__(self, _config) -> None:
            self.loaded = False

        def parameter_sha256(self) -> str:
            return "controller-sha" if self.loaded else "random-sha"

    class FakeBundle:
        def __init__(self, _model, controller) -> None:
            self.controller = controller

        def update(self, values) -> None:
            assert values == {"controller.x": parent_value}
            self.controller.loaded = True

        def parameters(self):
            return {"controller": self.controller}

    monkeypatch.setattr(
        evaluator,
        "resolve_checkpoint_generation",
        lambda *_args, **_kwargs: SimpleNamespace(
            receipt=receipt,
            weights_path=weights,
        ),
    )
    monkeypatch.setattr(evaluator, "UnifiedRecurrentController", FakeController)
    monkeypatch.setattr(evaluator, "UnifiedTrainingBundle", FakeBundle)
    monkeypatch.setattr(evaluator, "_controller_config", lambda *_args: object())
    monkeypatch.setattr(
        evaluator,
        "_trainable",
        lambda _bundle: {"controller.x": child_value},
    )
    monkeypatch.setattr(
        evaluator,
        "tree_unflatten",
        lambda rows: dict(rows),
    )
    monkeypatch.setattr(
        evaluator.mx,
        "load",
        lambda _path: {"bundle.controller.x": parent_value},
    )
    monkeypatch.setattr(evaluator.mx, "eval", lambda *_args: None)
    layout = evaluator.EvaluationLayout(
        checkpoint_dir=tmp_path,
        dataset_path=tmp_path / "dataset.json",
        tokenized_dataset_path=tmp_path / "tokenized.json",
        bootstrap_output_dir=bootstrap_output,
    )

    controller = evaluator._bootstrap_initial_controller(  # noqa: SLF001
        layout,
        object(),
        object(),
        identity,
        argparse.Namespace(digit_token_ids=()),
        argparse.Namespace(patterns=(), contexts=()),
    )

    assert controller is not None
    assert controller.loaded is True


def test_bootstrap_initial_controller_rejects_parent_commitment_drift(
    tmp_path,
) -> None:
    layout = evaluator.EvaluationLayout(
        checkpoint_dir=tmp_path,
        dataset_path=tmp_path / "dataset.json",
        tokenized_dataset_path=tmp_path / "tokenized.json",
        bootstrap_output_dir=None,
    )
    identity = {
        "bootstrap": {
            "schema": "aura.unified_intrinsic.bootstrap_tissue.v1",
            "stem": "checkpoint_latest",
        },
        "window_tissue_mode": "controller_only",
    }

    with pytest.raises(RuntimeError, match="bootstrap output is unavailable"):
        evaluator._bootstrap_initial_controller(  # noqa: SLF001
            layout,
            object(),
            object(),
            identity,
            object(),
            object(),
        )


def test_root_control_binding_requires_a_true_compatible_root(
    tmp_path,
    monkeypatch,
) -> None:
    weights = tmp_path / "root.safetensors"
    weights.write_bytes(b"root-controller-checkpoint")
    identity = {
        field: f"value-{field}"
        for field in evaluator._CONTROL_COMPATIBILITY_FIELDS  # noqa: SLF001
    }
    identity.update(
        {
            "bootstrap": None,
            "initial_controller_sha256": "a" * 64,
        }
    )
    identity["identity_sha256"] = evaluator._canonical_sha256(identity)  # noqa: SLF001
    receipt_body = {
        "step": 73,
        "checkpoint_sha256": evaluator._file_sha256(weights),  # noqa: SLF001
        "identity": identity,
    }
    receipt = {
        **receipt_body,
        "receipt_sha256": evaluator._canonical_sha256(receipt_body),  # noqa: SLF001
    }
    monkeypatch.setattr(
        evaluator,
        "resolve_checkpoint_generation",
        lambda *_args, **_kwargs: SimpleNamespace(
            receipt=receipt,
            weights_path=weights,
        ),
    )

    binding = evaluator.root_control_binding(
        tmp_path,
        stem="checkpoint_latest",
        target_identity=dict(identity),
    )

    assert binding["mode"] == "deterministic_pretraining_root"
    assert binding["checkpoint_step"] == 73
    assert binding["controller_sha256"] == "a" * 64
    assert binding["binding_sha256"] == evaluator._canonical_sha256(  # noqa: SLF001
        {key: value for key, value in binding.items() if key != "binding_sha256"}
    )

    target = dict(identity)
    target["controller_rank"] = "different"
    with pytest.raises(RuntimeError, match="topology differs: controller_rank"):
        evaluator.root_control_binding(
            tmp_path,
            stem="checkpoint_latest",
            target_identity=target,
        )

    identity["bootstrap"] = {"parent_step": 72}
    identity["identity_sha256"] = evaluator._canonical_sha256(  # noqa: SLF001
        {key: value for key, value in identity.items() if key != "identity_sha256"}
    )
    receipt_body["identity"] = identity
    receipt["identity"] = identity
    receipt["receipt_sha256"] = evaluator._canonical_sha256(receipt_body)  # noqa: SLF001
    with pytest.raises(RuntimeError, match="itself bootstrapped"):
        evaluator.root_control_binding(tmp_path, stem="checkpoint_latest")


def test_evaluation_without_external_guard_uses_live_pressure(monkeypatch) -> None:
    observed = {"available": True, "under_pressure": False, "source": "live"}
    monkeypatch.setattr(evaluator, "host_pressure", lambda **_kwargs: observed)
    monkeypatch.setattr(
        evaluator,
        "verify_release",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError),
    )

    pressure, release = _evaluation_preload_evidence(
        resource_enabled=False,
        preload_ready_path=None,
        preload_release_path=None,
        preload_key_path=None,
        preload_config_sha256=None,
    )

    assert pressure == observed
    assert release is None


def test_detached_evaluation_can_use_exact_brokered_pressure_evidence(
    tmp_path,
    monkeypatch,
) -> None:
    observed = {"available": True, "under_pressure": False, "source": "broker"}
    calls = []

    def fake_pressure(**kwargs):
        calls.append(kwargs)
        return observed

    monkeypatch.setattr(evaluator, "host_pressure", fake_pressure)
    vm_stat = tmp_path / "vm-stat.txt"
    swapusage = tmp_path / "swapusage.txt"

    pressure, release = _evaluation_preload_evidence(
        resource_enabled=False,
        preload_ready_path=None,
        preload_release_path=None,
        preload_key_path=None,
        preload_config_sha256=None,
        pressure_broker_vm_stat_path=vm_stat,
        pressure_broker_swapusage_path=swapusage,
    )

    assert pressure == observed
    assert release is None
    assert calls == [
        {
            "broker_vm_stat_path": vm_stat,
            "broker_swapusage_path": swapusage,
        }
    ]


def test_evaluation_rejects_partial_or_competing_pressure_authority(tmp_path) -> None:
    with pytest.raises(ValueError, match="broker arguments must be supplied together"):
        _evaluation_preload_evidence(
            resource_enabled=False,
            preload_ready_path=None,
            preload_release_path=None,
            preload_key_path=None,
            preload_config_sha256=None,
            pressure_broker_vm_stat_path=tmp_path / "vm-stat.txt",
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        _evaluation_preload_evidence(
            resource_enabled=True,
            preload_ready_path=tmp_path / "ready.json",
            preload_release_path=tmp_path / "release.json",
            preload_key_path=tmp_path / "key",
            preload_config_sha256="b" * 64,
            pressure_broker_vm_stat_path=tmp_path / "vm-stat.txt",
            pressure_broker_swapusage_path=tmp_path / "swapusage.txt",
        )


def test_external_evaluation_guard_requires_complete_signed_preload(tmp_path) -> None:
    with pytest.raises(ValueError, match="requires signed preload"):
        _evaluation_preload_evidence(
            resource_enabled=True,
            preload_ready_path=None,
            preload_release_path=None,
            preload_key_path=None,
            preload_config_sha256=None,
        )
    with pytest.raises(ValueError, match="supplied together"):
        _evaluation_preload_evidence(
            resource_enabled=True,
            preload_ready_path=tmp_path / "ready.json",
            preload_release_path=None,
            preload_key_path=None,
            preload_config_sha256=None,
        )


def test_signed_preload_is_the_detached_evaluation_pressure_authority(
    tmp_path,
    monkeypatch,
) -> None:
    release = {
        "host_pressure": {
            "available": True,
            "under_pressure": False,
            "source": "signed-external-sentinel",
        },
        "hmac_sha256": "a" * 64,
    }
    calls = []

    def fake_verify(path, **kwargs):
        calls.append((path, kwargs))
        return release

    monkeypatch.setattr(evaluator, "verify_release", fake_verify)
    monkeypatch.setattr(
        evaluator,
        "host_pressure",
        lambda: (_ for _ in ()).throw(AssertionError("live probe must not run")),
    )
    ready = tmp_path / "ready.json"
    signed = tmp_path / "release.json"
    key = tmp_path / "key"

    pressure, observed_release = _evaluation_preload_evidence(
        resource_enabled=True,
        preload_ready_path=ready,
        preload_release_path=signed,
        preload_key_path=key,
        preload_config_sha256="b" * 64,
    )

    assert pressure["source"] == "signed-external-sentinel"
    assert observed_release is release
    assert calls == [
        (
            signed,
            {
                "ready_path": ready,
                "key_path": key,
                "config_sha256": "b" * 64,
                "require_live_evidence": True,
            },
        )
    ]
