from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.learning.verified_trajectory_distillation import (
    DISTILLATION_SCHEMA,
    EPISODIC_TRANSPLANT_SCHEMA,
    build_verified_trajectory_artifact,
    compile_episodic_delta_factors,
    compile_episodic_delta_inventory,
    evaluate_verified_trajectory_transfer,
    fit_verified_trajectory_factors,
    fit_verified_trajectory_inventory,
    fit_verified_trajectory_sample_complexity,
    install_verified_trajectory_inventory,
    load_verified_trajectory_artifact,
    publish_verified_trajectory_artifact,
)


def _low_rank_problem(*, seed: int = 17):
    rng = np.random.default_rng(seed)
    inputs = rng.normal(size=(12, 9))
    left = rng.normal(size=(9, 3))
    right = rng.normal(size=(3, 7))
    corrections = inputs @ left @ right
    return inputs, corrections


def test_verified_trajectory_fit_recovers_low_rank_transition() -> None:
    inputs, corrections = _low_rank_problem()
    fitted = fit_verified_trajectory_factors(
        inputs,
        corrections,
        site="model.layers.3.self_attn.o_proj",
        rank=3,
        regularization=1e-8,
        gain=0.75,
        adapter_scale=20.0,
        normalize_corrections=False,
    )

    predicted = 20.0 * (inputs @ fitted.lora_a) @ fitted.lora_b

    assert np.allclose(predicted, 0.75 * corrections, rtol=2e-4, atol=2e-4)
    assert fitted.lora_a.shape == (9, 3)
    assert fitted.lora_b.shape == (3, 7)
    assert fitted.receipt["schema"] == DISTILLATION_SCHEMA
    assert fitted.receipt["effective_rank"] == 3
    assert fitted.receipt["training_relative_error"] < 1e-4
    assert len(fitted.receipt["receipt_sha256"]) == 64


def test_episodic_delta_compiles_to_exact_decode_scoped_operator() -> None:
    import mlx.core as mx
    import mlx.nn as nn

    from core.brain.llm.latent_cortex.fast_weights import EpisodicDeltaLinear
    from core.brain.llm.latent_cortex.recurrence_adapter import (
        ScopedCodaLoRALinear,
        coda_adapter_scope,
    )

    rng = np.random.default_rng(41)
    u = rng.normal(size=(7, 3)).astype(np.float32)
    v = rng.normal(size=(3, 9)).astype(np.float32)
    factors = compile_episodic_delta_factors(
        u,
        v,
        site="model.layers.0.self_attn.o_proj",
        episodic_scale=0.35,
        adapter_scale=0.35,
    )

    base = nn.Linear(9, 7)
    episodic = EpisodicDeltaLinear(
        base,
        rank=3,
        scale=0.35,
        seed_stat=0.5,
        tag="exact-transplant",
    )
    episodic.U = mx.array(u)
    episodic.V = mx.array(v)
    episodic.identity_bypass = False
    episodic.activation_policy = "decode_only"
    persistent = ScopedCodaLoRALinear.from_base(
        base,
        r=3,
        scale=0.35,
        block_index=0,
        site=factors.site,
    )
    persistent.lora_a = mx.array(factors.lora_a)
    persistent.lora_b = mx.array(factors.lora_b)
    persistent.exact_episodic_operation = True
    x = mx.array(rng.normal(size=(2, 4, 9)).astype(np.float16))

    ordinary = persistent(x)
    base_output = base(x)
    with coda_adapter_scope():
        episodic_output = episodic(x)
        persistent_output = persistent(x)
    mx.eval(ordinary, base_output, episodic_output, persistent_output)

    assert bool(mx.array_equal(ordinary, base_output))
    assert bool(mx.array_equal(persistent_output, episodic_output))
    assert persistent_output.dtype == episodic_output.dtype == mx.float32
    assert factors.target_phase == "decode"
    assert factors.receipt["schema"] == EPISODIC_TRANSPLANT_SCHEMA
    assert len(factors.receipt["receipt_sha256"]) == 64


def test_episodic_delta_transplant_rejects_collapsed_or_cross_rank_operator() -> None:
    kwargs = {
        "site": "model.layers.0.self_attn.o_proj",
        "episodic_scale": 0.35,
        "adapter_scale": 0.35,
    }
    with pytest.raises(ValueError, match="collapsed"):
        compile_episodic_delta_factors(
            np.ones((7, 3)),
            np.zeros((3, 9)),
            **kwargs,
        )
    with pytest.raises(ValueError, match="rank dimensions differ"):
        compile_episodic_delta_factors(
            np.ones((7, 3)),
            np.ones((2, 9)),
            **kwargs,
        )
    with pytest.raises(ValueError, match="original operator scale"):
        compile_episodic_delta_factors(
            np.ones((7, 3)),
            np.ones((3, 9)),
            **{**kwargs, "adapter_scale": 20.0},
        )


def test_episodic_delta_inventory_binds_exact_sites_and_scales() -> None:
    rng = np.random.default_rng(42)
    inventory = compile_episodic_delta_inventory(
        [
            {
                "layer": layer,
                "scale": 0.25,
                "U": rng.normal(size=(7, 3)),
                "V": rng.normal(size=(3, 9)),
            }
            for layer in (7, 11)
        ],
        target="o_proj",
    )

    assert tuple(inventory) == (
        "model.layers.7.self_attn.o_proj",
        "model.layers.11.self_attn.o_proj",
    )
    assert {factors.target_phase for factors in inventory.values()} == {"decode"}
    assert {
        factors.receipt["adapter_scale"] for factors in inventory.values()
    } == {0.25}

    with pytest.raises(ValueError, match="duplicates"):
        compile_episodic_delta_inventory(
            [
                {
                    "layer": 7,
                    "scale": 0.25,
                    "U": rng.normal(size=(7, 3)),
                    "V": rng.normal(size=(3, 9)),
                },
                {
                    "layer": 7,
                    "scale": 0.25,
                    "U": rng.normal(size=(7, 3)),
                    "V": rng.normal(size=(3, 9)),
                },
            ],
            target="o_proj",
        )


def test_verified_trajectory_artifact_is_deterministic_and_round_trips(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(43)
    inventory = compile_episodic_delta_inventory(
        [
            {
                "layer": layer,
                "scale": 0.25,
                "U": rng.normal(size=(7, 3)),
                "V": rng.normal(size=(3, 9)),
            }
            for layer in (7, 11)
        ],
        target="o_proj",
    )
    checkpoint = "a" * 64
    evidence = "b" * 64

    first = build_verified_trajectory_artifact(
        inventory,
        checkpoint_fingerprint=checkpoint,
        source_evidence_sha256=evidence,
    )
    second = build_verified_trajectory_artifact(
        inventory,
        checkpoint_fingerprint=checkpoint,
        source_evidence_sha256=evidence,
    )
    assert first == second

    publication = publish_verified_trajectory_artifact(
        tmp_path / "trajectory",
        inventory,
        checkpoint_fingerprint=checkpoint,
        source_evidence_sha256=evidence,
    )
    loaded, manifest = load_verified_trajectory_artifact(
        tmp_path / "trajectory",
        expected_checkpoint_fingerprint=checkpoint,
        expected_source_evidence_sha256=evidence,
    )

    assert tuple(loaded) == tuple(sorted(inventory))
    assert publication["manifest"] == manifest
    assert manifest["operation_modes"] == {
        site: "episodic_exact" for site in inventory
    }
    for site in inventory:
        assert np.array_equal(loaded[site].lora_a, inventory[site].lora_a)
        assert np.array_equal(loaded[site].lora_b, inventory[site].lora_b)


def test_verified_trajectory_artifact_rejects_wrong_checkpoint_and_tamper(
    tmp_path: Path,
) -> None:
    inventory = compile_episodic_delta_inventory(
        [
            {
                "layer": 7,
                "scale": 0.25,
                "U": np.ones((7, 3)),
                "V": np.ones((3, 9)),
            }
        ],
        target="o_proj",
    )
    artifact = tmp_path / "trajectory"
    publish_verified_trajectory_artifact(
        artifact,
        inventory,
        checkpoint_fingerprint="c" * 64,
        source_evidence_sha256="d" * 64,
    )

    with pytest.raises(ValueError, match="checkpoint fingerprint differs"):
        load_verified_trajectory_artifact(
            artifact,
            expected_checkpoint_fingerprint="e" * 64,
        )

    factors_path = artifact / "factors.npz"
    factors_path.write_bytes(factors_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="tensor artifact binding differs"):
        load_verified_trajectory_artifact(
            artifact,
            expected_checkpoint_fingerprint="c" * 64,
        )
def test_verified_trajectory_fit_normalizes_teacher_magnitude() -> None:
    inputs, corrections = _low_rank_problem()
    corrections[0] *= 1000.0
    fitted = fit_verified_trajectory_factors(
        inputs,
        corrections,
        site="model.layers.3.self_attn.o_proj",
        rank=8,
        regularization=1e-5,
        gain=0.25,
        adapter_scale=20.0,
    )

    target = corrections / np.linalg.norm(corrections, axis=1, keepdims=True)
    predicted = 20.0 * (inputs @ fitted.lora_a) @ fitted.lora_b

    assert fitted.receipt["corrections_normalized"] is True
    assert fitted.receipt["correction_norm_max"] > 100 * fitted.receipt["correction_norm_min"]
    assert np.linalg.norm(predicted - 0.25 * target) < np.linalg.norm(0.25 * target)


def test_verified_trajectory_inventory_rejects_partial_pair_counts() -> None:
    inputs, corrections = _low_rank_problem()

    with pytest.raises(ValueError, match="unequal pair counts"):
        fit_verified_trajectory_inventory(
            {
                "model.layers.3.self_attn.o_proj": (inputs, corrections),
                "model.layers.4.self_attn.o_proj": (inputs[:-1], corrections[:-1]),
            },
            rank=3,
            regularization=1e-5,
            gain=0.25,
            adapter_scale=20.0,
        )


def test_verified_trajectory_transfer_diagnostic_recovers_shared_rule() -> None:
    rng = np.random.default_rng(47)
    operator = rng.normal(size=(9, 7))
    train_inputs = rng.normal(size=(12, 9))
    validation_inputs = rng.normal(size=(6, 9))
    train_corrections = train_inputs @ operator
    validation_corrections = validation_inputs @ operator
    site = "model.layers.3.self_attn.o_proj"
    training = {site: (train_inputs, train_corrections)}
    validation = {site: (validation_inputs, validation_corrections)}
    fitted = fit_verified_trajectory_inventory(
        training,
        rank=7,
        regularization=1e-8,
        gain=0.75,
        adapter_scale=20.0,
        normalize_corrections=False,
    )

    diagnostic = evaluate_verified_trajectory_transfer(
        fitted,
        validation,
        training_pairs=training,
    )

    aggregate = diagnostic["aggregate"]
    assert fitted[site].receipt["corrections_normalized"] is False
    assert aggregate["relative_error"] < 1e-3
    assert aggregate["cosine"] > 0.999
    assert aggregate["better_than_zero_operator"] is True
    assert aggregate["input_subspace_coverage"] == pytest.approx(1.0)
    assert aggregate["correction_subspace_coverage"] == pytest.approx(1.0)


def test_verified_trajectory_sample_complexity_requires_fresh_cohort_transfer() -> None:
    rng = np.random.default_rng(731)
    operator = rng.normal(size=(12, 6))
    train_inputs = rng.normal(size=(64, 12))
    site = "model.layers.3.self_attn.o_proj"
    teaching = {site: (train_inputs, train_inputs @ operator)}
    cohorts = {}
    for index in range(3):
        inputs = rng.normal(size=(24, 12))
        cohorts[f"fresh-{index}"] = {site: (inputs, inputs @ operator)}

    report, fitted = fit_verified_trajectory_sample_complexity(
        teaching,
        cohorts,
        sample_rows=(4, 8, 16, 32, 64),
        rank=6,
        regularization=1e-8,
        gain=0.75,
        adapter_scale=20.0,
        site_phases={site: "decode"},
        normalize_corrections=False,
    )

    assert report["schema"] == "aura.verified_trajectory_sample_complexity.v1"
    assert report["sample_rows"] == [4, 8, 16, 32, 64]
    assert report["stages"][-1]["summary"]["worst_relative_error"] < 1e-5
    assert report["stages"][-1]["summary"]["all_sites_better_than_zero"] is True
    assert report["gates"] == {
        "fresh_cohorts_all_better_than_zero": True,
        "fresh_cohorts_all_direction_positive": True,
        "final_worst_case_beats_zero": True,
        "fresh_site_cells_all_better_than_zero": True,
        "fresh_site_cells_all_direction_positive": True,
        "mean_error_improves_with_more_evidence": True,
        "mean_direction_improves_with_more_evidence": True,
    }
    assert report["admitted"] is True
    assert fitted[site].target_phase == "decode"
    assert len(report["report_sha256"]) == 64


def test_verified_trajectory_sample_complexity_rejects_wrong_shared_rule() -> None:
    rng = np.random.default_rng(977)
    operator = rng.normal(size=(10, 7))
    train_inputs = rng.normal(size=(48, 10))
    site = "model.layers.5.self_attn.o_proj"
    teaching = {site: (train_inputs, train_inputs @ operator)}
    cohorts = {}
    for index in range(2):
        inputs = rng.normal(size=(16, 10))
        cohorts[f"opposite-{index}"] = {site: (inputs, -(inputs @ operator))}

    report, _fitted = fit_verified_trajectory_sample_complexity(
        teaching,
        cohorts,
        sample_rows=(8, 24, 48),
        rank=7,
        regularization=1e-8,
        gain=0.5,
        adapter_scale=20.0,
        normalize_corrections=False,
    )

    assert report["gates"]["fresh_cohorts_all_better_than_zero"] is False
    assert report["gates"]["fresh_cohorts_all_direction_positive"] is False
    assert report["admitted"] is False


@pytest.mark.parametrize(
    ("sample_rows", "message"),
    [
        ((2,), "at least two validation cohorts|sample rows"),
        ((2, 5), "ending at all rows"),
        ((4, 4, 6), "increasing complete prefixes"),
    ],
)
def test_verified_trajectory_sample_complexity_rejects_invalid_prefixes(
    sample_rows: tuple[int, ...],
    message: str,
) -> None:
    inputs = np.arange(18, dtype=np.float64).reshape(6, 3) + 1.0
    corrections = np.arange(24, dtype=np.float64).reshape(6, 4) + 2.0
    site = "model.layers.3.self_attn.o_proj"
    teaching = {site: (inputs, corrections)}
    cohorts = {
        "fresh-a": {site: (inputs[:2], corrections[:2])},
        "fresh-b": {site: (inputs[2:4], corrections[2:4])},
    }

    with pytest.raises(ValueError, match=message):
        fit_verified_trajectory_sample_complexity(
            teaching,
            cohorts,
            sample_rows=sample_rows,
            rank=2,
            regularization=1.0,
            gain=0.25,
            adapter_scale=1.0,
        )


def test_verified_trajectory_inventory_installs_exact_named_site() -> None:
    import mlx.core as mx
    import mlx.nn as nn

    from core.brain.llm.latent_cortex.recurrence_adapter import ScopedLoRALinear

    class _Attention:
        pass

    class _Layer:
        def __init__(self) -> None:
            self.self_attn = _Attention()
            self.self_attn.o_proj = ScopedLoRALinear.from_base(
                nn.Linear(9, 7),
                r=3,
                scale=20.0,
                block_index=0,
                site="model.layers.0.self_attn.o_proj",
            )

    class _Inner:
        def __init__(self) -> None:
            self.layers = [_Layer()]

    class _Model:
        def __init__(self) -> None:
            self.model = _Inner()

    inputs, corrections = _low_rank_problem()
    fitted = fit_verified_trajectory_factors(
        inputs,
        corrections,
        site="model.layers.0.self_attn.o_proj",
        rank=3,
        regularization=1e-8,
        gain=0.75,
        adapter_scale=20.0,
        normalize_corrections=False,
    )
    model = _Model()

    receipt = install_verified_trajectory_inventory(
        model,
        {fitted.site: fitted},
        expected_sites=[fitted.site],
    )

    projection = model.model.layers[0].self_attn.o_proj
    mx.eval(projection.lora_a, projection.lora_b)
    assert np.allclose(np.asarray(projection.lora_a), fitted.lora_a)
    assert np.allclose(np.asarray(projection.lora_b), fitted.lora_b)
    assert receipt["sites"] == [fitted.site]
    assert len(receipt["receipt_sha256"]) == 64


def test_verified_trajectory_inventory_binds_recurrence_and_decode_phases() -> None:
    import mlx.core as mx
    import mlx.nn as nn

    from core.brain.llm.latent_cortex.recurrence_adapter import (
        ScopedCodaLoRALinear,
        ScopedLoRALinear,
    )

    recurrence_site = "model.layers.0.self_attn.o_proj"
    decode_site = "model.layers.1.self_attn.o_proj"

    class _Attention:
        def __init__(self, projection) -> None:
            self.o_proj = projection

    class _Layer:
        def __init__(self, projection) -> None:
            self.self_attn = _Attention(projection)

    class _Inner:
        def __init__(self) -> None:
            self.layers = [
                _Layer(
                    ScopedLoRALinear.from_base(
                        nn.Linear(9, 7),
                        r=3,
                        scale=20.0,
                        block_index=0,
                        site=recurrence_site,
                    )
                ),
                _Layer(
                    ScopedCodaLoRALinear.from_base(
                        nn.Linear(9, 7),
                        r=3,
                        scale=20.0,
                        block_index=1,
                        site=decode_site,
                    )
                ),
            ]

    class _Model:
        def __init__(self) -> None:
            self.model = _Inner()

    inputs, corrections = _low_rank_problem()
    fitted = fit_verified_trajectory_inventory(
        {
            recurrence_site: (inputs, corrections),
            decode_site: (inputs, corrections),
        },
        rank=3,
        regularization=1e-8,
        gain=0.75,
        adapter_scale=20.0,
        site_phases={recurrence_site: "recurrence", decode_site: "decode"},
    )
    model = _Model()
    receipt = install_verified_trajectory_inventory(
        model,
        fitted,
        expected_sites=[recurrence_site, decode_site],
    )

    recurrence = model.model.layers[0].self_attn.o_proj
    decode = model.model.layers[1].self_attn.o_proj
    mx.eval(recurrence.lora_a, recurrence.lora_b, decode.lora_a, decode.lora_b)
    assert fitted[recurrence_site].target_phase == "recurrence"
    assert fitted[decode_site].target_phase == "decode"
    assert receipt["site_phases"] == {
        recurrence_site: "recurrence",
        decode_site: "decode",
    }
    assert np.allclose(np.asarray(recurrence.lora_a), fitted[recurrence_site].lora_a)
    assert np.allclose(np.asarray(decode.lora_b), fitted[decode_site].lora_b)

    crossed = dict(fitted)
    crossed[decode_site] = fit_verified_trajectory_factors(
        inputs,
        corrections,
        site=decode_site,
        rank=3,
        regularization=1e-8,
        gain=0.75,
        adapter_scale=20.0,
        normalize_corrections=False,
        target_phase="recurrence",
    )
    with pytest.raises(ValueError, match="phase differs"):
        install_verified_trajectory_inventory(
            model,
            crossed,
            expected_sites=[recurrence_site, decode_site],
        )

    decode.scale = 19.0
    with pytest.raises(ValueError, match="adapter scale differs"):
        install_verified_trajectory_inventory(
            model,
            fitted,
            expected_sites=[recurrence_site, decode_site],
        )


@pytest.mark.parametrize(
    ("inputs", "corrections", "message"),
    [
        (np.ones((1, 3)), np.ones((1, 4)), "at least two"),
        (np.ones((2, 3)), np.ones((3, 4)), "pair counts differ"),
        (np.ones((2, 3)), np.zeros((2, 4)), "collapsed row"),
    ],
)
def test_verified_trajectory_fit_rejects_invalid_evidence(
    inputs: np.ndarray,
    corrections: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        fit_verified_trajectory_factors(
            inputs,
            corrections,
            site="model.layers.3.self_attn.o_proj",
            rank=2,
            regularization=1e-5,
            gain=0.25,
            adapter_scale=20.0,
        )
