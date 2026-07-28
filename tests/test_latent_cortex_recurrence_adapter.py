"""Contracts for recurrence-only adapter activation."""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx import nn  # noqa: E402

from core.brain.llm.latent_cortex.recurrence import WindowRunner  # noqa: E402
from core.brain.llm.latent_cortex.recurrence_adapter import (  # noqa: E402
    ScopedLoRALinear,
    current_recurrence_adapter_scope,
    recurrence_adapter_disabled,
    recurrence_adapter_scope,
)
from core.brain.llm.latent_cortex.types import ComputeBudget, EpisodeReceipt  # noqa: E402


def _projection() -> tuple[nn.Linear, ScopedLoRALinear]:
    base = nn.Linear(4, 3, bias=False)
    wrapped = ScopedLoRALinear.from_base(base, r=2, scale=1.0)
    wrapped.lora_a = mx.ones_like(wrapped.lora_a)
    wrapped.lora_b = mx.ones_like(wrapped.lora_b)
    mx.eval(wrapped.parameters())
    return base, wrapped


def test_adapter_is_bit_exact_base_outside_scope():
    base, wrapped = _projection()
    x = mx.arange(24, dtype=mx.float32).reshape(2, 3, 4) / 10.0
    expected = base(x)
    actual = wrapped(x)
    mx.eval(expected, actual)
    assert bool(mx.array_equal(actual, expected))
    assert current_recurrence_adapter_scope() is None


def test_full_scope_applies_delta_and_receipts_every_position():
    base, wrapped = _projection()
    x = mx.ones((1, 3, 4))
    with recurrence_adapter_scope() as activation:
        actual = wrapped(x)
    expected = base(x)
    mx.eval(expected, actual)
    assert not bool(mx.array_equal(actual, expected))
    assert activation.to_dict() == {
        "start": None,
        "stop": None,
        "calls": 1,
        "adapted_positions": 3,
        "observed_positions": 3,
        # This projection was built without attachment identity, so the
        # application is counted but not attributable. Attachment sites that
        # supply block_index/site populate these; see the identity tests below.
        "applied_blocks": {},
        "applied_sites": {},
    }
    assert current_recurrence_adapter_scope() is None


def test_position_scope_changes_slots_only():
    base, wrapped = _projection()
    x = mx.ones((1, 7, 4))
    expected = base(x)
    with recurrence_adapter_scope(start=2, stop=5) as activation:
        actual = wrapped(x)
    mx.eval(expected, actual)
    assert bool(mx.array_equal(actual[:, :2], expected[:, :2]))
    assert not bool(mx.array_equal(actual[:, 2:5], expected[:, 2:5]))
    assert bool(mx.array_equal(actual[:, 5:], expected[:, 5:]))
    assert activation.adapted_positions == 3
    assert activation.observed_positions == 7


def test_scope_validation_and_restoration_are_fail_closed():
    with pytest.raises(ValueError, match="supplied together"):
        with recurrence_adapter_scope(start=1):
            pass
    with pytest.raises(ValueError, match="non-empty"):
        with recurrence_adapter_scope(start=2, stop=2):
            pass

    _, wrapped = _projection()
    with pytest.raises(ValueError, match="exceeds sequence"):
        with recurrence_adapter_scope(start=0, stop=4):
            wrapped(mx.ones((1, 3, 4)))
    assert current_recurrence_adapter_scope() is None


def test_nested_scope_restores_parent_activation():
    _, wrapped = _projection()
    with recurrence_adapter_scope() as outer:
        assert current_recurrence_adapter_scope() is outer
        with recurrence_adapter_scope(start=1, stop=2) as inner:
            assert current_recurrence_adapter_scope() is inner
            wrapped(mx.ones((1, 2, 4)))
        assert current_recurrence_adapter_scope() is outer
        assert outer.calls == inner.calls == 1
        assert outer.adapted_positions == inner.adapted_positions == 1
    assert current_recurrence_adapter_scope() is None


def test_disabled_reference_runs_the_same_scope_without_adapter_delta():
    base, wrapped = _projection()
    x = mx.ones((1, 3, 4))
    expected = base(x)

    with recurrence_adapter_disabled():
        with recurrence_adapter_scope() as activation:
            actual = wrapped(x)

    mx.eval(expected, actual)
    assert bool(mx.array_equal(actual, expected))
    assert activation.calls == 0
    assert current_recurrence_adapter_scope() is None


def test_nested_disable_boundaries_do_not_reenable_early():
    base, wrapped = _projection()
    x = mx.ones((1, 2, 4))
    expected = base(x)

    with recurrence_adapter_disabled():
        with recurrence_adapter_disabled():
            with recurrence_adapter_scope():
                nested = wrapped(x)
        with recurrence_adapter_scope():
            outer = wrapped(x)

    mx.eval(expected, nested, outer)
    assert bool(mx.array_equal(nested, expected))
    assert bool(mx.array_equal(outer, expected))


def test_window_runner_is_the_only_automatic_activation_boundary():
    _, projection = _projection()

    class Layer:
        def __call__(self, hidden, _mask, cache):
            cache.offset += int(hidden.shape[1])
            return projection(hidden)

    class Cache:
        offset = 0

    class Inner:
        layers = [Layer()]

    x = mx.ones((1, 3, 4))
    direct = projection(x)
    runner = WindowRunner(Inner(), ComputeBudget(), mask_fn=lambda *_: None)
    latent = runner.run(x, [Cache()], 0, 1, persist=True)
    mx.eval(direct, latent)
    assert not bool(mx.array_equal(direct, latent))
    assert runner.adapter_receipt() == {
        "schema": "aura.recurrence_adapter_activation.v1",
        "scope": "latent_slots_only",
        "calls": 1,
        "adapted_positions": 3,
        "observed_positions": 3,
        "active": True,
    }


def test_episode_receipt_exposes_adapter_activation_evidence():
    receipt = EpisodeReceipt(
        recurrence_adapter={
            "schema": "aura.recurrence_adapter_activation.v1",
            "scope": "latent_slots_only",
            "calls": 2,
        }
    )
    assert receipt.to_dict()["recurrence_adapter"]["calls"] == 2


# --- attributable activation (SPARK-066) ------------------------------------
#
# `calls` proves something fired. It cannot tell "all wrapped projections
# fired" from "one did and the rest were never wrapped", and that difference
# is a treatment that is really a fraction of a treatment -- the CP227 failure
# one level down. These contracts cover the identity that closes it.


def _identified(block_index: int, site: str) -> ScopedLoRALinear:
    base = nn.Linear(4, 3, bias=False)
    wrapped = ScopedLoRALinear.from_base(
        base, r=2, scale=1.0, block_index=block_index, site=site
    )
    wrapped.lora_a = mx.ones_like(wrapped.lora_a)
    wrapped.lora_b = mx.ones_like(wrapped.lora_b)
    mx.eval(wrapped.parameters())
    return wrapped


def test_an_identified_projection_names_itself_when_it_applies():
    wrapped = _identified(40, "model.layers.40.self_attn.o_proj")
    x = mx.ones((1, 3, 4))
    with recurrence_adapter_scope() as activation:
        wrapped(x)
    assert activation.activated_blocks() == [40]
    assert activation.to_dict()["applied_sites"] == {
        "model.layers.40.self_attn.o_proj": 1
    }


def test_a_projection_that_never_ran_is_named_as_unfired():
    fired = _identified(40, "model.layers.40.self_attn.o_proj")
    dark = _identified(41, "model.layers.41.self_attn.o_proj")
    expected = [
        "model.layers.40.self_attn.o_proj",
        "model.layers.41.self_attn.o_proj",
    ]
    x = mx.ones((1, 3, 4))
    with recurrence_adapter_scope() as activation:
        fired(x)
    assert activation.calls == 1  # the aggregate looks healthy
    assert activation.unfired_sites(expected) == [
        "model.layers.41.self_attn.o_proj"
    ]
    assert dark is not None


def test_a_fully_dark_adapter_activates_no_block_at_all():
    wrapped = _identified(40, "model.layers.40.self_attn.o_proj")
    x = mx.ones((1, 3, 4))
    # No scope open: this is exactly the CP227 shape, the projection passes
    # through to the base weights.
    wrapped(x)
    with recurrence_adapter_scope() as activation:
        with recurrence_adapter_disabled():
            wrapped(x)
    assert activation.calls == 0
    assert activation.activated_blocks() == []
    assert activation.unfired_sites(["model.layers.40.self_attn.o_proj"]) == [
        "model.layers.40.self_attn.o_proj"
    ]


def test_repeated_applications_are_counted_per_site():
    wrapped = _identified(40, "model.layers.40.self_attn.o_proj")
    x = mx.ones((1, 3, 4))
    with recurrence_adapter_scope() as activation:
        for _ in range(4):
            wrapped(x)
    assert activation.applied_blocks == {40: 4}
    assert activation.applied_sites == {"model.layers.40.self_attn.o_proj": 4}


def test_a_span_scoped_application_is_attributed_too():
    wrapped = _identified(7, "model.layers.7.mlp.down_proj")
    x = mx.ones((1, 6, 4))
    with recurrence_adapter_scope(start=2, stop=5) as activation:
        wrapped(x)
    assert activation.activated_blocks() == [7]
    assert activation.adapted_positions == 3


def test_attachment_identity_is_validated():
    base = nn.Linear(4, 3, bias=False)
    with pytest.raises(ValueError):
        ScopedLoRALinear.from_base(base, r=2, block_index=-1)
    with pytest.raises(ValueError):
        ScopedLoRALinear.from_base(base, r=2, site="   ")
