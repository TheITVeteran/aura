"""Transactional loading contracts for split recurrent and coda tissue."""
from __future__ import annotations

import hashlib

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx.utils import tree_flatten  # noqa: E402
from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from core.brain.llm.latent_cortex.adapter_identity import (  # noqa: E402
    inspect_mlx_tensor_metadata,
)
from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec  # noqa: E402
from core.brain.llm.latent_cortex.recurrence_adapter import (  # noqa: E402
    ScopedCodaLoRALinear,
    ScopedLoRALinear,
    coda_adapter_scope,
)
from core.brain.llm.latent_cortex.resident_adapter_loader import (  # noqa: E402
    load_resident_adapter,
)
from core.brain.llm.latent_cortex.resident_recurrent_sft_adapter_identity import (  # noqa: E402
    CODA_INTERPRETING_MANIFEST_SCHEMA,
)
from core.learning.recurrent_grpo import attach_recurrent_policy_adapters  # noqa: E402


def _model(seed: int) -> Model:
    mx.random.seed(seed)
    model = Model(
        ModelArgs(
            model_type="qwen2",
            hidden_size=16,
            num_hidden_layers=4,
            intermediate_size=32,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=32,
            num_key_value_heads=2,
            max_position_embeddings=64,
            rope_theta=10000.0,
        )
    )
    mx.eval(model.parameters())
    return model


def test_loader_reconstructs_split_scopes_and_keeps_ordinary_decode_exact(
    tmp_path,
) -> None:
    spec = RLCExecutionSpec(
        recurrent_steps=2,
        prelude_frac=0.25,
        coda_frac=0.25,
    )
    source = _model(991)
    sites = attach_recurrent_policy_adapters(
        source,
        spec,
        lora_rank=2,
        lora_scale=1.0,
        lora_dropout=0.0,
        lora_layers=1,
        lora_targets=("o_proj",),
        initialization_seed=17,
        depth_conditioned_steps=2,
        role_conditioned_branches=2,
        coda_lora_layers=1,
        coda_lora_targets=("down_proj",),
    )
    source.model.layers[2].self_attn.o_proj.lora_a = mx.ones((16, 2))
    source.model.layers[2].self_attn.o_proj.lora_b = mx.ones((2, 16))
    source.model.layers[3].mlp.down_proj.lora_a = mx.ones((32, 2))
    source.model.layers[3].mlp.down_proj.lora_b = mx.ones((2, 16))
    tensors = dict(tree_flatten(source.trainable_parameters()))
    mx.eval(tensors)

    package = tmp_path / "package"
    package.mkdir()
    adapter_path = package / "adapter.safetensors"
    mx.save_safetensors(str(adapter_path), tensors)
    payload = adapter_path.read_bytes()
    tensor_rows = [row.to_dict() for row in inspect_mlx_tensor_metadata(adapter_path)]
    recurrent_path, coda_path = sites
    manifest = {
        "schema": CODA_INTERPRETING_MANIFEST_SCHEMA,
        "bindings": {
            "adapter": {
                "path": "adapter.safetensors",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        },
        "lora": {
            "rank": 2,
            "scale": 1.0,
            "dropout": 0.0,
            "layers": 1,
            "targets": ["o_proj"],
            "wrapped_projections": 2,
            "projection_paths": list(sites),
            "depth_bank_size": 2,
            "role_bank_size": 2,
            "coda_layers": 1,
            "coda_targets": ["down_proj"],
            "coda_wrapped_projections": 1,
            "coda_projection_paths": [coda_path],
        },
        "tensors": tensor_rows,
    }

    target = _model(991)
    original_recurrent = target.model.layers[2].self_attn.o_proj
    original_coda = target.model.layers[3].mlp.down_proj
    recurrent_input = mx.ones((1, 1, 16))
    coda_input = mx.ones((1, 1, 32))
    recurrent_base = original_recurrent(recurrent_input)
    coda_base = original_coda(coda_input)

    assert load_resident_adapter(target, package, manifest) == 2
    recurrent = target.model.layers[2].self_attn.o_proj
    coda = target.model.layers[3].mlp.down_proj
    assert recurrent_path == "model.layers.2.self_attn.o_proj"
    assert coda_path == "model.layers.3.mlp.down_proj"
    assert isinstance(recurrent, ScopedLoRALinear)
    assert isinstance(coda, ScopedCodaLoRALinear)
    assert bool(mx.array_equal(recurrent(recurrent_input), recurrent_base))
    assert bool(mx.array_equal(coda(coda_input), coda_base))
    with coda_adapter_scope():
        assert not bool(mx.array_equal(coda(coda_input), coda_base))
        assert bool(mx.array_equal(recurrent(recurrent_input), recurrent_base))
