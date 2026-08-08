"""The RLC output floor reuses one bound ordinary-decode artifact."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))


class _Tokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):  # noqa: ARG002
        return [1 + (ord(character) % 100) for character in text]

    def decode(self, tokens):
        return "".join(chr(65 + (int(token) % 26)) for token in tokens)


@pytest.fixture(scope="module")
def tiny_model():
    mx = pytest.importorskip("mlx.core")
    from mlx_lm.models.qwen2 import Model, ModelArgs

    model = Model(
        ModelArgs(
            model_type="qwen2",
            hidden_size=32,
            num_hidden_layers=8,
            intermediate_size=64,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=128,
            num_key_value_heads=2,
            max_position_embeddings=512,
            rope_theta=10000.0,
        )
    )
    mx.eval(model.parameters())
    return model


def _artifact(model_dir: Path, *, prompt=(3, 8, 15, 22), output=(7, 9, 11)):
    from core.brain.llm.latent_cortex.governance import checkpoint_file_fingerprint
    from core.brain.llm.latent_cortex.incumbent_artifact import (
        build_incumbent_artifact,
    )

    checkpoint = checkpoint_file_fingerprint(model_dir)
    tokenizer = _Tokenizer()
    return build_incumbent_artifact(
        input_tokens=prompt,
        output_tokens=output,
        output_text=tokenizer.decode(output),
        checkpoint_fingerprint=checkpoint["fingerprint"],
        checkpoint_fingerprint_method=checkpoint["method"],
        max_tokens=24,
        n_layers=8,
        termination="contract_complete",
    )


def test_artifact_reconstruction_rejects_every_bound_dimension(tmp_path: Path):
    from core.brain.llm.latent_cortex.governance import checkpoint_file_fingerprint
    from core.brain.llm.latent_cortex.incumbent_artifact import (
        build_incumbent_artifact,
        validate_incumbent_artifact,
    )

    (tmp_path / "model.safetensors").write_bytes(b"frozen-checkpoint")
    checkpoint = checkpoint_file_fingerprint(tmp_path)
    artifact = _artifact(tmp_path)
    kwargs = {
        "input_tokens": [3, 8, 15, 22],
        "checkpoint_fingerprint": checkpoint["fingerprint"],
        "checkpoint_fingerprint_method": checkpoint["method"],
        "max_tokens": 24,
        "n_layers": 8,
        "decode": lambda values: _Tokenizer().decode(values),
    }
    assert validate_incumbent_artifact(artifact, **kwargs) == artifact

    with pytest.raises(ValueError, match="reconstruction differs"):
        validate_incumbent_artifact(artifact, **{**kwargs, "input_tokens": [3, 8, 15]})
    with pytest.raises(ValueError, match="reconstruction differs"):
        validate_incumbent_artifact(
            artifact,
            **{**kwargs, "checkpoint_fingerprint": "0" * 64},
        )
    with pytest.raises(ValueError, match="round trip differs"):
        validate_incumbent_artifact(
            build_incumbent_artifact(
                input_tokens=kwargs["input_tokens"],
                output_tokens=artifact.tokens,
                output_text="tampered",
                checkpoint_fingerprint=checkpoint["fingerprint"],
                checkpoint_fingerprint_method=checkpoint["method"],
                max_tokens=24,
                n_layers=8,
                termination="contract_complete",
            ),
            **kwargs,
        )


def test_public_receipt_rejects_compute_and_digest_tampering(tmp_path: Path):
    from core.brain.llm.latent_cortex.incumbent_artifact import (
        validate_incumbent_receipt,
    )

    (tmp_path / "model.safetensors").write_bytes(b"frozen-checkpoint")
    receipt = _artifact(tmp_path).receipt
    assert validate_incumbent_receipt(receipt) == receipt

    tampered = {**receipt, "compute": dict(receipt["compute"])}
    tampered["compute"]["transformer_layer_apps"] += 1
    with pytest.raises(ValueError, match="compute accounting"):
        validate_incumbent_receipt(tampered)

    tampered = {**receipt, "receipt_sha256": "0" * 64}
    with pytest.raises(ValueError, match="digest differs"):
        validate_incumbent_receipt(tampered)


def test_journal_round_trip_revalidates_private_output(tmp_path: Path):
    from core.brain.llm.latent_cortex.governance import checkpoint_file_fingerprint
    from core.brain.llm.latent_cortex.incumbent_artifact import (
        incumbent_artifact_from_value,
        incumbent_artifact_to_value,
        validate_incumbent_artifact,
    )

    (tmp_path / "model.safetensors").write_bytes(b"frozen-checkpoint")
    artifact = _artifact(tmp_path)
    restored = incumbent_artifact_from_value(incumbent_artifact_to_value(artifact))
    checkpoint = checkpoint_file_fingerprint(tmp_path)
    assert validate_incumbent_artifact(
        restored,
        input_tokens=[3, 8, 15, 22],
        checkpoint_fingerprint=checkpoint["fingerprint"],
        checkpoint_fingerprint_method=checkpoint["method"],
        max_tokens=24,
        n_layers=8,
        decode=lambda values: _Tokenizer().decode(values),
    ) == artifact

    value = incumbent_artifact_to_value(artifact)
    value["tokens"][0] += 1
    with pytest.raises(ValueError, match="reconstruction differs"):
        validate_incumbent_artifact(
            incumbent_artifact_from_value(value),
            input_tokens=[3, 8, 15, 22],
            checkpoint_fingerprint=checkpoint["fingerprint"],
            checkpoint_fingerprint_method=checkpoint["method"],
            max_tokens=24,
            n_layers=8,
        )


def test_engine_retains_the_exact_bound_incumbent(tiny_model, tmp_path: Path):
    import run_rlc_reconciliation_sweep as sweep

    from core.brain.llm.latent_cortex.engine import LatentCortexEngine
    from core.brain.llm.latent_cortex.types import ComputeBudget

    (tmp_path / "model.safetensors").write_bytes(b"frozen-checkpoint")
    prompt = [3, 8, 15, 22]
    artifact = _artifact(tmp_path, prompt=prompt, output=(7, 9, 11))
    config = sweep._build_config(4, 8, "suppressed", 24, profile="full")
    engine = LatentCortexEngine(
        tiny_model,
        config=config,
        tokenizer=_Tokenizer(),
        model_path=str(tmp_path),
    )
    # The synthetic tokenizer has no on-disk tokenizer identity. That is
    # intentionally outside this test; checkpoint binding was already armed
    # by pre_episode and the production tokenizer contract has its own suite.
    engine.invariant.post_episode = lambda _receipt: True
    result = engine.reason(
        token_ids=prompt,
        budget=ComputeBudget(max_layer_apps=4_000_000, wall_clock_s=30.0),
        incumbent_artifact=artifact,
    )

    assert result.ok is True
    assert result.tokens == list(artifact.tokens)
    assert result.text == artifact.text
    assert result.receipt.incumbent_artifact == artifact.receipt
    assert result.receipt.answer_replacement["decision"] == "retain"
    final_nodes = [
        node for node in result.receipt.kv_state_tree["nodes"] if node["final"]
    ]
    assert len(final_nodes) == 1
    assert final_nodes[0]["authority"] == "canonical_ordinary_decode_artifact"


def test_latent_owned_output_cannot_smuggle_an_incumbent(tiny_model, tmp_path: Path):
    import run_rlc_reconciliation_sweep as sweep

    from core.brain.llm.latent_cortex.engine import LatentCortexEngine

    (tmp_path / "model.safetensors").write_bytes(b"frozen-checkpoint")
    artifact = _artifact(tmp_path)
    config = sweep._build_config(4, 8, "suppressed", 24, profile="mechanism")
    engine = LatentCortexEngine(
        tiny_model,
        config=config,
        tokenizer=_Tokenizer(),
        model_path=str(tmp_path),
    )
    with pytest.raises(ValueError, match="decode_incumbent_policy"):
        engine.reason(token_ids=[3, 8, 15, 22], incumbent_artifact=artifact)
