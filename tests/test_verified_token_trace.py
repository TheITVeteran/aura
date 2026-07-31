from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pytest

from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes
from core.learning.verified_token_trace import (
    HuggingFaceTokenizerTraceAdapter,
    VerifiedTokenTraceError,
    build_tokenizer_bundle_identity,
    build_verified_token_trace,
    canonical_behavior_logprob,
    observable_completion_from_trace,
    tokenizer_adapter_source_sha256,
    tokenizer_file_bindings_from_bytes,
    validate_observable_completion,
    validate_verified_token_trace,
    validate_verified_token_trace_structure,
)


def test_behavior_logprob_canonicalization_is_stable_for_integral_values() -> None:
    assert canonical_behavior_logprob(-1.0) == "-1"
    assert canonical_behavior_logprob(0.0) == "0"
    assert canonical_behavior_logprob("-1.2500") == "-1.25"


def test_observable_completion_stops_at_first_valid_contract_before_eos() -> None:
    bundle = _bundle()
    bundle = build_tokenizer_bundle_identity(
        tokenizer_class=bundle["tokenizer_class"],
        tokenizer_files=bundle["tokenizer_files"],
        chat_template="{% for message in messages %}{{ message.content }}{% endfor %}",
        special_token_map={"eos_token_id": 99},
        encode_options=bundle["encode_options"],
        decode_options=bundle["decode_options"],
        implementation_source_sha256=bundle["implementation_source_sha256"],
    )
    observed = observable_completion_from_trace(
        token_ids=[10, 11, 99, 12],
        streaming_deltas=(
            "reason\n",
            'FINAL_ANSWER: {"value":1}',
            "<|im_end|>",
            " unreachable",
        ),
        tokenizer_bundle=bundle,
    )
    assert observed["optimization_token_count"] == 2
    assert observed["termination"] == "contract_complete"
    assert observed["response_text"].endswith('FINAL_ANSWER: {"value":1}')


def test_observable_completion_eos_prevents_late_answer_cherry_pick() -> None:
    bundle = build_tokenizer_bundle_identity(
        tokenizer_class="test.EosTokenizer",
        tokenizer_files=tokenizer_file_bindings_from_bytes(
            {
                "tokenizer.json": b"{}",
                "tokenizer_config.json": b"{}",
            }
        ),
        chat_template=None,
        special_token_map={"eos_token_id": 99},
        encode_options={},
        decode_options={},
        implementation_source_sha256="7" * 64,
    )
    observed = observable_completion_from_trace(
        token_ids=[10, 99, 11],
        streaming_deltas=(
            "unfinished",
            "<|im_end|>",
            '\nFINAL_ANSWER: {"value":1}',
        ),
        tokenizer_bundle=bundle,
    )
    assert observed["optimization_token_count"] == 2
    assert observed["termination"] == "eos_token"
    assert "FINAL_ANSWER" not in observed["response_text"]


def test_observable_completion_rejects_tampered_boundary() -> None:
    bundle = _bundle()
    observed = observable_completion_from_trace(
        token_ids=[10, 11],
        streaming_deltas=("reason", " only"),
        tokenizer_bundle=bundle,
    )
    tampered = copy.deepcopy(observed)
    tampered["optimization_token_count"] = 1
    with pytest.raises(
        VerifiedTokenTraceError,
        match="observable_completion_mismatch",
    ):
        validate_observable_completion(
            tampered,
            token_ids=[10, 11],
            streaming_deltas=("reason", " only"),
            tokenizer_bundle=bundle,
        )


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _reseal(trace: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in trace.items() if key != "trace_sha256"}
    trace["trace_sha256"] = _sha256(canonical_json_bytes(body))
    return trace


def _bundle(*, suffix: bytes = b"", implementation: str | None = None) -> dict[str, Any]:
    return build_tokenizer_bundle_identity(
        tokenizer_class="Qwen2TokenizerFast",
        tokenizer_files=tokenizer_file_bindings_from_bytes(
            {
                "tokenizer.json": b'{"model":"bpe"}' + suffix,
                "tokenizer_config.json": b'{"add_prefix_space":false}',
            }
        ),
        chat_template="{% for message in messages %}{{ message.content }}{% endfor %}",
        special_token_map={"eos_token": "<|im_end|>", "pad_token": None},
        encode_options={"add_special_tokens": False},
        decode_options={
            "skip_special_tokens": False,
            "clean_up_tokenization_spaces": False,
        },
        implementation_source_sha256=implementation or ("a" * 64),
    )


@dataclass
class StubAdapter:
    _bundle: Mapping[str, Any]
    prompt_encodings: Mapping[str, Sequence[int]]
    decodings: Mapping[tuple[int, ...], str]
    streams: Mapping[tuple[int, ...], Sequence[str]]

    @property
    def bundle_identity(self) -> Mapping[str, Any]:
        return self._bundle

    def encode_prompt(self, prompt_text: str) -> Sequence[int]:
        return self.prompt_encodings[prompt_text]

    def decode_output(self, token_ids: Sequence[int]) -> str:
        return self.decodings[tuple(token_ids)]

    def stream_decode_deltas(self, token_ids: Sequence[int]) -> Sequence[str]:
        return self.streams[tuple(token_ids)]


def _adapter(*, bundle: Mapping[str, Any] | None = None) -> StubAdapter:
    return StubAdapter(
        _bundle=bundle or _bundle(),
        prompt_encodings={"Prompt": [1, 2]},
        decodings={(10, 11): "é"},
        streams={(10, 11): ("", "é")},
    )


def _trace(adapter: StubAdapter | None = None) -> tuple[StubAdapter, dict[str, Any]]:
    selected = adapter or _adapter()
    return selected, build_verified_token_trace(
        adapter=selected,
        prompt_text="Prompt",
        prompt_token_ids=[1, 2],
        output_token_ids=[10, 11],
        behavior_logprobs=["-0.25", "-1.5"],
        response_text="é",
        streaming_deltas=["", "é"],
    )


def test_split_utf8_token_allows_empty_streaming_delta() -> None:
    adapter, trace = _trace()

    validated = validate_verified_token_trace(trace, adapter=adapter)

    assert validated["generation"]["streaming_deltas"] == ["", "é"]
    assert validated["generation"]["response_utf8_b64"] == "w6k="


def test_noncanonical_generated_token_segmentation_does_not_reencode_output() -> None:
    class NoncanonicalAdapter(StubAdapter):
        def encode_prompt(self, prompt_text: str) -> Sequence[int]:
            if prompt_text == "ab":
                raise AssertionError("generated output must never be re-encoded")
            return super().encode_prompt(prompt_text)

    adapter = NoncanonicalAdapter(
        _bundle=_bundle(),
        prompt_encodings={"Prompt": [1]},
        decodings={(20, 21): "ab"},
        streams={(20, 21): ("a", "b")},
    )
    trace = build_verified_token_trace(
        adapter=adapter,
        prompt_text="Prompt",
        prompt_token_ids=[1],
        output_token_ids=[20, 21],
        behavior_logprobs=[-0.2, -0.3],
        response_text="ab",
        streaming_deltas=["a", "b"],
    )

    validate_verified_token_trace(trace, adapter=adapter)


def test_resealed_forged_text_is_structural_only_and_fails_real_replay() -> None:
    adapter, trace = _trace()
    attacked = copy.deepcopy(trace)
    attacked["generation"]["response_text"] = "FORGED"
    attacked["generation"]["response_utf8_b64"] = "Rk9SR0VE"
    attacked["generation"]["streaming_deltas"] = ["FOR", "GED"]
    attacked["generation"]["streaming_deltas_utf8_b64"] = ["Rk9S", "R0VE"]
    attacked["generation"]["streaming_deltas_sha256"] = _sha256(
        canonical_json_bytes(["Rk9S", "R0VE"])
    )
    _reseal(attacked)

    validate_verified_token_trace_structure(
        attacked,
        expected_tokenizer_bundle_sha256=adapter.bundle_identity["bundle_sha256"],
    )
    with pytest.raises(VerifiedTokenTraceError, match="decode_mismatch"):
        validate_verified_token_trace(attacked, adapter=adapter)


@pytest.mark.parametrize("field", ["token_ids", "behavior_logprobs"])
def test_token_or_logprob_tamper_breaks_committed_trace(field: str) -> None:
    adapter, trace = _trace()
    commitment = trace["trace_sha256"]
    attacked = copy.deepcopy(trace)
    if field == "token_ids":
        attacked["generation"][field][0] = 99
        attacked["generation"]["token_ids_sha256"] = _sha256(
            canonical_json_bytes(attacked["generation"][field])
        )
    else:
        attacked["generation"][field][0] = "-0.75"
        attacked["generation"]["behavior_logprobs_sha256"] = _sha256(
            canonical_json_bytes(attacked["generation"][field])
        )
    _reseal(attacked)

    with pytest.raises(
        VerifiedTokenTraceError,
        match=(
            "verified_token_trace_commitment_mismatch"
            if field == "behavior_logprobs"
            else "verified_token_trace_commitment_mismatch"
        ),
    ):
        validate_verified_token_trace(
            attacked,
            adapter=adapter,
            expected_trace_sha256=commitment,
        )


def test_tokenizer_identity_substitution_is_rejected() -> None:
    adapter, trace = _trace()
    substituted = _adapter(bundle=_bundle(suffix=b"changed"))

    with pytest.raises(VerifiedTokenTraceError, match="tokenizer_bundle_substitution"):
        validate_verified_token_trace(trace, adapter=substituted)


def test_prompt_encoding_mismatch_is_rejected_after_valid_reseal() -> None:
    adapter, trace = _trace()
    attacked = copy.deepcopy(trace)
    attacked["prompt"]["token_ids"] = [1, 9]
    attacked["prompt"]["token_ids_sha256"] = _sha256(
        canonical_json_bytes(attacked["prompt"]["token_ids"])
    )
    _reseal(attacked)

    with pytest.raises(VerifiedTokenTraceError, match="prompt_encoding_mismatch"):
        validate_verified_token_trace(attacked, adapter=adapter)


def test_full_sequence_decode_mismatch_is_rejected_without_output_reencoding() -> None:
    adapter, trace = _trace()
    attacked = copy.deepcopy(trace)
    attacked["generation"]["response_text"] = "e"
    attacked["generation"]["response_utf8_b64"] = "ZQ=="
    attacked["generation"]["streaming_deltas"] = ["", "e"]
    attacked["generation"]["streaming_deltas_utf8_b64"] = ["", "ZQ=="]
    attacked["generation"]["streaming_deltas_sha256"] = _sha256(
        canonical_json_bytes(attacked["generation"]["streaming_deltas_utf8_b64"])
    )
    _reseal(attacked)

    with pytest.raises(VerifiedTokenTraceError, match="full_sequence_decode_mismatch"):
        validate_verified_token_trace(attacked, adapter=adapter)


def test_tokenizer_bundle_binds_template_special_tokens_options_and_source() -> None:
    original = _bundle()
    for field in (
        "chat_template",
        "special_token_map",
        "encode_options",
        "decode_options",
        "implementation_source_sha256",
    ):
        attacked = copy.deepcopy(original)
        if field == "chat_template":
            attacked[field]["size_bytes"] += 1
        elif field == "implementation_source_sha256":
            attacked[field] = "b" * 64
        else:
            attacked[field]["attacked"] = True
        with pytest.raises(VerifiedTokenTraceError):
            build_verified_token_trace(
                adapter=_adapter(bundle=attacked),
                prompt_text="Prompt",
                prompt_token_ids=[1, 2],
                output_token_ids=[10, 11],
                behavior_logprobs=[-0.25, -1.5],
                response_text="é",
                streaming_deltas=["", "é"],
            )


class FakeHuggingFaceTokenizer:
    def encode(self, text: str, **options: Any) -> list[int]:
        assert options == {"add_special_tokens": False}
        return [1] if text == "Prompt" else [999]

    def decode(self, token_ids: list[int], **options: Any) -> str:
        assert options == {
            "clean_up_tokenization_spaces": False,
            "skip_special_tokens": False,
        }
        return {
            (): "",
            (30,): "�",
            (30, 31): "€",
        }[tuple(token_ids)]


def test_huggingface_qwen_adapter_is_model_free_and_prefix_stable() -> None:
    bundle = _bundle(implementation=tokenizer_adapter_source_sha256())
    adapter = HuggingFaceTokenizerTraceAdapter(FakeHuggingFaceTokenizer(), bundle)

    trace = build_verified_token_trace(
        adapter=adapter,
        prompt_text="Prompt",
        prompt_token_ids=[1],
        output_token_ids=[30, 31],
        behavior_logprobs=[-0.4, -0.8],
        response_text="€",
    )

    assert trace["generation"]["streaming_deltas"] == ["", "€"]
    validate_verified_token_trace(trace, adapter=adapter)


def test_positive_or_nonfinite_behavior_logprobs_are_rejected() -> None:
    adapter = _adapter()
    for attacked in (0.01, float("nan"), float("inf")):
        with pytest.raises(VerifiedTokenTraceError, match="behavior_logprob_invalid"):
            build_verified_token_trace(
                adapter=adapter,
                prompt_text="Prompt",
                prompt_token_ids=[1, 2],
                output_token_ids=[10, 11],
                behavior_logprobs=[attacked, -1.0],
                response_text="é",
                streaming_deltas=["", "é"],
            )
