from __future__ import annotations

import copy
import json

import pytest

import core.learning.structured_sft as structured
from core.learning.structured_sft import (
    CODE_TOOL,
    CODE_TOOL_REPAIR,
    DERIVATION_TARGET,
    FORMAL_LOGIC,
    HOLDOUT_SPLIT,
    REPAIR_INTERPRETATION_TARGET,
    REPAIR_TOOL_CALL_TARGET,
    STRUCTURED_PROGRAM,
    STRUCTURED_SFT_EVALUATOR_FILES,
    TOOL_CALL_TARGET,
    TOOL_INTERPRETATION_TARGET,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    StructuredSFTCurriculumSpec,
    StructuredSFTError,
    build_candidate_dataset_artifacts,
    build_structured_sft_curriculum,
    build_structured_sft_custody_bundles,
    curriculum_manifest,
    generate_structured_sft_example,
    trainer_rows,
    validate_candidate_dataset_artifacts,
    validate_structured_sft_curriculum,
    validate_structured_sft_custody_pair,
    validate_structured_sft_example,
    validate_trainer_tokenization,
    verify_structured_sft_example_semantics,
)

HOLDOUT_SEED = bytes(range(32))

TARGETS = (
    (STRUCTURED_PROGRAM, DERIVATION_TARGET),
    (FORMAL_LOGIC, DERIVATION_TARGET),
    (CODE_TOOL, TOOL_CALL_TARGET),
    (CODE_TOOL, TOOL_INTERPRETATION_TARGET),
    (CODE_TOOL_REPAIR, REPAIR_TOOL_CALL_TARGET),
    (CODE_TOOL_REPAIR, REPAIR_INTERPRETATION_TARGET),
)


@pytest.mark.parametrize(("family", "target_kind"), TARGETS)
def test_examples_are_deterministic_and_replay_validated(family, target_kind):
    first = generate_structured_sft_example(
        family=family,
        target_kind=target_kind,
        seed=910,
    )
    second = generate_structured_sft_example(
        family=family,
        target_kind=target_kind,
        seed=910,
    )

    assert first == second
    assert validate_structured_sft_example(first) == first
    assert first["loss_policy"] == {
        "trainer": "mlx_lm.ChatDataset",
        "mask_prompt": True,
        "supervised_region": "final_assistant_message_only",
        "prior_assistant_failures_are_context_only": True,
    }
    assert first["privacy_governance_disposition"]["contains_user_content"] is False
    assert (
        first["privacy_governance_disposition"]["contains_hidden_chain_of_thought"]
        is False
    )
    assert "<thought>" not in json.dumps(first["messages"]).lower()


def test_source_binding_covers_semantic_dependency_closure() -> None:
    example = generate_structured_sft_example(
        family=STRUCTURED_PROGRAM,
        target_kind=DERIVATION_TARGET,
        seed=909,
    )
    binding = example["source_binding"]

    assert binding["schema"] == "aura.rlc.structured_sft_source_closure.v2"
    assert {row["path"] for row in binding["files"]} == set(
        structured._SOURCE_BINDING_PATHS
    )
    assert binding["runtime"]["implementation"]
    assert len(binding["sha256"]) == 64


def test_derivation_contains_replayable_logical_program_and_proof():
    example = generate_structured_sft_example(
        family=STRUCTURED_PROGRAM,
        target_kind=DERIVATION_TARGET,
        seed=41,
    )

    content = example["messages"][-1]["content"]
    assert "LOGICAL_FORM:" in content
    assert "PROGRAM:" in content
    assert "PROOF_STEPS:" in content
    assert content.endswith(
        f"FINAL_ANSWER: {example['oracle']['expected_final_answer']}"
    )
    assert example["oracle"]["checked"] is True
    assert example["oracle"]["executor"] == "deterministic_modular_state_machine"


def test_formal_logic_derivation_is_proof_kernel_certified():
    example = generate_structured_sft_example(
        family=FORMAL_LOGIC,
        target_kind=DERIVATION_TARGET,
        seed=52,
    )

    content = example["messages"][-1]["content"]
    assert "PROGRAM: analytic_tableau_with_independent_kernel_check" in content
    assert "KERNEL_CERTIFICATE:" in content
    assert content.endswith("FINAL_ANSWER: 1")
    assert example["oracle"]["kernel_verified"] is True
    assert (
        example["oracle"]["executor"]
        == "core.reasoning.proof_kernel.check_certificate"
    )
    assert example["oracle"]["certificate_sha256"]


def test_tool_call_uses_live_qwen_and_code_repl_message_shape():
    example = generate_structured_sft_example(
        family=CODE_TOOL,
        target_kind=TOOL_CALL_TARGET,
        seed=72,
    )

    assert example["tools"][0]["function"]["name"] == "code_repl"
    target = example["messages"][-1]
    assert target["role"] == "assistant"
    assert target["content"] == ""
    call = target["tool_calls"][0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "code_repl"
    arguments = json.loads(call["function"]["arguments"])
    assert set(arguments) == {"capture_files", "code", "timeout"}
    assert arguments["capture_files"] is False
    assert arguments["timeout"] == 10


def test_tool_interpretation_requires_real_success_receipt():
    example = generate_structured_sft_example(
        family=CODE_TOOL,
        target_kind=TOOL_INTERPRETATION_TARGET,
        seed=83,
    )

    tool_result = example["messages"][-2]
    result = json.loads(tool_result["content"])
    assert result["ok"] is True
    assert result["status"] == "ok"
    assert result["returncode"] == 0
    assert result["stderr"] == ""
    assert result["engine"] == "sandbox_runner"
    assert "TOOL_RESULT_INTERPRETATION:" in example["messages"][-1]["content"]


def test_repair_target_masks_failed_call_and_supervises_only_correction():
    example = generate_structured_sft_example(
        family=CODE_TOOL_REPAIR,
        target_kind=REPAIR_TOOL_CALL_TARGET,
        seed=94,
    )

    failed_call = example["messages"][2]
    failed_result = json.loads(example["messages"][3]["content"])
    corrected_call = example["messages"][-1]
    assert failed_result["ok"] is False
    assert failed_result["status"] == "error"
    assert failed_result["returncode"] == 0
    assert "missing_operand" in failed_call["tool_calls"][0]["function"]["arguments"]
    assert "missing_operand" not in corrected_call["tool_calls"][0]["function"]["arguments"]
    assert example["loss_policy"]["supervised_region"] == "final_assistant_message_only"
    assert example["oracle"]["corrected_transition_verified"] is True


def test_repair_interpretation_binds_both_execution_results():
    example = generate_structured_sft_example(
        family=CODE_TOOL_REPAIR,
        target_kind=REPAIR_INTERPRETATION_TARGET,
        seed=105,
    )

    assert [message["role"] for message in example["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    content = example["messages"][-1]["content"]
    assert "LOCAL_REPAIR:" in content
    assert "sandbox_execution_error.undefined_name" in content
    assert "same_executor_rechecked" in content
    assert example["oracle"]["failed_result_sha256"]
    assert example["oracle"]["execution_result_sha256"]


@pytest.mark.parametrize(
    "mutation",
    (
        "final_answer",
        "tool_result",
        "tool_arguments",
        "source_binding",
        "loss_policy",
    ),
)
def test_example_tampering_is_refused(mutation):
    example = generate_structured_sft_example(
        family=CODE_TOOL_REPAIR,
        target_kind=REPAIR_INTERPRETATION_TARGET,
        seed=116,
    )
    tampered = copy.deepcopy(example)
    if mutation == "final_answer":
        tampered["messages"][-1]["content"] = tampered["messages"][-1][
            "content"
        ].replace("FINAL_ANSWER:", "FINAL_ANSWER: 999 #")
    elif mutation == "tool_result":
        result = json.loads(tampered["messages"][-2]["content"])
        result["stdout"] = "999\n"
        tampered["messages"][-2]["content"] = json.dumps(result)
    elif mutation == "tool_arguments":
        tampered["messages"][-3]["tool_calls"][0]["function"]["arguments"] = (
            '{"capture_files":false,"code":"print(999)","timeout":10}'
        )
    elif mutation == "source_binding":
        tampered["source_binding"]["sha256"] = "0" * 64
    else:
        tampered["loss_policy"]["mask_prompt"] = False

    with pytest.raises(StructuredSFTError):
        validate_structured_sft_example(tampered)


def test_independent_semantic_verifier_rejects_self_consistent_bad_program() -> None:
    example = generate_structured_sft_example(
        family=STRUCTURED_PROGRAM,
        target_kind=DERIVATION_TARGET,
        seed=117,
    )
    tampered = copy.deepcopy(example)
    payload = tampered["verification_payload"]
    payload["program"][0]["after"] = (
        payload["program"][0]["after"] + 1
    ) % payload["logical_form"]["modulus"]
    tampered["oracle"]["program_sha256"] = structured._sha256(
        payload["program"]
    )

    with pytest.raises(
        StructuredSFTError,
        match="structured_sft_semantic_program_invalid",
    ):
        verify_structured_sft_example_semantics(tampered)


@pytest.mark.parametrize(("family", "target_kind"), TARGETS)
def test_independent_semantic_verifier_binds_entire_public_prompt(
    family,
    target_kind,
) -> None:
    example = generate_structured_sft_example(
        family=family,
        target_kind=target_kind,
        seed=118,
    )
    example["messages"][1]["content"] = "Unrelated self-consistent prompt."

    with pytest.raises(
        StructuredSFTError,
        match="structured_sft_semantic_message_binding_invalid",
    ):
        verify_structured_sft_example_semantics(example)


def test_repair_semantics_proves_exact_single_substitution() -> None:
    example = generate_structured_sft_example(
        family=CODE_TOOL_REPAIR,
        target_kind=REPAIR_INTERPRETATION_TARGET,
        seed=119,
    )
    payload = example["verification_payload"]
    payload["failed_code"] = "print(missing_operand + 999)"
    payload["failed_execution"] = structured._normalized_execution(
        payload["failed_code"],
        allow_missing_operand=True,
    )
    example["oracle"]["failed_executable_sha256"] = (
        structured.hashlib.sha256(payload["failed_code"].encode()).hexdigest()
    )
    example["oracle"]["failed_result_sha256"] = structured._sha256(
        payload["failed_execution"]
    )
    failed_call = example["messages"][2]
    failed_id = failed_call["tool_calls"][0]["id"]
    example["messages"][2] = structured._tool_call(
        failed_id,
        payload["failed_code"],
    )
    example["messages"][3] = structured._tool_result_message(
        call_id=failed_id,
        execution=payload["failed_execution"],
    )

    with pytest.raises(
        StructuredSFTError,
        match="structured_sft_semantic_repair_invalid",
    ):
        verify_structured_sft_example_semantics(example)


def test_malformed_tool_call_fails_closed_without_attribute_error() -> None:
    example = generate_structured_sft_example(
        family=CODE_TOOL,
        target_kind=TOOL_CALL_TARGET,
        seed=118,
    )
    example["messages"][-1]["tool_calls"] = [None]
    committed = dict(example)
    committed.pop("example_sha256")
    example["example_sha256"] = structured._sha256(committed)

    with pytest.raises(
        StructuredSFTError,
        match="structured_sft_tool_call_invalid",
    ):
        validate_structured_sft_example(example)


@pytest.mark.parametrize(
    "code",
    (
        "print(__import__('os'))",
        "print((1).__class__)",
        "print([x for x in range(3)])",
        "print(values[0])",
        "print(open('x'))",
        "print(2 ** 8)",
    ),
)
def test_semantic_verifier_rejects_non_arithmetic_code_before_execution(
    monkeypatch,
    code,
) -> None:
    executed = False

    def forbidden_execution(*_args, **_kwargs):
        nonlocal executed
        executed = True
        raise AssertionError("untrusted code must not execute")

    monkeypatch.setattr(structured, "run_untrusted", forbidden_execution)
    with pytest.raises(StructuredSFTError):
        structured._normalized_execution(code)
    assert executed is False


@pytest.fixture(scope="module")
def curriculum():
    return build_structured_sft_curriculum(
        StructuredSFTCurriculumSpec(
            seed=8128,
            train_cases_per_family=2,
            validation_cases_per_family=1,
            holdout_cases_per_family=1,
        ),
        holdout_seed=HOLDOUT_SEED,
    )


def test_curriculum_has_balanced_disjoint_splits(curriculum):
    validated = validate_structured_sft_curriculum(curriculum)

    assert {split: len(rows) for split, rows in validated["splits"].items()} == {
        TRAIN_SPLIT: 12,
        VALIDATION_SPLIT: 6,
        HOLDOUT_SPLIT: 6,
    }
    fingerprints = {
        split: {row["case_fingerprint"] for row in rows}
        for split, rows in validated["splits"].items()
    }
    assert fingerprints[TRAIN_SPLIT].isdisjoint(fingerprints[VALIDATION_SPLIT])
    assert fingerprints[TRAIN_SPLIT].isdisjoint(fingerprints[HOLDOUT_SPLIT])
    assert fingerprints[VALIDATION_SPLIT].isdisjoint(fingerprints[HOLDOUT_SPLIT])
    assert validated["internal_split_audit"]["status"] == "passed_zero_case_overlap"
    assert validated["training_authority"].startswith("none_pending_")


def test_curriculum_tamper_is_refused(curriculum):
    tampered = copy.deepcopy(curriculum)
    tampered["splits"][TRAIN_SPLIT][0]["case_fingerprint"] = "0" * 64

    with pytest.raises(
        StructuredSFTError,
        match="structured_sft_curriculum_replay_mismatch",
    ):
        validate_structured_sft_curriculum(tampered)


def test_trainer_rows_exclude_oracles_and_holdout(curriculum):
    rows = trainer_rows(curriculum, split=TRAIN_SPLIT)

    assert len(rows) == 12
    assert all(set(row) == {"messages", "tools", "_meta"} for row in rows)
    assert all("oracle" not in row for row in rows)
    assert all(row["_meta"]["loss_policy"]["mask_prompt"] is True for row in rows)
    with pytest.raises(
        StructuredSFTError,
        match="structured_sft_holdout_export_forbidden",
    ):
        trainer_rows(curriculum, split=HOLDOUT_SPLIT)


def test_trainer_tokenization_proves_exact_masked_prefix(curriculum):
    class FakeQwenTokenizer:
        @staticmethod
        def apply_chat_template(
            messages,
            *,
            tools,
            add_generation_prompt=False,
            return_dict=False,
        ):
            assert return_dict is False
            tokens = [1, len(tools)]
            role_tokens = {
                "system": 10,
                "user": 20,
                "assistant": 30,
                "tool": 40,
            }
            for message in messages:
                tokens.append(role_tokens[message["role"]])
                tokens.extend(str(message.get("content", "")).encode("utf-8"))
                if message.get("tool_calls"):
                    tokens.extend(
                        json.dumps(
                            message["tool_calls"],
                            sort_keys=True,
                        ).encode("utf-8")
                    )
                tokens.append(2)
            if add_generation_prompt:
                tokens.append(role_tokens["assistant"])
            return tokens

    report = validate_trainer_tokenization(
        curriculum,
        tokenizer=FakeQwenTokenizer(),
    )

    assert report["status"] == "passed_exact_masked_prefix"
    assert report["rows_checked"] == 18
    assert report["holdout_tokenized"] is False
    assert report["max_seq_length"] == 4096
    assert report["rows_with_truncation"] == 0
    assert set(report["groups"]) == {
        f"{family}:{target_kind}" for family, target_kind in TARGETS
    }
    assert all(
        group["min_target_tokens"] > 0
        for group in report["groups"].values()
    )
    assert len(report["report_sha256"]) == 64


def test_trainer_tokenization_refuses_rows_that_would_truncate() -> None:
    constrained = build_structured_sft_curriculum(
        StructuredSFTCurriculumSpec(
            seed=8129,
            train_cases_per_family=1,
            validation_cases_per_family=1,
            holdout_cases_per_family=1,
            max_seq_length=256,
        ),
        holdout_seed=HOLDOUT_SEED,
    )

    class TruncatingTokenizer:
        @staticmethod
        def apply_chat_template(
            messages,
            *,
            tools,
            add_generation_prompt=False,
            return_dict=False,
        ):
            del tools, return_dict
            size = 299 if add_generation_prompt else 300
            return [7] * size

    with pytest.raises(
        StructuredSFTError,
        match="structured_sft_sequence_would_truncate",
    ):
        validate_trainer_tokenization(
            constrained,
            tokenizer=TruncatingTokenizer(),
        )


@pytest.mark.parametrize("max_seq_length", (255, 65_537, True))
def test_curriculum_rejects_invalid_max_sequence_length(max_seq_length) -> None:
    with pytest.raises(
        StructuredSFTError,
        match="structured_sft_spec_max_seq_length_invalid",
    ):
        StructuredSFTCurriculumSpec(
            seed=1,
            max_seq_length=max_seq_length,
        )


def test_public_manifest_exposes_commitments_not_holdout_examples(curriculum):
    manifest = curriculum_manifest(curriculum)

    assert manifest["schema"] == structured.STRUCTURED_SFT_MANIFEST_SCHEMA
    assert "splits" not in manifest
    assert '"role":' not in json.dumps(manifest)
    assert "oracle" not in json.dumps(manifest)
    assert manifest["split_commitments"][HOLDOUT_SPLIT]["example_count"] == 6
    assert manifest["trainer_contract"]["holdout_visible_to_trainer"] is False
    assert len(manifest["manifest_sha256"]) == 64


def test_candidate_package_is_not_directly_trainer_loadable(curriculum):
    artifacts = build_candidate_dataset_artifacts(curriculum)
    manifest = validate_candidate_dataset_artifacts(artifacts)

    assert set(artifacts) == set(structured.STRUCTURED_SFT_CANDIDATE_FILES)
    assert "train.jsonl" not in artifacts
    assert "valid.jsonl" not in artifacts
    assert manifest["trainer_ready"] is False
    assert manifest["training_authority"].startswith("none_pending_")
    assert manifest["candidate_filenames"]["train"] == "candidate_train.jsonl"
    assert manifest["candidate_filenames"]["validation"] == "candidate_valid.jsonl"
    assert set(artifacts) == {
        "candidate_train.jsonl",
        "candidate_valid.jsonl",
        "manifest.json",
    }
    assert "sealed_holdout" not in manifest["candidate_filenames"]
    assert "private_curriculum" not in manifest["candidate_filenames"]
    assert manifest["validation_scope"] == "train_validation_replay_only"
    assert len(manifest["required_next_gates"]) == 10
    assert "external_replay_privacy_attestation" in manifest[
        "required_next_gates"
    ]
    assert "external_replay_sft_authority" in manifest["required_next_gates"]


def test_candidate_package_tamper_and_partial_sets_are_refused(curriculum):
    artifacts = build_candidate_dataset_artifacts(curriculum)
    tampered = dict(artifacts)
    tampered["candidate_train.jsonl"] += b'{"messages":[]}\n'

    with pytest.raises(
        StructuredSFTError,
        match="structured_sft_candidate_replay_mismatch",
    ):
        validate_candidate_dataset_artifacts(tampered)

    partial = dict(artifacts)
    partial.pop("manifest.json")
    with pytest.raises(
        StructuredSFTError,
        match="structured_sft_candidate_file_set_invalid",
    ):
        validate_candidate_dataset_artifacts(partial)


@pytest.mark.parametrize(
    "mutation",
    (
        "source_binding",
        "trainer_contract",
        "training_authority",
        "required_next_gates",
        "holdout_commitment",
    ),
)
def test_candidate_package_rejects_resigned_policy_tampering(
    curriculum,
    mutation,
):
    artifacts = build_candidate_dataset_artifacts(curriculum)
    manifest = json.loads(artifacts["manifest.json"])
    public = manifest["curriculum_manifest"]

    if mutation == "source_binding":
        public["source_binding"]["sha256"] = "0" * 64
    elif mutation == "trainer_contract":
        public["trainer_contract"]["mask_prompt"] = False
        manifest["trainer_contract"] = public["trainer_contract"]
    elif mutation == "training_authority":
        public["training_authority"] = "none_pending_untrusted_override"
        manifest["training_authority"] = public["training_authority"]
    elif mutation == "required_next_gates":
        manifest["required_next_gates"] = manifest["required_next_gates"][:-1]
    else:
        public["holdout_seed_commitment_sha256"] = "not-a-sha256"

    public_body = dict(public)
    public_body.pop("manifest_sha256")
    public["manifest_sha256"] = structured._sha256(public_body)
    manifest["custody_root_sha256"] = structured._sha256(
        {
            "domain": "AURA-SFT-CUSTODY-ROOT-v1",
            "curriculum_manifest": public,
            "candidate_artifacts": manifest["artifacts"],
            "holdout_seed_commitment_sha256": public[
                "holdout_seed_commitment_sha256"
            ],
        }
    )
    manifest_body = dict(manifest)
    manifest_body.pop("package_sha256")
    manifest["package_sha256"] = structured._sha256(manifest_body)
    tampered = {
        **artifacts,
        "manifest.json": structured.canonical_json_bytes(manifest),
    }

    with pytest.raises(StructuredSFTError):
        validate_candidate_dataset_artifacts(tampered)


@pytest.mark.parametrize("mutation", ("package_schema", "public_schema"))
def test_full_custody_rejects_resigned_unknown_schemas(mutation) -> None:
    bundles = build_structured_sft_custody_bundles(
        StructuredSFTCurriculumSpec(
            seed=31415,
            train_cases_per_family=1,
            validation_cases_per_family=1,
            holdout_cases_per_family=1,
        ),
        holdout_seed=HOLDOUT_SEED,
    )
    manifest = json.loads(bundles.candidate_artifacts["manifest.json"])
    public = manifest["curriculum_manifest"]
    if mutation == "package_schema":
        manifest["schema"] = "attacker.unknown.package"
    else:
        public["schema"] = "attacker.unknown.public_manifest"
        public_body = dict(public)
        public_body.pop("manifest_sha256")
        public["manifest_sha256"] = structured._sha256(public_body)
        manifest["custody_root_sha256"] = structured._sha256(
            {
                "domain": "AURA-SFT-CUSTODY-ROOT-v1",
                "curriculum_manifest": public,
                "candidate_artifacts": manifest["artifacts"],
                "holdout_seed_commitment_sha256": public[
                    "holdout_seed_commitment_sha256"
                ],
            }
        )
    manifest_body = dict(manifest)
    manifest_body.pop("package_sha256")
    manifest["package_sha256"] = structured._sha256(manifest_body)
    tampered_candidate = {
        **bundles.candidate_artifacts,
        "manifest.json": structured.canonical_json_bytes(manifest),
    }

    with pytest.raises(StructuredSFTError, match="schema"):
        validate_structured_sft_custody_pair(
            tampered_candidate,
            bundles.evaluator_artifacts,
        )


def test_custody_bundles_exclude_all_holdout_material_from_candidate() -> None:
    bundles = build_structured_sft_custody_bundles(
        StructuredSFTCurriculumSpec(
            seed=2718,
            train_cases_per_family=1,
            validation_cases_per_family=1,
            holdout_cases_per_family=1,
        ),
        holdout_seed=HOLDOUT_SEED,
    )
    report = validate_structured_sft_custody_pair(
        bundles.candidate_artifacts,
        bundles.evaluator_artifacts,
    )
    evaluator_holdout = json.loads(
        bundles.evaluator_artifacts["holdout.private.json"]
    )
    candidate_bytes = b"\n".join(
        bundles.candidate_artifacts.values()
    )

    assert set(bundles.evaluator_artifacts) == set(
        STRUCTURED_SFT_EVALUATOR_FILES
    )
    assert HOLDOUT_SEED not in candidate_bytes
    assert HOLDOUT_SEED.hex().encode() not in candidate_bytes
    assert all(
        row["example_id"].encode() not in candidate_bytes
        and row["case_fingerprint"].encode() not in candidate_bytes
        for row in evaluator_holdout["examples"]
    )
    assert report["status"] == "passed_artifact_noncontainment_shared_uid"
    assert report["access_isolation_enforced"] is False
    assert report["example_id_overlap_count"] == 0
    assert report["case_fingerprint_overlap_count"] == 0


def test_custody_pair_rejects_wrong_evaluator_bundle() -> None:
    first = build_structured_sft_custody_bundles(
        StructuredSFTCurriculumSpec(
            seed=314,
            train_cases_per_family=1,
            validation_cases_per_family=1,
            holdout_cases_per_family=1,
        ),
        holdout_seed=HOLDOUT_SEED,
    )
    second = build_structured_sft_custody_bundles(
        StructuredSFTCurriculumSpec(
            seed=315,
            train_cases_per_family=1,
            validation_cases_per_family=1,
            holdout_cases_per_family=1,
        ),
        holdout_seed=b"x" * 32,
    )

    with pytest.raises(StructuredSFTError):
        validate_structured_sft_custody_pair(
            first.candidate_artifacts,
            second.evaluator_artifacts,
        )


@pytest.mark.parametrize("bad_value", (1, None, "not-bytes"))
def test_custody_pair_rejects_non_bytes_evaluator_artifacts(
    bad_value,
) -> None:
    bundles = build_structured_sft_custody_bundles(
        StructuredSFTCurriculumSpec(
            seed=315,
            train_cases_per_family=1,
            validation_cases_per_family=1,
            holdout_cases_per_family=1,
        ),
        holdout_seed=HOLDOUT_SEED,
    )
    malformed = dict(bundles.evaluator_artifacts)
    malformed["holdout.private.json"] = bad_value

    with pytest.raises(
        StructuredSFTError,
        match="structured_sft_evaluator_file_invalid",
    ):
        validate_structured_sft_custody_pair(
            bundles.candidate_artifacts,
            malformed,
        )


def test_custody_pair_rejects_excessive_json_depth_stably() -> None:
    bundles = build_structured_sft_custody_bundles(
        StructuredSFTCurriculumSpec(
            seed=316,
            train_cases_per_family=1,
            validation_cases_per_family=1,
            holdout_cases_per_family=1,
        ),
        holdout_seed=HOLDOUT_SEED,
    )
    malformed = dict(bundles.evaluator_artifacts)
    malformed["holdout.private.json"] = (
        b"[" * 10_000 + b"0" + b"]" * 10_000
    )

    with pytest.raises(
        StructuredSFTError,
        match="structured_sft_evaluator_json_invalid",
    ):
        validate_structured_sft_custody_pair(
            bundles.candidate_artifacts,
            malformed,
        )


@pytest.mark.parametrize("holdout_seed", (b"", b"x" * 31, b"x" * 33))
def test_custody_rejects_invalid_holdout_secret_length(holdout_seed) -> None:
    with pytest.raises(
        StructuredSFTError,
        match="structured_sft_holdout_seed_invalid",
    ):
        build_structured_sft_custody_bundles(
            StructuredSFTCurriculumSpec(seed=316),
            holdout_seed=holdout_seed,
        )


def test_tool_schema_is_not_shared_mutable_state():
    first = generate_structured_sft_example(
        family=CODE_TOOL,
        target_kind=TOOL_CALL_TARGET,
        seed=127,
    )
    first["tools"][0]["function"]["name"] = "tampered"
    second = generate_structured_sft_example(
        family=CODE_TOOL,
        target_kind=TOOL_CALL_TARGET,
        seed=127,
    )

    assert second["tools"][0]["function"]["name"] == "code_repl"
    with pytest.raises(StructuredSFTError):
        validate_structured_sft_example(first)


@pytest.mark.parametrize(
    ("family", "target_kind", "seed"),
    (
        ("unknown", DERIVATION_TARGET, 1),
        (STRUCTURED_PROGRAM, TOOL_CALL_TARGET, 1),
        (CODE_TOOL, TOOL_CALL_TARGET, -1),
    ),
)
def test_invalid_example_coordinates_fail_closed(family, target_kind, seed):
    with pytest.raises(StructuredSFTError):
        generate_structured_sft_example(
            family=family,
            target_kind=target_kind,
            seed=seed,
        )
