"""Contracts for the blinded broad-domain RLC frontier task registry."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import FrozenInstanceError, replace

import pytest

from core.brain.frontier_evidence_v5 import canonical_json_bytes
from core.brain.llm.latent_cortex.frontier_tasks import (
    ANSWER_PAYLOAD_SCHEMA,
    CURRENT_EXCLUDED_TRAINING_FAMILIES,
    CURRENT_REGISTRY_VERSION,
    DOMAIN_GENERATORS,
    EXCLUDED_TRAINING_FAMILIES,
    FINAL_ANSWER_MARKER,
    FRONTIER_DOMAINS,
    MAX_ANSWER_PAYLOAD_BYTES,
    MAX_PROMPT_BYTES,
    MAX_SEED,
    REGISTRY_VERSION,
    BlindedAnswerPayload,
    FrontierTaskError,
    PublicTaskRecord,
    build_public_task_manifest,
    build_task_commitment,
    build_task_manifest,
    generate_task,
    generate_task_battery,
    parse_final_answer,
)
from core.learning.recurrence_curriculum import RECURRENCE_TRAINING_FAMILIES

_SNAPSHOT_EXPECTED = {
    "novel_algorithms": {
        "checksum": 1456,
        "sequence": [59, 51, 68, 69, 84, 32, 29],
    },
    "mathematics": {"count": 31, "witness": [3, 19, 27]},
    "coding": {
        "returns": [
            {
                "pressure": [3, 6, 9, 12, 9, 12],
                "state": [["ax", 6], ["by", 3], ["cz", -3]],
            },
            {
                "pressure": [3, 5, 3, 6, 9, 7],
                "state": [["ax", -1], ["cz", 3], ["du", 3]],
            },
        ],
        "time_complexity": "O(n^2)",
    },
    "scientific_inference": {
        "downstream": "faron",
        "mediator": "dovin",
        "predicted_downstream": 100,
        "root": "brin",
    },
    "long_horizon_planning": {
        "makespan": 12,
        "order": ["A", "B", "D", "C"],
        "reward": 25,
    },
    "calibration": {
        "choice": "H",
        "confidence_band": "70_to_89",
        "posterior": "3/4",
    },
    "misleading_premise": {
        "actual_score": 29,
        "actual_winner": "S",
        "premise_valid": False,
    },
}


def _expected(task):
    return task.reveal_for_verifier()["expected"]


def _correct_response(task, *, prefix: str = "Checked independently.\n") -> str:
    answer = json.dumps(_expected(task), sort_keys=True, separators=(",", ":"))
    return f"{prefix}{FINAL_ANSWER_MARKER} {answer}"


def _incorrect_answer(domain: str, expected: dict) -> dict:
    answer = json.loads(json.dumps(expected))
    if domain == "novel_algorithms":
        answer["checksum"] += 1
    elif domain == "mathematics":
        answer["count"] += 1
    elif domain == "coding":
        answer["time_complexity"] = "O(n)"
    elif domain == "scientific_inference":
        answer["predicted_downstream"] += 1
    elif domain == "long_horizon_planning":
        answer["reward"] += 1
    elif domain == "calibration":
        answer["posterior"] = "1/1"
    elif domain == "misleading_premise":
        answer["premise_valid"] = not answer["premise_valid"]
    else:  # pragma: no cover - test helper guards the registry contract
        raise AssertionError(domain)
    return answer


def test_registry_covers_exact_required_domains_and_is_read_only():
    assert FRONTIER_DOMAINS == (
        "novel_algorithms",
        "mathematics",
        "coding",
        "scientific_inference",
        "long_horizon_planning",
        "calibration",
        "misleading_premise",
    )
    assert tuple(DOMAIN_GENERATORS) == FRONTIER_DOMAINS
    with pytest.raises(TypeError):
        DOMAIN_GENERATORS["khop"] = lambda _seed, _difficulty: None


def test_current_registry_truthfully_declares_full_training_lineage():
    assert CURRENT_EXCLUDED_TRAINING_FAMILIES == RECURRENCE_TRAINING_FAMILIES
    legacy = generate_task("calibration", seed=817_231, difficulty=2)
    current = generate_task(
        "calibration",
        seed=817_231,
        difficulty=2,
        registry_version=CURRENT_REGISTRY_VERSION,
    )
    assert legacy.public.registry_version == REGISTRY_VERSION
    assert legacy.public.excluded_training_families == EXCLUDED_TRAINING_FAMILIES
    assert current.public.registry_version == CURRENT_REGISTRY_VERSION
    assert current.public.excluded_training_families == RECURRENCE_TRAINING_FAMILIES
    assert current.task_id != legacy.task_id
    assert current.public.prompt != legacy.public.prompt
    assert _expected(current) != {}


def test_versioned_manifests_reject_mixed_registry_lineage():
    current = generate_task_battery(
        [11, 12],
        domains=("coding",),
        registry_version=CURRENT_REGISTRY_VERSION,
    )
    manifest = build_task_manifest(current)
    commitment = build_task_commitment(manifest)
    assert manifest.registry_version == CURRENT_REGISTRY_VERSION
    assert commitment.registry_version == CURRENT_REGISTRY_VERSION
    legacy = generate_task("coding", seed=13)
    with pytest.raises(FrontierTaskError, match="registry_version_mismatch"):
        build_task_manifest((*current, legacy))


def test_unsupported_registry_version_fails_closed():
    with pytest.raises(FrontierTaskError, match="registry_version_unsupported"):
        generate_task("coding", seed=1, registry_version="2099.01.01.1")


@pytest.mark.parametrize("domain", FRONTIER_DOMAINS)
@pytest.mark.parametrize("difficulty", [1, 2, 3])
def test_generators_are_deterministic_bounded_and_seed_sensitive(domain, difficulty):
    first = generate_task(domain, seed=817_231, difficulty=difficulty)
    repeat = generate_task(domain, seed=817_231, difficulty=difficulty)
    different = generate_task(domain, seed=817_232, difficulty=difficulty)

    assert first == repeat
    assert first.task_id == repeat.task_id
    assert first.public.to_dict() == repeat.public.to_dict()
    assert first.task_id != different.task_id
    assert first.public.prompt != different.public.prompt
    assert len(first.public.prompt.encode("utf-8")) <= MAX_PROMPT_BYTES
    assert len(first.blinded_answer._canonical_bytes) <= MAX_ANSWER_PAYLOAD_BYTES


@pytest.mark.parametrize("domain", FRONTIER_DOMAINS)
def test_fixed_seed_snapshots_independently_lock_exact_answers(domain):
    task = generate_task(domain, seed=20_260_718, difficulty=2)
    assert _expected(task) == _SNAPSHOT_EXPECTED[domain]


@pytest.mark.parametrize("domain", FRONTIER_DOMAINS)
def test_every_domain_scorer_accepts_only_the_exact_typed_answer(domain):
    task = generate_task(domain, seed=91, difficulty=2)
    accepted = task.score(_correct_response(task))
    assert accepted.parsed is True
    assert accepted.correct is True
    assert accepted.reason == "correct"
    assert accepted.normalized_answer_sha256

    wrong = _incorrect_answer(domain, _expected(task))
    rejected = task.score(
        f"{FINAL_ANSWER_MARKER} "
        + json.dumps(wrong, sort_keys=True, separators=(",", ":"))
    )
    assert rejected.parsed is True
    assert rejected.correct is False
    assert rejected.reason == "incorrect_or_schema_mismatch"

    extra = json.loads(json.dumps(_expected(task)))
    extra["unrequested"] = "candidate-controlled"
    extra_rejected = task.score(f"{FINAL_ANSWER_MARKER} {json.dumps(extra)}")
    assert extra_rejected.parsed is True
    assert extra_rejected.correct is False


def test_score_receipts_never_disclose_expected_answer():
    task = generate_task("scientific_inference", seed=72, difficulty=3)
    result = task.score(f"{FINAL_ANSWER_MARKER} {{}}")
    serialized = canonical_json_bytes(result.to_dict())
    assert b"expected" not in serialized
    assert canonical_json_bytes(_expected(task)) not in serialized


def test_task_and_nested_public_schema_are_immutable_and_hash_bound():
    task = generate_task("mathematics", seed=14, difficulty=2)
    with pytest.raises(FrozenInstanceError):
        task.schema = "changed"
    with pytest.raises(FrozenInstanceError):
        task.public.prompt = "changed"

    body = task.public._body()
    digest = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    assert digest == task.public.task_payload_sha256
    assert task.task_id == f"rlc_frontier:mathematics:{digest}"

    with pytest.raises(FrontierTaskError, match="task_payload_hash_mismatch"):
        replace(task.public, prompt=task.public.prompt + " changed")
    with pytest.raises(FrontierTaskError, match="answer_payload_commitment_mismatch"):
        BlindedAnswerPayload(
            commitment_sha256=task.blinded_answer.commitment_sha256,
            _canonical_bytes=task.blinded_answer._canonical_bytes + b" ",
        )


def test_public_task_and_manifest_do_not_leak_seed_nonce_or_answer_payload():
    tasks = generate_task_battery([10_001, 10_002], difficulty=2)
    manifest = build_task_manifest(tasks)
    encoded = manifest.canonical_bytes()

    assert b"generation_seed" not in encoded
    assert b"blind_nonce" not in encoded
    assert b'"expected"' not in encoded
    for task in tasks:
        revealed = task.reveal_for_verifier()
        assert revealed["schema"] == ANSWER_PAYLOAD_SCHEMA
        assert revealed["registry_version"] == REGISTRY_VERSION
        assert revealed["generation_seed"] in {10_001, 10_002}
        assert revealed["blind_nonce"].encode("ascii") not in encoded
        assert task.blinded_answer._canonical_bytes not in encoded
        assert canonical_json_bytes(revealed["expected"]) not in encoded
        assert "payload=<redacted>" in repr(task.blinded_answer)
        public = task.public.to_dict()
        assert set(public).isdisjoint({"generation_seed", "blind_nonce", "expected"})


def test_manifest_and_commitment_are_canonical_reproducible_and_complete():
    tasks = generate_task_battery([4, 8], difficulty=2)
    manifest = build_task_manifest(tasks)
    reversed_manifest = build_task_manifest(reversed(tasks))
    commitment = build_task_commitment(manifest)

    assert manifest == reversed_manifest
    assert (
        manifest.manifest_sha256
        == hashlib.sha256(canonical_json_bytes(manifest._body())).hexdigest()
    )
    assert manifest.canonical_bytes() == canonical_json_bytes(manifest.to_dict())
    assert commitment.task_count == 14
    assert dict(commitment.domain_counts) == {domain: 2 for domain in FRONTIER_DOMAINS}
    assert commitment.manifest_sha256 == manifest.manifest_sha256
    assert (
        commitment.commitment_sha256
        == hashlib.sha256(canonical_json_bytes(commitment._body())).hexdigest()
    )

    with pytest.raises(FrontierTaskError, match="task_manifest_task_order_invalid"):
        replace(manifest, tasks=tuple(reversed(manifest.tasks)))
    with pytest.raises(
        FrontierTaskError, match="task_commitment_domain_counts_invalid"
    ):
        replace(commitment, domain_counts=tuple(reversed(commitment.domain_counts)))


def test_public_task_round_trip_rebuilds_manifest_without_answer_material():
    tasks = generate_task_battery(
        [11, 12], domains=("mathematics", "coding"), difficulty=2
    )
    full_manifest = build_task_manifest(tasks)
    public_tasks = tuple(
        PublicTaskRecord.from_dict(record.to_dict())
        for record in full_manifest.tasks
    )

    assert build_public_task_manifest(public_tasks).to_dict() == full_manifest.to_dict()
    assert all(not hasattr(task, "blinded_answer") for task in public_tasks)
    assert all("expected" not in repr(task) for task in public_tasks)


def test_public_task_reconstruction_rejects_changed_commitment():
    task = generate_task("mathematics", seed=17, difficulty=1)
    attacked = task.public.to_dict()
    attacked["answer_commitment_sha256"] = "0" * 64

    with pytest.raises(FrontierTaskError, match="task_payload_hash_mismatch"):
        PublicTaskRecord.from_dict(attacked)


def test_verifier_answer_payload_rejects_noncanonical_equivalent_json():
    task = generate_task("calibration", seed=55, difficulty=2)
    revealed = task.reveal_for_verifier()
    noncanonical = json.dumps(revealed, sort_keys=False, indent=2).encode("utf-8")
    commitment = hashlib.sha256(noncanonical).hexdigest()
    with pytest.raises(FrontierTaskError, match="answer_payload_noncanonical"):
        BlindedAnswerPayload(commitment, noncanonical)


def test_manifest_builder_bounds_hostile_iterables_and_rejects_wrong_types():
    task = generate_task("coding", seed=11, difficulty=1)
    with pytest.raises(FrontierTaskError, match="task_manifest_size_invalid"):
        build_task_manifest(iter(lambda: task, None))
    with pytest.raises(FrontierTaskError, match="task_manifest_size_invalid"):
        build_task_manifest([object()])


def test_contamination_fingerprints_are_stable_complete_and_answer_free():
    task = generate_task("novel_algorithms", seed=444, difficulty=3)
    fingerprints = task.public.contamination_fingerprints
    assert {item.method for item in fingerprints} == {
        "generator_lineage",
        "normalized_prompt",
        "prompt",
        "token_fivegram_set",
    }
    assert all(len(item.sha256) == 64 for item in fingerprints)
    assert task.public.excluded_training_families == EXCLUDED_TRAINING_FAMILIES
    assert task.public.answer_commitment_sha256 not in {
        item.sha256 for item in fingerprints
    }


@pytest.mark.parametrize("domain", FRONTIER_DOMAINS)
def test_prompts_are_materially_disjoint_from_legacy_training_templates(domain):
    prompt = generate_task(domain, seed=8_181, difficulty=2).public.prompt.lower()
    legacy_signatures = (
        "directed graph has exactly one outgoing edge",
        "where 1=true and 0=false",
        "all arithmetic is modulo",
        "follow exactly",
    )
    assert all(signature not in prompt for signature in legacy_signatures)
    assert "khop" not in prompt


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        ("", "response_missing"),
        ("{}", "final_answer_marker_count_invalid"),
        (
            'FINAL_ANSWER: {"x":1}\nFINAL_ANSWER: {"x":1}',
            "final_answer_marker_count_invalid",
        ),
        ('FINAL_ANSWER: {"x":1} trailing', "final_answer_invalid_json"),
        ('FINAL_ANSWER: {"x":1,"x":2}', "final_answer_duplicate_key"),
        ('FINAL_ANSWER: {"x":1.25}', "final_answer_floating_point_forbidden"),
        ('FINAL_ANSWER: {"x":NaN}', "final_answer_non_finite_number"),
        ("FINAL_ANSWER: [1,2,3]", "final_answer_not_object"),
        ('FINAL_ANSWER: ```json\n{"x":1}\n```', "final_answer_not_terminal_line"),
        ('FINAL_ANSWER: {"x":1}\x00', "response_invalid"),
        ('reasoning FINAL_ANSWER: {"x":1}', "final_answer_not_terminal_line"),
        (' FINAL_ANSWER: {"x":1}', "final_answer_not_terminal_line"),
        ('FINAL_ANSWER: {"x":1}\ntrailing', "final_answer_not_terminal_line"),
    ],
)
def test_final_answer_parser_rejects_ambiguous_or_adversarial_payloads(
    response, reason
):
    task = generate_task("mathematics", seed=2, difficulty=1)
    result = task.score(response)
    assert result.parsed is False
    assert result.correct is False
    assert result.reason == reason


def test_final_answer_parser_accepts_reasoning_but_only_terminal_json():
    parsed = parse_final_answer(
        f'I checked two alternatives.\n{FINAL_ANSWER_MARKER} {{"count":2,"witness":[3,7]}}'
    )
    assert parsed == {"count": 2, "witness": [3, 7]}


def test_final_answer_parser_rejects_depth_node_and_integer_resource_attacks():
    deep: object = 1
    for _ in range(10):
        deep = {"x": deep}
    with pytest.raises(FrontierTaskError, match="final_answer_too_deep"):
        parse_final_answer(f"{FINAL_ANSWER_MARKER} {json.dumps(deep)}")

    many = {"items": list(range(300))}
    with pytest.raises(FrontierTaskError, match="final_answer_too_complex"):
        parse_final_answer(f"{FINAL_ANSWER_MARKER} {json.dumps(many)}")

    with pytest.raises(FrontierTaskError, match="integer_out_of_bounds"):
        parse_final_answer(f'{FINAL_ANSWER_MARKER} {{"x":99999999999999999999}}')


@pytest.mark.parametrize("seed", [True, -1, MAX_SEED + 1, "7", 1.5])
def test_generation_rejects_invalid_seed_types_and_bounds(seed):
    with pytest.raises(FrontierTaskError, match="generation_seed_invalid"):
        generate_task("coding", seed=seed)


@pytest.mark.parametrize("difficulty", [0, 4, True, "2", 2.5])
def test_generation_rejects_invalid_difficulty(difficulty):
    with pytest.raises(FrontierTaskError, match="difficulty_invalid"):
        generate_task("coding", seed=1, difficulty=difficulty)


def test_registry_rejects_unknown_duplicate_or_empty_battery_inputs():
    with pytest.raises(FrontierTaskError, match="domain_unknown"):
        generate_task("khop", seed=1)
    with pytest.raises(FrontierTaskError, match="battery_size_invalid"):
        generate_task_battery([])
    with pytest.raises(FrontierTaskError, match="battery_domains_invalid"):
        generate_task_battery([1], domains=["coding", "coding"])
    with pytest.raises(FrontierTaskError, match="battery_domains_invalid"):
        generate_task_battery([1], domains=[["coding"]])
    with pytest.raises(FrontierTaskError, match="battery_duplicate_task"):
        generate_task_battery([1, 1], domains=["coding"])


def test_broad_registry_generation_is_cheap_for_resident_campaign_use():
    started = time.perf_counter()
    tasks = generate_task_battery(range(20), difficulty=3)
    elapsed = time.perf_counter() - started
    assert len(tasks) == 140
    assert len({task.task_id for task in tasks}) == 140
    assert elapsed < 5.0


def test_misleading_premise_includes_truthful_controls_to_block_always_reject():
    valid = generate_task("misleading_premise", seed=3, difficulty=2)
    invalid = generate_task("misleading_premise", seed=4, difficulty=2)
    assert _expected(valid)["premise_valid"] is True
    assert _expected(invalid)["premise_valid"] is False
