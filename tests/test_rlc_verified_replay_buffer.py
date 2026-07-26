from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from core.brain.latent_cortex_service import LatentCortexService
from core.brain.llm.latent_cortex.answer_replacement import (
    build_answer_replacement_receipt,
)
from core.brain.llm.latent_cortex.atomic_decomposition import (
    build_atomic_decomposition,
)
from core.brain.llm.latent_cortex.diagnostic_action_selector import (
    build_candidate_routes,
    build_diagnostic_action_selector_receipt,
)
from core.brain.llm.latent_cortex.epistemic_state import OperationKind
from core.brain.llm.latent_cortex.local_repair import (
    build_local_repair_receipt,
    prepare_local_repair_requests,
)
from core.brain.llm.latent_cortex.output_quality import evaluate_latent_output
from core.brain.llm.latent_cortex.value_of_computation import (
    build_evidence_snapshot,
)
from core.brain.llm.latent_cortex.verified_replay_buffer import (
    ReplayEncryptionUnavailableError,
    ReplayStoreCorruptError,
    VerifiedReplayBuffer,
    build_verified_replay_entry,
    extract_verified_replay_payload,
    materialize_verified_replay_entry,
    validate_verified_replay_entry,
    validate_verified_replay_payload,
    validate_verified_replay_receipt,
    validate_verified_replay_store,
)


def _canonical(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha(value) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _text_sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _encode(value: str) -> list[int]:
    return list(value.encode("utf-8"))


def _decode(tokens) -> str:
    return bytes(tokens).decode("utf-8")


class _Protector:
    encryption_active = True
    key_provenance = "test_ephemeral"

    def __init__(self) -> None:
        self._cipher = AESGCM(b"v" * 32)

    def encrypt(self, data: bytes) -> bytes:
        nonce = os.urandom(12)
        return nonce + self._cipher.encrypt(nonce, data, None)

    def decrypt(self, blob: bytes) -> bytes:
        return self._cipher.decrypt(blob[:12], blob[12:], None)


class _InactiveProtector:
    encryption_active = False
    key_provenance = "none"

    @staticmethod
    def encrypt(_data: bytes) -> bytes:
        raise AssertionError("inactive protector must not be called")

    @staticmethod
    def decrypt(_data: bytes) -> bytes:
        raise AssertionError("inactive protector must not be called")


def _verified_episode() -> tuple[dict, dict, str, str, list[int], dict]:
    objective = "Return every arithmetic statement with its exactly correct result."
    failed = "2 + 2 = 5. 3 + 3 = 6. 4 + 4 = 8."
    corrected = "2 + 2 = 4. 3 + 3 = 6. 4 + 4 = 8."
    candidates = {0: failed, 1: corrected}
    decompositions = {
        str(index): build_atomic_decomposition(text, objective=objective)
        for index, text in candidates.items()
    }
    graph_payload = {
        "n_branches": 2,
        "candidate_decompositions": decompositions,
        "branches": [
            {
                "index": index,
                "operator_transition_count": 1,
                "operator_program_sha256": _text_sha(f"program-{index}"),
                "candidate_decomposition_sha256": decompositions[str(index)]["receipt_sha256"],
            }
            for index in range(2)
        ],
        "pairwise": [
            {
                "left": 0,
                "right": 1,
                "localized": True,
                "causal_divergence": {
                    "available": True,
                    "kind": "causal_transition",
                    "action_step": 1,
                },
                "candidate_divergence": {
                    "available": True,
                    "kind": "atomic_claim",
                    "atom_ordinal": 0,
                    "left": {
                        "atom_id": "a000",
                        "text_sha256": decompositions["0"]["atoms"][0]["text_sha256"],
                    },
                    "right": {
                        "atom_id": "a000",
                        "text_sha256": decompositions["1"]["atoms"][0]["text_sha256"],
                    },
                },
            }
        ],
    }
    graph = {**graph_payload, "receipt_sha256": _text_sha("verified-replay-graph")}
    routes = build_candidate_routes(
        candidates,
        objective=objective,
        candidate_decompositions=decompositions,
    )
    snapshot = build_evidence_snapshot(bucket="verified-replay", cells={})
    selector = build_diagnostic_action_selector_receipt(
        disagreement_graph=graph,
        candidate_routes=routes,
        action_policy_evidence=snapshot,
        value_policy={
            "snapshot_sha256": snapshot["snapshot_sha256"],
            "executors": [
                OperationKind.CHECK_ASSUMPTION.value,
                OperationKind.REGENERATE_FROM_PREFIX.value,
            ],
        },
        action_trace=[
            {
                "state_signal": {
                    "has_memory": False,
                    "has_evidence": False,
                    "has_verifier": True,
                    "has_savepoint": True,
                }
            }
        ],
    )
    request = prepare_local_repair_requests(
        disagreement_graph=graph,
        diagnostic_selection=selector,
        branch_candidates=candidates,
        objective=objective,
    )[0]
    generated = {
        request["request_id"]: {
            "candidate": corrected,
            "generation_context": {
                "prompt_sha256": request["prompt_sha256"],
                "generated_token_count": 32,
                "termination": "contract_complete",
                "initial_cache_offsets": [0, 0],
                "final_cache_offsets": [32, 32],
                "all_initial_offsets_zero": True,
                "solver_context_imported": False,
                "parameter_relation": "shared_resident_checkpoint",
            },
        }
    }
    local_repair = build_local_repair_receipt(
        disagreement_graph=graph,
        diagnostic_selection=selector,
        branch_candidates=candidates,
        objective=objective,
        generated_repairs=generated,
    )
    tokens = _encode(corrected)
    replacement, accepted, private = build_answer_replacement_receipt(
        disagreement_graph=graph,
        diagnostic_selection=selector,
        local_repair=local_repair,
        selected_branch=0,
        branch_candidates=candidates,
        generated_repairs=generated,
        objective=objective,
        baseline_text=failed,
        baseline_tokens=_encode(failed),
        encode=_encode,
        decode=_decode,
        enabled=True,
        margin=0.05,
        max_output_tokens=128,
    )
    assert replacement["decision"] == "replace"
    assert accepted == tokens
    quality = evaluate_latent_output(
        corrected,
        generated_tokens=len(tokens),
        termination="eos",
        objective=objective,
    )
    assert quality["passed"] is True, quality["reasons"]
    receipt = {
        "episode_id": "episode-verified-replay-1",
        "input_tokens_sha256": _text_sha("input-tokens"),
        "checkpoint_fingerprint": _text_sha("checkpoint"),
        "checkpoint_fingerprint_method": "sha256",
        "checkpoint_file_count": 7,
        "worker_identity": {
            "model": "Qwen2.5-32B",
            "worker_pid": 123,
        },
        "selected_branch": 0,
        "disagreement_graph": graph,
        "diagnostic_action_selection": selector,
        "local_repair": local_repair,
        "answer_replacement": replacement,
    }
    return receipt, private, objective, corrected, tokens, quality


def _payload() -> dict:
    receipt, private, objective, corrected, tokens, quality = _verified_episode()
    return extract_verified_replay_payload(
        receipt=receipt,
        private_evidence=private,
        objective=objective,
        output_text=corrected,
        output_tokens=tokens,
        output_quality=quality,
    )


def _distinct_payload(base: dict, index: int) -> dict:
    payload = copy.deepcopy(base)
    payload["provenance"]["episode_id"] = f"episode-verified-replay-{index}"
    return validate_verified_replay_payload(payload)


def test_extracts_complete_causal_repair_from_applied_answer():
    payload = _payload()

    assert payload["initial_failure"]["failed_atom"]["text"] == "2 + 2 = 5."
    assert payload["task_context"]["objective"].startswith("Return every arithmetic statement")
    assert payload["earliest_causal_error"]["atom_ordinal"] == 0
    assert payload["earliest_causal_error"]["earlier_exact_refutation_count"] == 0
    assert payload["discriminating_test"]["original_outcome"] == "refuted"
    assert payload["discriminating_test"]["corrected_outcome"] == "verified"
    assert payload["corrected_transition"]["corrected_atom"]["text"] == "2 + 2 = 4."
    assert payload["verified_solution"]["text"].startswith("2 + 2 = 4.")
    assert payload["error_class"] == "reasoning.exact_integer_arithmetic"
    assert payload["privacy_governance_disposition"]["export_allowed"] is False
    assert (
        payload["privacy_governance_disposition"]["training_authority"]
        == "none_pending_independent_transfer_validation"
    )


def test_unapplied_or_tampered_repair_cannot_enter_replay():
    receipt, private, objective, corrected, tokens, quality = _verified_episode()
    unapplied = copy.deepcopy(receipt)
    unapplied["answer_replacement"]["decision"] = "retain"
    with pytest.raises(ValueError):
        extract_verified_replay_payload(
            receipt=unapplied,
            private_evidence=private,
            objective=objective,
            output_text=corrected,
            output_tokens=tokens,
            output_quality=quality,
        )

    tampered = copy.deepcopy(private)
    tampered["generated_repairs"][receipt["answer_replacement"]["selected_request_id"]] = (
        "2 + 2 = 9."
    )
    with pytest.raises(ValueError):
        extract_verified_replay_payload(
            receipt=receipt,
            private_evidence=tampered,
            objective=objective,
            output_text=corrected,
            output_tokens=tokens,
            output_quality=quality,
        )


def test_payload_rejects_privacy_or_causal_claim_tamper():
    payload = _payload()
    privacy_tamper = copy.deepcopy(payload)
    privacy_tamper["privacy_governance_disposition"]["export_allowed"] = True
    with pytest.raises(ValueError, match="privacy disposition"):
        validate_verified_replay_payload(privacy_tamper)

    causal_tamper = copy.deepcopy(payload)
    causal_tamper["earliest_causal_error"]["earlier_exact_refutation_count"] = 1
    with pytest.raises(ValueError, match="discriminating test"):
        validate_verified_replay_payload(causal_tamper)


def test_entry_roundtrip_is_encrypted_and_commitment_bound():
    payload = _payload()
    protector = _Protector()
    entry = build_verified_replay_entry(
        payload,
        protector=protector,
        sequence=1,
        previous_entry_sha256="0" * 64,
        created_at_unix_ns=42,
    )

    rendered = _canonical(entry)
    assert b"2 + 2 = 5" not in rendered
    assert b"2 + 2 = 4" not in rendered
    assert materialize_verified_replay_entry(entry, protector=protector) == payload
    tampered = copy.deepcopy(entry)
    tampered["commitments"]["verified_solution_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="entry commitment"):
        validate_verified_replay_entry(tampered)


def test_inactive_encryption_fails_before_plaintext_write(tmp_path: Path):
    path = tmp_path / "replay.json"
    buffer = VerifiedReplayBuffer(path)

    with pytest.raises(ReplayEncryptionUnavailableError):
        buffer.append(_payload(), protector=_InactiveProtector())
    assert not path.exists()


def test_store_persists_deduplicates_and_materializes_without_plaintext(tmp_path: Path):
    path = tmp_path / "replay.json"
    buffer = VerifiedReplayBuffer(path)
    protector = _Protector()
    payload = _payload()

    first = buffer.append(payload, protector=protector, created_at_unix_ns=100)
    second = buffer.append(payload, protector=protector, created_at_unix_ns=101)

    assert first["status"] == "stored"
    assert second["status"] == "duplicate"
    assert first["entry_sha256"] == second["entry_sha256"]
    raw = path.read_bytes()
    assert b"2 + 2 = 5" not in raw
    assert b"2 + 2 = 4" not in raw
    store = buffer.load()
    assert store["entry_count"] == 1
    assert buffer.materialize(protector=protector) == [payload]


def test_retention_prunes_with_auditable_chain_anchor(tmp_path: Path):
    buffer = VerifiedReplayBuffer(
        tmp_path / "replay.json",
        max_entries=2,
    )
    protector = _Protector()
    base = _payload()

    for index in range(1, 4):
        buffer.append(
            _distinct_payload(base, index),
            protector=protector,
            created_at_unix_ns=100 + index,
        )

    store = buffer.load()
    assert store["entry_count"] == 2
    assert store["retired_count"] == 1
    assert store["retired_tail_entry_sha256"] != "0" * 64
    assert store["retired_accumulator_sha256"] != "0" * 64
    assert store["entries"][0]["previous_entry_sha256"] == store["retired_tail_entry_sha256"]
    assert [row["provenance"]["episode_id"] for row in buffer.materialize(protector=protector)] == [
        "episode-verified-replay-2",
        "episode-verified-replay-3",
    ]


def test_corrupt_store_refuses_overwrite_and_preserves_bytes(tmp_path: Path):
    path = tmp_path / "replay.json"
    original = b'{"not":"a replay store"}'
    path.write_bytes(original)
    buffer = VerifiedReplayBuffer(path)

    with pytest.raises(ReplayStoreCorruptError):
        buffer.append(_payload(), protector=_Protector())
    assert path.read_bytes() == original


def test_symlink_store_is_rejected_without_touching_target(tmp_path: Path):
    target = tmp_path / "target.json"
    target.write_text("owner-data", encoding="utf-8")
    link = tmp_path / "replay.json"
    link.symlink_to(target)
    buffer = VerifiedReplayBuffer(link)

    with pytest.raises(ReplayStoreCorruptError):
        buffer.append(_payload(), protector=_Protector())
    assert target.read_text(encoding="utf-8") == "owner-data"


def test_store_rejects_group_or_world_readable_private_state(tmp_path: Path):
    path = tmp_path / "replay.json"
    buffer = VerifiedReplayBuffer(path)
    protector = _Protector()
    buffer.append(_payload(), protector=protector)
    path.chmod(0o644)
    original = path.read_bytes()

    with pytest.raises(ReplayStoreCorruptError, match="owner-private"):
        buffer.load()
    with pytest.raises(ReplayStoreCorruptError, match="owner-private"):
        buffer.append(_distinct_payload(_payload(), 2), protector=protector)
    assert path.read_bytes() == original


def test_ciphertext_tamper_fails_authentication_even_if_public_hashes_are_resealed(
    tmp_path: Path,
):
    path = tmp_path / "replay.json"
    buffer = VerifiedReplayBuffer(path)
    protector = _Protector()
    buffer.append(_payload(), protector=protector, created_at_unix_ns=100)
    store = json.loads(path.read_text(encoding="utf-8"))
    entry = store["entries"][0]
    ciphertext = bytearray(base64.b64decode(entry["encryption"]["ciphertext_b64"], validate=True))
    ciphertext[-1] ^= 1
    entry["encryption"]["ciphertext_b64"] = base64.b64encode(ciphertext).decode("ascii")
    entry["encryption"]["ciphertext_sha256"] = hashlib.sha256(ciphertext).hexdigest()
    entry["entry_sha256"] = _sha(
        {key: value for key, value in entry.items() if key != "entry_sha256"}
    )
    store["head_entry_sha256"] = entry["entry_sha256"]
    store["store_sha256"] = _sha(
        {key: value for key, value in store.items() if key != "store_sha256"}
    )
    path.write_bytes(_canonical(store))
    validate_verified_replay_store(store)

    with pytest.raises(ReplayStoreCorruptError, match="authenticated"):
        buffer.materialize(protector=protector)


def test_concurrent_writers_serialize_without_loss(tmp_path: Path):
    path = tmp_path / "replay.json"
    protector = _Protector()
    base = _payload()

    def append(index: int) -> dict:
        return VerifiedReplayBuffer(path, max_entries=16).append(
            _distinct_payload(base, index),
            protector=protector,
            created_at_unix_ns=1_000 + index,
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        receipts = list(executor.map(append, range(1, 9)))

    assert all(receipt["status"] == "stored" for receipt in receipts)
    store = VerifiedReplayBuffer(path, max_entries=16).load()
    assert store["entry_count"] == 8
    assert [entry["sequence"] for entry in store["entries"]] == list(range(1, 9))
    assert len({entry["experience_sha256"] for entry in store["entries"]}) == 8


@pytest.mark.asyncio
async def test_service_capture_runs_persistence_off_event_loop(monkeypatch):
    import core.brain.llm.latent_cortex.verified_replay_buffer as replay_module

    receipt, private, objective, corrected, tokens, quality = _verified_episode()

    def slow_persist(**_kwargs):
        time.sleep(0.06)
        payload = {
            "schema": "aura.rlc.verified_replay_receipt.v1",
            "status": "stored",
            "experience_sha256": "1" * 64,
            "entry_sha256": "2" * 64,
            "sequence": 1,
            "store_revision": 1,
            "store_sha256": "3" * 64,
            "entry_count": 1,
            "retired_count": 0,
            "persistence_transaction_id": "transaction-1",
        }
        return {**payload, "receipt_sha256": _sha(payload)}

    monkeypatch.setattr(replay_module, "persist_runtime_verified_replay", slow_persist)
    ticks = 0
    stop = False

    async def heartbeat() -> None:
        nonlocal ticks
        while not stop:
            ticks += 1
            await asyncio.sleep(0.005)

    pulse = asyncio.create_task(heartbeat())
    result = await LatentCortexService._capture_verified_replay(
        receipt=receipt,
        private_evidence=private,
        objective=objective,
        output_text=corrected,
        output_tokens=tokens,
        output_quality=quality,
    )
    stop = True
    await pulse

    assert result["status"] == "stored"
    assert result["learning_effect"] == "encrypted_verified_experience_retained"
    assert ticks >= 5


def test_public_persistence_receipt_rejects_resealed_status_lie(tmp_path: Path):
    receipt = VerifiedReplayBuffer(tmp_path / "replay.json").append(
        _payload(),
        protector=_Protector(),
    )
    tampered = copy.deepcopy(receipt)
    tampered["status"] = "duplicate"
    tampered["receipt_sha256"] = _sha(
        {key: value for key, value in tampered.items() if key != "receipt_sha256"}
    )

    with pytest.raises(ValueError, match="persistence transaction"):
        validate_verified_replay_receipt(tampered)


@pytest.mark.asyncio
async def test_service_discloses_nonapplicable_and_failed_persistence(monkeypatch):
    receipt, private, objective, corrected, tokens, quality = _verified_episode()
    no_repair = copy.deepcopy(receipt)
    no_repair["answer_replacement"]["decision"] = "retain"
    result = await LatentCortexService._capture_verified_replay(
        receipt=no_repair,
        private_evidence=private,
        objective=objective,
        output_text=corrected,
        output_tokens=tokens,
        output_quality=quality,
    )
    assert result == {
        "schema": "aura.rlc.verified_replay_host.v1",
        "status": "not_applicable",
        "reason": "no_applied_verified_local_repair",
        "learning_effect": "none",
    }

    import core.brain.llm.latent_cortex.verified_replay_buffer as replay_module

    monkeypatch.setattr(
        replay_module,
        "persist_runtime_verified_replay",
        lambda **_kwargs: (_ for _ in ()).throw(ReplayEncryptionUnavailableError("missing key")),
    )
    failed = await LatentCortexService._capture_verified_replay(
        receipt=receipt,
        private_evidence=private,
        objective=objective,
        output_text=corrected,
        output_tokens=tokens,
        output_quality=quality,
    )
    assert failed["status"] == "not_persisted"
    assert failed["reason"] == "ReplayEncryptionUnavailableError"
    assert failed["learning_effect"] == "none"
