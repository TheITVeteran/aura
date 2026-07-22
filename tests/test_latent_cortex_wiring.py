"""Contract tests: latent-cortex runtime wiring (service, handler, economy).

No worker processes are spawned here — the worker/client IPC bodies are
exercised through the handler function and a mocked client, which is exactly
the seam the live path uses.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import queue
from types import SimpleNamespace

import pytest

from core.brain.latent_cortex_service import LatentCortexService
from core.brain.llm.latent_cortex.runtime_identity import (
    latent_request_payload_sha256,
)
from core.brain.llm.latent_cortex.worker_handler import (
    budget_from_job,
    config_from_job,
    cortex_enabled,
    handle_latent_reason,
)
from core.brain.llm.mlx_client import MLXLocalClient


class _ResidentProcess:
    def __init__(self) -> None:
        self.alive = True

    def is_alive(self) -> bool:
        return self.alive

    def kill(self) -> None:
        self.alive = False

    def join(self, timeout=None) -> None:
        self.alive = False


_WORKER_IDENTITY = {
    "schema": "aura.latent_cortex.worker_identity.v1",
    "worker_boot_id": "1" * 32,
    "worker_pid": 4242,
    "worker_model_path": "/models/test-32b",
    "worker_model_parameter_count": 32_000_000_000,
    "worker_model_stored_parameter_element_count": 5_000_000_000,
    "worker_model_parameter_count_basis": "architecture_config_logical",
    "worker_source_sha256": "2" * 64,
    "worker_affective_steering_active": True,
    "worker_affective_steering_alpha": 0.30,
}

_RUNTIME_IDENTITY = {
    "schema": "aura.latent_cortex.runtime_identity.v1",
    "identity_bound": True,
    "launch_mode": "direct",
    "installed_app_required": False,
    "installed_app_verified": False,
    "source_verified": True,
    "source_commit": "3" * 40,
    "workspace_state_sha256": "4" * 64,
    "shell_assets_sha256": "5" * 64,
    "issues": [],
}


def _identity_receipt(**overrides):
    receipt = {
        **_WORKER_IDENTITY,
        "request_payload_sha256": "6" * 64,
        "input_tokens_sha256": "7" * 64,
        "input_token_count": 64,
        "episode_affective_steering_applied": True,
        "episode_affective_steering_alpha": 0.30,
        "runtime_identity": dict(_RUNTIME_IDENTITY),
    }
    receipt.update(overrides)
    return receipt


def _identity_receipt_for_request(request, **overrides):
    receipt = _identity_receipt(
        request_payload_sha256=latent_request_payload_sha256(
            prompt=request.get("prompt"),
            messages=request.get("messages"),
            domain=request.get("domain", "general"),
            config=request.get("config"),
            budget=request.get("budget"),
            runtime_controls=request.get("runtime_controls"),
            cognitive_context=request.get("cognitive_context"),
            operation_authority=request.get("operation_authority"),
            action_policy_evidence=request.get("action_policy_evidence"),
            response_contract=request.get("response_contract"),
        )
    )
    receipt.update(overrides)
    return receipt


def _branch_isolation_fields(config, *, exchanges=0):
    count = config["n_branches"]
    required = config["isolation_steps"]

    def digest(label):
        return hashlib.sha256(label.encode("utf-8")).hexdigest()

    return {
        "exchanges": exchanges,
        "branch_isolation": {
            "schema": "aura.rlc.branch_isolation.v1",
            "n_branches": count,
            "required_steps": required,
            "sealed": True,
            "certified": True,
            "reason": "certified",
            "configured_role_lesion": False,
            "seed_alias_free": True,
            "seed_states_unique": True,
            "rng_streams_unique": True,
            "cross_exposure_started": exchanges > 0,
            "first_exchange_step": required if exchanges else None,
            "blocked_cross_exposures": 0,
            "candidates": [
                {
                    "index": index,
                    "role": f"role-{index}",
                    "context_sha256": digest("shared-context"),
                    "rng_stream_sha256": digest(f"rng-{index}"),
                    "seed_sha256": digest(f"seed-{index}"),
                    "candidate_sha256": digest(f"candidate-{index}"),
                    "candidate_step": required,
                }
                for index in range(count)
            ],
            "cache_discipline": {
                "schema": "aura.rlc.cache_discipline.v1",
                "nonpersistent_calls": count + required,
                "restored_calls": count + required,
                "restore_failures": 0,
                "all_restored": True,
            },
        },
    }


def _bind_test_client_identity(monkeypatch, client):
    from core.brain.llm.latent_cortex import runtime_identity

    client._worker_identity = dict(_WORKER_IDENTITY)
    monkeypatch.setattr(
        runtime_identity,
        "collect_latent_runtime_identity",
        lambda *_args, **_kwargs: dict(_RUNTIME_IDENTITY),
    )

# ── Worker handler ──────────────────────────────────────────────────────


def test_config_from_job_defaults_are_conservative():
    cfg = config_from_job(None)
    assert cfg.workspace.n_slots == 16
    assert cfg.recurrence.max_steps == 8
    assert cfg.branches.n_branches == 2
    assert cfg.branches.isolation_steps == 2
    assert cfg.latent_opt.enabled is False
    assert cfg.fast_weights.enabled is False
    assert cfg.verifier_probe_max_tokens == 48
    assert cfg.verifier_accept_non_regression is False
    assert cfg.validate() == []


def test_config_from_job_rejects_out_of_band_requests():
    with pytest.raises(ValueError):
        config_from_job({"n_branches": 640})
    with pytest.raises(ValueError):
        config_from_job({"max_steps": 100000})
    with pytest.raises(ValueError, match="JSON boolean"):
        config_from_job({"fast_weights": "false"})
    with pytest.raises(ValueError, match="unknown keys"):
        config_from_job({"fast_weight": True})
    with pytest.raises(ValueError):
        config_from_job({"exchange_interval": 0})
    with pytest.raises(ValueError):
        config_from_job({"isolation_steps": 9, "max_steps": 8})
    with pytest.raises(ValueError):
        config_from_job({"decode_temperature": float("nan")})
    with pytest.raises(ValueError):
        config_from_job({"verifier_probe_max_tokens": 15})
    with pytest.raises(ValueError, match="JSON boolean"):
        config_from_job({"verifier_accept_non_regression": "true"})


def test_config_from_job_maps_every_advanced_mechanism():
    cfg = config_from_job(
        {
            "latent_opt": True,
            "latent_opt_steps": 6,
            "latent_opt_lr": 0.02,
            "fast_weights": True,
            "fast_weights_opt_steps": 3,
            "fast_weights_lr": 0.005,
            "fast_weights_max_layers": 4,
            "fast_weights_canary_max_delta_rms": 0.025,
            "exchange_gamma": 0.2,
            "convergence_eps": 0.01,
            "decode_top_p": 0.82,
            "verifier_probe_max_tokens": 24,
            "verifier_accept_non_regression": True,
            "input_context_max_chars": 4096,
            "allow_vanilla_fallback": False,
        }
    )
    assert cfg.latent_opt.enabled is True and cfg.latent_opt.steps == 6
    assert cfg.latent_opt.lr == 0.02
    assert cfg.fast_weights.enabled is True and cfg.fast_weights.opt_steps == 3
    assert cfg.fast_weights.lr == 0.005 and cfg.fast_weights.max_wrapped_layers == 4
    assert cfg.fast_weights.canary_max_effective_delta_rms == 0.025
    assert cfg.branches.exchange_gamma == 0.2
    assert cfg.recurrence.convergence_eps == 0.01
    assert cfg.decode_top_p == 0.82
    assert cfg.verifier_probe_max_tokens == 24
    assert cfg.verifier_accept_non_regression is True
    assert cfg.input_context_max_chars == 4096
    assert cfg.allow_vanilla_fallback is False


def test_budget_from_job_caps_apply():
    budget = budget_from_job({"max_layer_apps": 10**15, "wall_clock_s": 5.0})
    assert budget.wall_clock_s == 5.0
    assert budget.max_layer_apps == 500_000_000
    assert budget.remaining_layer_apps == 500_000_000


@pytest.mark.parametrize(
    "payload",
    [
        {"max_layer_apps": -1},
        {"wall_clock_s": 0},
        {"wall_clock_s": float("inf")},
        {"max_layer_apps": "1000"},
        {"typo": 1},
    ],
)
def test_budget_from_job_rejects_invalid_values(payload):
    with pytest.raises((TypeError, ValueError)):
        budget_from_job(payload)


def test_kill_switch_refuses_honestly(monkeypatch):
    monkeypatch.setenv("AURA_LATENT_CORTEX", "0")
    assert cortex_enabled() is False
    body = handle_latent_reason(
        {"prompt": "hi"}, model=None, tokenizer=None, model_path=""
    )
    assert body["status"] == "error"
    assert "latent_cortex_disabled" in body["message"]


def test_handler_requires_prompt(monkeypatch):
    monkeypatch.delenv("AURA_LATENT_CORTEX", raising=False)
    body = handle_latent_reason({}, model=None, tokenizer=None, model_path="")
    assert body["status"] == "error"
    assert "requires prompt" in body["message"]


def test_handler_rejects_malformed_response_contract_before_engine(monkeypatch):
    monkeypatch.delenv("AURA_LATENT_CORTEX", raising=False)
    constructed = False

    class ForbiddenEngine:
        def __init__(self, *args, **kwargs):
            nonlocal constructed
            constructed = True

    import core.brain.llm.latent_cortex.worker_handler as handler_mod

    monkeypatch.setattr(handler_mod, "LatentCortexEngine", ForbiddenEngine)
    body = handle_latent_reason(
        {"prompt": "answer", "response_contract": '{"answer":not_a_type}'},
        model=object(),
        tokenizer=object(),
        model_path="/models/test-32b",
    )

    assert body["status"] == "error"
    assert "response_contract rejected" in body["message"]
    assert constructed is False


def test_handler_wires_response_contract_into_config_and_verifier(monkeypatch):
    from core.brain.llm.latent_cortex.types import EpisodeReceipt, LatentReasoningResult

    monkeypatch.delenv("AURA_LATENT_CORTEX", raising=False)
    captured: dict = {}

    class StubEngine:
        def __init__(self, model, tokenizer, config, **kwargs):
            captured["config"] = config

        def reason(self, **kwargs):
            captured.update(kwargs)
            return LatentReasoningResult(
                ok=True,
                text='FINAL_ANSWER: {"answer":7}',
                receipt=EpisodeReceipt(),
            )

    import core.brain.llm.latent_cortex.worker_handler as handler_mod

    monkeypatch.setattr(handler_mod, "LatentCortexEngine", StubEngine)
    body = handle_latent_reason(
        {
            "prompt": "answer with an integer",
            "response_contract": '{"answer":int}',
            "config": {"decode_max_tokens": 96},
        },
        model=object(),
        tokenizer=object(),
        model_path="/models/test-32b",
        worker_identity=dict(_WORKER_IDENTITY),
    )

    assert body["status"] == "ok"
    assert captured["config"].decode_contract == "final_answer_v1"
    assert captured["config"].decode_contract_grace_tokens == 96
    verifier = captured["verifier"]
    assert verifier is not None
    assert verifier.response_contract == '{"answer":int}'
    assert body["receipt"]["verifier_guidance"][
        "response_contract_required"
    ] is True


def test_handler_compacts_messages_but_hashes_the_original_request(monkeypatch):
    from core.brain.llm.latent_cortex.types import (
        EpisodeReceipt,
        LatentReasoningResult,
    )

    monkeypatch.delenv("AURA_LATENT_CORTEX", raising=False)
    captured: dict = {}

    class StubEngine:
        def __init__(self, model, tokenizer, config, **kwargs):
            captured["config"] = config

        def reason(self, **kwargs):
            captured.update(kwargs)
            return LatentReasoningResult(
                ok=True,
                text="bounded",
                receipt=EpisodeReceipt(),
            )

    import core.brain.llm.latent_cortex.worker_handler as handler_mod

    monkeypatch.setattr(handler_mod, "LatentCortexEngine", StubEngine)
    messages = [
        {"role": "system", "content": "system " + "s" * 4000},
        {"role": "user", "content": "question " + "u" * 4000},
    ]
    config = {"input_context_max_chars": 2048}
    job = {
        "messages": messages,
        "config": config,
        "domain": "unit",
    }
    body = handle_latent_reason(
        job,
        model=object(),
        tokenizer=object(),
        model_path="/models/test-32b",
        worker_identity=dict(_WORKER_IDENTITY),
    )

    compacted = captured["messages"]
    assert sum(len(item["content"]) for item in compacted) <= 2048
    receipt = body["receipt"]
    assert receipt["input_context_compaction"]["applied"] is True
    assert receipt["input_context_compaction"]["compacted_char_count"] <= 2048
    assert receipt["request_payload_sha256"] == latent_request_payload_sha256(
        prompt=None,
        messages=messages,
        domain="unit",
        config=config,
        budget=None,
        runtime_controls=None,
    )


def test_handler_runs_full_episode_on_tiny_model(monkeypatch, tmp_path):
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from mlx_lm.models.qwen2 import Model, ModelArgs

    monkeypatch.delenv("AURA_LATENT_CORTEX", raising=False)
    args = ModelArgs(
        model_type="qwen2", hidden_size=64, num_hidden_layers=8,
        intermediate_size=128, num_attention_heads=4, rms_norm_eps=1e-6,
        vocab_size=128, num_key_value_heads=2, max_position_embeddings=512,
        rope_theta=10000.0,
    )
    model = Model(args)
    mx.eval(model.parameters())

    class StubTokenizer:
        eos_token_id = 0

        def encode(self, text):
            return [ord(c) % 128 for c in text][:16]

        def decode(self, ids):
            return " ".join(str(i) for i in ids)

    body = handle_latent_reason(
        {
            "prompt": "compose the deepest thought",
            "config": {"n_slots": 4, "n_branches": 2, "max_steps": 4, "decode_max_tokens": 6},
            "budget": {"wall_clock_s": 30.0},
            "domain": "unit",
            "verifier_guidance": True,
        },
        model=model,
        tokenizer=StubTokenizer(),
        model_path="",
    )
    assert body["status"] == "ok", body
    assert body["receipt"]["params_unchanged"] is True
    assert body["receipt"]["steps_taken"] >= 2
    assert body["receipt"]["branch_isolation"]["certified"] is True
    policy = body["receipt"]["value_of_computation"]
    trace = body["receipt"]["cognitive_action_trace"]
    assert policy["active"] is True
    assert policy["actions_selected"] == len(trace) >= 2
    assert policy["actions_selected"] <= 4
    assert "execute" not in policy["executors"]
    assert all(row["decision"]["action"] in policy["executors"] for row in trace)
    operators = body["receipt"]["cognitive_operator_trace"]
    assert operators
    assert {row["operator"] for row in operators} == {
        "constructive_solution",
        "counterexample",
    }
    structure = body["receipt"]["structural_diversity"]
    assert structure["certified"] is True
    assert structure["independent_support_count"] == 2
    assert structure["wording_counted"] is False
    assert body["receipt"]["correlated_support"]["raw_support_count"] == 2
    assert body["receipt"]["blind_review"]["deranged_order"] is True
    assert body["receipt"]["blind_review"]["first_answer_designated"] is False
    from core.brain.llm.latent_cortex.blind_review import (
        validate_blind_review_receipt,
    )

    validate_blind_review_receipt(
        body["receipt"]["blind_review"],
        n_branches=2,
        branch_scores=body["receipt"]["branch_scores"],
        isolation_receipt=body["receipt"]["branch_isolation"],
        objective_sha256=body["receipt"]["input_tokens_sha256"],
    )
    contract_config = {
        "n_slots": 4,
        "n_branches": 2,
        "decode_max_tokens": 6,
    }
    assert "cognitive_operator_execution_unproven" not in (
        LatentCortexService._receipt_contract_errors(
            body["receipt"],
            contract_config,
        )
    )
    assert "blind_branch_review_unproven" not in (
        LatentCortexService._receipt_contract_errors(
            body["receipt"],
            contract_config,
        )
    )
    tampered = copy.deepcopy(body["receipt"])
    tampered["cognitive_operator_trace"][0]["receipt_sha256"] = "0" * 64
    assert "cognitive_operator_execution_unproven" in (
        LatentCortexService._receipt_contract_errors(tampered, contract_config)
    )
    tampered = copy.deepcopy(body["receipt"])
    tampered["structural_diversity"]["wording_counted"] = True
    assert "structural_diversity_unproven" in (
        LatentCortexService._receipt_contract_errors(tampered, contract_config)
    )
    tampered = copy.deepcopy(body["receipt"])
    tampered["correlated_support"]["effective_support_count"] = 1.0
    assert "correlated_support_unproven" in (
        LatentCortexService._receipt_contract_errors(tampered, contract_config)
    )
    tampered = copy.deepcopy(body["receipt"])
    tampered["blind_review"]["ownership_framing_supplied"] = True
    assert "blind_branch_review_unproven" in (
        LatentCortexService._receipt_contract_errors(tampered, contract_config)
    )
    assert body["requires_cache_clear"] is False


@pytest.mark.asyncio
async def test_client_latent_reason_owns_and_releases_resident_lane(monkeypatch):
    from core.brain.llm import mlx_client

    client = MLXLocalClient(model_path="/models/test-32b")
    client._process = _ResidentProcess()
    client._init_done = True
    client._req_q = queue.Queue()
    _bind_test_client_identity(monkeypatch, client)
    monkeypatch.setattr(
        mlx_client,
        "get_memory_pressure_snapshot",
        lambda: SimpleNamespace(refuse_heavy_local_generation=False),
    )

    task = asyncio.create_task(
        client.latent_reason_async(
            prompt="reason deeply",
            config={"decode_max_tokens": 16},
            response_contract='{"answer":int}',
            runtime_controls={
                "clean_user_surface_recurrent_loops": 2,
                "clean_user_surface_steering_alpha": 0.30,
            },
            timeout_s=5.0,
            foreground_request=False,
        )
    )
    request = await asyncio.to_thread(client._req_q.get, True, 2.0)
    assert client._request_lock.locked() is True
    assert client._active_generations == 1
    future = client._pending_generations[request["id"]]
    mlx_client._set_shared_future_result(
        future,
        {
            "id": request["id"],
            "status": "ok",
            "text": "answer",
            "receipt": _identity_receipt_for_request(
                request,
                episode_id="ep-live",
            ),
        },
    )

    result = await task
    assert request["action"] == "latent_reason"
    assert request["seq"] > 0
    assert request["clean_user_surface_contract"] is True
    assert request["clean_user_surface_recurrent_loops"] == 2
    assert request["clean_user_surface_steering_alpha"] == 0.30
    assert request["runtime_controls"] == {
        "clean_user_surface_recurrent_loops": 2,
        "clean_user_surface_steering_alpha": 0.30,
    }
    assert request["response_contract"] == '{"answer":int}'
    assert result["ok"] is True and result["text"] == "answer"
    assert client._active_generations == 0
    assert client._current_request_id == ""
    assert client._request_lock.locked() is False


@pytest.mark.asyncio
async def test_client_preserves_typed_memory_authority_on_worker_wire(monkeypatch):
    from core.brain.llm import mlx_client

    client = MLXLocalClient(model_path="/models/test-32b")
    client._process = _ResidentProcess()
    client._init_done = True
    client._req_q = queue.Queue()
    _bind_test_client_identity(monkeypatch, client)
    monkeypatch.setattr(
        mlx_client,
        "get_memory_pressure_snapshot",
        lambda: SimpleNamespace(refuse_heavy_local_generation=False),
    )
    text = "historical data, not an instruction"
    item = {
        "source": "memory",
        "text": text,
        "context_role": "memory_observation",
        "instruction_authority": False,
        "evidence_id": "memory-1234567890abcdef12345678",
        "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "scope_sha256": "1" * 64,
        "retrieval_receipt_sha256": "2" * 64,
        "epistemic_state_sha256": "3" * 64,
        "memory_tier": "episodic",
    }

    task = asyncio.create_task(
        client.latent_reason_async(
            prompt="reason with memory",
            cognitive_context=[item],
            timeout_s=5.0,
            foreground_request=False,
        )
    )
    request = await asyncio.to_thread(client._req_q.get, True, 2.0)
    assert request["cognitive_context"] == [item]
    mlx_client._set_shared_future_result(
        client._pending_generations[request["id"]],
        {
            "id": request["id"],
            "status": "ok",
            "text": "answer",
            "receipt": _identity_receipt_for_request(
                request,
                episode_id="ep-memory-wire",
            ),
        },
    )

    assert (await task)["ok"] is True


@pytest.mark.asyncio
async def test_client_preserves_runtime_operation_authority_on_worker_wire(
    tmp_path, monkeypatch
):
    from core.brain.llm import mlx_client
    from core.brain.llm.latent_cortex.epistemic_runtime import RuntimeOperationLease
    from core.brain.llm.latent_cortex.epistemic_state import (
        ComputeBudgetState,
        EpistemicState,
        EpistemicTransaction,
        OperationKind,
        OperationOutcome,
        OperationRecord,
        ProblemFrame,
        text_sha256,
    )
    from core.brain.llm.latent_cortex.value_of_computation import (
        build_evidence_snapshot,
    )

    objective = "reason with a state-bound operation"
    genesis = EpistemicState.genesis(
        episode_id="rlc-client-operation-wire",
        problem=ProblemFrame.create(objective),
        budget=ComputeBudgetState(total=1.0),
    )
    memory = OperationRecord.create(
        operation_id="client-wire-memory-search",
        kind=OperationKind.SEARCH_MEMORY,
        outcome=OperationOutcome.SUCCEEDED,
        input_state_sha256=genesis.state_sha256,
        cost=0.01,
        operator_id="selective_memory_bridge",
        operator_version="v1",
        input_payload_sha256=text_sha256("client wire memory"),
        started_at=1.0,
        completed_at=2.0,
    )
    state = EpistemicTransaction(genesis).add_operation(memory).commit()
    config = {"decode_max_tokens": 16, "n_branches": 2}
    budget = {"max_layer_apps": 1000, "wall_clock_s": 30.0}
    action_policy = build_evidence_snapshot(
        bucket="unit|none|short|s:mid|u:mid",
        cells={},
    )
    lease = RuntimeOperationLease.begin(
        genesis=genesis,
        state=state,
        decision={
            "schema": "aura.latent_execution_controller.v1",
            "bucket": "unit|none|short|s:mid|u:mid",
            "arm": "base",
            "mode": "observe",
            "evidence": {},
        },
        config=config,
        budget=budget,
        action_policy_evidence=action_policy,
        root=tmp_path / "runtime",
        started_at=10.0,
    )

    client = MLXLocalClient(model_path="/models/test-32b")
    client._process = _ResidentProcess()
    client._init_done = True
    client._req_q = queue.Queue()
    _bind_test_client_identity(monkeypatch, client)
    monkeypatch.setattr(
        mlx_client,
        "get_memory_pressure_snapshot",
        lambda: SimpleNamespace(refuse_heavy_local_generation=False),
    )

    task = asyncio.create_task(
        client.latent_reason_async(
            prompt=objective,
            config=config,
            budget=budget,
            operation_authority=lease.authority,
            action_policy_evidence=action_policy,
            timeout_s=5.0,
            foreground_request=False,
        )
    )
    request = await asyncio.to_thread(client._req_q.get, True, 2.0)
    assert request["operation_authority"] == lease.authority
    assert request["action_policy_evidence"] == action_policy
    mlx_client._set_shared_future_result(
        client._pending_generations[request["id"]],
        {
            "id": request["id"],
            "status": "ok",
            "text": "answer",
            "receipt": _identity_receipt_for_request(
                request,
                episode_id="ep-operation-wire",
                runtime_operation_authority=request["operation_authority"],
            ),
        },
    )

    result = await task
    assert result["ok"] is True
    assert result["receipt"]["runtime_operation_authority"] == lease.authority


@pytest.mark.asyncio
async def test_client_latent_reason_serializes_concurrent_requests(monkeypatch):
    from core.brain.llm import mlx_client

    client = MLXLocalClient(model_path="/models/test-32b")
    client._process = _ResidentProcess()
    client._init_done = True
    client._req_q = queue.Queue()
    _bind_test_client_identity(monkeypatch, client)
    monkeypatch.setattr(
        mlx_client,
        "get_memory_pressure_snapshot",
        lambda: SimpleNamespace(refuse_heavy_local_generation=False),
    )

    first = asyncio.create_task(
        client.latent_reason_async(
            prompt="first", timeout_s=5.0, foreground_request=False
        )
    )
    first_request = await asyncio.to_thread(client._req_q.get, True, 2.0)
    second = asyncio.create_task(
        client.latent_reason_async(
            prompt="second", timeout_s=5.0, foreground_request=False
        )
    )
    await asyncio.sleep(0.05)
    assert client._req_q.empty(), "second episode must wait behind request ownership"

    mlx_client._set_shared_future_result(
        client._pending_generations[first_request["id"]],
        {
            "id": first_request["id"],
            "status": "ok",
            "text": "one",
            "receipt": _identity_receipt_for_request(
                first_request,
                episode_id="first",
            ),
        },
    )
    assert (await first)["ok"] is True
    second_request = await asyncio.to_thread(client._req_q.get, True, 2.0)
    mlx_client._set_shared_future_result(
        client._pending_generations[second_request["id"]],
        {
            "id": second_request["id"],
            "status": "ok",
            "text": "two",
            "receipt": _identity_receipt_for_request(
                second_request,
                episode_id="second",
            ),
        },
    )
    assert (await second)["ok"] is True
    assert client._request_lock.locked() is False


@pytest.mark.asyncio
async def test_client_latent_reason_timeout_cancels_recycles_and_releases(monkeypatch):
    from core.brain.llm import mlx_client

    client = MLXLocalClient(model_path="/models/test-32b")
    client._process = _ResidentProcess()
    client._init_done = True
    client._req_q = queue.Queue()
    reboot_reasons: list[str] = []

    monkeypatch.setattr(
        mlx_client,
        "get_memory_pressure_snapshot",
        lambda: SimpleNamespace(refuse_heavy_local_generation=False),
    )

    async def timeout(*args, **kwargs):
        raise TimeoutError

    async def record_reboot(reason, mark_failed=False):
        reboot_reasons.append(reason)

    monkeypatch.setattr(mlx_client, "_await_shared_future", timeout)
    monkeypatch.setattr(client, "reboot_worker", record_reboot)

    result = await client.latent_reason_async(
        prompt="bounded episode", timeout_s=5.0, foreground_request=False
    )

    assert result["reason"] == "latent_timeout:TimeoutError"
    assert reboot_reasons == ["latent_reason_deadline_unacknowledged"]
    assert client._active_generations == 0
    assert client._pending_generations == {}
    assert client._current_request_id == ""
    assert client._request_lock.locked() is False


@pytest.mark.asyncio
async def test_client_latent_reason_timeout_keeps_clean_cooperatively_cancelled_worker(
    monkeypatch,
):
    from core.brain.llm import mlx_client

    client = MLXLocalClient(model_path="/models/test-32b")
    client._process = _ResidentProcess()
    client._init_done = True
    client._req_q = queue.Queue()
    reboot_reasons: list[str] = []
    await_count = 0
    _captured: dict[str, str] = {"expected_sha256": ""}

    monkeypatch.setattr(
        mlx_client,
        "get_memory_pressure_snapshot",
        lambda: SimpleNamespace(refuse_heavy_local_generation=False),
    )

    async def timeout_then_ack(future, *, timeout_s=None):
        nonlocal await_count
        await_count += 1
        if await_count == 1:
            client._record_latent_progress(
                {
                    "id": client._current_request_id,
                    "action": "latent_reason",
                    "status": "progress",
                    "stage": "prefill",
                    "elapsed_s": 4.8,
                    "input_tokens": 4096,
                    "untrusted": "must-not-escape",
                }
            )
            raise TimeoutError
        # CP126 07d62d51: a clean-cancel acknowledgement must be BOUND to
        # this request, this payload and this worker. The real worker sets
        # "id" on every response and its receipt carries the worker identity
        # and payload digest, so the fake models that rather than the
        # unbound shape an attacker (or a stale reply) could produce.
        return {
            "id": client._current_request_id,
            "status": "error",
            "message": "soft_cancelled",
                    "receipt": {
                "params_unchanged": True,
                "fast_weights_applied": True,
                "fast_weights_erased": True,
                "last_stage": "prefill",
                "input_token_count": 4096,
                "request_payload_sha256": _captured["expected_sha256"],
                "worker_boot_id": "b" * 32,
                "worker_pid": 4242,
                "worker_model_path": "/models/test-32b",
            },
        }

    async def record_reboot(reason, mark_failed=False):
        reboot_reasons.append(reason)

    monkeypatch.setattr(mlx_client, "_await_shared_future", timeout_then_ack)
    monkeypatch.setattr(client, "reboot_worker", record_reboot)
    client._worker_identity = {"worker_boot_id": "b" * 32, "worker_pid": 4242}

    # The client imports this from runtime_identity inside the call, so the
    # patch has to land on the source module.
    from core.brain.llm.latent_cortex import runtime_identity as _runtime_identity

    original_sha = _runtime_identity.latent_request_payload_sha256

    def _capture_sha(*args, **kwargs):
        digest = original_sha(*args, **kwargs)
        _captured["expected_sha256"] = digest
        return digest

    monkeypatch.setattr(
        _runtime_identity, "latent_request_payload_sha256", _capture_sha,
    )

    result = await client.latent_reason_async(
        prompt="bounded episode", timeout_s=5.0, foreground_request=False
    )

    assert result["reason"] == "latent_timeout:cooperative_cancelled"
    assert result["receipt"]["params_unchanged"] is True
    assert result["progress"]["stage"] == "prefill"
    assert result["progress"]["input_tokens"] == 4096
    assert "untrusted" not in result["progress"]
    assert reboot_reasons == []
    assert client._active_generations == 0
    assert client._request_lock.locked() is False


@pytest.mark.asyncio
async def test_client_latent_reason_caller_cancel_recycles_and_releases(monkeypatch):
    from core.brain.llm import mlx_client

    client = MLXLocalClient(model_path="/models/test-32b")
    client._process = _ResidentProcess()
    client._init_done = True
    client._req_q = queue.Queue()
    reboot_reasons: list[str] = []
    monkeypatch.setattr(
        mlx_client,
        "get_memory_pressure_snapshot",
        lambda: SimpleNamespace(refuse_heavy_local_generation=False),
    )

    async def record_reboot(reason, mark_failed=False):
        reboot_reasons.append(reason)

    monkeypatch.setattr(client, "reboot_worker", record_reboot)
    task = asyncio.create_task(
        client.latent_reason_async(
            prompt="cancel this episode", timeout_s=30.0, foreground_request=False
        )
    )
    request = await asyncio.to_thread(client._req_q.get, True, 2.0)
    assert request["action"] == "latent_reason"
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert reboot_reasons == ["latent_reason_caller_cancelled"]
    assert client._active_generations == 0
    assert client._pending_generations == {}
    assert client._current_request_id == ""
    assert client._request_lock.locked() is False


@pytest.mark.asyncio
async def test_client_latent_reason_cancel_while_queued_releases_foreground_owner(
    monkeypatch,
):
    from core.brain.llm import mlx_client

    client = MLXLocalClient(model_path="/models/test-32b")
    client._process = _ResidentProcess()
    client._init_done = True
    client._req_q = queue.Queue()
    lock_wait_started = asyncio.Event()
    owner_events: list[str] = []
    monkeypatch.setattr(
        mlx_client,
        "get_memory_pressure_snapshot",
        lambda: SimpleNamespace(refuse_heavy_local_generation=False),
    )

    class OwnerContext:
        async def __aenter__(self):
            owner_events.append("entered")

        async def __aexit__(self, exc_type, exc, traceback):
            owner_events.append("exited")

    async def wait_for_lane(**kwargs):
        lock_wait_started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(mlx_client, "_foreground_owner_context", lambda *a, **k: OwnerContext())
    monkeypatch.setattr(client, "_acquire_request_lock", wait_for_lane)
    task = asyncio.create_task(
        client.latent_reason_async(prompt="queued", foreground_request=True)
    )
    await lock_wait_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert owner_events == ["entered", "exited"]
    assert client._request_lock.locked() is False


@pytest.mark.asyncio
async def test_client_latent_reason_integrity_failure_recycles_resident(monkeypatch):
    from core.brain.llm import mlx_client

    client = MLXLocalClient(model_path="/models/test-32b")
    client._process = _ResidentProcess()
    client._init_done = True
    client._req_q = queue.Queue()
    reboot_reasons: list[str] = []
    lifecycle_events: list[str] = []
    monkeypatch.setattr(
        mlx_client,
        "get_memory_pressure_snapshot",
        lambda: SimpleNamespace(refuse_heavy_local_generation=False),
    )

    async def record_reboot(reason, mark_failed=False):
        assert client._request_lock.locked() is True
        lifecycle_events.append("reboot")
        reboot_reasons.append(reason)

    async def record_fence(preemptible):
        lifecycle_events.append(f"fence:{preemptible}")
        return True

    monkeypatch.setattr(client, "reboot_worker", record_reboot)
    monkeypatch.setattr(client, "_set_durable_lane_preemptible", record_fence)
    task = asyncio.create_task(
        client.latent_reason_async(
            prompt="prove cleanup", timeout_s=5.0, foreground_request=False
        )
    )
    request = await asyncio.to_thread(client._req_q.get, True, 2.0)
    mlx_client._set_shared_future_result(
        client._pending_generations[request["id"]],
        {
            "id": request["id"],
            "status": "error",
            "message": "fast_weight_cleanup_unproven",
            "receipt": {"fast_weights_erased": False},
        },
    )

    result = await task
    assert result["reason"] == "fast_weight_cleanup_unproven"
    assert reboot_reasons == ["latent_integrity:fast_weight_cleanup_unproven"]
    assert lifecycle_events == ["fence:False", "reboot"]
    assert client._active_generations == 0
    assert client._request_lock.locked() is False


@pytest.mark.asyncio
async def test_client_latent_reason_rejects_invalid_inputs_before_lane_fence(monkeypatch):
    from core.brain.llm import mlx_client

    client = MLXLocalClient(model_path="/models/test-32b")
    client._process = _ResidentProcess()
    client._init_done = True
    client._req_q = queue.Queue()
    fence_calls: list[bool] = []
    monkeypatch.setattr(
        mlx_client,
        "get_memory_pressure_snapshot",
        lambda: SimpleNamespace(refuse_heavy_local_generation=False),
    )

    async def record_fence(preemptible):
        fence_calls.append(preemptible)
        return True

    monkeypatch.setattr(client, "_set_durable_lane_preemptible", record_fence)

    assert (
        await client.latent_reason_async(
            prompt="q", config="bad", foreground_request=False
        )
    )["reason"] == "invalid_config"
    assert (
        await client.latent_reason_async(
            prompt="q", budget="bad", foreground_request=False
        )
    )["reason"] == "invalid_budget"
    assert (
        await client.latent_reason_async(
            prompt="q",
            runtime_controls={"clean_user_surface_steering_alpha": 0.3},
            foreground_request=False,
        )
    )["reason"] == "invalid_runtime_controls"
    assert (
        await client.latent_reason_async(
            prompt="q", foreground_request="false"
        )
    )["reason"] == "invalid_foreground_request"
    assert (
        await client.latent_reason_async(
            prompt="q",
            response_contract='{"answer":unknown}',
            foreground_request=False,
        )
    )["reason"] == "invalid_response_contract"
    assert fence_calls == []
    assert client._request_lock.locked() is False


@pytest.mark.asyncio
async def test_client_latent_reason_contains_malformed_worker_receipt(monkeypatch):
    from core.brain.llm import mlx_client

    client = MLXLocalClient(model_path="/models/test-32b")
    client._process = _ResidentProcess()
    client._init_done = True
    client._req_q = queue.Queue()
    monkeypatch.setattr(
        mlx_client,
        "get_memory_pressure_snapshot",
        lambda: SimpleNamespace(refuse_heavy_local_generation=False),
    )
    task = asyncio.create_task(
        client.latent_reason_async(
            prompt="q",
            config={"decode_max_tokens": "malformed"},
            timeout_s=5.0,
            foreground_request=False,
        )
    )
    request = await asyncio.to_thread(client._req_q.get, True, 2.0)
    mlx_client._set_shared_future_result(
        client._pending_generations[request["id"]],
        {"id": request["id"], "status": "ok", "text": "bad", "receipt": "bad"},
    )

    result = await task
    assert result["reason"] == "invalid_worker_receipt"
    assert client._active_generations == 0
    assert client._request_lock.locked() is False


@pytest.mark.asyncio
async def test_client_recycles_worker_on_identity_receipt_mismatch(monkeypatch):
    from core.brain.llm import mlx_client

    client = MLXLocalClient(model_path="/models/test-32b")
    client._process = _ResidentProcess()
    client._init_done = True
    client._req_q = queue.Queue()
    _bind_test_client_identity(monkeypatch, client)
    reboot_reasons = []
    monkeypatch.setattr(
        mlx_client,
        "get_memory_pressure_snapshot",
        lambda: SimpleNamespace(refuse_heavy_local_generation=False),
    )

    async def record_reboot(reason, mark_failed=False):
        reboot_reasons.append(reason)

    monkeypatch.setattr(client, "reboot_worker", record_reboot)
    task = asyncio.create_task(
        client.latent_reason_async(
            prompt="bind this episode",
            timeout_s=5.0,
            foreground_request=False,
        )
    )
    request = await asyncio.to_thread(client._req_q.get, True, 2.0)
    mlx_client._set_shared_future_result(
        client._pending_generations[request["id"]],
        {
            "id": request["id"],
            "status": "ok",
            "text": "untrusted",
            "receipt": _identity_receipt_for_request(
                request,
                worker_boot_id="9" * 32,
            ),
        },
    )

    result = await task

    assert result["ok"] is False
    assert "worker_boot_id_mismatch" in result["reason"]
    assert reboot_reasons == ["latent_integrity:worker_identity_mismatch"]


@pytest.mark.asyncio
async def test_client_recycles_worker_on_request_digest_mismatch(monkeypatch):
    from core.brain.llm import mlx_client

    client = MLXLocalClient(model_path="/models/test-32b")
    client._process = _ResidentProcess()
    client._init_done = True
    client._req_q = queue.Queue()
    _bind_test_client_identity(monkeypatch, client)
    reboot_reasons = []
    monkeypatch.setattr(
        mlx_client,
        "get_memory_pressure_snapshot",
        lambda: SimpleNamespace(refuse_heavy_local_generation=False),
    )

    async def record_reboot(reason, mark_failed=False):
        reboot_reasons.append(reason)

    monkeypatch.setattr(client, "reboot_worker", record_reboot)
    task = asyncio.create_task(
        client.latent_reason_async(
            prompt="bind the exact request",
            runtime_controls={
                "clean_user_surface_recurrent_loops": 2,
                "clean_user_surface_steering_alpha": 0.30,
            },
            timeout_s=5.0,
            foreground_request=False,
        )
    )
    request = await asyncio.to_thread(client._req_q.get, True, 2.0)
    mlx_client._set_shared_future_result(
        client._pending_generations[request["id"]],
        {
            "id": request["id"],
            "status": "ok",
            "text": "tampered",
            "receipt": _identity_receipt_for_request(
                request,
                request_payload_sha256="0" * 64,
            ),
        },
    )

    result = await task

    assert result["ok"] is False
    assert "request_payload_sha256_mismatch" in result["reason"]
    assert reboot_reasons == ["latent_integrity:worker_identity_mismatch"]


# ── Service economy ─────────────────────────────────────────────────────


def test_allocation_scales_with_stakes_and_uncertainty():
    svc = LatentCortexService()
    low_cfg, low_budget = svc.allocate(stakes=0.1, uncertainty=0.1)
    high_cfg, high_budget = svc.allocate(stakes=0.9, uncertainty=0.9)
    assert high_cfg["max_steps"] > low_cfg["max_steps"]
    assert high_cfg["n_branches"] > low_cfg["n_branches"]
    assert high_budget["max_layer_apps"] > low_budget["max_layer_apps"]
    assert high_budget["wall_clock_s"] > low_budget["wall_clock_s"]
    assert low_cfg["latent_opt"] is True and low_cfg["fast_weights"] is True
    assert high_cfg["latent_opt_steps"] >= low_cfg["latent_opt_steps"]
    assert high_cfg["fast_weights_max_layers"] >= low_cfg["fast_weights_max_layers"]


def test_resident_32b_interactive_allocation_keeps_full_stack_inside_live_budget():
    svc = LatentCortexService()

    cfg, budget = svc.allocate(
        stakes=0.7,
        uncertainty=0.8,
        model_parameter_count=32_000_000_000,
        foreground_request=True,
        timeout_s=128.0,
    )

    assert cfg["n_slots"] == 4
    assert cfg["n_branches"] == 2
    assert cfg["max_steps"] == cfg["min_steps"] == 2
    assert cfg["exchange_interval"] == 1
    assert cfg["latent_opt"] is True and cfg["latent_opt_steps"] == 1
    assert cfg["fast_weights"] is True
    assert cfg["fast_weights_opt_steps"] == 1
    assert cfg["fast_weights_max_layers"] == 2
    assert cfg["decode_max_tokens"] == 256
    assert cfg["decode_bridge_policy"] == "assistant_answer_v1"
    assert cfg["verifier_probe_max_tokens"] == 24
    assert cfg["verifier_accept_non_regression"] is True
    assert cfg["input_context_max_chars"] == 9000
    assert cfg["allow_vanilla_fallback"] is False
    assert budget["wall_clock_s"] <= 120.0
    assert (
        svc.get_status()["last_allocation"]["allocation_profile"]
        == "resident_32b_interactive_full_stack_v2"
    )


def test_service_applies_resident_identity_profile_before_worker_ipc(monkeypatch):
    svc = LatentCortexService()
    captured: dict = {}

    class Resident32Client:
        def get_worker_identity_snapshot(self):
            return {"worker_model_parameter_count": 32_500_000_000}

        async def latent_reason_async(self, **kwargs):
            captured.update(kwargs)
            return {"ok": False, "reason": "profile_observed"}

    import core.brain.llm.mlx_client as mlx_client_mod

    monkeypatch.setattr(
        mlx_client_mod,
        "get_mlx_client",
        lambda *args, **kwargs: Resident32Client(),
    )

    result = asyncio.run(
        svc.deep_reason(
            "hard live question",
            stakes=0.7,
            uncertainty=0.8,
            config_overrides={"decode_max_tokens": 2048},
            timeout_s=128.0,
            foreground_request=True,
        )
    )

    assert result == {"ok": False, "reason": "profile_observed"}
    assert captured["config"]["decode_max_tokens"] == 256
    assert captured["config"]["decode_bridge_policy"] == "assistant_answer_v1"
    assert captured["config"]["verifier_probe_max_tokens"] == 24
    assert captured["config"]["verifier_accept_non_regression"] is True
    assert captured["config"]["input_context_max_chars"] == 9000
    assert captured["config"]["allow_vanilla_fallback"] is False
    assert captured["config"]["max_steps"] == 2
    assert captured["config"]["exchange_interval"] == 1
    assert captured["budget"]["wall_clock_s"] <= 120.0


def test_compound_objective_expands_answer_surface(monkeypatch):
    """A request the quality gate will judge on 4 facets must be provisioned
    for 4 facets: more decode room, lower temperature, the coverage-demanding
    v2 bridge, and a wall clock that admits the bigger decode."""
    svc = LatentCortexService()
    captured: dict = {}

    class Resident32Client:
        def get_worker_identity_snapshot(self):
            return {"worker_model_parameter_count": 32_500_000_000}

        async def latent_reason_async(self, **kwargs):
            captured.update(kwargs)
            return {"ok": False, "reason": "profile_observed"}

    import core.brain.llm.mlx_client as mlx_client_mod

    monkeypatch.setattr(
        mlx_client_mod,
        "get_mlx_client",
        lambda *args, **kwargs: Resident32Client(),
    )

    compound = (
        "Compare an early single-owner design with a late deduplication design, "
        "then choose the stronger architecture and explain how you would verify "
        "it under cancellation, timeout, and worker-restart faults."
    )
    result = asyncio.run(
        svc.deep_reason(
            compound,
            stakes=0.75,
            uncertainty=0.8,
            config_overrides={
                "decode_max_tokens": 256,
                "decode_temperature": 0.58,
                "decode_top_p": 0.85,
            },
            timeout_s=157.0,
            foreground_request=True,
        )
    )
    assert result == {"ok": False, "reason": "profile_observed"}
    assert 320 <= captured["config"]["decode_max_tokens"] <= 384
    # Compound answers decode near-greedy for coverage determinism — safe
    # now that the repetition penalty, EOS floor, and newline discipline
    # guard against the degeneration CP105 measured at low temperature.
    assert captured["config"]["decode_temperature"] == 0.3
    assert captured["config"]["decode_repetition_penalty"] == 1.25
    assert captured["config"]["decode_repetition_window"] == 72
    assert captured["config"]["decode_bridge_policy"] == "assistant_answer_v3"
    assert captured["budget"]["wall_clock_s"] >= 140.0
    assert captured["budget"]["wall_clock_s"] <= 157.0 - 8.0
    allocation = svc._last_allocation
    assert allocation["compound_objective"] is True
    assert set(allocation["objective_facets"]) >= {"compare", "select", "verify"}

    # A simple objective keeps the tight interactive profile.
    captured.clear()
    asyncio.run(
        svc.deep_reason(
            "What time zone does the scheduler use?",
            stakes=0.7,
            uncertainty=0.8,
            timeout_s=128.0,
            foreground_request=True,
        )
    )
    assert captured["config"]["decode_max_tokens"] == 256
    assert captured["config"]["decode_bridge_policy"] == "assistant_answer_v1"


def test_allocation_damped_by_body_pressure(monkeypatch):
    svc = LatentCortexService()
    monkeypatch.setattr(svc, "_body_pressure", lambda: 0.0)
    calm_cfg, calm_budget = svc.allocate(stakes=0.8, uncertainty=0.8)
    monkeypatch.setattr(svc, "_body_pressure", lambda: 1.0)
    strained_cfg, strained_budget = svc.allocate(stakes=0.8, uncertainty=0.8)
    assert strained_cfg["max_steps"] < calm_cfg["max_steps"]
    assert strained_budget["max_layer_apps"] < calm_budget["max_layer_apps"]
    assert strained_cfg["n_branches"] <= calm_cfg["n_branches"]


def test_service_kill_switch_and_status(monkeypatch):
    monkeypatch.setenv("AURA_LATENT_CORTEX", "0")
    svc = LatentCortexService()
    result = asyncio.run(svc.deep_reason("why?"))
    assert result["ok"] is False and "disabled" in result["reason"]
    status = svc.get_status()
    assert status["enabled"] is False
    assert status["healthy"] is False
    assert status["state"] == "disabled"


def test_service_idle_state_is_explicitly_unproven_not_healthy(monkeypatch):
    monkeypatch.delenv("AURA_LATENT_CORTEX", raising=False)
    status = LatentCortexService().get_status()
    assert status["state"] == "idle_unproven"
    assert status["healthy"] is False


@pytest.mark.parametrize(
    ("kwargs", "selected", "reason"),
    [
        (
            {
                "foreground": True,
                "desktop_required": True,
                "cognitive_mode": "deliberate",
                "prompt_shape": {},
                "compact_contract": False,
                "strict_output_contract": False,
                "incompatible_contract": False,
                "proof_or_benchmark": False,
            },
            True,
            "deliberate_cognitive_mode",
        ),
        (
            {
                "foreground": True,
                "desktop_required": True,
                "cognitive_mode": "reactive",
                "prompt_shape": {"question_parts": 3},
                "compact_contract": False,
                "strict_output_contract": False,
                "incompatible_contract": False,
                "proof_or_benchmark": False,
            },
            True,
            "multipart_or_extended_prompt",
        ),
        (
            {
                "foreground": True,
                "desktop_required": True,
                "cognitive_mode": "deliberate",
                "prompt_shape": {},
                "compact_contract": False,
                "strict_output_contract": True,
                "incompatible_contract": False,
                "proof_or_benchmark": False,
            },
            False,
            "strict_output_contract",
        ),
        (
            {
                "foreground": True,
                "desktop_required": True,
                "cognitive_mode": "deliberate",
                "prompt_shape": {},
                "compact_contract": False,
                "strict_output_contract": False,
                "incompatible_contract": False,
                "proof_or_benchmark": True,
            },
            False,
            "proof_lane_not_explicitly_opted_in",
        ),
    ],
)
def test_foreground_selection_is_bounded_and_auditable(kwargs, selected, reason):
    decision = LatentCortexService.select_foreground_episode(**kwargs)
    assert decision["latent_cortex_selected"] is selected
    assert decision["latent_cortex_selection_reason"] == reason


def test_explicit_proof_lane_requirement_selects_latent_episode():
    decision = LatentCortexService.select_foreground_episode(
        foreground=True,
        desktop_required=True,
        cognitive_mode="reactive",
        prompt_shape={},
        compact_contract=False,
        strict_output_contract=False,
        incompatible_contract=False,
        proof_or_benchmark=True,
        explicitly_required=True,
    )
    assert decision["latent_cortex_selected"] is True
    assert decision["latent_cortex_selection_reason"] == "explicit_requirement"


def test_selection_recomputes_visible_compound_depth_when_caller_shape_is_stale():
    objective = (
        "Compare optimistic and pessimistic locking for a hot task queue, choose "
        "which one you would use in a single-host async runtime, explain why, and "
        "verify your choice with one concrete failure scenario."
    )

    decision = LatentCortexService.select_foreground_episode(
        foreground=True,
        desktop_required=True,
        cognitive_mode="reactive",
        prompt_shape={},
        compact_contract=True,
        strict_output_contract=False,
        incompatible_contract=False,
        proof_or_benchmark=False,
        visible_objective=objective,
    )

    assert decision["latent_cortex_selected"] is True
    assert decision["latent_cortex_selection_reason"] == "multipart_or_extended_prompt"
    assert decision["latent_cortex_prompt_shape"]["imperative_parts"] == 4
    assert decision["latent_cortex_prompt_shape"]["question_parts"] == 4


def test_service_routes_through_client_and_records_receipt(monkeypatch):
    monkeypatch.delenv("AURA_LATENT_CORTEX", raising=False)
    svc = LatentCortexService()

    captured = {}

    class StubClient:
        async def latent_reason_async(self, prompt=None, **kwargs):
            from core.brain.llm_health_router import generation_gate_snapshot

            captured["prompt"] = prompt
            captured["config"] = kwargs.get("config")
            captured["budget"] = kwargs.get("budget")
            captured["runtime_controls"] = kwargs.get("runtime_controls")
            captured["gate_snapshot"] = generation_gate_snapshot()
            return {
                "ok": True,
                "text": (
                    "The deep answer explains the architecture and preserves its evidence."
                ),
                "receipt": {
                    "steps_taken": 7,
                    "halting_reason": "converged",
                    "n_branches": kwargs["config"]["n_branches"],
                    "n_slots": kwargs["config"]["n_slots"],
                    "episode_id": "abc",
                    "schedule_hash": "b" * 64,
                    "checkpoint_fingerprint": "a" * 64,
                    "checkpoint_fingerprint_method": "sha256",
                    "checkpoint_file_count": 8,
                        **_identity_receipt(),
                        **_branch_isolation_fields(kwargs["config"]),
                    "params_unchanged": True,
                    "budget": {
                        "max_layer_apps": 1_000,
                        "spent_layer_apps": 100,
                        "exhausted": False,
                    },
                    "decode_requested_tokens": kwargs["config"]["decode_max_tokens"],
                    "decode_generated_tokens": 12,
                    "decode_termination": "eos",
                    "decode_newline_suppressions": 0,
                    "decode_repetition_penalty_applied": kwargs["config"].get(
                        "decode_repetition_penalty", 1.0
                    ),
                    "decode_temperature": kwargs["config"].get(
                        "decode_temperature", 0.0
                    ),
                    "decode_top_p": kwargs["config"].get("decode_top_p", 1.0),
                    "verifier_probe_max_tokens": kwargs["config"].get(
                        "verifier_probe_max_tokens", 48
                    ),
                    "latent_opt_applied": True,
                    "latent_opt_mode": "gradient",
                    "latent_opt_attempts": 2,
                    "latent_opt_steps": 2,
                    "latent_opt_rejected": 0,
                    "latent_opt_budget_exhausted": False,
                    "fast_weights_applied": True,
                    "fast_weights_erased": True,
                    "fast_weights_layers": 2,
                    "fast_weight_optimization_attempts": 2,
                    "fast_weight_optimized_steps": 2,
                    "fast_weight_rejected_steps": 0,
                    "fast_weight_budget_exhausted": False,
                    "fast_weight_optimizer": "rms_normalized_sgd_backtracking_v1",
                    "fast_weight_loss_trail": [2.0, 1.5, 1.0],
                    "fast_weight_gradient_norm_trail": [3.0, 2.0],
                    "fast_weight_accepted_step_sizes": [0.005, 0.0025],
                    "fast_weight_line_search_backtracks": 1,
                    "honest_flags": [],
                },
                "reason": "",
            }

    import core.brain.llm.mlx_client as mlx_client_mod

    monkeypatch.setattr(mlx_client_mod, "get_mlx_client", lambda *a, **k: StubClient())
    result = asyncio.run(
        svc.deep_reason(
            "hard question",
            stakes=0.9,
            uncertainty=0.9,
            runtime_controls={
                "clean_user_surface_recurrent_loops": 2,
                "clean_user_surface_steering_alpha": 0.30,
            },
        )
    )
    assert result["ok"]
    assert result["text"].startswith("The deep answer explains")
    assert result["receipt"]["output_quality"]["passed"] is True
    assert captured["prompt"] == "hard question"
    assert captured["config"]["n_branches"] >= 2
    assert captured["config"]["latent_opt"] is True
    assert captured["config"]["fast_weights"] is True
    assert captured["config"]["branch_correlation_evidence"]["schema"] == (
        "aura.rlc.branch_error_correlation.v1"
    )
    assert captured["config"]["branch_correlation_evidence"]["evidence_state"] in {
        "bootstrap_unmeasured",
        "measured",
    }
    assert captured["budget"]["max_layer_apps"] > 0
    assert captured["runtime_controls"] == {
        "clean_user_surface_recurrent_loops": 2,
        "clean_user_surface_steering_alpha": 0.30,
    }
    assert captured["gate_snapshot"]["active_count"] >= 1
    assert "latent_cortex_foreground:episode" in {
        item["owner"]
        for item in captured["gate_snapshot"]["active"].values()
    }
    assert svc.get_status()["last_receipt"]["halting_reason"] == "converged"


def test_service_rejects_nominal_full_stack_without_accepted_optimization():
    config = {
        "n_slots": 16,
        "n_branches": 2,
        "latent_opt": True,
        "fast_weights": True,
    }
    receipt = {
        "episode_id": "ep-noop",
        "checkpoint_fingerprint": "a" * 64,
        "checkpoint_fingerprint_method": "sha256",
        "checkpoint_file_count": 8,
        "params_unchanged": True,
        "schedule_hash": "b" * 64,
        "steps_taken": 4,
        "n_slots": 16,
        "n_branches": 2,
        "budget": {
            "max_layer_apps": 1_000,
            "spent_layer_apps": 100,
            "exhausted": False,
        },
        "decode_requested_tokens": 512,
        "decode_generated_tokens": 12,
        "decode_termination": "eos",
        "honest_flags": [],
        "latent_opt_applied": True,
        "latent_opt_mode": "gradient",
        "latent_opt_attempts": 1,
        "latent_opt_steps": 0,
        "latent_opt_rejected": 1,
        "latent_opt_budget_exhausted": False,
        "fast_weights_applied": True,
        "fast_weights_erased": True,
        "fast_weights_layers": 2,
        "fast_weight_optimization_attempts": 1,
        "fast_weight_optimized_steps": 0,
        "fast_weight_rejected_steps": 1,
        "fast_weight_budget_exhausted": False,
    }

    errors = LatentCortexService._receipt_contract_errors(receipt, config)

    assert "latent_optimization_no_accepted_steps" in errors
    assert "fast_weight_optimization_no_accepted_steps" in errors


def test_service_reconstructs_and_rejects_branch_isolation_tampering():
    config = {"n_branches": 2, "isolation_steps": 2}
    receipt = {
        "n_branches": 2,
        **_branch_isolation_fields(config, exchanges=1),
    }
    assert "branch_isolation_unproven" not in (
        LatentCortexService._receipt_contract_errors(receipt, config)
    )

    tampered = {
        **receipt,
        "branch_isolation": {
            **receipt["branch_isolation"],
            "first_exchange_step": 1,
        },
    }
    assert "branch_isolation_unproven" in (
        LatentCortexService._receipt_contract_errors(tampered, config)
    )
    candidates = [dict(row) for row in receipt["branch_isolation"]["candidates"]]
    candidates[1]["candidate_sha256"] = candidates[0]["candidate_sha256"]
    tampered = {
        **receipt,
        "branch_isolation": {
            **receipt["branch_isolation"],
            "candidates": candidates,
        },
    }
    assert "branch_isolation_unproven" in (
        LatentCortexService._receipt_contract_errors(tampered, config)
    )


def test_service_validates_interactive_verifier_profile_and_acceptance_receipt():
    config = {
        "latent_opt": True,
        "verifier_probe_max_tokens": 24,
        "verifier_accept_non_regression": True,
    }
    receipt = {
        "latent_opt_applied": True,
        "latent_opt_mode": "gradient",
        "latent_opt_attempts": 1,
        "latent_opt_steps": 1,
        "latent_opt_rejected": 0,
        "latent_opt_budget_exhausted": False,
        "verifier_probe_max_tokens": 24,
        "latent_opt_verifier": {
            "policy": "task_score_nonregression_with_proxy_descent_v1",
            "baseline_source": "caller_reused_verified_branch",
            "score_tolerance": 1e-9,
            "proxy_tolerance_scale": 1e-9,
            "score_trail": [0.5, 0.5],
            "decisions": [
                {
                    "proposal": 0,
                    "baseline_score": 0.5,
                    "candidate_score": 0.5,
                    "current_proxy_loss": 1.0,
                    "candidate_proxy_loss": 0.9,
                    "proxy_required_delta": 1e-9,
                    "decision": (
                        "accepted_task_score_nonregression_with_proxy_descent"
                    ),
                }
            ],
            "score_improvement_accepts": 0,
            "proxy_nonregression_accepts": 1,
        },
    }

    errors = LatentCortexService._receipt_contract_errors(receipt, config)

    assert "verifier_probe_profile_mismatch" not in errors
    assert "latent_optimization_verifier_receipt_invalid" not in errors
    tampered = dict(receipt)
    tampered["verifier_probe_max_tokens"] = 48
    assert "verifier_probe_profile_mismatch" in (
        LatentCortexService._receipt_contract_errors(tampered, config)
    )
    tampered = dict(receipt)
    tampered["latent_opt_verifier"] = {
        **receipt["latent_opt_verifier"],
        "proxy_nonregression_accepts": 0,
    }
    assert "latent_optimization_verifier_receipt_invalid" in (
        LatentCortexService._receipt_contract_errors(tampered, config)
    )
    tampered = dict(receipt)
    tampered["latent_opt_verifier"] = {
        **receipt["latent_opt_verifier"],
        "decisions": [
            {
                **receipt["latent_opt_verifier"]["decisions"][0],
                "candidate_proxy_loss": 1.1,
            }
        ],
    }
    assert "latent_optimization_verifier_receipt_invalid" in (
        LatentCortexService._receipt_contract_errors(tampered, config)
    )
    tampered = dict(receipt)
    tampered["latent_opt_verifier"] = {
        **receipt["latent_opt_verifier"],
        "score_trail": [0.5, 0.6],
    }
    assert "latent_optimization_verifier_receipt_invalid" in (
        LatentCortexService._receipt_contract_errors(tampered, config)
    )


def test_service_enforces_default_verifier_probe_profile():
    assert "verifier_probe_profile_mismatch" in (
        LatentCortexService._receipt_contract_errors({}, {})
    )
    assert "verifier_probe_profile_mismatch" not in (
        LatentCortexService._receipt_contract_errors(
            {"verifier_probe_max_tokens": 48},
            {},
        )
    )


@pytest.mark.parametrize(
    ("override", "expected_error"),
    [
        (
            {"fast_weight_optimizer": "plain_sgd"},
            "fast_weight_optimizer_unproven",
        ),
        (
            {"fast_weight_loss_trail": [2.0, 2.0]},
            "fast_weight_loss_descent_unproven",
        ),
        (
            {"fast_weight_gradient_norm_trail": []},
            "fast_weight_gradient_evidence_invalid",
        ),
        (
            {"fast_weight_accepted_step_sizes": [0.0]},
            "fast_weight_step_evidence_invalid",
        ),
        (
            {"fast_weight_line_search_backtracks": -1},
            "fast_weight_line_search_evidence_invalid",
        ),
    ],
)
def test_service_rejects_unproven_fast_weight_descent(override, expected_error):
    config = {"fast_weights": True}
    receipt = {
        "fast_weights_applied": True,
        "fast_weights_erased": True,
        "fast_weights_layers": 2,
        "fast_weight_optimization_attempts": 1,
        "fast_weight_optimized_steps": 1,
        "fast_weight_rejected_steps": 0,
        "fast_weight_budget_exhausted": False,
        "fast_weight_optimizer": "rms_normalized_sgd_backtracking_v1",
        "fast_weight_loss_trail": [2.0, 1.0],
        "fast_weight_gradient_norm_trail": [3.0],
        "fast_weight_accepted_step_sizes": [0.005],
        "fast_weight_line_search_backtracks": 0,
        **override,
    }

    errors = LatentCortexService._receipt_contract_errors(receipt, config)

    assert expected_error in errors


@pytest.mark.parametrize(
    ("termination", "exhausted", "expected"),
    [
        ("budget_exhausted", True, "decode_incomplete"),
        ("budget_unaffordable", False, "decode_incomplete"),
        ("token_limit", True, "incomplete_or_exhausted_compute_receipt"),
    ],
)
def test_service_rejects_truncated_or_exhausted_decode_receipts(
    termination, exhausted, expected
):
    config = {
        "n_slots": 16,
        "n_branches": 2,
        "decode_max_tokens": 512,
    }
    receipt = {
        "episode_id": "ep-truncated",
        "checkpoint_fingerprint": "a" * 64,
        "checkpoint_fingerprint_method": "sha256",
        "checkpoint_file_count": 8,
        "params_unchanged": True,
        "schedule_hash": "b" * 64,
        "steps_taken": 4,
        "n_slots": 16,
        "n_branches": 2,
        "budget": {
            "max_layer_apps": 1_000,
            "spent_layer_apps": 1_000 if exhausted else 900,
            "exhausted": exhausted,
        },
        "decode_requested_tokens": 512,
        "decode_generated_tokens": 20,
        "decode_termination": termination,
        "decode_newline_suppressions": 0,
        "decode_repetition_penalty_applied": 1.25,
        "honest_flags": [],
    }

    errors = LatentCortexService._receipt_contract_errors(receipt, config)

    assert expected in errors


def test_service_reports_refusals_honestly(monkeypatch):
    monkeypatch.delenv("AURA_LATENT_CORTEX", raising=False)
    svc = LatentCortexService()

    class BusyClient:
        async def latent_reason_async(self, **kwargs):
            return {"ok": False, "reason": "generation_active"}

    import core.brain.llm.mlx_client as mlx_client_mod

    monkeypatch.setattr(mlx_client_mod, "get_mlx_client", lambda *a, **k: BusyClient())
    result = asyncio.run(svc.deep_reason("q"))
    assert result["ok"] is False and result["reason"] == "generation_active"
    assert svc.get_status()["last_refusal"] == "generation_active"


@pytest.mark.parametrize(
    ("kwargs", "expected_reason"),
    [
        ({"stakes": float("nan")}, "invalid_cognitive_economy"),
        ({"uncertainty": "high"}, "invalid_cognitive_economy"),
        ({"config_overrides": []}, "invalid_config_overrides"),
        ({"runtime_controls": []}, "invalid_runtime_controls"),
        ({"runtime_controls": {}}, "invalid_runtime_controls"),
        (
            {
                "runtime_controls": {
                    "clean_user_surface_recurrent_loops": 3,
                    "clean_user_surface_steering_alpha": 0.30,
                }
            },
            "invalid_runtime_controls",
        ),
        ({"require_full_stack": "yes"}, "invalid_require_full_stack"),
        ({"foreground_request": "yes"}, "invalid_foreground_request"),
        ({"question": 7}, "invalid_question"),
        ({"messages": "not-a-list"}, "invalid_messages"),
    ],
)
def test_service_rejects_malformed_inputs_before_model_client_lookup(
    monkeypatch, kwargs, expected_reason
):
    monkeypatch.delenv("AURA_LATENT_CORTEX", raising=False)
    svc = LatentCortexService()

    import core.brain.llm.mlx_client as mlx_client_mod

    def unexpected_lookup(*args, **kwargs):
        raise AssertionError("malformed input must not touch the model client")

    monkeypatch.setattr(mlx_client_mod, "get_mlx_client", unexpected_lookup)
    question = kwargs.pop("question", "q")
    result = asyncio.run(svc.deep_reason(question, **kwargs))

    assert result == {"ok": False, "reason": expected_reason}


def test_service_propagates_background_lane_priority(monkeypatch):
    monkeypatch.delenv("AURA_LATENT_CORTEX", raising=False)
    svc = LatentCortexService()
    captured: dict[str, object] = {}

    class BackgroundClient:
        async def latent_reason_async(self, **kwargs):
            captured.update(kwargs)
            return {"ok": False, "reason": "generation_active"}

    import core.brain.llm.mlx_client as mlx_client_mod

    monkeypatch.setattr(
        mlx_client_mod,
        "get_mlx_client",
        lambda *args, **kwargs: BackgroundClient(),
    )

    result = asyncio.run(svc.deep_reason("idle thought", foreground_request=False))

    assert result == {"ok": False, "reason": "generation_active"}
    assert captured["foreground_request"] is False


def test_service_rejects_incomplete_success_receipt(monkeypatch):
    monkeypatch.delenv("AURA_LATENT_CORTEX", raising=False)
    svc = LatentCortexService()

    class ShallowClient:
        async def latent_reason_async(self, **kwargs):
            return {"ok": True, "text": "shallow", "receipt": {"episode_id": "x"}}

    import core.brain.llm.mlx_client as mlx_client_mod

    monkeypatch.setattr(mlx_client_mod, "get_mlx_client", lambda *a, **k: ShallowClient())
    result = asyncio.run(svc.deep_reason("q"))
    assert result["ok"] is False
    assert result["reason"].startswith("receipt_contract_failed:")
    assert svc.get_status()["ok_episodes"] == 0


@pytest.mark.parametrize(
    ("worker_result", "expected_reason"),
    [
        (
            {"ok": True, "text": "bad", "receipt": "not-a-mapping"},
            "receipt_not_mapping",
        ),
        (
            {
                "ok": True,
                "text": "bad",
                "receipt": {
                    "episode_id": "x",
                    "steps_taken": "7",
                    "n_slots": 16,
                    "n_branches": 2,
                    "budget": {"spent_layer_apps": "100"},
                    "honest_flags": "none",
                },
            },
            "no_recurrent_steps",
        ),
        ("not-a-mapping", "invalid_client_response"),
    ],
)
def test_service_contains_malformed_worker_response(
    monkeypatch, worker_result, expected_reason
):
    monkeypatch.delenv("AURA_LATENT_CORTEX", raising=False)
    svc = LatentCortexService()

    class MalformedClient:
        async def latent_reason_async(self, **kwargs):
            return worker_result

    import core.brain.llm.mlx_client as mlx_client_mod

    monkeypatch.setattr(
        mlx_client_mod, "get_mlx_client", lambda *a, **k: MalformedClient()
    )
    result = asyncio.run(svc.deep_reason("q"))
    assert result["ok"] is False
    assert expected_reason in result["reason"]
    assert svc.get_status()["failure_streak"] == 1


def test_service_contains_client_exception_and_degrades_health(monkeypatch):
    monkeypatch.delenv("AURA_LATENT_CORTEX", raising=False)
    svc = LatentCortexService()

    class BrokenClient:
        async def latent_reason_async(self, **kwargs):
            raise RuntimeError("worker exploded")

    import core.brain.llm.mlx_client as mlx_client_mod

    monkeypatch.setattr(mlx_client_mod, "get_mlx_client", lambda *a, **k: BrokenClient())
    for _ in range(3):
        result = asyncio.run(svc.deep_reason("q"))
        assert result["ok"] is False
    status = svc.get_status()
    assert status["failure_streak"] == 3
    assert status["healthy"] is False and status["state"] == "degraded"


def test_service_name_registered_in_spine():
    from core.service_names import ServiceNames

    assert ServiceNames.LATENT_CORTEX == "latent_cortex"


def test_handler_builds_task_verifier_when_guided(monkeypatch):
    from core.brain.llm.latent_cortex.types import (
        EpisodeReceipt,
        LatentReasoningResult,
    )

    monkeypatch.delenv("AURA_LATENT_CORTEX", raising=False)
    captured: dict = {}

    class StubEngine:
        def __init__(self, model, tokenizer, config, **kwargs):
            """Accept the engine construction contract; state is unused."""

        def reason(self, **kwargs):
            captured.update(kwargs)
            if kwargs.get("verifier") is not None:
                kwargs["verifier"]("probe with 2 + 2 = 4")
            return LatentReasoningResult(ok=True, text="ok", receipt=EpisodeReceipt())

    import core.brain.llm.latent_cortex.worker_handler as handler_mod

    monkeypatch.setattr(handler_mod, "LatentCortexEngine", StubEngine)

    class StubTokenizer:
        eos_token_id = 0

        def encode(self, text, **kwargs):
            return [1, 2, 3]

        def decode(self, ids):
            return "x"

    body = handler_mod.handle_latent_reason(
        {"prompt": "verify that 2 + 2 = 4", "verifier_guidance": True},
        model=object(),
        tokenizer=StubTokenizer(),
        model_path="",
        worker_identity=dict(_WORKER_IDENTITY),
    )
    assert body["status"] == "ok"
    assert captured["verifier"] is not None
    guidance = body["receipt"]["verifier_guidance"]
    assert guidance["evaluations"] == 1
    assert "arithmetic" in guidance["best_applicable_checks"]
    assert not guidance.get("best_failures"), "correct arithmetic must not be flagged"
    assert guidance["outcome_checked"] is False
    assert guidance["outcome_passed"] is None
    assert guidance["outcome_reason"] == "candidate_checks_are_not_task_ground_truth"

    # Without the flag, no verifier is constructed.
    captured.clear()
    handler_mod.handle_latent_reason(
        {"prompt": "verify that 2 + 2 = 4"},
        model=object(),
        tokenizer=StubTokenizer(),
        model_path="",
        worker_identity=dict(_WORKER_IDENTITY),
    )
    assert captured["verifier"] is None


def test_service_requests_verifier_guidance_for_resident_profile(monkeypatch):
    svc = LatentCortexService()
    captured: dict = {}

    class Resident32Client:
        def get_worker_identity_snapshot(self):
            return {"worker_model_parameter_count": 32_500_000_000}

        async def latent_reason_async(self, **kwargs):
            captured.update(kwargs)
            return {"ok": False, "reason": "profile_observed"}

    import core.brain.llm.mlx_client as mlx_client_mod

    monkeypatch.setattr(
        mlx_client_mod, "get_mlx_client", lambda *a, **k: Resident32Client()
    )
    asyncio.run(
        svc.deep_reason(
            "hard live question",
            stakes=0.7,
            uncertainty=0.8,
            timeout_s=128.0,
            foreground_request=True,
        )
    )
    assert captured["verifier_guidance"] is True


# ── GWT ↔ RLC coupling gating ───────────────────────────────────────────


def _full_success_stub_client(captured):
    class StubClient:
        async def latent_reason_async(self, prompt=None, **kwargs):
            captured["prompt"] = prompt
            captured["config"] = kwargs.get("config")
            return {
                "ok": True,
                "text": "A deliberate conclusion that answers the question.",
                "receipt": {
                    "steps_taken": 7,
                    "halting_reason": "converged",
                    "n_branches": kwargs["config"]["n_branches"],
                    "n_slots": kwargs["config"]["n_slots"],
                    "episode_id": "ep-gwt",
                    "schedule_hash": "b" * 64,
                    "checkpoint_fingerprint": "a" * 64,
                    "checkpoint_fingerprint_method": "sha256",
                    "checkpoint_file_count": 8,
                    **_identity_receipt(),
                    **_branch_isolation_fields(kwargs["config"]),
                    "params_unchanged": True,
                    "budget": {
                        "max_layer_apps": 1_000,
                        "spent_layer_apps": 100,
                        "exhausted": False,
                    },
                    "decode_requested_tokens": kwargs["config"][
                        "decode_max_tokens"
                    ],
                    "decode_generated_tokens": 12,
                    "decode_termination": "eos",
                    "decode_newline_suppressions": 0,
                    "decode_repetition_penalty_applied": 1.0,
                    "decode_temperature": 0.0,
                    "decode_top_p": 1.0,
                    "verifier_probe_max_tokens": kwargs["config"].get(
                        "verifier_probe_max_tokens", 48
                    ),
                    "latent_opt_applied": True,
                    "latent_opt_mode": "gradient",
                    "latent_opt_attempts": 2,
                    "latent_opt_steps": 2,
                    "latent_opt_rejected": 0,
                    "latent_opt_budget_exhausted": False,
                    "fast_weights_applied": True,
                    "fast_weights_erased": True,
                    "fast_weights_layers": 2,
                    "fast_weight_optimization_attempts": 2,
                    "fast_weight_optimized_steps": 2,
                    "fast_weight_rejected_steps": 0,
                    "fast_weight_budget_exhausted": False,
                    "fast_weight_optimizer": (
                        "rms_normalized_sgd_backtracking_v1"
                    ),
                    "fast_weight_loss_trail": [2.0, 1.5, 1.0],
                    "fast_weight_gradient_norm_trail": [3.0, 2.0],
                    "fast_weight_accepted_step_sizes": [0.005, 0.0025],
                    "fast_weight_line_search_backtracks": 1,
                    "honest_flags": [],
                },
                "reason": "",
            }

    return StubClient


def _run_episode_with_coupling_probes(monkeypatch, *, foreground: bool):
    import core.brain.gwt_rlc_coupling as coupling_mod
    import core.brain.llm.mlx_client as mlx_client_mod

    monkeypatch.delenv("AURA_LATENT_CORTEX", raising=False)
    svc = LatentCortexService()
    captured: dict = {}
    calls = {"merge": 0, "broadcast": 0}

    def _fake_merge(items, **kwargs):
        calls["merge"] += 1
        return items

    async def _fake_broadcast(objective, text, receipt, *, stakes=0.5):
        calls["broadcast"] += 1
        return {
            "schema": coupling_mod.GWT_RLC_SCHEMA,
            "submitted": True,
            "accepted": True,
            "priority": 0.7,
            "pricing": {"verified": False},
        }

    monkeypatch.setattr(coupling_mod, "merge_cognitive_context", _fake_merge)
    monkeypatch.setattr(
        coupling_mod, "broadcast_episode_conclusion", _fake_broadcast
    )
    monkeypatch.setattr(
        mlx_client_mod,
        "get_mlx_client",
        lambda *a, **k: _full_success_stub_client(captured)(),
    )
    result = asyncio.run(
        svc.deep_reason(
            "hard question",
            stakes=0.9,
            uncertainty=0.9,
            foreground_request=foreground,
            runtime_controls={
                "clean_user_surface_recurrent_loops": 2,
                "clean_user_surface_steering_alpha": 0.30,
            },
        )
    )
    return result, calls


def test_foreground_episode_couples_to_workspace(monkeypatch):
    result, calls = _run_episode_with_coupling_probes(
        monkeypatch, foreground=True
    )
    assert result["ok"]
    assert calls["merge"] == 1
    assert calls["broadcast"] == 1
    broadcast = result["receipt"]["workspace_broadcast"]
    assert broadcast["submitted"] is True
    assert broadcast["accepted"] is True


def test_background_episode_stays_decoupled_from_live_mind(monkeypatch):
    result, calls = _run_episode_with_coupling_probes(
        monkeypatch, foreground=False
    )
    assert result["ok"]
    assert calls["merge"] == 0
    assert calls["broadcast"] == 0
    assert "workspace_broadcast" not in result["receipt"]


# ── Held-out facet grading loop (service ↔ Foundry) ─────────────────────


def test_facet_weights_stay_none_until_foundry_has_graded_evidence(monkeypatch):
    class NeutralFoundry:
        def weight_for(self, verifier, domain):
            return 1.0

    import core.brain.verifiers.foundry as foundry_mod

    monkeypatch.setattr(
        foundry_mod, "get_verifier_foundry", lambda: NeutralFoundry()
    )
    assert LatentCortexService._facet_reliability_weights("general") is None

    class MeasuredFoundry:
        def weight_for(self, verifier, domain):
            return 0.4 if verifier == "latent_facet_explain" else 1.0

    monkeypatch.setattr(
        foundry_mod, "get_verifier_foundry", lambda: MeasuredFoundry()
    )
    weights = LatentCortexService._facet_reliability_weights("general")
    assert weights is not None
    assert weights["explain"] == 0.4
    assert weights["compare"] == 1.0


def test_successful_episode_queues_facet_judgments_for_grading(monkeypatch):
    recorded: list[dict] = []

    class RecordingFoundry:
        def record_verdict(self, **kwargs):
            recorded.append(kwargs)
            return f"vd-{len(recorded)}"

        def weight_for(self, verifier, domain):
            return 1.0

    import core.brain.verifiers.foundry as foundry_mod

    monkeypatch.setattr(
        foundry_mod, "get_verifier_foundry", lambda: RecordingFoundry()
    )
    svc = LatentCortexService()
    receipt = {
        "verifier_guidance": {
            "evaluations": 3,
            "best_score": 0.8,
            "facet_judgments": [
                {
                    "facet": "explain",
                    "satisfied": True,
                    "excerpt": "because the lease ordering bounds waiting",
                },
                {"facet": "compare", "satisfied": False, "excerpt": ""},
                {"facet": 42, "satisfied": True},  # junk row is skipped
            ],
        }
    }
    svc._record_facet_judgments(receipt, "general", "why prefer older leases?")
    assert len(recorded) == 2
    by_verifier = {row["verifier"]: row for row in recorded}
    explain = by_verifier["latent_facet_explain"]
    assert explain["hard_pass"] is True and explain["score"] == 1.0
    assert explain["checked"] is True
    assert "lease ordering" in explain["meta"]["excerpt"]
    compare = by_verifier["latent_facet_compare"]
    assert compare["hard_pass"] is False and compare["score"] == 0.0
    assert explain["task_key"] == compare["task_key"] != ""
