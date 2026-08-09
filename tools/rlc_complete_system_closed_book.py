"""Closed-book composition of Aura's complete reasoning treatment.

This module is imported lazily by the frozen reconciliation sweep. It owns the
service-side experimental composition while the sweep retains campaign
identity, controls, journaling, and grading.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any


def _task_type_for_domain(domain: str) -> str:
    """Map the frozen battery taxonomy onto the production amplifier taxonomy."""

    return {
        "coding": "code",
        "mathematics": "math",
        "calibration": "math",
        "long_horizon_planning": "planning",
        "scientific_inference": "factual",
        "misleading_premise": "logic",
        "novel_algorithms": "logic",
    }.get(str(domain or "").strip(), "logic")


def _candidate_quality_assessment(row: dict[str, Any]) -> dict[str, Any]:
    """Separate useful candidate screening from exact correctness authority."""

    checks = row.get("checks") or {}
    contract = checks.get("response_contract") or {}
    atomic = checks.get("atomic_decomposition") or {}
    routed = checks.get("deterministic_router") or {}
    arithmetic = checks.get("arithmetic") or {}
    code = checks.get("code") or {}
    routes = (routed.get("receipt") or {}).get("routes") or []
    exact_routes = [route for route in routes if route.get("verifier") == "exact_objective_program"]
    exact_public_objective_proof = bool(exact_routes) and all(
        route.get("outcome") == "verified" for route in exact_routes
    )
    issues: list[str] = []
    if contract and contract.get("valid") is not True:
        issues.append("response_contract_failed")
    if atomic.get("valid") is not True:
        issues.append("atomic_decomposition_failed")
    if routed.get("valid") is not True:
        issues.append("deterministic_route_failed")
    if arithmetic.get("failures"):
        issues.append("arithmetic_claim_failed")
    if code.get("failures"):
        issues.append("code_check_failed")
    applicable = bool(row.get("applicable_checks"))
    return {
        "schema": "aura.rlc.candidate_quality_assessment.v1",
        "proxy_admitted": applicable and not issues,
        "exact_public_objective_proof": exact_public_objective_proof,
        "ground_truth_verified": exact_public_objective_proof,
        "answer_key_used": False,
        "issues": issues,
        "score": float(row.get("score") or 0.0),
    }


class _ClosedBookVerifierRegistry:
    """Production-amplifier adapter for the answer-key-free episode verifier."""

    def __init__(self, verifier: Any, *, domain: str) -> None:
        self.verifier = verifier
        self.domain = str(domain or "general")
        self.calls = 0
        self.input_bytes = 0
        self.output_bytes = 0

    async def verify(
        self,
        candidate: str,
        *,
        task_type: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> Any:
        from core.brain.verifiers.base import VerificationResult

        del task_type, context
        self.calls += 1
        self.input_bytes += len(candidate.encode("utf-8"))
        row = self.verifier.evaluate(candidate)
        assessment = _candidate_quality_assessment(row)
        self.output_bytes += len(
            json.dumps(
                {"evaluation": row, "assessment": assessment},
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        )
        checked = bool(row.get("applicable_checks"))
        return VerificationResult(
            domain=self.domain,
            ok=bool(assessment["proxy_admitted"]),
            checked=checked,
            score=float(row.get("score") or 0.0),
            engine="rlc_candidate_quality_proxy_not_ground_truth",
            issues=list(assessment["issues"]),
            evidence=[
                f"applicable_checks={','.join(row.get('applicable_checks') or [])}",
                f"candidate_score={float(row.get('score') or 0.0):.6f}",
                f"ground_truth_verified={str(assessment['ground_truth_verified']).lower()}",
            ],
            detail={
                "answer_key_used": False,
                "authority": "candidate_quality_proxy_not_ground_truth",
                "assessment": assessment,
                "evaluation": row,
            },
        )


def _run_closed_book_sample(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    seed: int,
) -> tuple[str, dict[str, Any]]:
    """Generate one deterministic in-process amplifier candidate.

    The benchmark owns exactly one model. Calling the regular batched/live
    helper here would cross into the resident MLX client and either load a
    second checkpoint or measure a different process. This adapter exercises
    the same checkpoint already owned by the frozen campaign.
    """

    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    from core.brain.llm.latent_cortex.answer_contract import is_contract_complete

    mx.random.seed(int(seed) & 0xFFFFFFFF)
    messages = [{"role": "user", "content": str(prompt)}]
    rendered = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    prompt_token_ids = list(
        tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
        )
    )
    pieces: list[str] = []
    generated_tokens = 0
    for response in stream_generate(
        model,
        tokenizer,
        prompt=rendered,
        max_tokens=max_tokens,
        sampler=make_sampler(
            temp=max(0.01, float(temperature)),
            top_p=0.95,
        ),
    ):
        pieces.append(response.text)
        generated_tokens = int(response.generation_tokens)
        if "}" in response.text and is_contract_complete("".join(pieces)):
            break
    text = "".join(pieces).strip()

    from core.brain.llm.latent_cortex.resource_accounting import (
        ModelComputeProfile,
        ResourceLedger,
        triangular_attention_pairs,
    )

    profile = ModelComputeProfile.from_model(model)
    ledger = ResourceLedger(profile)
    prompt_tokens = len(prompt_token_ids)
    output_tokens = max(1, generated_tokens)
    decode_forwards = max(0, output_tokens - 1)
    n_layers = len(model.model.layers)
    ledger.charge(
        "amplifier_prefill",
        transformer_layer_apps=prompt_tokens * n_layers,
        attention_query_key_pairs=triangular_attention_pairs(prompt_tokens) * n_layers,
        output_head_tokens=1,
    )
    decode_pairs = sum(prompt_tokens + index + 1 for index in range(decode_forwards))
    ledger.charge(
        "amplifier_decode",
        transformer_layer_apps=decode_forwards * n_layers,
        attention_query_key_pairs=decode_pairs * n_layers,
        output_head_tokens=decode_forwards,
        tensor_element_reads=output_tokens * profile.vocab_size,
        tensor_element_writes=output_tokens * profile.vocab_size,
        host_scalar_ops=output_tokens * profile.vocab_size * 8,
    )
    return text, ledger.to_receipt()


def _promotion_assessment(
    *,
    verifier: Any,
    incumbent_text: str,
    candidate_text: str,
    candidate_verified: bool,
    authority: str = "candidate_quality_proxy_not_ground_truth",
) -> tuple[str, dict[str, Any]]:
    """Apply the same answer-key-free monotonic promotion rule after amplification."""

    incumbent_row = verifier.evaluate(incumbent_text, _record=False)
    candidate_row = verifier.evaluate(candidate_text, _record=False) if candidate_text else {}
    incumbent_score = float(incumbent_row.get("score") or 0.0)
    candidate_score = float(candidate_row.get("score") or 0.0)
    candidate_contract = bool(
        (candidate_row.get("checks") or {}).get("response_contract", {}).get("valid", False)
    )
    replace = bool(
        candidate_text
        and candidate_verified
        and candidate_contract
        and candidate_score > incumbent_score + 1e-9
    )
    final_text = candidate_text if replace else incumbent_text
    receipt = {
        "schema": "aura.rlc.closed_book_promotion.v1",
        "decision": "replace" if replace else "retain",
        "reason": (
            "verified_candidate_score_improved"
            if replace
            else "candidate_not_verified"
            if not candidate_verified
            else "candidate_contract_invalid"
            if not candidate_contract
            else "candidate_score_did_not_improve"
        ),
        "answer_key_used": False,
        "authority": authority,
        "ground_truth_verified": authority == "public_objective_deterministic_execution",
        "no_regression_guaranteed": authority == "public_objective_deterministic_execution",
        "incumbent_score": round(incumbent_score, 6),
        "candidate_score": round(candidate_score, 6),
        "candidate_verified": bool(candidate_verified),
        "candidate_contract_valid": candidate_contract,
        "incumbent_text_sha256": hashlib.sha256(incumbent_text.encode()).hexdigest(),
        "candidate_text_sha256": (
            hashlib.sha256(candidate_text.encode()).hexdigest() if candidate_text else ""
        ),
        "final_text_sha256": hashlib.sha256(final_text.encode()).hexdigest(),
    }
    return final_text, receipt


def _run_complete_system_closed_book(
    model: Any,
    config: Any,
    prompt_tokens: list[int],
    tokenizer: Any,
    *,
    task: Any,
    max_tokens: int,
    wall_clock_s: float,
    model_path: str,
    incumbent_artifact: Any,
    worker_identity: dict[str, Any],
    runtime_identity: dict[str, Any],
    campaign_seed: int,
) -> tuple[str, dict[str, Any]]:
    """Measure the same-information RLC, acquisition, and amplifier composition."""

    from core.brain.llm.latent_cortex.cognitive_acquisition import (
        COGNITIVE_COMPUTE_ACTIONS,
        build_acquisition_receipt,
        build_acquisition_request,
        build_continuation_receipt,
        validate_acquisition_receipt,
        validate_continuation_receipt,
    )
    from core.brain.llm.latent_cortex.epistemic_state import OperationKind
    from core.brain.llm.latent_cortex.incumbent_artifact import (
        incumbent_artifact_to_value,
    )
    from core.brain.llm.latent_cortex.task_verifiers import EpisodeTaskVerifier
    from core.brain.reasoning_amplifier_v2 import (
        AmplificationRequest,
        ReasoningAmplifierV2,
    )
    from tools import run_rlc_reconciliation_sweep as sweep
    from tools.rlc_reconciliation_evidence import full_stack_evidence

    objective = task.public.prompt
    domain = task.domain
    incumbent_value = incumbent_artifact_to_value(incumbent_artifact)
    incumbent_text = str(incumbent_value.get("text") or "")
    if not incumbent_text:
        raise sweep.EpisodeFault("complete-system incumbent text is absent")

    first_verifier = EpisodeTaskVerifier(
        objective,
        response_contract=task.public.response_contract,
    )
    first_text, first_receipt = sweep._run_rlc(
        model,
        config,
        prompt_tokens,
        tokenizer,
        wall_clock_s=wall_clock_s,
        verifier=first_verifier,
        objective=objective,
        model_path=model_path,
        incumbent_artifact=incumbent_artifact,
        worker_identity=worker_identity,
        runtime_identity=runtime_identity,
        domain=domain,
    )
    final_rlc_text = first_text
    final_receipt = first_receipt
    acquisition_evidence: dict[str, Any] = {
        "status": "not_requested",
        "request": None,
        "receipt": None,
        "continuation_executed": False,
        "closed_book_external_sources_withheld": True,
    }
    request = build_acquisition_request(
        objective=objective,
        first_text=first_text,
        first_receipt=first_receipt,
        cognitive_context=None,
    )
    if request is not None:
        action = OperationKind(request["action"])
        acquisition_evidence["request"] = request
        if action in COGNITIVE_COMPUTE_ACTIONS:
            from core.brain.cortex_compute_acquisition import acquire_cognitive_compute

            acquisition_started = time.monotonic()
            compute = asyncio.run(
                acquire_cognitive_compute(
                    objective=objective,
                    first_text=first_text,
                    action=action,
                    timeout_s=min(12.0, max(0.1, wall_clock_s * 0.05)),
                )
            )
            ingress_receipt = {
                "schema": "aura.rlc.compute_ingress.v1",
                "compute": compute.receipt,
                "absent_sources": [] if compute.context else ["symbolic_compute"],
            }
            acquisition = build_acquisition_receipt(
                request,
                acquired_context=compute.context,
                ingress_receipt=ingress_receipt,
                elapsed_s=min(30.0, time.monotonic() - acquisition_started),
            )
            validate_acquisition_receipt(acquisition, request=request)
            acquisition_evidence.update(
                {
                    "status": acquisition["status"],
                    "receipt": acquisition,
                    "compute": compute.receipt,
                }
            )
            if compute.context:
                continuation_verifier = EpisodeTaskVerifier(
                    objective,
                    response_contract=task.public.response_contract,
                )
                second_text, second_receipt = sweep._run_rlc(
                    model,
                    config,
                    prompt_tokens,
                    tokenizer,
                    wall_clock_s=wall_clock_s,
                    verifier=continuation_verifier,
                    objective=objective,
                    model_path=model_path,
                    incumbent_artifact=incumbent_artifact,
                    worker_identity=worker_identity,
                    runtime_identity=runtime_identity,
                    domain=domain,
                    cognitive_context=list(compute.context),
                )
                final_rlc_text = second_text
                final_receipt = second_receipt
                acquisition_evidence["continuation_executed"] = True
                continuation = build_continuation_receipt(
                    request,
                    acquisition,
                    first_result={
                        "ok": True,
                        "text": first_text,
                        "receipt": first_receipt,
                    },
                    second_result={
                        "ok": True,
                        "text": second_text,
                        "receipt": second_receipt,
                    },
                    returned_round=2,
                    continuation_reason="second_episode_succeeded",
                )
                validate_continuation_receipt(continuation)
                acquisition_evidence["continuation"] = continuation
                acquisition_evidence["first_receipt_sha256"] = hashlib.sha256(
                    json.dumps(
                        first_receipt,
                        sort_keys=True,
                        separators=(",", ":"),
                        default=str,
                    ).encode()
                ).hexdigest()
        else:
            acquisition_evidence["status"] = "withheld_by_closed_book_contract"
            acquisition_evidence["withheld_action"] = action.value

    generation_calls = 0
    amplifier_generation_resources: list[dict[str, Any]] = []

    async def generate(prompt: str, temperature: float) -> str:
        nonlocal generation_calls
        index = generation_calls
        generation_calls += 1
        text, resource = _run_closed_book_sample(
            model,
            tokenizer,
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=sweep._equal_compute_seed(
                campaign_seed,
                task.task_id,
                1000 + index,
            ),
        )
        amplifier_generation_resources.append(resource)
        return text

    amplifier_verifier = EpisodeTaskVerifier(
        objective,
        response_contract=task.public.response_contract,
    )
    verifier_registry = _ClosedBookVerifierRegistry(
        amplifier_verifier,
        domain=domain,
    )
    seeds = list(dict.fromkeys([incumbent_text, final_rlc_text]))
    amplifier = ReasoningAmplifierV2(generate, verifier=verifier_registry)
    amplified = asyncio.run(
        amplifier.amplify(
            AmplificationRequest(
                objective=objective,
                task_type=_task_type_for_domain(domain),
                time_budget_s=max(30.0, min(900.0, wall_clock_s)),
                sample_budget=3,
                context={
                    "seed_candidates": seeds,
                    "read_only_evaluation": True,
                    "sealed_evaluation": True,
                    "skip_evidence": True,
                    "skip_cache": True,
                    "skip_precompute_enqueue": True,
                    "disable_batched_candidates": True,
                    "generation_max_tokens": max_tokens,
                },
            )
        )
    )
    amplifier_candidate = str(amplified.source_answer or "").strip()
    amplifier_candidate_verifier = EpisodeTaskVerifier(
        objective,
        response_contract=task.public.response_contract,
    )
    amplifier_candidate_evaluation = amplifier_candidate_verifier.evaluate(
        amplifier_candidate,
        _record=False,
    )
    amplifier_candidate_quality = _candidate_quality_assessment(amplifier_candidate_evaluation)
    if bool(amplified.verified) != bool(amplifier_candidate_quality["proxy_admitted"]):
        raise sweep.EpisodeFault(
            "amplifier verdict differs from reconstructed candidate-quality gate"
        )
    promotion_authority = (
        "public_objective_deterministic_execution"
        if amplifier_candidate_quality["exact_public_objective_proof"]
        else "candidate_quality_proxy_not_ground_truth"
    )
    final_text, promotion = _promotion_assessment(
        verifier=EpisodeTaskVerifier(
            objective,
            response_contract=task.public.response_contract,
        ),
        incumbent_text=incumbent_text,
        candidate_text=amplifier_candidate,
        candidate_verified=bool(amplifier_candidate_quality["proxy_admitted"]),
        authority=promotion_authority,
    )
    from core.brain.llm.latent_cortex.resource_accounting import (
        ResourceLedger,
        validate_information_receipt,
        validate_resource_receipt,
    )

    rlc_receipts = [
        first_receipt["budget"]["resource_accounting"],
        *(
            [final_receipt["budget"]["resource_accounting"]]
            if acquisition_evidence["continuation_executed"]
            else []
        ),
    ]
    complete_ledger = ResourceLedger.aggregate([*rlc_receipts, *amplifier_generation_resources])
    verifier_input_bytes = (
        verifier_registry.input_bytes
        + len(amplifier_candidate.encode("utf-8"))
        + len(incumbent_text.encode("utf-8"))
        + (len(amplifier_candidate.encode("utf-8")) if amplifier_candidate else 0)
    )
    verifier_calls = verifier_registry.calls + 2 + int(bool(amplifier_candidate))
    complete_ledger.charge(
        "complete_system_verification_and_promotion",
        verifier_calls=verifier_calls,
        verifier_input_bytes=verifier_input_bytes,
        verifier_output_bytes=(verifier_registry.output_bytes + verifier_calls * 8),
        host_scalar_ops=max(1, verifier_input_bytes * 4),
    )
    if acquisition_evidence.get("request"):
        request_bytes = len(
            json.dumps(
                acquisition_evidence["request"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        result_bytes = len(
            json.dumps(
                acquisition_evidence.get("compute") or acquisition_evidence,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        )
        complete_ledger.charge(
            "cognitive_acquisition",
            tool_calls=1,
            tool_input_bytes=request_bytes,
            tool_result_bytes=result_bytes,
            host_scalar_ops=max(1, request_bytes + result_bytes),
        )
    complete_resource_accounting = validate_resource_receipt(complete_ledger.to_receipt())
    complete_information_accounting = validate_information_receipt(
        final_receipt["budget"]["information_accounting"]
    )
    final_receipt["complete_system_closed_book"] = {
        "schema": "aura.rlc.complete_system_closed_book.v1",
        "contract": "same_information_no_memory_rag_web_or_answer_key",
        "objective": objective,
        "objective_sha256": hashlib.sha256(objective.encode()).hexdigest(),
        "response_contract": task.public.response_contract,
        "single_model_owner": True,
        "first_rlc_runtime": full_stack_evidence(first_receipt),
        "first_rlc_receipt": (
            first_receipt if acquisition_evidence["continuation_executed"] else None
        ),
        "rlc_rounds": 2 if acquisition_evidence["continuation_executed"] else 1,
        "cognitive_acquisition": acquisition_evidence,
        "reasoning_amplifier": amplified.receipt.to_dict(),
        "amplifier_verified": bool(amplified.verified),
        "amplifier_candidate": {
            "schema": "aura.rlc.closed_book_amplifier_candidate.v1",
            "text": amplifier_candidate,
            "text_sha256": hashlib.sha256(amplifier_candidate.encode()).hexdigest(),
            "evaluation": amplifier_candidate_evaluation,
            "quality_assessment": amplifier_candidate_quality,
        },
        "amplifier_verifier_calls": verifier_registry.calls,
        "in_process_generation_calls": generation_calls,
        "seed_candidate_count": len(seeds),
        "resource_accounting": complete_resource_accounting,
        "information_accounting": complete_information_accounting,
        "promotion": promotion,
    }
    return final_text, final_receipt


def _complete_system_evidence(
    receipt: dict[str, Any],
    *,
    engine_evidence: dict[str, Any],
) -> dict[str, Any]:
    """Prove the closed-book arm ran beyond the neural engine boundary."""

    engine = dict(engine_evidence)
    system = receipt.get("complete_system_closed_book") or {}
    acquisition = system.get("cognitive_acquisition") or {}
    amplifier = system.get("reasoning_amplifier") or {}
    promotion = system.get("promotion") or {}
    issues = list(engine.get("issues") or [])

    def require(condition: bool, issue: str) -> None:
        if not condition:
            issues.append(issue)

    require(
        system.get("schema") == "aura.rlc.complete_system_closed_book.v1",
        "complete_system_receipt_absent",
    )
    require(
        system.get("contract") == "same_information_no_memory_rag_web_or_answer_key",
        "closed_book_contract_not_bound",
    )
    objective = str(system.get("objective") or "")
    response_contract = str(system.get("response_contract") or "")
    require(bool(objective), "complete_system_objective_absent")
    require(bool(response_contract), "complete_system_response_contract_absent")
    require(
        system.get("objective_sha256") == hashlib.sha256(objective.encode()).hexdigest(),
        "complete_system_objective_digest_mismatch",
    )
    require(system.get("single_model_owner") is True, "single_model_owner_not_proven")
    first_runtime = system.get("first_rlc_runtime") or {}
    require(first_runtime.get("valid") is True, "first_rlc_round_not_measured")
    status = str(acquisition.get("status") or "")
    require(
        status
        in {
            "not_requested",
            "completed_new_context",
            "completed_no_new_context",
            "withheld_by_closed_book_contract",
        },
        "cognitive_acquisition_not_measured",
    )
    if status == "completed_new_context":
        require(
            acquisition.get("continuation_executed") is True,
            "cognitive_continuation_not_executed",
        )
        require(system.get("rlc_rounds") == 2, "cognitive_continuation_round_count_invalid")
        first_receipt = system.get("first_rlc_receipt")
        require(isinstance(first_receipt, dict), "first_rlc_receipt_absent")
        if isinstance(first_receipt, dict):
            from tools.rlc_reconciliation_evidence import full_stack_evidence

            require(
                full_stack_evidence(first_receipt) == first_runtime,
                "first_rlc_runtime_summary_mismatch",
            )
    else:
        require(system.get("rlc_rounds") == 1, "closed_book_rlc_round_count_invalid")
        require(system.get("first_rlc_receipt") is None, "unexpected_first_rlc_receipt_copy")
        require(first_runtime == engine, "first_rlc_runtime_summary_mismatch")
    if status == "withheld_by_closed_book_contract":
        require(
            acquisition.get("withheld_action") in {"search_memory", "retrieve_evidence"},
            "closed_book_withholding_scope_invalid",
        )
    require(
        acquisition.get("closed_book_external_sources_withheld") is True,
        "external_information_not_sealed",
    )
    require(int(amplifier.get("num_candidates") or 0) > 0, "amplifier_candidates_absent")
    require(
        str(amplifier.get("strategy_used") or "") not in {"", "none"},
        "amplifier_strategy_not_executed",
    )
    require(
        int(system.get("amplifier_verifier_calls") or 0) > 0,
        "amplifier_verifier_not_executed",
    )
    require(int(system.get("seed_candidate_count") or 0) > 0, "amplifier_seed_absent")
    candidate = system.get("amplifier_candidate") or {}
    candidate_text = str(candidate.get("text") or "")
    candidate_quality = candidate.get("quality_assessment") or {}
    from core.brain.llm.latent_cortex.task_verifiers import EpisodeTaskVerifier

    reconstructed_evaluation = EpisodeTaskVerifier(
        objective,
        response_contract=response_contract,
    ).evaluate(candidate_text, _record=False)
    require(
        candidate.get("schema") == "aura.rlc.closed_book_amplifier_candidate.v1",
        "amplifier_candidate_receipt_absent",
    )
    require(
        candidate.get("text_sha256") == hashlib.sha256(candidate_text.encode()).hexdigest(),
        "amplifier_candidate_digest_mismatch",
    )
    require(
        candidate.get("evaluation") == reconstructed_evaluation,
        "amplifier_candidate_evaluation_mismatch",
    )
    require(
        _candidate_quality_assessment(reconstructed_evaluation) == candidate_quality,
        "amplifier_candidate_quality_mismatch",
    )
    require(
        system.get("amplifier_verified") is candidate_quality.get("proxy_admitted"),
        "amplifier_proxy_verdict_mismatch",
    )
    require(
        promotion.get("schema") == "aura.rlc.closed_book_promotion.v1",
        "system_promotion_not_measured",
    )
    require(promotion.get("answer_key_used") is False, "answer_key_contaminated")
    require(
        promotion.get("authority")
        in {
            "candidate_quality_proxy_not_ground_truth",
            "public_objective_deterministic_execution",
        },
        "system_promotion_authority_invalid",
    )
    require(
        promotion.get("candidate_text_sha256") == candidate.get("text_sha256"),
        "system_promotion_candidate_mismatch",
    )
    require(
        promotion.get("decision") in {"retain", "replace"},
        "system_promotion_decision_invalid",
    )
    require(bool(promotion.get("final_text_sha256")), "system_final_answer_unbound")
    from core.brain.llm.latent_cortex.resource_accounting import (
        validate_information_receipt,
        validate_resource_receipt,
    )

    try:
        resource_accounting = validate_resource_receipt(system.get("resource_accounting"))
    except (TypeError, ValueError):
        resource_accounting = {}
        issues.append("complete_system_resource_accounting_invalid")
    try:
        information_accounting = validate_information_receipt(system.get("information_accounting"))
    except (TypeError, ValueError):
        information_accounting = {}
        issues.append("complete_system_information_accounting_invalid")
    require(
        resource_accounting.get("accounting_complete") is True,
        "complete_system_resource_accounting_incomplete",
    )
    require(
        information_accounting.get("accounting_complete") is True,
        "complete_system_information_accounting_incomplete",
    )

    return {
        "valid": not issues,
        "issues": sorted(set(issues)),
        "engine": engine,
        "closed_book_contract": str(system.get("contract") or ""),
        "single_model_owner": system.get("single_model_owner") is True,
        "rlc_rounds": int(system.get("rlc_rounds") or 0),
        "acquisition_status": status,
        "continuation_executed": acquisition.get("continuation_executed") is True,
        "amplifier_mode": str(amplifier.get("mode") or ""),
        "amplifier_strategy": str(amplifier.get("strategy_used") or ""),
        "amplifier_candidates": int(amplifier.get("num_candidates") or 0),
        "amplifier_verified": system.get("amplifier_verified") is True,
        "amplifier_ground_truth_verified": candidate_quality.get("ground_truth_verified") is True,
        "amplifier_verifier_calls": int(system.get("amplifier_verifier_calls") or 0),
        "in_process_generation_calls": int(system.get("in_process_generation_calls") or 0),
        "promotion_decision": str(promotion.get("decision") or ""),
        "promotion_authority": str(promotion.get("authority") or ""),
        "no_regression_guaranteed": promotion.get("no_regression_guaranteed") is True,
        "resource_accounting_sha256": str(resource_accounting.get("receipt_sha256") or ""),
        "information_accounting_sha256": str(information_accounting.get("receipt_sha256") or ""),
        "estimated_flops": resource_accounting.get("estimated_flops"),
        "final_text_sha256": str(promotion.get("final_text_sha256") or ""),
    }


__all__ = [
    "_candidate_quality_assessment",
    "_promotion_assessment",
    "_complete_system_evidence",
    "_run_complete_system_closed_book",
]
