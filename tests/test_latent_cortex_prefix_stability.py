from __future__ import annotations

import copy
import hashlib
import json
import re

import pytest

from core.brain.llm.latent_cortex.prefix_stability import (
    PREFIX_STABILITY_CONTEXT_SCHEMA,
    build_prefix_prompt,
    run_prefix_stability_verifier,
    validate_prefix_stability_envelope,
)
from core.learning.prefix_stability import (
    CALIBRATION_TARGET,
    PrefixStabilityCalibrator,
    PrefixStabilityExample,
)


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _receipt_sha(value: dict) -> str:
    payload = {key: value[key] for key in value if key != "receipt_sha256"}
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _context(seed: int, temperature: float, top_p: float) -> dict:
    return {
        "schema": PREFIX_STABILITY_CONTEXT_SCHEMA,
        "prompt_token_count": 40,
        "generated_token_count": 20,
        "termination": "contract_complete",
        "initial_cache_offsets": [0, 0, 0],
        "final_cache_offsets": [60, 60, 60],
        "all_initial_offsets_zero": True,
        "solver_context_imported": False,
        "parameter_relation": "shared_resident_checkpoint",
        "sample_seed": seed,
        "temperature": temperature,
        "top_p": top_p,
    }


def _generator(conclusions: list[str]):
    calls: list[tuple[str, int, float, float]] = []

    def generate(prompt: str, seed: int, temperature: float, top_p: float) -> dict:
        index = len(calls)
        calls.append((prompt, seed, temperature, top_p))
        candidate = re.search(r"CANDIDATE_SHA256: ([0-9a-f]{64})", prompt)
        prefix = re.search(r"VERIFIED_PREFIX_SHA256: ([0-9a-f]{64})", prompt)
        assert candidate and prefix
        payload = {
            "candidate_sha256": candidate.group(1),
            "prefix_sha256": prefix.group(1),
            "conclusion": conclusions[index],
        }
        return {
            "text": "FINAL_ANSWER: " + json.dumps(payload, separators=(",", ":")),
            "context": _context(seed, temperature, top_p),
        }

    return generate, calls


def _candidate() -> str:
    return "The premise is 2 + 2 = 4. Therefore the result is 3 + 3 = 6."


def _examples(prefix: str, *, calibration: bool) -> list[PrefixStabilityExample]:
    examples: list[PrefixStabilityExample] = []
    for index in range(32):
        raw = index / 31
        match = index >= 16
        identity = f"{prefix}-{index}"
        examples.append(
            PrefixStabilityExample(
                example_id=identity,
                task_id=f"{prefix}-task-{index % 8}",
                domain="arithmetic",
                raw_stability=raw,
                future_conclusion_match=match,
                probe_receipt_sha256=_sha_text(f"{identity}-probe"),
                future_receipt_sha256=_sha_text(f"{identity}-future"),
            )
        )
    assert calibration is (prefix == "cal")
    return examples


def test_prompt_withholds_source_conclusion_and_binds_prefix():
    prompt = build_prefix_prompt(
        objective="Compute both equalities.",
        candidate_sha256="a" * 64,
        prefix="The premise is 2 + 2 = 4.",
        prefix_sha256="b" * 64,
    )
    assert "3 + 3 = 6" not in prompt
    assert "no access to the source conclusion" in prompt
    assert "a" * 64 in prompt
    assert "b" * 64 in prompt


def test_complete_prefix_recurrence_is_measured_without_authority():
    generate, calls = _generator(["3 + 3 = 6"] * 3)
    receipt = run_prefix_stability_verifier(
        _candidate(),
        objective="Check the arithmetic chain.",
        generate=generate,
    )
    assert len(calls) == 3
    assert len({call[1] for call in calls}) == 3
    assert all(call[2:] == (0.35, 0.9) for call in calls)
    assert receipt["measurement_admitted"] is True
    assert receipt["metrics"]["raw_stability"] == 1.0
    assert receipt["metrics"]["normalized_entropy"] == 0.0
    assert receipt["selection_authority_admitted"] is False
    assert receipt["correctness_authority_admitted"] is False
    assert receipt["calibration"]["target"] == CALIBRATION_TARGET
    assert receipt["calibration"]["future_recurrence_probability"] is None
    assert validate_prefix_stability_envelope(receipt) == receipt


def test_disagreement_is_conservatively_combined_and_not_called_correctness():
    generate, _calls = _generator(["3 + 3 = 6", "3 + 3 = 6", "3 + 3 = 7"])
    receipt = run_prefix_stability_verifier(
        _candidate(),
        objective="Check the arithmetic chain.",
        generate=generate,
    )
    assert receipt["metrics"]["reference_agreement"] == pytest.approx(2 / 3)
    assert receipt["metrics"]["pairwise_agreement"] == pytest.approx(1 / 3)
    assert receipt["metrics"]["modal_fraction"] == pytest.approx(2 / 3)
    assert receipt["metrics"]["raw_stability"] == pytest.approx(1 / 3)
    encoded = json.dumps(receipt, sort_keys=True)
    assert "correctness_probability" not in encoded
    assert "diagnostic_conclusion_recurrence_only" in encoded


def test_unverified_prefix_spends_no_generation():
    called = False

    def generate(_prompt: str, _seed: int, _temperature: float, _top_p: float) -> dict:
        nonlocal called
        called = True
        raise AssertionError("unverified prose prefix must not regenerate")

    receipt = run_prefix_stability_verifier(
        "The sky appears blue. Therefore the weather is pleasant.",
        objective="Describe the observation.",
        generate=generate,
    )
    assert called is False
    assert receipt == {
        "requested": True,
        "available": False,
        "reason": "verified_prefix_unavailable",
        "selection_effect": "none",
        "correctness_effect": "none",
    }


def test_partial_samples_withhold_metric_instead_of_cherry_picking():
    generate, _calls = _generator(["3 + 3 = 6"] * 3)
    calls = 0

    def fail_middle(prompt: str, seed: int, temperature: float, top_p: float) -> dict:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("sample failed")
        return generate(prompt, seed, temperature, top_p)

    receipt = run_prefix_stability_verifier(
        _candidate(),
        objective="Check the arithmetic chain.",
        generate=fail_middle,
    )
    assert receipt["measurement_admitted"] is False
    assert receipt["metrics"]["sample_count"] == 2
    assert receipt["metrics"]["raw_stability"] is None
    assert receipt["calibration"]["reason"] == "measurement_unavailable"
    assert validate_prefix_stability_envelope(receipt) == receipt


def test_contract_refusal_retains_bounded_generation_evidence():
    generate, _calls = _generator(["3 + 3 = 6"] * 3)
    calls = 0

    def malformed(prompt: str, seed: int, temperature: float, top_p: float) -> dict:
        nonlocal calls
        calls += 1
        if calls == 2:
            return {
                "text": "not the required contract",
                "context": _context(seed, temperature, top_p),
            }
        return generate(prompt, seed, temperature, top_p)

    receipt = run_prefix_stability_verifier(
        _candidate(),
        objective="Check the arithmetic chain.",
        generate=malformed,
    )
    refused = receipt["samples"][1]
    assert refused["status"] == "contract_refused"
    assert refused["generated_text"] == "not the required contract"
    assert refused["generated_text_sha256"] == _sha_text(refused["generated_text"])
    assert refused["context"]["sample_seed"] == refused["sample_seed"]
    assert receipt["measurement_admitted"] is False
    assert validate_prefix_stability_envelope(receipt) == receipt


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda value: value["samples"][0]["context"].update({"sample_seed": 7}),
            "context isolation",
        ),
        (
            lambda value: value["metrics"].update({"raw_stability": 0.99}),
            "metric",
        ),
        (
            lambda value: value.update({"correctness_authority_admitted": True}),
            "authority boundary",
        ),
    ],
)
def test_recommitted_prefix_stability_tampering_fails_closed(mutation, match):
    generate, _calls = _generator(["3 + 3 = 6"] * 3)
    receipt = run_prefix_stability_verifier(
        _candidate(),
        objective="Check the arithmetic chain.",
        generate=generate,
    )
    forged = copy.deepcopy(receipt)
    mutation(forged)
    forged["receipt_sha256"] = _receipt_sha(forged)
    with pytest.raises(ValueError, match=match):
        validate_prefix_stability_envelope(forged)


def test_calibrator_uses_disjoint_future_recurrence_evidence_and_round_trips(tmp_path):
    fit = _examples("fit", calibration=False)
    calibration = _examples("cal", calibration=True)
    calibrator = PrefixStabilityCalibrator.fit(fit, calibration)
    assert calibrator.admitted is True
    assert calibrator.probability(0.1) == 0.0
    assert calibrator.probability(0.9) == 1.0
    encoded = json.dumps(calibrator.to_payload(), sort_keys=True)
    assert '"correct":' not in encoded
    assert "correctness_probability" not in encoded
    assert CALIBRATION_TARGET in encoded
    target = tmp_path / "prefix-stability.json"
    digest = calibrator.save(target)
    loaded = PrefixStabilityCalibrator.load(target, expected_sha256=digest)
    assert loaded.to_payload() == calibrator.to_payload()


def test_calibrator_rejects_task_or_evidence_leakage():
    fit = _examples("fit", calibration=False)
    calibration = _examples("cal", calibration=True)
    leaked_task = [
        (
            PrefixStabilityExample(
                example_id=example.example_id,
                task_id=fit[0].task_id if index == 0 else example.task_id,
                domain=example.domain,
                raw_stability=example.raw_stability,
                future_conclusion_match=example.future_conclusion_match,
                probe_receipt_sha256=example.probe_receipt_sha256,
                future_receipt_sha256=example.future_receipt_sha256,
            )
        )
        for index, example in enumerate(calibration)
    ]
    with pytest.raises(ValueError, match="overlap"):
        PrefixStabilityCalibrator.fit(fit, leaked_task)

    leaked_evidence = list(calibration)
    source = leaked_evidence[0]
    leaked_evidence[0] = PrefixStabilityExample(
        example_id=source.example_id,
        task_id=source.task_id,
        domain=source.domain,
        raw_stability=source.raw_stability,
        future_conclusion_match=source.future_conclusion_match,
        probe_receipt_sha256=fit[0].probe_receipt_sha256,
        future_receipt_sha256=source.future_receipt_sha256,
    )
    with pytest.raises(ValueError, match="overlap"):
        PrefixStabilityCalibrator.fit(fit, leaked_evidence)


def test_runtime_applies_only_pinned_admitted_recurrence_calibration(tmp_path):
    calibrator = PrefixStabilityCalibrator.fit(
        _examples("fit", calibration=False),
        _examples("cal", calibration=True),
    )
    path = tmp_path / "prefix-stability.json"
    digest = calibrator.save(path)
    config = {
        "mode": "learned",
        "artifact_path": str(path),
        "artifact_sha256": digest,
    }
    generate, _calls = _generator(["3 + 3 = 6"] * 3)
    receipt = run_prefix_stability_verifier(
        _candidate(),
        objective="Check the arithmetic chain.",
        generate=generate,
        calibrator_config=config,
    )
    assert receipt["calibration"]["calibrated"] is True
    assert receipt["calibration"]["future_recurrence_probability"] == 1.0
    assert validate_prefix_stability_envelope(
        receipt,
        expected_calibrator_config=config,
    ) == receipt
    with pytest.raises(ValueError, match="calibration"):
        validate_prefix_stability_envelope(receipt)


def test_service_reconstructs_prefix_stability_and_rejects_tampering():
    from core.brain.latent_cortex_service import LatentCortexService

    generate, _calls = _generator(["3 + 3 = 6"] * 3)
    prefix = run_prefix_stability_verifier(
        _candidate(),
        objective="Check the arithmetic chain.",
        generate=generate,
    )
    config = {
        "prefix_stability_enabled": True,
        "prefix_stability_calibrator": None,
        "generative_verifier_enabled": False,
        "counterfactual_verifier_enabled": False,
    }
    errors = LatentCortexService._receipt_contract_errors(
        {"prefix_stability": prefix},
        config,
    )
    assert "prefix_stability_unproven" not in errors
    forged = copy.deepcopy(prefix)
    forged["samples"][0]["context"]["solver_context_imported"] = True
    forged["receipt_sha256"] = _receipt_sha(forged)
    errors = LatentCortexService._receipt_contract_errors(
        {"prefix_stability": forged},
        config,
    )
    assert "prefix_stability_unproven" in errors


def test_training_cli_emits_runtime_pinned_artifact_and_report(tmp_path):
    from tools.train_prefix_stability_calibrator import train

    fit_path = tmp_path / "fit.jsonl"
    calibration_path = tmp_path / "calibration.jsonl"
    artifact_path = tmp_path / "calibrator.json"
    report_path = tmp_path / "report.json"
    fit_path.write_text(
        "\n".join(
            json.dumps(example.to_dict(), sort_keys=True)
            for example in _examples("fit", calibration=False)
        )
        + "\n",
        encoding="ascii",
    )
    calibration_path.write_text(
        "\n".join(
            json.dumps(example.to_dict(), sort_keys=True)
            for example in _examples("cal", calibration=True)
        )
        + "\n",
        encoding="ascii",
    )
    report = train(
        fit_path=fit_path,
        calibration_path=calibration_path,
        output_path=artifact_path,
        report_path=report_path,
    )
    assert report["runtime_eligible"] is True
    assert report["target"] == CALIBRATION_TARGET
    assert hashlib.sha256(artifact_path.read_bytes()).hexdigest() == report["artifact_sha256"]
    assert json.loads(report_path.read_text(encoding="ascii")) == report


def test_training_cli_rejects_duplicate_json_keys(tmp_path):
    from tools.train_prefix_stability_calibrator import _load_examples

    source = tmp_path / "duplicate.jsonl"
    source.write_text('{"schema":"a","schema":"b"}\n', encoding="ascii")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        _load_examples(source)
