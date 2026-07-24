from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from core.brain.llm.latent_cortex.bidirectional_reflector import (
    observe_reflector_vectors,
)
from core.learning.adversarial_verifier_curriculum import (
    AdversarialTraceCapture,
    AdversarialVerifierCurriculum,
    VerifiedAdversarialPair,
    VerifiedNegativeStore,
)
from core.learning.recurrence_curriculum import TASK_GENERATORS


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _observation(
    step: int,
    prior: list[float],
    proposal: list[float],
) -> dict:
    return observe_reflector_vectors(
        prior,
        proposal,
        proposal,
        branch_index=0,
        branch_step=step,
        prior_state_sha256=_digest(f"prior-{step}-{prior}"),
        proposal_state_sha256=_digest(f"proposal-{step}-{proposal}"),
        admitted_state_sha256=_digest(f"proposal-{step}-{proposal}"),
        accepted=True,
    )


def _capture(
    relation: str,
    family: str,
    index: int,
    *,
    mutation_family: str | None,
    error_index: int,
    candidate: int,
    repeat: int = 0,
) -> AdversarialTraceCapture:
    seed = 10_000 + {"train": 0, "in_domain": 10_000, "out_of_domain": 20_000}[relation] + index
    task = TASK_GENERATORS[family](4, seed)
    prefix = f"{relation}-{family}-{index}"
    rows = []
    for step in range(4):
        prior = [1.0 + index * 0.01, 1.2 + step * 0.02, 0.8, 1.1]
        proposal = [prior[0] + 0.01, prior[1] - 0.01, prior[2] + 0.005, prior[3]]
        if mutation_family is not None and step >= error_index:
            if step == error_index:
                if mutation_family in {"premise_flip", "sign_inversion"}:
                    proposal[0] += 0.12 + candidate * 0.005
                else:
                    proposal[1] -= 0.12 + candidate * 0.005
            else:
                prior[2] += 0.03 * (step - error_index)
                proposal[2] += 0.03 * (step - error_index)
        rows.append(
            _observation(
                step,
                [round(value, 8) for value in prior],
                [round(value, 8) for value in proposal],
            )
        )
    variant = "clean" if mutation_family is None else f"{mutation_family}-{candidate}"
    return AdversarialTraceCapture(
        capture_id=f"{prefix}-{variant}-repeat-{repeat}",
        relation=relation,
        family=family,
        depth=4,
        seed=seed,
        execution_seed=77 + index,
        model_stack_sha256=_digest("model-stack"),
        schedule_sha256=_digest("schedule"),
        config_sha256=_digest("config"),
        source_manifest_sha256=_digest("source-manifest"),
        answer=(task.answer if mutation_family is None else 'FINAL_ANSWER: {"wrong":1}'),
        observations=tuple(rows),
    )


def _pool() -> list[VerifiedAdversarialPair]:
    result = []
    domains = {
        "train": ("modular", "boolean"),
        "in_domain": ("modular", "boolean"),
        "out_of_domain": ("khop", "code_trace"),
    }
    for relation, families in domains.items():
        for index in range(8):
            family = families[index % 2]
            clean = _capture(
                relation,
                family,
                index,
                mutation_family=None,
                error_index=index % 4,
                candidate=0,
            )
            clean_repeat = _capture(
                relation,
                family,
                index,
                mutation_family=None,
                error_index=index % 4,
                candidate=0,
                repeat=1,
            )
            mutation_families = (
                ("sign_inversion", "dependency_skip")
                if relation == "out_of_domain"
                else ("premise_flip", "operator_swap")
            )
            for candidate, mutation_family in enumerate(mutation_families):
                mutant = _capture(
                    relation,
                    family,
                    index,
                    mutation_family=mutation_family,
                    error_index=index % 4,
                    candidate=candidate,
                )
                mutant_repeat = _capture(
                    relation,
                    family,
                    index,
                    mutation_family=mutation_family,
                    error_index=index % 4,
                    candidate=candidate,
                    repeat=1,
                )
                result.append(
                    VerifiedAdversarialPair(
                        pair_id=f"{relation}-{index}-{candidate}",
                        mutation_family=mutation_family,
                        error_index=index % 4,
                        clean=clean,
                        mutant=mutant,
                        clean_repeat=clean_repeat,
                        mutant_repeat=mutant_repeat,
                    )
                )
    return result


def test_pair_requires_independent_failure_matching_controls_and_subtle_first_divergence():
    pair = _pool()[0]
    assert pair.clean.correct is True
    assert pair.mutant.correct is False
    assert pair.relative_delta < pair.max_relative_delta
    clean_rows, mutant_rows = pair.examples()
    assert all(row.error_index is None for row in clean_rows)
    assert mutant_rows[pair.error_index].is_error
    assert VerifiedAdversarialPair.from_dict(pair.to_input_dict()) == pair

    with pytest.raises(ValueError, match="controls differ"):
        replace(pair, mutant=replace(pair.mutant, execution_seed=999))
    with pytest.raises(ValueError, match="repeat-determinism"):
        replace(
            pair,
            mutant_repeat=replace(
                pair.mutant_repeat,
                answer='FINAL_ANSWER: {"different_wrong":2}',
            ),
        )
    with pytest.raises(ValueError, match="subtle first divergence"):
        replace(pair, error_index=(pair.error_index + 1) % 4)


def test_curriculum_coadapts_under_frozen_sandboxed_ood_and_retains_only_train_misses(
    tmp_path: Path,
):
    store = VerifiedNegativeStore(tmp_path / "negative-store")
    try:
        result = AdversarialVerifierCurriculum(rounds=2, seed=19).run(
            _pool(),
            repo_root=Path(__file__).parents[1],
            negative_store=store,
        )
        report = result.report
        assert result.head.manifest()["input_representation"] == "reflector_hidden_sketch_v1"
        assert result.head.admitted is True
        assert report["heldout_frozen_before_training"] is True
        assert report["heldout_used_for_weight_updates"] is False
        assert report["sandbox_evaluation"]["sandbox_contract"] == {
            "network": "denied",
            "file_write": "denied",
            "head_frozen": True,
            "training_examples_available": False,
        }
        assert report["retained_negative_store_verified"] is True
        assert report["claims"]["resident_32b_reasoning_gain"] is False
        assert set(report["sandbox_evaluation"]["evaluation"]["by_mutation_family"]) == {
            "clean_sham",
            "dependency_skip",
            "sign_inversion",
        }
        assert len(report["round_history"]) == 2
        assert report["round_history"][0]["policy"]["policy_sha256"]
        assert (
            report["report_sha256"]
            == hashlib.sha256(
                json.dumps(
                    {key: report[key] for key in report if key != "report_sha256"},
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ).encode("ascii")
            ).hexdigest()
        )
    finally:
        store.close()


def test_negative_store_rejects_heldout_and_detects_record_tampering(tmp_path: Path):
    pair = _pool()[0]
    _clean, mutant = pair.examples()
    store = VerifiedNegativeStore(tmp_path / "store")
    try:
        record_id = store.append(
            pair,
            mutant,
            error_probability=0.2,
            threshold=0.5,
            round_index=0,
        )
        assert store.verify() == (True, [])
        with pytest.raises(ValueError, match="do not match"):
            store.append(
                pair,
                mutant[:-1],
                error_probability=0.2,
                threshold=0.5,
                round_index=0,
            )
        heldout = next(item for item in _pool() if item.clean.relation == "out_of_domain")
        with pytest.raises(ValueError, match="held-out evidence"):
            store.append(
                heldout,
                heldout.examples()[1],
                error_probability=0.2,
                threshold=0.5,
                round_index=0,
            )
        path = tmp_path / "store" / "records" / f"{record_id}.json"
        path.write_bytes(path.read_bytes().replace(b'"round_index":0', b'"round_index":1'))
        ok, problems = store.verify()
        assert ok is False
        assert problems
    finally:
        store.close()


def test_reflector_sketch_head_runs_on_live_tensor_observation():
    mx = pytest.importorskip("mlx.core")
    from core.brain.llm.latent_cortex.mistake_locator import MistakeLocatorRuntime
    from core.learning.mistake_locator import MistakeLocatorHead

    pool = _pool()

    def examples(relation: str):
        rows = []
        selected = {}
        for pair in pool:
            if pair.clean.relation == relation and pair.clean.task_id not in selected:
                selected[pair.clean.task_id] = pair
        for pair in selected.values():
            clean, mutant = pair.examples()
            rows.extend(clean)
            rows.extend(mutant)
        return rows

    head = MistakeLocatorHead.fit(
        examples("train"),
        examples("in_domain"),
        examples("out_of_domain"),
        hidden_width=8,
        steps=100,
        input_representation="reflector_hidden_sketch_v1",
    )
    runtime = MistakeLocatorRuntime(
        mode="learned",
        head=head,
        head_sha256=_digest("head"),
    )
    prior = mx.array([[[0.9, 1.1, 0.8, 1.0], [1.1, 1.3, 0.9, 1.2]]])
    proposal = prior + mx.array([[[0.01, -0.01, 0.005, 0.0], [0.01, -0.01, 0.005, 0.0]]])
    observation = runtime.observe(
        prior,
        proposal,
        branch_index=0,
        branch_step=0,
        prior_state_sha256=_digest("prior"),
        proposal_state_sha256=_digest("proposal"),
        admitted_state_sha256=_digest("proposal"),
        accepted=True,
    )
    assert len(observation["prior_pooled_hidden"]) == 8
    assert observation["error_probability"] == pytest.approx(
        head.probability(
            observation["prior_pooled_hidden"],
            observation["proposal_pooled_hidden"],
        )
    )


def test_training_cli_consumes_round_trip_capture_bundle(tmp_path: Path):
    from core.learning.mistake_locator import MistakeLocatorHead
    from core.runtime.subprocess_gateway import get_subprocess_gateway

    source = tmp_path / "pairs.json"
    head_path = tmp_path / "head.json"
    report_path = tmp_path / "report.json"
    source.write_text(
        json.dumps(
            {
                "schema": "aura.rlc.adversarial_curriculum_input.v1",
                "pairs": [pair.to_input_dict() for pair in _pool()],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    result = get_subprocess_gateway().run(
        [
            sys.executable,
            "tools/train_adversarial_verifier.py",
            "--input",
            str(source),
            "--head",
            str(head_path),
            "--report",
            str(report_path),
            "--negative-store",
            str(tmp_path / "negatives"),
            "--rounds",
            "1",
        ],
        cwd=Path(__file__).parents[1],
        timeout=60.0,
        capture_output=True,
        offline_tooling=True,
        source="training_tooling:test_adversarial_verifier_cli",
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["admitted"] is True
    loaded = MistakeLocatorHead.load(
        head_path,
        expected_sha256=summary["head_sha256"],
    )
    assert loaded.manifest()["input_representation"] == "reflector_hidden_sketch_v1"
    envelope = json.loads(report_path.read_text(encoding="utf-8"))
    assert envelope["payload"]["sandbox_evaluation"]["sandbox_contract"]["network"] == "denied"
