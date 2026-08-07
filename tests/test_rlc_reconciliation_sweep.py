"""Contracts for the frozen-checkpoint reconciliation sweep.

The sweep decides whether a twelve-hour resident run is worth launching, so
its journal, resumption, and grading must be correct before it consumes any
32B time.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS = REPO_ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_rlc_reconciliation_sweep as sweep  # noqa: E402


def test_the_full_stack_arm_enables_every_pillar_that_was_built():
    """Every run before 2026-08-07 measured recurrence with hidden-state
    optimization, fast weights and adaptive halting switched OFF, and reported
    the result as a property of "the recurrent path". The program's own claim
    is that reasoning is a unified system, so the arm under test has to be the
    union, not one component of it."""
    cfg = sweep._build_config(8, 16, "applied", 512, profile="full")
    assert cfg.latent_opt.enabled is True, "hidden-state optimization"
    assert cfg.fast_weights.enabled is True, "temporary fast weights"
    assert cfg.local_repair_enabled is True, "local repair"
    assert cfg.answer_replacement_enabled is True, "evidence-bound acceptance"
    assert cfg.generative_verifier_enabled is True
    assert cfg.counterfactual_verifier_enabled is True
    assert cfg.prefix_stability_enabled is True
    # Adaptive halting: the depth is a ceiling, not a floor.
    assert cfg.recurrence.min_steps == 2
    assert cfg.recurrence.max_steps == 8
    assert cfg.recurrence.fixed_depth is False


def test_the_mechanism_arm_stays_an_ablation():
    """The stripped configuration is retained, but only underneath the full
    arm, and it must not silently acquire the pillars."""
    cfg = sweep._build_config(4, 16, "suppressed", 512, profile="mechanism")
    assert cfg.latent_opt.enabled is False
    assert cfg.fast_weights.enabled is False
    assert cfg.local_repair_enabled is False
    assert cfg.answer_replacement_enabled is False
    # Forced depth: no early halting, which is what makes it an ablation.
    assert cfg.recurrence.min_steps == cfg.recurrence.max_steps == 4
    assert cfg.recurrence.fixed_depth is True


def test_the_battery_leads_with_the_unified_system_not_the_ablation():
    by_name = {a.name: a for a in sweep.ARMS}
    assert by_name["vanilla"].profile == "ordinary"
    assert by_name["full_stack"].profile == "full"
    assert by_name["full_stack_oracle"].profile == "full_oracle"
    assert by_name["rlc_mechanism"].profile == "mechanism"
    # The control and the unified arm must share a decode budget, or the
    # contrast measures the budget instead of the system.
    assert by_name["vanilla"].max_tokens == by_name["full_stack"].max_tokens
    # The oracle arm is a diagnostic ceiling and is never promotable.
    assert "oracle" in by_name["full_stack_oracle"].name


def test_config_carries_the_arm_policy_and_validates():
    for steps, policy in ((4, "applied"), (1, "suppressed")):
        config = sweep._build_config(steps, 16, policy, 320, profile="mechanism")
        assert config.validate() == []
        assert config.terminal_instruction_policy == policy
        assert config.recurrence.max_steps == steps
        # The bridge stays off in every arm so the only prefix difference
        # between arms is the disposition being measured.
        assert config.decode_bridge_policy == "none"


def test_journal_resumes_and_ignores_a_torn_final_line(tmp_path: Path):
    path = tmp_path / "journal.jsonl"
    journal = sweep.Journal(path)
    journal.append({"event": "CELL", "arm": "vanilla", "task_id": "t1", "text": "x"})
    journal.append({"event": "CELL", "arm": "vanilla", "task_id": "t2", "text": "y"})
    # Simulate a hard kill mid-write.
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"event": "CELL", "arm": "vanil')

    resumed = sweep.Journal(path)
    assert ("vanilla", "t1") in resumed.done
    assert ("vanilla", "t2") in resumed.done
    assert len(resumed.done) == 2
    assert len(resumed.cells()) == 2


def test_grade_refuses_to_credit_a_harness_fault_as_a_wrong_answer(tmp_path: Path):
    """A crashed cell is an error, never a scored zero."""
    from core.brain.llm.latent_cortex import frontier_tasks as ft

    tasks = ft.generate_task_battery([20260807], difficulty=2)
    task = tasks[0]

    journal = sweep.Journal(tmp_path / "journal.jsonl")
    journal.append(
        {
            "event": "CELL",
            "arm": "rlc_asrun",
            "task_id": task.task_id,
            "domain": task.domain,
            "text": "",
            "error": "RuntimeError: worker died",
        }
    )
    verdict = sweep.grade(tmp_path, [task])
    bucket = verdict["arms"]["rlc_asrun"]
    assert bucket["errors"] == 1
    assert bucket["total"] == 1
    # The failure did not become a graded observation.
    assert bucket["correct"] == 0
    assert bucket["reasons"] == {}


def test_grade_decides_against_the_ordinary_decode_not_against_itself(tmp_path: Path):
    """Parity is measured against vanilla. A recurrent arm cannot grade itself."""
    from core.brain.llm.latent_cortex import frontier_tasks as ft

    tasks = ft.generate_task_battery([20260807], difficulty=2)
    journal = sweep.Journal(tmp_path / "journal.jsonl")

    for task in tasks:
        reveal = task.reveal_for_verifier()
        correct_line = "FINAL_ANSWER: " + json.dumps(reveal["expected"])
        # Vanilla answers everything correctly; the recurrent arm answers none.
        journal.append(
            {
                "event": "CELL",
                "arm": "vanilla",
                "task_id": task.task_id,
                "domain": task.domain,
                "text": correct_line,
                "error": "",
            }
        )
        journal.append(
            {
                "event": "CELL",
                "arm": "rlc_asrun",
                "task_id": task.task_id,
                "domain": task.domain,
                "text": 'FINAL_ANSWER: {"wrong": 1}',
                "error": "",
            }
        )

    verdict = sweep.grade(tmp_path, tasks)
    assert verdict["vanilla_correct"] == len(tasks)
    assert verdict["best_recurrent_correct"] == 0
    assert verdict["reaches_parity_with_ordinary_decode"] is False
    assert verdict["decision"] == "recurrent_path_below_ordinary_decode"
    # A negative sweep never authorizes downstream work.
    assert verdict["claims"]["fusion_authorized"] is False
    assert verdict["claims"]["reasoning_gain_proven"] is False


def test_grade_promotes_only_when_a_recurrent_arm_reaches_parity(tmp_path: Path):
    from core.brain.llm.latent_cortex import frontier_tasks as ft

    tasks = ft.generate_task_battery([20260807], difficulty=2)
    journal = sweep.Journal(tmp_path / "journal.jsonl")
    for task in tasks:
        reveal = task.reveal_for_verifier()
        correct_line = "FINAL_ANSWER: " + json.dumps(reveal["expected"])
        for arm in ("vanilla", "rlc_nodisp"):
            journal.append(
                {
                    "event": "CELL",
                    "arm": arm,
                    "task_id": task.task_id,
                    "domain": task.domain,
                    "text": correct_line,
                    "error": "",
                }
            )
    verdict = sweep.grade(tmp_path, tasks)
    assert verdict["reaches_parity_with_ordinary_decode"] is True
    assert verdict["decision"] == "proceed_to_checkpoint_phase"
    # Parity on a frozen path still proves no gain and authorizes no fusion.
    assert verdict["claims"]["fusion_authorized"] is False


class _StubTokenizer:
    """Enough tokenizer to make the disposition text real tokens."""

    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):  # noqa: D102, ARG002
        return [1 + (ord(ch) % 100) for ch in text[:64]]

    def decode(self, tokens):  # noqa: D102
        return "".join(chr(65 + (int(t) % 26)) for t in tokens)


def _tiny_model():
    mx = pytest.importorskip("mlx.core")
    from mlx_lm.models.qwen2 import Model, ModelArgs

    args = ModelArgs(
        model_type="qwen2",
        hidden_size=64,
        num_hidden_layers=8,
        intermediate_size=128,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=128,
        num_key_value_heads=2,
        max_position_embeddings=512,
        rope_theta=10000.0,
    )
    model = Model(args)
    mx.eval(model.parameters())
    return model


def test_disposition_injection_is_real_when_applied_and_absent_when_suppressed():
    """The load-bearing claim: the arms differ by exactly this injection.

    With no tokenizer the engine cannot encode the disposition at all, so a
    test that omits one proves nothing -- both arms trivially report zero.
    """
    model = _tiny_model()

    applied_config = sweep._build_config(2, 4, "applied", 8, decode_contract="none")
    _, applied = sweep._run_rlc(model, applied_config, [1, 2, 3, 4, 5], _StubTokenizer())

    suppressed_config = sweep._build_config(
        2, 4, "suppressed", 8, decode_contract="none"
    )
    _, suppressed = sweep._run_rlc(
        model, suppressed_config, [1, 2, 3, 4, 5], _StubTokenizer()
    )

    applied_prefix = applied["decode_prefix_composition"]
    suppressed_prefix = suppressed["decode_prefix_composition"]

    assert applied_prefix["terminal_instruction_tokens"] > 0
    assert suppressed_prefix["terminal_instruction_tokens"] == 0
    assert applied["decode_prefix_token_count"] > 0
    assert suppressed["decode_prefix_token_count"] == 0
    # Neither arm configured a bridge policy, so neither may claim one.
    assert applied["decode_bridge_applied"] is False
    assert suppressed["decode_bridge_applied"] is False
    assert applied["decode_bridge_token_count"] == 0


def _run_with_dead_engine(model, config, reason: str, termination: str):
    class _DeadResult:
        ok = False
        tokens: list[int] = []
        text = ""

        def __init__(self):
            self.reason = reason

        class receipt:  # noqa: N801
            @staticmethod
            def to_dict():
                return {"decode_termination": termination}

    class _DeadEngine:
        def __init__(self, *args, **kwargs):
            pass

        def reason(self, **kwargs):  # noqa: ARG002
            return _DeadResult()

    import core.brain.llm.latent_cortex.engine as engine_module

    original = engine_module.LatentCortexEngine
    engine_module.LatentCortexEngine = _DeadEngine
    try:
        return sweep._run_rlc(model, config, [1, 2, 3], _StubTokenizer())
    finally:
        engine_module.LatentCortexEngine = original


def test_infrastructure_failure_raises_instead_of_scoring_zero():
    """A broken harness must not be gradeable as a wrong answer.

    The first live sweep died on the engine's default 120s episode wall clock,
    which is smaller than these episodes need -- the 2026-08-06 campaign's
    median recurrent episode ran 298s.
    """
    model = _tiny_model()
    config = sweep._build_config(2, 4, "applied", 8, decode_contract="none")

    with pytest.raises(sweep.EpisodeFault) as excinfo:
        _run_with_dead_engine(
            model, config, "latent_phase_failed:ValueError:boom", "not_reached"
        )
    assert "latent_phase_failed" in str(excinfo.value)


def test_a_model_that_cannot_finish_its_answer_is_scored_not_excluded():
    """An unfinished decode is the arm failing to answer, which is a result.

    CP420S12 settled this: bounded abstentions and incomplete decodes are
    scored as incorrect policy observations, while cancellation, latent-phase,
    worker and invariant failures stay fatal. The 2026-08-06 base_rlc arm
    carried nine such policy failures out of 28, so excluding them would
    flatter the recurrent path rather than measure it.
    """
    model = _tiny_model()
    config = sweep._build_config(2, 4, "applied", 8, decode_contract="none")

    text, receipt = _run_with_dead_engine(
        model,
        config,
        "decode_incomplete:contract_irrecoverable",
        "contract_irrecoverable",
    )
    assert text == ""
    assert receipt["decode_termination"] == "contract_irrecoverable"

    assert sweep._is_policy_failure("decode_incomplete:x", "contract_irrecoverable")
    assert sweep._is_policy_failure("", "token_limit_contract_incomplete")
    # Infrastructure never counts as policy, even when it mentions a budget.
    assert not sweep._is_policy_failure("latent_phase_failed:budget_exhausted", "")
    assert not sweep._is_policy_failure("worker_died", "budget_exhausted")


def test_a_faulted_arm_makes_the_sweep_inconclusive_not_negative(tmp_path: Path):
    """An arm that did not run has not lost. It has not been measured."""
    from core.brain.llm.latent_cortex import frontier_tasks as ft

    tasks = ft.generate_task_battery([20260807], difficulty=2)
    journal = sweep.Journal(tmp_path / "journal.jsonl")
    for task in tasks:
        reveal = task.reveal_for_verifier()
        journal.append(
            {
                "event": "CELL",
                "arm": "vanilla",
                "task_id": task.task_id,
                "domain": task.domain,
                "text": "FINAL_ANSWER: " + json.dumps(reveal["expected"]),
                "error": "",
            }
        )
        journal.append(
            {
                "event": "CELL",
                "arm": "rlc_asrun",
                "task_id": task.task_id,
                "domain": task.domain,
                "text": "",
                "error": "EpisodeFault: episode produced no answer",
            }
        )

    verdict = sweep.grade(tmp_path, tasks)
    assert verdict["arms_complete"] is False
    assert verdict["faulted_arms"]["rlc_asrun"] == len(tasks)
    assert verdict["decision"] == "inconclusive_arms_carry_harness_faults"
    # Crucially it does NOT report the recurrent path as below vanilla.
    assert verdict["reaches_parity_with_ordinary_decode"] is False
    assert verdict["arms"]["rlc_asrun"]["correct"] == 0


def test_mutual_failure_is_not_parity(tmp_path: Path):
    """0 >= 0 satisfies the parity inequality. It must not satisfy the gate.

    A battery the ordinary decode cannot score on has not measured recurrence
    at all, and promoting on it would advance a model that answered nothing.
    """
    from core.brain.llm.latent_cortex import frontier_tasks as ft

    tasks = ft.generate_task_battery([20260807], difficulty=2)
    journal = sweep.Journal(tmp_path / "journal.jsonl")

    for task in tasks:
        for arm in ("vanilla", "rlc_asrun"):
            journal.append(
                {
                    "event": "CELL",
                    "arm": arm,
                    "task_id": task.task_id,
                    "domain": task.domain,
                    "text": "I am not able to answer this.",
                    "error": "",
                }
            )

    verdict = sweep.grade(tmp_path, tasks)
    assert verdict["vanilla_correct"] == 0
    assert verdict["best_recurrent_correct"] == 0
    assert verdict["battery_informative"] is False
    assert verdict["reaches_parity_with_ordinary_decode"] is False
    assert (
        verdict["decision"]
        == "inconclusive_battery_uninformative_ordinary_decode_scored_zero"
    )
    assert verdict["claims"]["fusion_authorized"] is False


def test_one_solved_control_task_makes_the_battery_informative(tmp_path: Path):
    """The floor is structural: a baseline exists, or it does not."""
    from core.brain.llm.latent_cortex import frontier_tasks as ft

    tasks = ft.generate_task_battery([20260807], difficulty=2)
    journal = sweep.Journal(tmp_path / "journal.jsonl")

    for index, task in enumerate(tasks):
        reveal = task.reveal_for_verifier()
        correct_line = "FINAL_ANSWER: " + json.dumps(reveal["expected"])
        # Exactly one control task is solved, by both arms.
        text = correct_line if index == 0 else "no answer"
        for arm in ("vanilla", "rlc_asrun"):
            journal.append(
                {
                    "event": "CELL",
                    "arm": arm,
                    "task_id": task.task_id,
                    "domain": task.domain,
                    "text": text,
                    "error": "",
                }
            )

    verdict = sweep.grade(tmp_path, tasks)
    assert verdict["vanilla_correct"] == 1
    assert verdict["battery_informative"] is True
    assert verdict["reaches_parity_with_ordinary_decode"] is True
    assert verdict["decision"] == "proceed_to_checkpoint_phase"


def test_a_cell_from_a_superseded_decode_configuration_is_re_run(tmp_path: Path):
    """The defect that cost two restarts: a resumed run reused cells produced
    under an older decode configuration, so the control and the treatment were
    compared across different rules. Configuration identity travels with the
    cell, and a mismatch is treated as absent rather than as evidence."""
    current = sweep.decode_fingerprint(
        model="/models/resident",
        n_slots=16,
        max_tokens=320,
        episode_wall_s=720.0,
        seed=20260807,
        per_domain=4,
    )
    superseded = sweep.decode_fingerprint(
        model="/models/resident",
        n_slots=16,
        max_tokens=192,  # the only difference, and it changes every answer
        episode_wall_s=720.0,
        seed=20260807,
        per_domain=4,
    )
    assert current != superseded

    path = tmp_path / "journal.jsonl"
    stale = sweep.Journal(path, superseded)
    stale.append(
        {
            "event": "CELL",
            "arm": "vanilla",
            "task_id": "task-a",
            "domain": "mathematics",
            "decode_fingerprint": superseded,
            "text": "FINAL_ANSWER: {}",
            "error": "",
        }
    )

    resumed = sweep.Journal(path, current)
    assert resumed.done == set(), "a superseded cell must not count as committed"
    assert resumed.superseded == 1
    assert resumed.cells() == [], "and must not be graded"

    # Under its own fingerprint it is still perfectly good evidence.
    replayed = sweep.Journal(path, superseded)
    assert replayed.done == {("vanilla", "task-a")}


def test_an_unfingerprinted_journal_is_still_readable(tmp_path: Path):
    """Older runs and unit fixtures carry no fingerprint; they are admitted."""
    path = tmp_path / "journal.jsonl"
    journal = sweep.Journal(path)
    journal.append(
        {
            "event": "CELL",
            "arm": "vanilla",
            "task_id": "task-a",
            "domain": "mathematics",
            "text": "FINAL_ANSWER: {}",
            "error": "",
        }
    )
    assert sweep.Journal(path).done == {("vanilla", "task-a")}
    assert sweep.Journal(path).superseded == 0


def test_no_recurrent_arm_is_not_a_verdict_about_recurrence(tmp_path: Path):
    """Grading a vanilla-only run reported recurrent_path_below_ordinary_decode
    off a -1 sentinel. A conclusion about the recurrent path requires having
    run one."""
    from core.brain.llm.latent_cortex import frontier_tasks as ft

    tasks = ft.generate_task_battery([20260807], difficulty=2)
    journal = sweep.Journal(tmp_path / "journal.jsonl")
    for task in tasks:
        reveal = task.reveal_for_verifier()
        journal.append(
            {
                "event": "CELL",
                "arm": "vanilla",
                "task_id": task.task_id,
                "domain": task.domain,
                "text": "FINAL_ANSWER: " + json.dumps(reveal["expected"]),
                "error": "",
            }
        )

    verdict = sweep.grade(tmp_path, tasks)
    assert verdict["vanilla_correct"] == len(tasks)
    assert verdict["best_recurrent_correct"] == -1
    assert verdict["reaches_parity_with_ordinary_decode"] is False
    assert verdict["decision"] == "inconclusive_no_recurrent_arm_measured"
    assert verdict["claims"]["fusion_authorized"] is False


def test_per_arm_fingerprints_retire_only_the_arm_that_changed(tmp_path: Path):
    """Production passes a per-arm mapping. Raising one arm's budget must not
    discard the arms whose configuration is untouched."""
    common = dict(
        model="/models/resident",
        n_slots=16,
        episode_wall_s=720.0,
        seed=20260807,
        per_domain=4,
    )
    vanilla_fp = sweep.decode_fingerprint(max_tokens=512, arm="vanilla", **common)
    long_512 = sweep.decode_fingerprint(max_tokens=512, arm="vanilla_long", **common)
    long_1024 = sweep.decode_fingerprint(max_tokens=1024, arm="vanilla_long", **common)

    path = tmp_path / "journal.jsonl"
    writer = sweep.Journal(path)
    for arm, fp in (("vanilla", vanilla_fp), ("vanilla_long", long_512)):
        writer.append(
            {
                "event": "CELL",
                "arm": arm,
                "task_id": "task-a",
                "domain": "mathematics",
                "decode_fingerprint": fp,
                "text": "FINAL_ANSWER: {}",
                "error": "",
            }
        )

    resumed = sweep.Journal(
        path, {"vanilla": vanilla_fp, "vanilla_long": long_1024}
    )
    assert resumed.done == {("vanilla", "task-a")}
    assert resumed.superseded == 1

    # An arm dropped from the configuration stops counting as evidence.
    narrowed = sweep.Journal(path, {"vanilla": vanilla_fp})
    assert narrowed.done == {("vanilla", "task-a")}
    assert narrowed.superseded == 1


def test_latency_is_reported_beside_accuracy(tmp_path: Path):
    """A unified system that answers better but takes ten minutes has not been
    shown to be deployable. The program's own standard requires equal-latency
    and equal-compute comparisons, so cost travels with the verdict."""
    from core.brain.llm.latent_cortex import frontier_tasks as ft

    tasks = ft.generate_task_battery([20260807], difficulty=2)
    journal = sweep.Journal(tmp_path / "journal.jsonl")
    for task in tasks:
        reveal = task.reveal_for_verifier()
        correct = "FINAL_ANSWER: " + json.dumps(reveal["expected"])
        journal.append(
            {
                "event": "CELL", "arm": "vanilla", "task_id": task.task_id,
                "domain": task.domain, "text": correct, "error": "",
                "latency_s": 40.0,
            }
        )
        journal.append(
            {
                "event": "CELL", "arm": "full_stack", "task_id": task.task_id,
                "domain": task.domain, "text": correct, "error": "",
                "latency_s": 400.0, "steps_taken": 3, "halted_early": True,
            }
        )

    verdict = sweep.grade(tmp_path, tasks)
    full = verdict["arms"]["full_stack"]
    assert full["latency_median_s"] == 400.0
    assert full["steps_median"] == 3
    # Adaptive halting is the latency lever; its use must be visible.
    assert full["halted_early_fraction"] == 1.0
    # Ten times the cost for the same score is a reportable result.
    assert verdict["latency_ratio_vs_ordinary_decode"]["full_stack"] == 10.0
    assert verdict["latency_ratio_vs_ordinary_decode"]["vanilla"] == 1.0


def test_the_operator_can_reclaim_the_machine_between_cells(tmp_path: Path):
    """The host cannot hold two 32B models, so the campaign and the live
    instance are exclusive. The campaign must therefore be able to leave on
    request rather than requiring long contiguous blocks."""
    assert sweep.yield_requested(tmp_path) is False
    (tmp_path / sweep.YIELD_SENTINEL).touch()
    assert sweep.yield_requested(tmp_path) is True
    # Committed work survives a yield: identity travels with the cell, so a
    # resumed run re-admits exactly the cells it already paid for.
    (tmp_path / sweep.YIELD_SENTINEL).unlink()
    assert sweep.yield_requested(tmp_path) is False


def test_the_product_arm_keeps_ordinary_decode_as_the_incumbent():
    """The stack scored HALF of plain greedy decode because the product arm ran
    decode_incumbent_policy="latent": the recurrent path owned the answer
    unconditionally, so ordinary decode's answer was never a candidate. Adding
    verifiers to that cannot help -- selection cannot exceed the best candidate
    in the pool, and the good one was not in it.

    The deployed system runs "vanilla_incumbent": every subsystem still
    executes and is receipted, the answer decodes from the clean prompt root,
    and a latent answer takes over only when a gain gate promotes it. That is
    monotonic by construction."""
    full = sweep._build_config(8, 16, "applied", 512, profile="full")
    assert full.decode_incumbent_policy == "vanilla_incumbent"
    # The acceptance rule is what lets a latent answer win at all.
    assert full.answer_replacement_enabled is True

    # The ablation deliberately keeps the latent path owning the answer, so a
    # degraded episode cannot silently serve vanilla and read as a result.
    mech = sweep._build_config(4, 16, "suppressed", 512, profile="mechanism")
    assert mech.decode_incumbent_policy == "latent"
    assert mech.answer_replacement_enabled is False
