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


def test_arms_cross_disposition_against_recurrence_depth():
    """The sweep must be a factorial, or it cannot attribute the deficit."""
    by_name = {name: (steps, policy) for name, steps, policy in sweep.ARMS}
    assert by_name["vanilla"][0] is None
    # Both factors must vary, and vary independently.
    assert by_name["rlc_asrun"] == (4, "applied")
    assert by_name["rlc_nodisp"] == (4, "suppressed")
    assert by_name["rlc_shallow"] == (1, "applied")
    assert by_name["rlc_shallow_nodisp"] == (1, "suppressed")


def test_config_carries_the_arm_policy_and_validates():
    for steps, policy in ((4, "applied"), (1, "suppressed")):
        config = sweep._build_config(steps, 16, policy, 320)
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
