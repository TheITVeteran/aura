"""PlanBench's reply parsing and plan checking, restated, agree with VAL where VAL's verdicts are stored."""

from __future__ import annotations

import json

import pytest

from tools.planbench_restated import (
    ROOT,
    grade,
    plan_as_text,
    plan_reaches_goal,
    tasks,
    text_to_plan,
)

needs_planbench = pytest.mark.skipif(not (ROOT / "configs").exists(), reason="PlanBench is not downloaded here")


@needs_planbench
def test_the_restated_checker_agrees_with_val_on_every_stored_gpt4_reply() -> None:
    rows = json.loads((ROOT / "results/blocksworld/gpt-4_chat/task_1_plan_generation.json").read_text())["instances"]
    verdicts = [grade(r["llm_raw_response"], {"domain": "blocksworld", "instance_id": r["instance_id"]})[0]
                for r in rows]
    assert verdicts == [str(r["llm_correct"]) == "True" for r in rows]
    assert sum(verdicts) == 157 and len(rows) == 500


@needs_planbench
@pytest.mark.parametrize("domain", ["blocksworld", "mystery_blocksworld"])
def test_every_gold_plan_reaches_its_goal_and_a_cut_one_does_not(domain: str) -> None:
    for task in tasks(domain):
        text = plan_as_text(task["truth"]["ground_truth_plan"], domain)
        assert grade(text, task["truth"])[0]
        steps = text_to_plan(text.split("[PLAN END]")[0], domain)
        assert not plan_reaches_goal(steps[:-1], domain, int(task["truth"]["instance_id"]))
