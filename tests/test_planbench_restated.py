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


@pytest.mark.skipif(not ROOT.exists(), reason="PlanBench is not on this host")
@pytest.mark.parametrize("domain", ["logistics", "depots", "obfuscated_deceptive_logistics"])
def test_every_gold_plan_reaches_its_goal_through_text_and_a_broken_one_does_not(domain) -> None:
    for task in tasks(domain):
        text = plan_as_text(task["truth"]["ground_truth_plan"], domain)
        assert grade(text, task["truth"])[0], task["id"]
        assert not grade("\n".join(text.split("\n")[1:]), task["truth"])[0], task["id"]


@pytest.mark.skipif(not ROOT.exists(), reason="PlanBench is not on this host")
def test_a_depots_step_whose_objects_are_the_wrong_type_fails_as_val_fails_it() -> None:
    task = next(t for t in tasks("depots") if t["truth"]["ground_truth_plan"].startswith("(lift"))
    first = task["truth"]["ground_truth_plan"].split("\n")[0].strip("()").split()
    plan = text_to_plan(plan_as_text(task["truth"]["ground_truth_plan"], "depots"), "depots")
    assert plan[0] == tuple(first)
    hoist, crate = first[1], first[2]
    swapped = [(first[0], crate, hoist, *first[3:]), *plan[1:]]
    assert not plan_reaches_goal(swapped, "depots", int(task["truth"]["instance_id"]))
