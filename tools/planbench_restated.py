"""PlanBench plan generation (Valmeekam et al. 2023), its reply parsing and plan checking restated.

PlanBench grades a reply by turning its text into a PDDL plan
(utils/text_to_pddl.text_to_plan_blocksworld) and validating that plan with
VAL against the instance. VAL is a C++ binary and the parser needs tarski;
neither is in her runtime. Here the parser is restated line for line and
VAL is replaced by a STRIPS simulator read from the domain's own PDDL: every
action's preconditions must hold when it is taken, and the goal must hold at
the end. The two Blocksworld domains PlanBench uses are plain STRIPS (no
types, conditional effects or numbers), which is what the simulator reads.
The restatement is checked against VAL's verdicts on GPT-4's stored replies
(tests/test_planbench_restated.py).
"""

from __future__ import annotations

import functools
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path("~/.aura/benchmarks/g09/planbench/src/plan-bench").expanduser()
DOMAINS = ("blocksworld", "mystery_blocksworld")


def _sexp(text: str) -> list:
    tokens = re.findall(r"\(|\)|[^\s()]+", re.sub(r";[^\n]*", "", text).lower())
    stack: list[list] = [[]]
    for token in tokens:
        if token == "(":
            stack.append([])
        elif token == ")":
            done = stack.pop()
            stack[-1].append(done)
        else:
            stack[-1].append(token)
    return stack[0][0]


def _literals(node: Any) -> tuple[set[tuple[str, ...]], set[tuple[str, ...]]]:
    """Positive and negated atoms of a conjunction."""
    if not node:
        return set(), set()
    if node[0] == "and":
        positive, negative = set(), set()
        for part in node[1:]:
            more_positive, more_negative = _literals(part)
            positive |= more_positive
            negative |= more_negative
        return positive, negative
    if node[0] == "not":
        return set(), {tuple(node[1])}
    return {tuple(node)}, set()


@functools.cache
def config(domain: str) -> dict[str, Any]:
    """The domain's PlanBench config: its action phrases, object names and files."""
    import yaml

    return yaml.safe_load((ROOT / "configs" / f"{domain}.yaml").read_text(encoding="utf-8"))


@functools.cache
def actions(domain: str) -> dict[str, dict[str, Any]]:
    """Each action of the domain PDDL: its parameters, preconditions, additions and deletions, in file order."""
    tree = _sexp((ROOT / "instances" / config(domain)["domain_file"]).read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for part in tree:
        if isinstance(part, list) and part and part[0] == ":action":
            name, fields = part[1], dict(zip(part[2::2], part[3::2], strict=False))
            precondition, _ = _literals(fields.get(":precondition", []))
            additions, deletions = _literals(fields.get(":effect", []))
            out[name] = {"parameters": [p for p in fields.get(":parameters", []) if p.startswith("?")],
                         "precondition": precondition, "add": additions, "delete": deletions}
    return out


@functools.cache
def instance(domain: str, instance_id: int) -> tuple[frozenset, frozenset]:
    """The instance's initial facts and goal facts."""
    path = ROOT / "instances" / config(domain)["instance_dir"] / config(domain)["instances_template"].format(instance_id)
    tree = _sexp(path.read_text(encoding="utf-8"))
    fields = {part[0]: part[1:] for part in tree if isinstance(part, list) and part}
    initial = frozenset(tuple(atom) for atom in fields[":init"])
    goal, _ = _literals(fields[":goal"][0])
    return initial, frozenset(goal)


def text_to_plan(text: str, domain: str) -> list[tuple[str, ...]]:
    """PlanBench's text_to_plan_blocksworld, restated: action phrases replaced by action names, then on each
    line the first action named and the objects in the order they appear."""
    settings = config(domain)
    letters = settings["encoded_objects"]
    blocks = {name.lower(): letter for letter, name in letters.items()}
    phrases = {}
    for name, template in settings["actions"].items():
        word = template.split(" ")[0]
        phrases[name] = name.replace("-", " ") if word in name else word
    names = [name.lower() for name in actions(domain)]
    text = text.lower().strip()
    for name in names:
        text = text.replace(phrases[name], name)
    object_names = [name.lower() for name in letters.values()]
    plan = []
    for line in (line.strip() for line in text.split("\n")):
        if "[cost]" in line:
            break
        found = [name for name in names if name in line.split()]
        if not found:
            continue
        name = found[0]
        positions = sorted((line.index(obj), obj) for obj in object_names if obj in line)
        objects = [obj for _, obj in positions]
        if len(objects) != len(actions(domain)[name]["parameters"]):
            continue
        plan.append((name, *[blocks[obj] for obj in objects]))
    return plan


def _ground(atoms: set, binding: dict[str, str]) -> set:
    return {tuple(binding.get(term, term) for term in atom) for atom in atoms}


def plan_reaches_goal(plan: list[tuple[str, ...]], domain: str, instance_id: int) -> bool:
    """VAL's verdict, restated for plain STRIPS: every step applicable in turn, and the goal true at the end."""
    state, goal = instance(domain, instance_id)
    facts = set(state)
    for name, *objects in plan:
        action = actions(domain).get(name)
        if action is None or len(objects) != len(action["parameters"]):
            return False
        binding = dict(zip(action["parameters"], objects, strict=True))
        if not _ground(action["precondition"], binding) <= facts:
            return False
        facts = (facts - _ground(action["delete"], binding)) | _ground(action["add"], binding)
    return goal <= facts


def grade(text: str, truth: dict[str, Any]) -> tuple[bool, str]:
    """A reply is right when the plan read from it, up to the "[PLAN END]" PlanBench's own prompts close a
    plan with, reaches the goal from the instance's start."""
    reply = text.split("[PLAN END]")[0]
    plan = text_to_plan(reply, truth["domain"])
    return bool(plan) and plan_reaches_goal(plan, truth["domain"], int(truth["instance_id"])), f"{len(plan)} steps"


def plan_as_text(pddl_plan: str, domain: str) -> str:
    """A PDDL plan written the way PlanBench's prompts write one, closed with "[PLAN END]"."""
    settings = config(domain)
    names = list(settings["encoded_objects"].values())

    def spoken(obj: str) -> str:
        # Gold plans name an object by its letter, its first word ("yellow") or its last ("a").
        if obj in settings["encoded_objects"]:
            return settings["encoded_objects"][obj]
        matches = [name for name in names if obj in (name.split()[0], name.split()[-1])]
        if len(matches) != 1:
            raise ValueError(f"cannot name object {obj!r} in {domain}")
        return matches[0]

    lines = []
    for step in re.findall(r"\(([^)]*)\)", pddl_plan):
        name, *objects = step.split()
        lines.append(settings["actions"][name].format(*[spoken(obj) for obj in objects]))
    return "\n".join(lines) + "\n[PLAN END]"


def tasks(domain: str) -> list[dict[str, Any]]:
    data = json.loads((ROOT / "prompts" / domain / "task_1_plan_generation.json").read_text(encoding="utf-8"))
    return [{"id": f"{domain}:{item['instance_id']}", "request": item["query"],
             "truth": {"domain": domain, "instance_id": item["instance_id"], "ground_truth_plan": item["ground_truth_plan"]},
             "group": domain} for item in data["instances"]]
