"""Answer-blind semantic surface adaptation for causal recurrent tissue.

The activated semantic runtime intentionally admits exact issuer grammars. This
module is the measured bridge toward less-constrained language: it accepts
several independently rendered public evidence styles, derives a typed causal
graph from unordered intervention records, and only then invokes the existing
recurrent tissue. It never receives a verifier answer or private state trace.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from typing import Any, Final

from core.brain.llm.latent_cortex.semantic_neural_decode_context import (
    SemanticNeuralDecodeState,
    execute_semantic_neural_decode_state,
)
from core.learning.public_frontier_action_compiler import (
    MAX_PROCESS_INTEGER,
    public_frontier_operands,
)
from core.learning.recurrent_action_schema import ACTION_NULL
from core.learning.semantic_neural_machine import SemanticNeuralMachine

SEMANTIC_SURFACE_SCHEMA: Final = "aura.semantic_surface_adapter.v1"
SEMANTIC_SURFACE_PROFILES: Final = ("lab_report", "narrative", "compact")
SCIENTIFIC_FAMILY: Final = "frontier_scientific_inference"

_IDENTIFIER = r"[a-z][a-z0-9_]*"
_CONTRACT: Final = (
    "Return exactly one final line beginning FINAL_ANSWER:, followed by a JSON "
    "object with root (string), mediator (string), downstream (string), and "
    "predicted_downstream (integer), and no trailing text."
)
_DISTRACTORS: Final = (
    "Variable names are arbitrary labels and do not encode causal rank.",
    "Observation order is arbitrary and carries no causal information.",
    "All measurements use the same unnamed unit.",
)
_LAB_BASE = re.compile(rf"Baseline readings: (?P<body>{_IDENTIFIER}=-?\d+(?:; {_IDENTIFIER}=-?\d+){{2}})\.")
_NARRATIVE_BASE = re.compile(
    rf"Before intervention, (?P<a>{_IDENTIFIER}) was (?P<av>-?\d+), "
    rf"(?P<b>{_IDENTIFIER}) was (?P<bv>-?\d+), and "
    rf"(?P<c>{_IDENTIFIER}) was (?P<cv>-?\d+)\."
)
_COMPACT_BASE = re.compile(
    rf"BASE (?P<body>{_IDENTIFIER}=-?\d+(?: \| {_IDENTIFIER}=-?\d+){{2}})"
)
_LAB_INTERVENTION = re.compile(
    rf"- Raising (?P<actor>{_IDENTIFIER}) by (?P<step>\d+) produced deltas "
    rf"(?P<body>{_IDENTIFIER}:[+-]?\d+(?:; {_IDENTIFIER}:[+-]?\d+){{2}})\."
)
_NARRATIVE_INTERVENTION = re.compile(
    rf"When (?P<actor>{_IDENTIFIER}) increased by (?P<step>\d+), "
    rf"(?P<body>{_IDENTIFIER} changed by [+-]?\d+(?:, {_IDENTIFIER} changed by [+-]?\d+){{2}})\."
)
_COMPACT_INTERVENTION = re.compile(
    rf"DO (?P<actor>{_IDENTIFIER}) \+(?P<step>\d+) => "
    rf"(?P<body>{_IDENTIFIER}:[+-]?\d+(?:, {_IDENTIFIER}:[+-]?\d+){{2}})"
)
_LAB_QUERY = re.compile(
    rf"Prediction request: set (?P<actor>{_IDENTIFIER}) to baseline\+(?P<step>\d+); "
    rf"report the absolute (?P<target>{_IDENTIFIER}) reading\."
)
_NARRATIVE_QUERY = re.compile(
    rf"What absolute value should (?P<target>{_IDENTIFIER}) have if "
    rf"(?P<actor>{_IDENTIFIER}) is (?P<step>\d+) units above its baseline\?"
)
_COMPACT_QUERY = re.compile(
    rf"QUERY (?P<actor>{_IDENTIFIER})=BASE\+(?P<step>\d+) -> ABS (?P<target>{_IDENTIFIER})"
)


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _prompt_sha(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ScientificSurfaceProgram:
    """A typed public causal graph reconstructed without an answer channel."""

    profile: str
    public_prompt: str
    canonical_prompt: str
    labels: tuple[str, str, str]
    baselines: tuple[int, int, int]
    root: str
    mediator: str
    downstream: str
    root_step: int
    root_edges: tuple[tuple[str, int], tuple[str, int]]
    mediator_step: int
    mediator_change: int
    downstream_step: int
    query_step: int
    schema: str = SEMANTIC_SURFACE_SCHEMA

    def receipt(self) -> dict[str, Any]:
        facts = {
            "labels": self.labels,
            "baselines": self.baselines,
            "root": self.root,
            "mediator": self.mediator,
            "downstream": self.downstream,
            "root_step": self.root_step,
            "root_edges": self.root_edges,
            "mediator_step": self.mediator_step,
            "mediator_change": self.mediator_change,
            "downstream_step": self.downstream_step,
            "query_step": self.query_step,
        }
        body = {
            "schema": self.schema,
            "profile": self.profile,
            "public_prompt_sha256": _prompt_sha(self.public_prompt),
            "canonical_prompt_sha256": _prompt_sha(self.canonical_prompt),
            "public_fact_graph_sha256": _sha(facts),
            "answer_available": False,
            "verifier_available": False,
            "private_trace_available": False,
        }
        return {**body, "receipt_sha256": _sha(body)}


@dataclass(frozen=True, slots=True)
class ScientificSurfaceDecode:
    """Recurrent state plus a receipt for the surface-to-program bridge."""

    program: ScientificSurfaceProgram
    state: SemanticNeuralDecodeState

    def receipt(self) -> dict[str, Any]:
        body = {
            "schema": "aura.semantic_surface_decode.v1",
            "surface_receipt_sha256": self.program.receipt()["receipt_sha256"],
            "semantic_state_receipt_sha256": self.state.receipt()["receipt_sha256"],
            "surface_objective_sha256": _prompt_sha(self.program.public_prompt),
            "canonical_objective_sha256": self.state.objective_sha256,
            "teacher_available": False,
            "verifier_available": False,
            "answer_key_available": False,
        }
        return {**body, "receipt_sha256": _sha(body)}


def _pairs(text: str, *, item_separator: str, value_separator: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for item in text.split(item_separator):
        name, raw = item.split(value_separator, 1)
        name = name.strip()
        raw = raw.strip().removeprefix("changed by ")
        if not re.fullmatch(_IDENTIFIER, name) or name in values:
            raise ValueError("scientific surface contains duplicate or invalid labels")
        values[name] = int(raw)
    return values


def _context(line: str, prefix: str) -> str:
    if not line.startswith(prefix):
        raise ValueError("scientific surface context marker is invalid")
    value = line.removeprefix(prefix)
    if value not in _DISTRACTORS:
        raise ValueError("scientific surface context is not an admitted irrelevant fact")
    return value


def _extract_sections(prompt: str) -> tuple[str, dict[str, int], list[tuple[str, int, dict[str, int]]], str, int, str]:
    if prompt.startswith("Causal study report.\n"):
        profile = "lab_report"
        lines = prompt.splitlines()
        if len(lines) != 9:
            raise ValueError("lab-report scientific surface has invalid sections")
        _context(lines[2], "Intervention observations, listed in arbitrary order. Context: ")
        baseline_match = _LAB_BASE.fullmatch(lines[1])
        records = [_LAB_INTERVENTION.fullmatch(line) for line in lines[3:6]]
        query_match = _LAB_QUERY.fullmatch(lines[6])
        assumption = lines[7]
        contract = lines[8]
        baseline = (
            {} if baseline_match is None else _pairs(baseline_match.group("body"), item_separator="; ", value_separator="=")
        )
        deltas = [
            _pairs(match.group("body"), item_separator="; ", value_separator=":")
            if match is not None
            else {}
            for match in records
        ]
    elif prompt.startswith("Controlled causal field note.\n"):
        profile = "narrative"
        lines = prompt.splitlines()
        if len(lines) != 9:
            raise ValueError("narrative scientific surface has invalid sections")
        _context(lines[2], "The observations below are deliberately unordered. Context: ")
        baseline_match = _NARRATIVE_BASE.fullmatch(lines[1])
        records = [_NARRATIVE_INTERVENTION.fullmatch(line) for line in lines[3:6]]
        query_match = _NARRATIVE_QUERY.fullmatch(lines[6])
        assumption = lines[7]
        contract = lines[8]
        baseline = (
            {}
            if baseline_match is None
            else {
                baseline_match.group("a"): int(baseline_match.group("av")),
                baseline_match.group("b"): int(baseline_match.group("bv")),
                baseline_match.group("c"): int(baseline_match.group("cv")),
            }
        )
        deltas = [
            _pairs(
                match.group("body"),
                item_separator=", ",
                value_separator=" ",
            )
            if match is not None
            else {}
            for match in records
        ]
    elif prompt.startswith("CAUSAL_FACTS_V1\n"):
        profile = "compact"
        lines = prompt.splitlines()
        if len(lines) != 9:
            raise ValueError("compact scientific surface has invalid sections")
        _context(lines[2], "META ")
        baseline_match = _COMPACT_BASE.fullmatch(lines[1])
        records = [_COMPACT_INTERVENTION.fullmatch(line) for line in lines[3:6]]
        query_match = _COMPACT_QUERY.fullmatch(lines[6])
        assumption = lines[7]
        contract = lines[8]
        baseline = (
            {} if baseline_match is None else _pairs(baseline_match.group("body"), item_separator=" | ", value_separator="=")
        )
        deltas = [
            _pairs(match.group("body"), item_separator=", ", value_separator=":")
            if match is not None
            else {}
            for match in records
        ]
    else:
        raise ValueError("scientific surface profile is unsupported")
    if (
        baseline_match is None
        or query_match is None
        or any(match is None for match in records)
        or assumption
        != "Assumptions: effects are deterministic and linear; there is no hidden common cause."
        or contract != _CONTRACT
    ):
        raise ValueError("scientific surface contract is incomplete")
    if len(baseline) != 3:
        raise ValueError("scientific surface requires exactly three baselines")
    parsed_records = [
        (match.group("actor"), int(match.group("step")), delta)
        for match, delta in zip(records, deltas, strict=True)
        if match is not None
    ]
    return (
        profile,
        baseline,
        parsed_records,
        query_match.group("actor"),
        int(query_match.group("step")),
        query_match.group("target"),
    )


def _canonical_prompt(operands: dict[str, Any]) -> str:
    labels = operands["labels"]
    baselines = operands["baselines"]
    root_edges = operands["root_edges"]
    return (
        "Fresh causal-inference task. Three measured variables have baseline values "
        f"{labels[0]}={baselines[0]}, {labels[1]}={baselines[1]}, {labels[2]}={baselines[2]}. "
        "Independent interventions produced these changes relative to baseline: "
        f"setting {operands['root']} up by {operands['root_step']} changed "
        f"{root_edges[0][0]} by +{root_edges[0][1]} and {root_edges[1][0]} by "
        f"+{root_edges[1][1]}; setting {operands['mediator']} up by "
        f"{operands['mediator_step']} left {operands['root']} unchanged and changed "
        f"{operands['downstream']} by +{operands['mediator_change']}; setting "
        f"{operands['downstream']} up by {operands['downstream_step']} left both "
        "other variables unchanged. Assume deterministic linear effects and no hidden "
        "common cause. Identify root, mediator, and downstream variables, then predict "
        f"the absolute value of {operands['downstream']} when {operands['root']} is set "
        f"{operands['query_step']} above baseline. You may reason before the answer. "
        "End with exactly one line beginning FINAL_ANSWER:, followed by one JSON object "
        "and no trailing text. Required JSON keys and value types: root (string), "
        "mediator (string), downstream (string), predicted_downstream (integer)."
    )


def parse_scientific_surface(prompt: str) -> ScientificSurfaceProgram:
    """Parse and validate one alternate public surface without solving its answer."""

    if (
        not isinstance(prompt, str)
        or not prompt
        or prompt != prompt.strip()
        or "\x00" in prompt
        or len(prompt.encode("utf-8")) > 32_768
    ):
        raise ValueError("scientific surface prompt is invalid")
    profile, baseline, records, query_actor, query_step, query_target = _extract_sections(prompt)
    labels = tuple(baseline)
    label_set = set(labels)
    if (
        any(not 0 <= value <= MAX_PROCESS_INTEGER for value in baseline.values())
        or len(records) != 3
        or {actor for actor, _step, _delta in records} != label_set
        or any(
            step < 1
            or step >= ACTION_NULL
            or set(delta) != label_set
            or delta[actor] != 0
            or any(abs(value) > MAX_PROCESS_INTEGER for value in delta.values())
            for actor, step, delta in records
        )
    ):
        raise ValueError("scientific surface evidence is outside the admitted domain")
    changed = {
        actor: {target for target, value in delta.items() if value != 0}
        for actor, _step, delta in records
    }
    by_degree = {degree: [actor for actor, targets in changed.items() if len(targets) == degree] for degree in (0, 1, 2)}
    if any(len(by_degree[degree]) != 1 for degree in (0, 1, 2)):
        raise ValueError("scientific surface does not define a unique three-node chain")
    downstream = by_degree[0][0]
    mediator = by_degree[1][0]
    root = by_degree[2][0]
    if changed[root] != {mediator, downstream} or changed[mediator] != {downstream}:
        raise ValueError("scientific surface causal graph is inconsistent")
    record_by_actor = {actor: (step, delta) for actor, step, delta in records}
    root_step, root_delta = record_by_actor[root]
    mediator_step, mediator_delta = record_by_actor[mediator]
    downstream_step, _downstream_delta = record_by_actor[downstream]
    root_mediator_change = root_delta[mediator]
    root_downstream_change = root_delta[downstream]
    mediator_change = mediator_delta[downstream]
    if (
        min(root_mediator_change, root_downstream_change, mediator_change) <= 0
        or root_mediator_change >= ACTION_NULL
        or mediator_change >= ACTION_NULL
        or root_mediator_change % root_step
        or mediator_change % mediator_step
    ):
        raise ValueError("scientific surface effects do not define positive exact gains")
    mediator_gain = root_mediator_change // root_step
    downstream_gain = mediator_change // mediator_step
    if root_downstream_change != root_step * mediator_gain * downstream_gain:
        raise ValueError("scientific surface direct and mediated effects disagree")
    if query_actor != root or query_target != downstream or not 1 <= query_step < ACTION_NULL:
        raise ValueError("scientific surface query does not follow the admitted causal chain")
    predicted = baseline[downstream] + query_step * mediator_gain * downstream_gain
    if predicted > MAX_PROCESS_INTEGER:
        raise ValueError("scientific surface prediction exceeds recurrent state capacity")
    operands = {
        "labels": list(labels),
        "baselines": [baseline[label] for label in labels],
        "root": root,
        "mediator": mediator,
        "downstream": downstream,
        "root_step": root_step,
        "root_edges": [(target, root_delta[target]) for target in labels if target != root],
        "mediator_step": mediator_step,
        "mediator_change": mediator_change,
        "downstream_step": downstream_step,
        "query_step": query_step,
    }
    canonical = _canonical_prompt(operands)
    # The existing compiler is an independent structural validator. It must
    # accept the normalized prompt before the adapter can issue a receipt.
    validated = public_frontier_operands(canonical, SCIENTIFIC_FAMILY)
    if validated != operands:
        raise RuntimeError("scientific surface normalization changed public facts")
    return ScientificSurfaceProgram(
        profile=profile,
        public_prompt=prompt,
        canonical_prompt=canonical,
        labels=labels,
        baselines=tuple(operands["baselines"]),
        root=root,
        mediator=mediator,
        downstream=downstream,
        root_step=root_step,
        root_edges=tuple(operands["root_edges"]),
        mediator_step=mediator_step,
        mediator_change=mediator_change,
        downstream_step=downstream_step,
        query_step=query_step,
    )


def render_scientific_surface(
    canonical_prompt: str,
    *,
    profile: str,
    permutation_seed: int,
) -> str:
    """Render answer-blind canonical evidence into a controlled alternate surface."""

    if profile not in SEMANTIC_SURFACE_PROFILES or type(permutation_seed) is not int:
        raise ValueError("scientific surface render contract is invalid")
    operands = public_frontier_operands(canonical_prompt, SCIENTIFIC_FAMILY)
    labels = list(operands["labels"])
    baselines = dict(zip(labels, operands["baselines"], strict=True))
    root = operands["root"]
    mediator = operands["mediator"]
    downstream = operands["downstream"]
    root_delta = {label: 0 for label in labels}
    root_delta.update(dict(operands["root_edges"]))
    mediator_delta = {label: 0 for label in labels}
    mediator_delta[downstream] = operands["mediator_change"]
    records = [
        (root, operands["root_step"], root_delta),
        (mediator, operands["mediator_step"], mediator_delta),
        (downstream, operands["downstream_step"], {label: 0 for label in labels}),
    ]
    rng = random.Random(
        int.from_bytes(
            hashlib.sha256(f"{profile}:{permutation_seed}:{_prompt_sha(canonical_prompt)}".encode()).digest(),
            "big",
        )
    )
    rng.shuffle(labels)
    rng.shuffle(records)
    render_records = []
    for actor, step, delta in records:
        ordered = list(delta)
        rng.shuffle(ordered)
        render_records.append((actor, step, delta, ordered))

    distractor = _DISTRACTORS[permutation_seed % len(_DISTRACTORS)]
    assumption = "Assumptions: effects are deterministic and linear; there is no hidden common cause."
    if profile == "lab_report":
        lines = [
            "Causal study report.",
            "Baseline readings: " + "; ".join(f"{label}={baselines[label]}" for label in labels) + ".",
            f"Intervention observations, listed in arbitrary order. Context: {distractor}",
        ]
        for actor, step, delta, order in render_records:
            lines.append(
                f"- Raising {actor} by {step} produced deltas "
                + "; ".join(f"{label}:{delta[label]:+d}" for label in order)
                + "."
            )
        lines.extend(
            (
                f"Prediction request: set {root} to baseline+{operands['query_step']}; report the absolute {downstream} reading.",
                assumption,
                _CONTRACT,
            )
        )
    elif profile == "narrative":
        lines = [
            "Controlled causal field note.",
            f"Before intervention, {labels[0]} was {baselines[labels[0]]}, {labels[1]} was {baselines[labels[1]]}, and {labels[2]} was {baselines[labels[2]]}.",
            f"The observations below are deliberately unordered. Context: {distractor}",
        ]
        for actor, step, delta, order in render_records:
            lines.append(
                f"When {actor} increased by {step}, "
                + ", ".join(f"{label} changed by {delta[label]:+d}" for label in order)
                + "."
            )
        lines.extend(
            (
                f"What absolute value should {downstream} have if {root} is {operands['query_step']} units above its baseline?",
                assumption,
                _CONTRACT,
            )
        )
    else:
        lines = [
            "CAUSAL_FACTS_V1",
            "BASE " + " | ".join(f"{label}={baselines[label]}" for label in labels),
            f"META {distractor}",
        ]
        for actor, step, delta, order in render_records:
            lines.append(
                f"DO {actor} +{step} => "
                + ", ".join(f"{label}:{delta[label]:+d}" for label in order)
            )
        lines.extend(
            (
                f"QUERY {root}=BASE+{operands['query_step']} -> ABS {downstream}",
                assumption,
                _CONTRACT,
            )
        )
    rendered = "\n".join(lines)
    parse_scientific_surface(rendered)
    return rendered


def execute_scientific_surface(
    prompt: str,
    *,
    machine: SemanticNeuralMachine | None = None,
) -> ScientificSurfaceDecode:
    """Execute an alternate surface through the unchanged recurrent tissue."""

    program = parse_scientific_surface(prompt)
    state = execute_semantic_neural_decode_state(
        program.canonical_prompt,
        SCIENTIFIC_FAMILY,
        machine=machine,
    )
    if state.objective_sha256 != _prompt_sha(program.canonical_prompt):
        raise RuntimeError("scientific surface state is bound to the wrong objective")
    return ScientificSurfaceDecode(program=program, state=state)


__all__ = [
    "SCIENTIFIC_FAMILY",
    "SEMANTIC_SURFACE_PROFILES",
    "SEMANTIC_SURFACE_SCHEMA",
    "ScientificSurfaceDecode",
    "ScientificSurfaceProgram",
    "execute_scientific_surface",
    "parse_scientific_surface",
    "render_scientific_surface",
]
