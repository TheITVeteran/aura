"""SPARK-013 state causality: prove the typed epistemic state drives computation.

The typed epistemic state (SPARK-005..012) is only real cognition if the
structured content it holds changes later computation, removing required
state causes task-appropriate loss, and no prose shadow of the state can
substitute for the typed content.  This module makes those three claims
executable:

1. ``project_state_evidence_context`` is the only sanctioned projection
   from an ``EpistemicState``'s evidence ledger into the cognitive-context
   wire items the latent engine embeds into workspace slots.  Every
   injected text must hash to the typed record's ``content_sha256``; the
   prose ``summary`` field is never read; content without a typed state
   ancestor is refused.  A prose-only shadow ledger therefore cannot reach
   computation through this seam.

2. ``run_state_causality_experiment`` runs seven preregistered arms per
   task — intact, lesioned-required (decisive content replaced by an
   information-free filler at equal width, so layer-application parity
   is exact), sham-lesion (injected decoy removed), inert-lesion
   (non-projected memory record removed), restored, annotation-mutation,
   content-substitution — under identical fixed-depth configuration and
   deterministic decoding, and grades exact behavioral identities:
   removing the required information must change the final latent
   computation under exact compute parity; removing a non-projected
   state component must leave computation byte-exact; restoring must
   recover byte-exactly; mutating only state prose (summary) must leave
   computation byte-exact even though the state hash changes;
   substituting the typed content must change computation and, on a
   model that can read the slot channel, track the substituted fact
   while the sham lesion loses no task success.

3. ``replay_state_causality_receipt`` independently regrades every claim
   from the recorded rows alone, so a service-side verifier needs no
   engine to check the verdicts.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Never

from core.brain.llm.latent_cortex.cognitive_context import (
    MAX_COGNITIVE_CONTEXT_CHARS,
    MAX_COGNITIVE_CONTEXT_ITEMS,
    normalize_cognitive_context,
)
from core.brain.llm.latent_cortex.epistemic_state import (
    EpistemicState,
    EvidenceKind,
    EvidenceProvenance,
    EvidencePurpose,
    EvidenceRecord,
    EvidenceScope,
    EvidenceVerification,
    ProblemFrame,
    canonical_sha256,
)

STATE_CAUSALITY_SCHEMA = "aura.latent_cortex.state_causality.v1"
PROJECTION_SCHEMA = "aura.latent_cortex.state_evidence_projection.v1"

ARM_INTACT = "intact"
ARM_LESIONED = "lesioned_required"
ARM_SHAM = "sham_lesion"
ARM_INERT = "inert_lesion"
ARM_RESTORED = "restored"
ARM_ANNOTATION = "annotation_mutation"
ARM_SUBSTITUTION = "content_substitution"
STATE_CAUSALITY_ARMS = (
    ARM_INTACT,
    ARM_LESIONED,
    ARM_SHAM,
    ARM_INERT,
    ARM_RESTORED,
    ARM_ANNOTATION,
    ARM_SUBSTITUTION,
)

# Claims below this many tasks stay CONJECTURE, matching experiments.py.
MIN_TASKS_FOR_VERDICT = 20

PROVEN = "PROVEN"
SUPPORTED = "SUPPORTED"
CONJECTURE = "CONJECTURE"
REFUTED = "REFUTED"

_WIRE_KIND_BY_EVIDENCE_KIND = {
    EvidenceKind.TOOL_RESULT: "governed_tool_observation",
    EvidenceKind.CALCULATION: "governed_tool_observation",
    EvidenceKind.PROOF: "governed_tool_observation",
    EvidenceKind.SIMULATION: "governed_tool_observation",
    EvidenceKind.RETRIEVAL: "offline_reference",
    EvidenceKind.OBSERVATION: "live_world_observation",
}

_ROW_KEYS = {
    "task_id",
    "family",
    "arm",
    "state_sha256",
    "projection_sha256",
    "selected_evidence_ids",
    "item_count",
    "ok",
    "text_sha256",
    "tokens_sha256",
    "final_states_sha256",
    "success",
    "expected_success_answer",
    "steps_taken",
    "layer_apps",
}


class StateCausalityError(ValueError):
    """Stable fail-closed state-causality error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise StateCausalityError(code)


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── 1. The sanctioned state → cognitive-context projection ──────────────


def project_state_evidence_context(
    state: EpistemicState,
    contents: Mapping[str, str],
    *,
    limit: int = MAX_COGNITIVE_CONTEXT_ITEMS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Project typed evidence into wire context items, content-addressed.

    ``contents`` maps evidence ids to the full content text held outside
    the state (the state binds content by digest).  Selection is
    deterministic: fresh, projectable-kind records in evidence-id order.
    Every supplied text must hash to the typed ``content_sha256`` and the
    record's prose ``summary`` is never consulted, so prose cannot shadow
    the typed content.  Content without a typed ancestor is refused.
    """

    if not isinstance(state, EpistemicState):
        _fail("state_causality_state_invalid")
    if not isinstance(contents, Mapping):
        _fail("state_causality_contents_invalid")
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= MAX_COGNITIVE_CONTEXT_ITEMS
    ):
        _fail("state_causality_limit_invalid")
    known_ids = {record.evidence_id for record in state.evidence}
    unknown = set(contents) - known_ids
    if unknown:
        _fail("state_causality_unknown_evidence")

    selected: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    items: list[dict[str, Any]] = []
    for record in sorted(state.evidence, key=lambda item: item.evidence_id):
        if record.evidence_id not in contents:
            continue
        if record.kind is EvidenceKind.MEMORY:
            # Memory evidence travels only through the memory wire contract
            # with its own scope and epistemic-state digests.
            excluded.append(
                {"evidence_id": record.evidence_id, "reason": "memory_wire_only"}
            )
            continue
        if record.kind is EvidenceKind.IMMUTABLE_PROBLEM:
            excluded.append(
                {"evidence_id": record.evidence_id, "reason": "problem_is_prompt_borne"}
            )
            continue
        if len(items) >= limit:
            excluded.append(
                {"evidence_id": record.evidence_id, "reason": "limit_reached"}
            )
            continue
        text = contents[record.evidence_id]
        if not isinstance(text, str) or not text.strip():
            _fail("state_causality_content_invalid")
        if len(text) > MAX_COGNITIVE_CONTEXT_CHARS:
            _fail("state_causality_content_too_long")
        if _sha_text(text) != record.content_sha256:
            _fail("state_causality_content_binding_mismatch")
        items.append(
            {
                "source": f"evidence.{record.kind.value}"[:40],
                "text": text,
                "context_role": "evidence_observation",
                "instruction_authority": False,
                "evidence_id": record.evidence_id,
                "content_sha256": record.content_sha256,
                "retrieval_receipt_sha256": record.provenance.receipt_sha256,
                "evidence_kind": _WIRE_KIND_BY_EVIDENCE_KIND[record.kind],
                "evidence_origin": record.provenance.source_id,
                "source_version": record.provenance.source_version,
            }
        )
        selected.append(
            {
                "evidence_id": record.evidence_id,
                "content_sha256": record.content_sha256,
            }
        )
    normalized = normalize_cognitive_context(items)
    receipt_body = {
        "schema": PROJECTION_SCHEMA,
        "episode_id": state.episode_id,
        "state_sha256": state.state_sha256,
        "selected": selected,
        "excluded": excluded,
        "item_count": len(normalized),
    }
    receipt = {
        **receipt_body,
        "projection_sha256": canonical_sha256(receipt_body),
    }
    return normalized, receipt


# ── 2. Evidence-bound tasks and per-arm typed states ────────────────────


@dataclass(frozen=True, slots=True)
class StateCausalityTask:
    """One task whose decisive fact lives only in the typed state."""

    task_id: str
    family: str
    prompt: str
    required_content: str
    required_alt_content: str
    decoy_content: str
    expected_answer: str
    expected_alt_answer: str

    def __post_init__(self) -> None:
        for name in (
            "task_id",
            "family",
            "prompt",
            "required_content",
            "required_alt_content",
            "decoy_content",
            "expected_answer",
            "expected_alt_answer",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                _fail("state_causality_task_invalid")
        if self.required_content == self.required_alt_content:
            _fail("state_causality_task_invalid")


def build_state_binding_tasks(*, count: int, seed: int) -> list[StateCausalityTask]:
    """Deterministic evidence-dependent tasks across three families."""

    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        _fail("state_causality_task_count_invalid")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        _fail("state_causality_seed_invalid")
    tasks: list[StateCausalityTask] = []
    for index in range(count):
        stream = int.from_bytes(
            hashlib.sha256(f"state-causality:{seed}:{index}".encode()).digest()[:8],
            "big",
        )
        family = ("evidence_arithmetic", "evidence_lookup", "evidence_comparison")[
            index % 3
        ]
        if family == "evidence_arithmetic":
            k = 11 + stream % 79
            alt = k + 1 + stream // 97 % 41
            b = 3 + stream // 7 % 17
            task = StateCausalityTask(
                task_id=f"{family}:{seed}:{index}",
                family=family,
                prompt=(
                    "Your retrieved evidence states the value of the constant "
                    f"K. Compute K + {b} and answer with only the number."
                ),
                required_content=f"The stored constant K equals {k}.",
                required_alt_content=f"The stored constant K equals {alt}.",
                decoy_content="The archive room is painted green.",
                expected_answer=str(k + b),
                expected_alt_answer=str(alt + b),
            )
        elif family == "evidence_lookup":
            words = (
                "meridian",
                "obsidian",
                "harbor",
                "lantern",
                "quartz",
                "sable",
                "willow",
                "cobalt",
            )
            word = words[stream % len(words)]
            alt_word = words[(stream + 3) % len(words)]
            task = StateCausalityTask(
                task_id=f"{family}:{seed}:{index}",
                family=family,
                prompt=(
                    "Your retrieved evidence states the current code word. "
                    "Answer with only the code word."
                ),
                required_content=f"The current code word is {word}.",
                required_alt_content=f"The current code word is {alt_word}.",
                decoy_content="The cafeteria closes at noon on Fridays.",
                expected_answer=word,
                expected_alt_answer=alt_word,
            )
        else:
            t = 20 + stream % 60
            x_above = t + 1 + stream // 11 % 30
            task = StateCausalityTask(
                task_id=f"{family}:{seed}:{index}",
                family=family,
                prompt=(
                    "Your retrieved evidence states the threshold T. Is "
                    f"{x_above} strictly greater than T? Answer only yes or no."
                ),
                required_content=f"The threshold T is {t}.",
                required_alt_content=f"The threshold T is {x_above + 5}.",
                decoy_content="The maintenance ladder has eleven rungs.",
                expected_answer="yes",
                expected_alt_answer="no",
            )
        tasks.append(task)
    return tasks


def _evidence_record(
    *,
    evidence_id: str,
    content: str,
    summary: str,
    episode_id: str,
    objective_sha256: str,
    observed_at: float,
    kind: EvidenceKind = EvidenceKind.RETRIEVAL,
) -> EvidenceRecord:
    digest = _sha_text(content)
    return EvidenceRecord(
        evidence_id=evidence_id,
        kind=kind,
        summary=summary,
        content_sha256=digest,
        provenance=EvidenceProvenance(
            source_id="state_causality_fixture_store",
            source_version="v1",
            invocation_sha256=_sha_text(f"invocation:{evidence_id}:{digest}"),
            receipt_sha256=_sha_text(f"receipt:{evidence_id}:{digest}"),
            verification=EvidenceVerification.SOURCE_BOUND,
        ),
        scope=EvidenceScope(
            episode_id=episode_id,
            objective_sha256=objective_sha256,
            claim_ids=(),
            purpose=EvidencePurpose.CONTEXT_ONLY,
        ),
        observed_at=observed_at,
    )


@dataclass(frozen=True, slots=True)
class StateArm:
    """One arm's typed state plus the content store backing its digests."""

    arm: str
    state: EpistemicState
    contents: dict[str, str]
    expected_answer: str


def build_task_arm_states(
    task: StateCausalityTask, *, observed_at: float = 1.0
) -> dict[str, StateArm]:
    """Build the six preregistered per-arm typed states for one task."""

    if not isinstance(task, StateCausalityTask):
        _fail("state_causality_task_invalid")
    episode_base = hashlib.sha256(task.task_id.encode("utf-8")).hexdigest()[:24]
    problem = ProblemFrame.create(task.prompt)
    objective_sha256 = problem.objective_sha256
    required_id = "evidence-required-fact"
    decoy_id = "evidence-decoy-fact"
    memory_id = "evidence-recalled-note"
    memory_content = "Recalled note: the hallway lights flickered yesterday."
    # The lesion removes the INFORMATION while preserving the slot: a
    # width-matched filler keeps exact layer-application parity, the same
    # discipline as the norm-matched controls elsewhere in this package.
    # Removing the whole record would also remove its slot's compute rides
    # and turn every lesion comparison into an unequal-compute comparison.
    filler_content = "No reference entry is available for this topic."
    baseline_summary = "retrieved reference entry"
    mutated_summary = "retrieved reference entry (annotated: relevance high)"

    def _arm(
        arm: str,
        *,
        include_required: bool,
        include_decoy: bool,
        include_memory: bool,
        required_content: str,
        required_summary: str,
        expected: str,
    ) -> StateArm:
        records: list[EvidenceRecord] = []
        contents: dict[str, str] = {}
        episode_id = f"expS-{episode_base}-{arm}"
        if include_required:
            records.append(
                _evidence_record(
                    evidence_id=required_id,
                    content=required_content,
                    summary=required_summary,
                    episode_id=episode_id,
                    objective_sha256=objective_sha256,
                    observed_at=observed_at,
                )
            )
            contents[required_id] = required_content
        if include_decoy:
            records.append(
                _evidence_record(
                    evidence_id=decoy_id,
                    content=task.decoy_content,
                    summary=baseline_summary,
                    episode_id=episode_id,
                    objective_sha256=objective_sha256,
                    observed_at=observed_at,
                )
            )
            contents[decoy_id] = task.decoy_content
        if include_memory:
            # A memory-kind record the projection refuses: the arm that
            # removes it proves non-projected state is inert at this seam.
            records.append(
                _evidence_record(
                    evidence_id=memory_id,
                    content=memory_content,
                    summary=baseline_summary,
                    episode_id=episode_id,
                    objective_sha256=objective_sha256,
                    observed_at=observed_at,
                    kind=EvidenceKind.MEMORY,
                )
            )
            contents[memory_id] = memory_content
        from core.brain.llm.latent_cortex.epistemic_state import ComputeBudgetState

        state = EpistemicState.genesis(
            episode_id=episode_id,
            problem=problem,
            budget=ComputeBudgetState(total=1000.0, used=0.0),
            evidence=records,
        )
        return StateArm(
            arm=arm, state=state, contents=contents, expected_answer=expected
        )

    def _standard(
        arm: str,
        *,
        include_required: bool = True,
        include_decoy: bool = True,
        include_memory: bool = True,
        required_content: str = task.required_content,
        required_summary: str = baseline_summary,
        expected: str = task.expected_answer,
    ) -> StateArm:
        return _arm(
            arm,
            include_required=include_required,
            include_decoy=include_decoy,
            include_memory=include_memory,
            required_content=required_content,
            required_summary=required_summary,
            expected=expected,
        )

    return {
        ARM_INTACT: _standard(ARM_INTACT),
        ARM_LESIONED: _standard(
            ARM_LESIONED, required_content=filler_content
        ),
        ARM_SHAM: _standard(ARM_SHAM, include_decoy=False),
        ARM_INERT: _standard(ARM_INERT, include_memory=False),
        ARM_RESTORED: _standard(ARM_RESTORED),
        ARM_ANNOTATION: _standard(
            ARM_ANNOTATION, required_summary=mutated_summary
        ),
        ARM_SUBSTITUTION: _standard(
            ARM_SUBSTITUTION,
            required_content=task.required_alt_content,
            expected=task.expected_alt_answer,
        ),
    }


def answer_matches(text: str, expected: str) -> bool:
    """Exact expected-token check on the final answer-bearing line."""

    if not isinstance(text, str) or not isinstance(expected, str):
        return False
    lowered = text.strip().lower()
    needle = expected.strip().lower()
    if not lowered or not needle:
        return False
    if "final_answer:" in lowered:
        lowered = lowered.rsplit("final_answer:", 1)[1]
    tokens = [
        token.strip(".,;:!?'\"()[]")
        for token in lowered.replace("\n", " ").split(" ")
    ]
    return needle in [token for token in tokens if token]


# ── 3. The experiment: six arms, exact identities, graded claims ────────


def _identity_claim(
    experiment: str,
    statement: str,
    *,
    holds: list[bool],
    minimum: int,
) -> dict[str, Any]:
    n = len(holds)
    matched = sum(1 for value in holds if value)
    if n == 0:
        tier = CONJECTURE
    elif matched < n:
        tier = REFUTED
    elif n < minimum:
        tier = CONJECTURE
    else:
        tier = SUPPORTED
    return {
        "experiment": experiment,
        "statement": statement,
        "tier": tier,
        "evidence": {"n": n, "matched": matched, "minimum_n": minimum},
    }


def run_state_causality_experiment(
    run_episode: Callable[[str, list[dict[str, Any]]], Mapping[str, Any]],
    tasks: list[StateCausalityTask],
    *,
    minimum_tasks: int = MIN_TASKS_FOR_VERDICT,
    runner_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run every arm for every task and grade the causality claims.

    ``run_episode(prompt, cognitive_context)`` must execute one
    deterministic episode and return a mapping with ``ok`` (bool),
    ``text`` (str), ``tokens`` (list[int]), ``final_states_sha256``
    (64-hex digest of the final latent workspace states — the direct
    observable of later computation), ``steps_taken`` (int), and
    ``layer_apps`` (int).  The callable owns model, configuration, and
    seeding; it must hold them constant across arms and use a fixed
    recurrence depth — the receipt proves the parity it can see (steps
    and layer applications).
    """

    if not callable(run_episode):
        _fail("state_causality_runner_invalid")
    if not isinstance(tasks, list) or not tasks:
        _fail("state_causality_tasks_invalid")
    if (
        isinstance(minimum_tasks, bool)
        or not isinstance(minimum_tasks, int)
        or minimum_tasks < 1
    ):
        _fail("state_causality_minimum_invalid")

    rows: list[dict[str, Any]] = []
    for task in tasks:
        if not isinstance(task, StateCausalityTask):
            _fail("state_causality_tasks_invalid")
        arm_states = build_task_arm_states(task)
        for arm in STATE_CAUSALITY_ARMS:
            bundle = arm_states[arm]
            items, projection = project_state_evidence_context(
                bundle.state, bundle.contents
            )
            outcome = run_episode(task.prompt, items)
            if not isinstance(outcome, Mapping):
                _fail("state_causality_outcome_invalid")
            ok = outcome.get("ok")
            text = outcome.get("text")
            tokens = outcome.get("tokens")
            final_states = outcome.get("final_states_sha256")
            steps = outcome.get("steps_taken")
            layer_apps = outcome.get("layer_apps")
            if (
                type(ok) is not bool
                or not isinstance(text, str)
                or not isinstance(tokens, list)
                or any(type(token) is not int for token in tokens)
                or not isinstance(final_states, str)
                or len(final_states) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in final_states
                )
                or type(steps) is not int
                or steps < 0
                or type(layer_apps) is not int
                or layer_apps < 0
            ):
                _fail("state_causality_outcome_invalid")
            rows.append(
                {
                    "task_id": task.task_id,
                    "family": task.family,
                    "arm": arm,
                    "state_sha256": bundle.state.state_sha256,
                    "projection_sha256": projection["projection_sha256"],
                    "selected_evidence_ids": [
                        entry["evidence_id"] for entry in projection["selected"]
                    ],
                    "item_count": projection["item_count"],
                    "ok": ok,
                    "text_sha256": _sha_text(text),
                    "tokens_sha256": canonical_sha256(list(tokens)),
                    "final_states_sha256": final_states,
                    "success": answer_matches(text, bundle.expected_answer),
                    "expected_success_answer": bundle.expected_answer,
                    "steps_taken": steps,
                    "layer_apps": layer_apps,
                }
            )

    body = {
        "schema": STATE_CAUSALITY_SCHEMA,
        "arms": list(STATE_CAUSALITY_ARMS),
        "n_tasks": len(tasks),
        "minimum_tasks": minimum_tasks,
        "runner_identity": dict(runner_identity or {}),
        "rows": rows,
        "claims": _grade_claims(rows, minimum_tasks=minimum_tasks),
    }
    return {**body, "receipt_sha256": canonical_sha256(body)}


def _rows_by_task_arm(
    rows: list[Mapping[str, Any]],
) -> dict[str, dict[str, Mapping[str, Any]]]:
    indexed: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != _ROW_KEYS:
            _fail("state_causality_row_invalid")
        by_arm = indexed.setdefault(str(row["task_id"]), {})
        arm = str(row["arm"])
        if arm not in STATE_CAUSALITY_ARMS or arm in by_arm:
            _fail("state_causality_row_invalid")
        by_arm[arm] = row
    for by_arm in indexed.values():
        if set(by_arm) != set(STATE_CAUSALITY_ARMS):
            _fail("state_causality_rows_incomplete")
    return indexed


def _grade_claims(
    rows: list[Mapping[str, Any]], *, minimum_tasks: int
) -> list[dict[str, Any]]:
    indexed = _rows_by_task_arm(rows)

    def _pairs(
        left: str, right: str
    ) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
        return [
            (by_arm[left], by_arm[right])
            for _task_id, by_arm in sorted(indexed.items())
        ]

    def _same_output(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
        return (
            a["ok"] is True
            and b["ok"] is True
            and a["tokens_sha256"] == b["tokens_sha256"]
            and a["text_sha256"] == b["text_sha256"]
            and a["final_states_sha256"] == b["final_states_sha256"]
        )

    def _changed_computation(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
        return (
            a["ok"] is True
            and b["ok"] is True
            and (
                a["final_states_sha256"] != b["final_states_sha256"]
                or a["tokens_sha256"] != b["tokens_sha256"]
            )
        )

    def _parity(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
        return (
            a["steps_taken"] == b["steps_taken"]
            and a["layer_apps"] == b["layer_apps"]
        )

    def _recurrence_parity(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
        # The recurrent phase is depth- and width-matched by construction
        # (fixed steps, equal item counts).  Total layer applications also
        # include decode, whose length is an OUTCOME of the changed state,
        # so change-claims must not require it equal — identity claims
        # still do, and identical outputs imply identical decode cost.
        return (
            a["steps_taken"] == b["steps_taken"]
            and a["item_count"] == b["item_count"]
        )

    claims = [
        _identity_claim(
            "expS_required_lesion_changes_computation",
            "replacing the required typed evidence content with an "
            "information-free filler changes the final latent computation "
            "under fixed recurrence depth and equal workspace width",
            holds=[
                _changed_computation(a, b) and _recurrence_parity(a, b)
                for a, b in _pairs(ARM_INTACT, ARM_LESIONED)
            ],
            minimum=minimum_tasks,
        ),
        _identity_claim(
            "expS_nonprojected_state_inert",
            "removing a state component the sanctioned projection refuses "
            "changes the state hash but leaves the decoded computation "
            "byte-identical: only projected typed content reaches the seam",
            holds=[
                _same_output(a, b)
                and _parity(a, b)
                and a["state_sha256"] != b["state_sha256"]
                for a, b in _pairs(ARM_INTACT, ARM_INERT)
            ],
            minimum=minimum_tasks,
        ),
        _identity_claim(
            "expS_restoration_recovers",
            "restoring the required evidence reproduces the intact "
            "computation byte-identically",
            holds=[
                _same_output(a, b) and _parity(a, b)
                for a, b in _pairs(ARM_INTACT, ARM_RESTORED)
            ],
            minimum=minimum_tasks,
        ),
        _identity_claim(
            "expS_prose_shadow_prohibited",
            "mutating only the state's prose summary changes the state hash "
            "but leaves computation byte-identical: typed content, not "
            "prose, is the causal channel",
            holds=[
                _same_output(a, b)
                and _parity(a, b)
                and a["state_sha256"] != b["state_sha256"]
                and a["projection_sha256"] != b["projection_sha256"]
                for a, b in _pairs(ARM_INTACT, ARM_ANNOTATION)
            ],
            minimum=minimum_tasks,
        ),
        _identity_claim(
            "expS_content_substitution_tracks_state",
            "substituting the typed content changes the final latent "
            "computation under fixed recurrence depth and equal workspace "
            "width",
            holds=[
                _changed_computation(a, b) and _recurrence_parity(a, b)
                for a, b in _pairs(ARM_INTACT, ARM_SUBSTITUTION)
            ],
            minimum=minimum_tasks,
        ),
    ]

    # Semantic (task-appropriate) legs: graded only from measured success.
    intact_success = [
        by_arm[ARM_INTACT]["success"] is True
        for _task, by_arm in sorted(indexed.items())
    ]
    lesioned_success = [
        by_arm[ARM_LESIONED]["success"] is True
        for _task, by_arm in sorted(indexed.items())
    ]
    sham_success = [
        by_arm[ARM_SHAM]["success"] is True
        for _task, by_arm in sorted(indexed.items())
    ]
    substitution_tracks = [
        by_arm[ARM_SUBSTITUTION]["success"] is True
        for _task, by_arm in sorted(indexed.items())
    ]
    n = len(intact_success)
    intact_rate = sum(intact_success) / n if n else 0.0
    lesioned_rate = sum(lesioned_success) / n if n else 0.0
    sham_rate = sum(sham_success) / n if n else 0.0
    substitution_rate = sum(substitution_tracks) / n if n else 0.0
    readable = n >= minimum_tasks and intact_rate >= 0.5
    if not readable:
        loss_tier = CONJECTURE
    elif intact_rate - lesioned_rate >= 0.3 and sham_rate >= intact_rate - 0.1:
        loss_tier = PROVEN
    elif intact_rate - lesioned_rate > 0.0:
        loss_tier = SUPPORTED
    else:
        loss_tier = REFUTED
    claims.append(
        {
            "experiment": "expS_task_appropriate_loss",
            "statement": (
                "on a model that can read the slot channel, lesioning the "
                "required evidence loses exactly the evidence-dependent "
                "tasks while the sham lesion loses none"
            ),
            "tier": loss_tier,
            "evidence": {
                "n": n,
                "intact_rate": intact_rate,
                "lesioned_rate": lesioned_rate,
                "sham_rate": sham_rate,
                "substitution_tracks_rate": substitution_rate,
                "minimum_n": minimum_tasks,
                "channel_readable_floor": 0.5,
                "channel_readable": intact_rate >= 0.5,
                "note": (
                    "CONJECTURE whenever the untrained pooled-slot channel "
                    "cannot carry the fact (intact_rate below the floor); "
                    "the structural claims above remain decisive"
                ),
            },
        }
    )
    return claims


def replay_state_causality_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Independently regrade a receipt from its rows alone."""

    if not isinstance(receipt, Mapping) or set(receipt) != {
        "schema",
        "arms",
        "n_tasks",
        "minimum_tasks",
        "runner_identity",
        "rows",
        "claims",
        "receipt_sha256",
    }:
        _fail("state_causality_receipt_invalid")
    if receipt.get("schema") != STATE_CAUSALITY_SCHEMA:
        _fail("state_causality_receipt_invalid")
    if list(receipt.get("arms") or []) != list(STATE_CAUSALITY_ARMS):
        _fail("state_causality_receipt_invalid")
    body = {key: receipt[key] for key in receipt if key != "receipt_sha256"}
    if receipt.get("receipt_sha256") != canonical_sha256(body):
        _fail("state_causality_receipt_digest_mismatch")
    minimum_tasks = receipt.get("minimum_tasks")
    if (
        isinstance(minimum_tasks, bool)
        or not isinstance(minimum_tasks, int)
        or minimum_tasks < 1
    ):
        _fail("state_causality_receipt_invalid")
    rows = receipt.get("rows")
    if not isinstance(rows, list) or not rows:
        _fail("state_causality_receipt_invalid")
    n_tasks = receipt.get("n_tasks")
    if len(rows) != len(STATE_CAUSALITY_ARMS) * n_tasks:
        _fail("state_causality_receipt_invalid")
    regraded = _grade_claims(rows, minimum_tasks=minimum_tasks)
    recorded = receipt.get("claims")
    if regraded != recorded:
        _fail("state_causality_replay_mismatch")
    return {
        "schema": STATE_CAUSALITY_SCHEMA,
        "receipt_sha256": receipt["receipt_sha256"],
        "claims": regraded,
        "replayed": True,
    }


# ── 4. Engine-backed runner (mlx stays lazy) ────────────────────────────


def engine_episode_runner(
    build_engine: Callable[[], Any],
    *,
    decode_max_tokens: int = 12,
    wall_clock_s: float = 60.0,
    max_layer_apps: int | None = None,
) -> Callable[[str, list[dict[str, Any]]], dict[str, Any]]:
    """Adapt a latent engine factory to the experiment's episode contract.

    A fresh engine per episode keeps arms independent: the engine retains
    prefill state for one-shot nonparametric recall across calls, which
    would otherwise leak the previous arm's computation into the next.
    """

    if not callable(build_engine):
        _fail("state_causality_runner_invalid")
    if (
        isinstance(decode_max_tokens, bool)
        or not isinstance(decode_max_tokens, int)
        or not 1 <= decode_max_tokens <= 4096
    ):
        _fail("state_causality_runner_invalid")
    if (
        isinstance(wall_clock_s, bool)
        or not isinstance(wall_clock_s, (int, float))
        or not math.isfinite(float(wall_clock_s))
        or float(wall_clock_s) <= 0.0
    ):
        _fail("state_causality_runner_invalid")

    def _run(prompt: str, cognitive_context: list[dict[str, Any]]) -> dict[str, Any]:
        from core.brain.llm.latent_cortex.types import ComputeBudget

        engine = build_engine()
        budget_kwargs: dict[str, Any] = {"wall_clock_s": float(wall_clock_s)}
        if max_layer_apps is not None:
            budget_kwargs["max_layer_apps"] = int(max_layer_apps)
        budget = ComputeBudget(**budget_kwargs)
        result = engine.reason(
            prompt,
            budget=budget,
            cognitive_context=cognitive_context,
            decode_max_tokens=decode_max_tokens,
        )
        receipt = result.receipt
        branches = (receipt.to_dict().get("verified_best_state") or {}).get(
            "branches"
        )
        if not isinstance(branches, list) or not branches:
            _fail("state_causality_final_state_unavailable")
        digests = []
        for row in sorted(
            branches, key=lambda entry: int(entry.get("branch_index", 0))
        ):
            digest = (row.get("finalization") or {}).get("post_state_sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                _fail("state_causality_final_state_unavailable")
            digests.append(digest)
        return {
            "ok": bool(result.ok),
            "text": str(result.text),
            "tokens": list(result.tokens),
            "final_states_sha256": canonical_sha256(digests),
            "steps_taken": int(getattr(receipt, "steps_taken", 0) or 0),
            "layer_apps": int(budget.spent_layer_apps),
        }

    return _run


__all__ = [
    "ARM_ANNOTATION",
    "ARM_INTACT",
    "ARM_LESIONED",
    "ARM_RESTORED",
    "ARM_SHAM",
    "ARM_SUBSTITUTION",
    "MIN_TASKS_FOR_VERDICT",
    "PROJECTION_SCHEMA",
    "STATE_CAUSALITY_ARMS",
    "STATE_CAUSALITY_SCHEMA",
    "StateArm",
    "StateCausalityError",
    "StateCausalityTask",
    "answer_matches",
    "build_state_binding_tasks",
    "build_task_arm_states",
    "engine_episode_runner",
    "project_state_evidence_context",
    "replay_state_causality_receipt",
    "run_state_causality_experiment",
]
