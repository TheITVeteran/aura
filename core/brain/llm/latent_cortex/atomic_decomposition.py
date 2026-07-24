"""Machine-checkable atomic decomposition for candidate verification.

SPARK-039 requires verifier inputs to be explicit before any holistic score can
acquire authority.  This module converts a bounded candidate into text-free,
content-addressed claim spans and typed dependency transitions.  Its validator
reconstructs source coverage, graph topology, connective obligations, and
commitments from the original candidate instead of trusting producer totals.

The boundary is deliberately structural.  A complete receipt proves that the
candidate was decomposed without hidden text gaps or omitted *declared*
dependencies; it does not prove that any claim is true.  Domain verifiers in
later SPARK checkpoints grade the resulting atoms independently.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from enum import StrEnum
from typing import Any

ATOMIC_DECOMPOSITION_SCHEMA = "aura.rlc.atomic_decomposition.v1"
MAX_SOURCE_CHARS = 32_768
MAX_ATOMS = 256
MAX_ATOM_CHARS = 512
MAX_TRANSITIONS = 512

_WORD_RE = re.compile(r"[^\W\d_]{3,}", re.UNICODE)
_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\n.*?```", re.DOTALL)
_BOUNDARY_RE = re.compile(
    r"(?<=[.!?;])(?=\s|$)|\n+|(?=\b(?:because|therefore|thus|hence|however|"
    r"although|unless|consequently|which\s+means|so\s+that)\b)|"
    r"(?<=,)(?=\s+(?:and|but|or)\b)",
    re.IGNORECASE,
)
_CONCLUSION_RE = re.compile(r"^(?:therefore|thus|hence|consequently|so)\b", re.I)
_SUPPORT_RE = re.compile(r"^(?:because|since|given\s+that|as\s+a\s+result\s+of)\b", re.I)
_CONTRAST_RE = re.compile(r"^(?:however|although|but|yet|nevertheless)\b", re.I)
_CONDITION_RE = re.compile(r"^(?:if|unless|provided\s+that|assuming)\b", re.I)
_REFERENCE_RE = re.compile(
    r"^(?:this|that|these|those|it|they|which|the\s+former|the\s+latter)\b",
    re.I,
)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class AtomKind(StrEnum):
    ASSERTION = "assertion"
    CONDITION = "condition"
    SUPPORT = "support"
    CONCLUSION = "conclusion"
    CONTRAST = "contrast"
    CODE = "code"


class TransitionKind(StrEnum):
    SUPPORTS = "supports"
    DERIVES = "derives"
    CONDITIONS = "conditions"
    QUALIFIES = "qualifies"
    REFERENCES = "references"


def _canonical_sha256(value: Any) -> str:
    rendered = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _trimmed_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


def _bounded_plain_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = start
    for match in _BOUNDARY_RE.finditer(text, start, end):
        boundary = match.start()
        span = _trimmed_span(text, cursor, boundary)
        if span is not None:
            spans.extend(_split_oversized_span(text, *span))
        cursor = match.end() if match.end() > boundary else boundary
    span = _trimmed_span(text, cursor, end)
    if span is not None:
        spans.extend(_split_oversized_span(text, *span))
    return spans


def _split_oversized_span(text: str, start: int, end: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = start
    while end - cursor > MAX_ATOM_CHARS:
        ceiling = cursor + MAX_ATOM_CHARS
        split = text.rfind(" ", cursor + 1, ceiling + 1)
        if split <= cursor:
            split = ceiling
        span = _trimmed_span(text, cursor, split)
        if span is not None:
            spans.append(span)
        cursor = split
        while cursor < end and text[cursor].isspace():
            cursor += 1
    span = _trimmed_span(text, cursor, end)
    if span is not None:
        spans.append(span)
    return spans


def _candidate_spans(text: str) -> list[tuple[int, int, bool]]:
    spans: list[tuple[int, int, bool]] = []
    cursor = 0
    for fence in _FENCE_RE.finditer(text):
        spans.extend(
            (start, end, False) for start, end in _bounded_plain_spans(text, cursor, fence.start())
        )
        trimmed = _trimmed_span(text, fence.start(), fence.end())
        if trimmed is not None:
            spans.extend((start, end, True) for start, end in _split_oversized_span(text, *trimmed))
        cursor = fence.end()
    spans.extend(
        (start, end, False) for start, end in _bounded_plain_spans(text, cursor, len(text))
    )
    if len(spans) > MAX_ATOMS:
        raise ValueError(f"atomic decomposition exceeds {MAX_ATOMS} atoms")
    return spans


def _dependency_cues(fragment: str) -> tuple[str, ...]:
    cues: list[str] = []
    for name, pattern in (
        ("conclusion", _CONCLUSION_RE),
        ("support", _SUPPORT_RE),
        ("contrast", _CONTRAST_RE),
        ("condition", _CONDITION_RE),
        ("reference", _REFERENCE_RE),
    ):
        if pattern.search(fragment):
            cues.append(name)
    return tuple(cues)


def _atom_kind(fragment: str, *, code: bool) -> AtomKind:
    if code:
        return AtomKind.CODE
    if _CONCLUSION_RE.search(fragment):
        return AtomKind.CONCLUSION
    if _SUPPORT_RE.search(fragment):
        return AtomKind.SUPPORT
    if _CONTRAST_RE.search(fragment):
        return AtomKind.CONTRAST
    if _CONDITION_RE.search(fragment):
        return AtomKind.CONDITION
    return AtomKind.ASSERTION


def _transition_for_cue(
    *,
    atom_index: int,
    atom_id: str,
    previous_id: str | None,
    cue: str,
) -> dict[str, Any] | None:
    if previous_id is None:
        return None
    if cue == "support":
        premise_ids = [atom_id]
        output_id = previous_id
        kind = TransitionKind.SUPPORTS
    else:
        premise_ids = [previous_id]
        output_id = atom_id
        kind = {
            "conclusion": TransitionKind.DERIVES,
            "condition": TransitionKind.CONDITIONS,
            "contrast": TransitionKind.QUALIFIES,
            "reference": TransitionKind.REFERENCES,
        }[cue]
    payload = {
        "transition_id": f"t{atom_index:03d}.{cue}",
        "kind": kind.value,
        "premise_ids": premise_ids,
        "output_id": output_id,
        "cue": cue,
    }
    return {**payload, "transition_sha256": _canonical_sha256(payload)}


def _meaningful_indices(text: str) -> set[int]:
    return {index for index, char in enumerate(text) if not char.isspace()}


def _objective_terms(objective: str) -> set[str]:
    return {word.lower() for word in _WORD_RE.findall(objective) if len(word) >= 4}


def build_atomic_decomposition(candidate: str, *, objective: str = "") -> dict[str, Any]:
    """Build and validate a text-free decomposition receipt."""

    if not isinstance(candidate, str):
        raise TypeError("candidate must be a string")
    if not isinstance(objective, str):
        raise TypeError("objective must be a string")
    if len(candidate) > MAX_SOURCE_CHARS:
        raise ValueError(f"candidate exceeds {MAX_SOURCE_CHARS} characters")
    if _CONTROL_RE.search(candidate):
        raise ValueError("candidate contains control characters")

    atoms: list[dict[str, Any]] = []
    for index, (start, end, code) in enumerate(_candidate_spans(candidate)):
        fragment = candidate[start:end]
        atom_id = f"a{index:03d}"
        payload = {
            "atom_id": atom_id,
            "kind": _atom_kind(fragment, code=code).value,
            "start": start,
            "end": end,
            "chars": end - start,
            "text_sha256": _text_sha256(fragment),
            "dependency_cues": list(_dependency_cues(fragment)),
        }
        atoms.append({**payload, "atom_sha256": _canonical_sha256(payload)})

    transitions: list[dict[str, Any]] = []
    for index, atom in enumerate(atoms):
        previous_id = atoms[index - 1]["atom_id"] if index else None
        for cue in atom["dependency_cues"]:
            transition = _transition_for_cue(
                atom_index=index,
                atom_id=atom["atom_id"],
                previous_id=previous_id,
                cue=cue,
            )
            if transition is not None:
                transitions.append(transition)
    if len(transitions) > MAX_TRANSITIONS:
        raise ValueError(f"atomic decomposition exceeds {MAX_TRANSITIONS} transitions")

    covered = {
        index
        for atom in atoms
        for index in range(int(atom["start"]), int(atom["end"]))
        if not candidate[index].isspace()
    }
    meaningful = _meaningful_indices(candidate)
    linked_cues = {(row["output_id"], row["cue"]) for row in transitions} | {
        (row["premise_ids"][0], row["cue"]) for row in transitions if row["cue"] == "support"
    }
    if atoms and objective.strip():
        # A sentence-initial connective can be grounded by the immutable
        # objective rather than a prior candidate atom. The objective hash in
        # the receipt is the external dependency commitment.
        linked_cues.update((atoms[0]["atom_id"], cue) for cue in atoms[0]["dependency_cues"])
    omitted = [
        atom["atom_id"]
        for atom in atoms
        if any((atom["atom_id"], cue) not in linked_cues for cue in atom["dependency_cues"])
    ]
    candidate_terms = {word.lower() for word in _WORD_RE.findall(candidate)}
    objective_terms = _objective_terms(objective)
    grounded_terms = sorted(objective_terms & candidate_terms)
    payload = {
        "schema": ATOMIC_DECOMPOSITION_SCHEMA,
        "source_sha256": _text_sha256(candidate),
        "objective_sha256": _text_sha256(objective),
        "source_chars": len(candidate),
        "atoms": atoms,
        "transitions": transitions,
        "coverage": {
            "meaningful_chars": len(meaningful),
            "covered_chars": len(covered),
            "coverage_ratio": round(len(covered) / len(meaningful), 8) if meaningful else 0.0,
            "uncovered_indices": sorted(meaningful - covered)[:64],
        },
        "dependencies": {
            "cue_count": sum(len(atom["dependency_cues"]) for atom in atoms),
            "linked_cue_count": sum(len(atom["dependency_cues"]) for atom in atoms) - len(omitted),
            "omitted_dependency_atom_ids": omitted,
        },
        "objective_grounding": {
            "objective_term_count": len(objective_terms),
            "grounded_term_count": len(grounded_terms),
            "grounded_terms_sha256": _canonical_sha256(grounded_terms),
        },
        "grade_admissible": bool(atoms) and covered == meaningful and not omitted,
    }
    receipt = {**payload, "receipt_sha256": _canonical_sha256(payload)}
    return validate_atomic_decomposition(receipt, candidate=candidate, objective=objective)


def _validate_transition_graph(
    atoms: list[Mapping[str, Any]],
    transitions: list[Mapping[str, Any]],
) -> None:
    atom_ids = {str(atom["atom_id"]) for atom in atoms}
    edges: dict[str, set[str]] = {atom_id: set() for atom_id in atom_ids}
    for transition in transitions:
        premises = transition.get("premise_ids")
        output_id = transition.get("output_id")
        if (
            not isinstance(premises, list)
            or not premises
            or not all(isinstance(item, str) and item in atom_ids for item in premises)
            or not isinstance(output_id, str)
            or output_id not in atom_ids
            or output_id in premises
        ):
            raise ValueError("atomic transition references invalid claims")
        for premise in premises:
            edges[premise].add(output_id)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise ValueError("atomic dependency graph contains a cycle")
        if node in visited:
            return
        visiting.add(node)
        for child in edges[node]:
            visit(child)
        visiting.remove(node)
        visited.add(node)

    for atom_id in sorted(atom_ids):
        visit(atom_id)


def validate_atomic_decomposition_envelope(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the text-free cross-process receipt envelope.

    The worker performs source reconstruction with the private candidate.  The
    service independently rechecks commitments, bounded spans, topology,
    omission accounting, and grading authority without receiving candidate
    prose or hidden reasoning.
    """

    if not isinstance(value, Mapping):
        raise ValueError("atomic decomposition must be a mapping")
    fields = {
        "schema",
        "source_sha256",
        "objective_sha256",
        "source_chars",
        "atoms",
        "transitions",
        "coverage",
        "dependencies",
        "objective_grounding",
        "grade_admissible",
        "receipt_sha256",
    }
    if set(value) != fields:
        raise ValueError("atomic decomposition fields do not match schema")
    payload = {key: value[key] for key in fields - {"receipt_sha256"}}
    if value["receipt_sha256"] != _canonical_sha256(payload):
        raise ValueError("atomic decomposition receipt commitment mismatch")
    source_chars = value["source_chars"]
    if (
        value["schema"] != ATOMIC_DECOMPOSITION_SCHEMA
        or not re.fullmatch(r"[0-9a-f]{64}", str(value["source_sha256"]))
        or not re.fullmatch(r"[0-9a-f]{64}", str(value["objective_sha256"]))
        or type(source_chars) is not int
        or not 0 <= source_chars <= MAX_SOURCE_CHARS
    ):
        raise ValueError("atomic decomposition envelope identity is invalid")
    atoms = value["atoms"]
    transitions = value["transitions"]
    if not isinstance(atoms, list) or len(atoms) > MAX_ATOMS:
        raise ValueError("atomic decomposition atom inventory is invalid")
    if not isinstance(transitions, list) or len(transitions) > MAX_TRANSITIONS:
        raise ValueError("atomic decomposition transition inventory is invalid")

    expected_ids = [f"a{index:03d}" for index in range(len(atoms))]
    prior_end = -1
    allowed_cues = {"conclusion", "support", "contrast", "condition", "reference"}
    for expected_id, atom in zip(expected_ids, atoms, strict=True):
        atom_fields = {
            "atom_id",
            "kind",
            "start",
            "end",
            "chars",
            "text_sha256",
            "dependency_cues",
            "atom_sha256",
        }
        if not isinstance(atom, Mapping) or set(atom) != atom_fields:
            raise ValueError("atomic claim fields do not match schema")
        atom_payload = {key: atom[key] for key in atom_fields - {"atom_sha256"}}
        if atom["atom_sha256"] != _canonical_sha256(atom_payload):
            raise ValueError("atomic claim commitment mismatch")
        start, end = atom["start"], atom["end"]
        cues = atom["dependency_cues"]
        if (
            atom["atom_id"] != expected_id
            or atom["kind"] not in {kind.value for kind in AtomKind}
            or type(start) is not int
            or type(end) is not int
            or not 0 <= start < end <= source_chars
            or start < prior_end
            or end - start > MAX_ATOM_CHARS
            or atom["chars"] != end - start
            or not re.fullmatch(r"[0-9a-f]{64}", str(atom["text_sha256"]))
            or not isinstance(cues, list)
            or len(cues) != len(set(cues))
            or any(cue not in allowed_cues for cue in cues)
        ):
            raise ValueError("atomic claim envelope is invalid")
        prior_end = end

    transition_ids: set[str] = set()
    linked_cues: set[tuple[str, str]] = set()
    for transition in transitions:
        transition_fields = {
            "transition_id",
            "kind",
            "premise_ids",
            "output_id",
            "cue",
            "transition_sha256",
        }
        if not isinstance(transition, Mapping) or set(transition) != transition_fields:
            raise ValueError("atomic transition fields do not match schema")
        transition_payload = {
            key: transition[key] for key in transition_fields - {"transition_sha256"}
        }
        if transition["transition_sha256"] != _canonical_sha256(transition_payload):
            raise ValueError("atomic transition commitment mismatch")
        transition_id = transition["transition_id"]
        cue = transition["cue"]
        expected_kind = {
            "conclusion": TransitionKind.DERIVES.value,
            "support": TransitionKind.SUPPORTS.value,
            "contrast": TransitionKind.QUALIFIES.value,
            "condition": TransitionKind.CONDITIONS.value,
            "reference": TransitionKind.REFERENCES.value,
        }.get(cue)
        if (
            not isinstance(transition_id, str)
            or transition_id in transition_ids
            or transition["kind"] != expected_kind
            or cue not in allowed_cues
        ):
            raise ValueError("atomic transition identity is invalid")
        transition_ids.add(transition_id)
        if cue == "support":
            linked_cues.add((transition["premise_ids"][0], cue))
        else:
            linked_cues.add((transition["output_id"], cue))
    _validate_transition_graph(atoms, transitions)

    empty_objective_sha256 = _text_sha256("")
    if atoms and value["objective_sha256"] != empty_objective_sha256:
        linked_cues.update((atoms[0]["atom_id"], cue) for cue in atoms[0]["dependency_cues"])
    omitted = [
        atom["atom_id"]
        for atom in atoms
        if any((atom["atom_id"], cue) not in linked_cues for cue in atom["dependency_cues"])
    ]
    expected_dependencies = {
        "cue_count": sum(len(atom["dependency_cues"]) for atom in atoms),
        "linked_cue_count": sum(len(atom["dependency_cues"]) for atom in atoms) - len(omitted),
        "omitted_dependency_atom_ids": omitted,
    }
    coverage = value["coverage"]
    grounding = value["objective_grounding"]
    if (
        value["dependencies"] != expected_dependencies
        or not isinstance(coverage, Mapping)
        or set(coverage)
        != {"meaningful_chars", "covered_chars", "coverage_ratio", "uncovered_indices"}
        or type(coverage["meaningful_chars"]) is not int
        or type(coverage["covered_chars"]) is not int
        or not 0 <= coverage["covered_chars"] <= coverage["meaningful_chars"] <= source_chars
        or not isinstance(coverage["coverage_ratio"], (int, float))
        or isinstance(coverage["coverage_ratio"], bool)
        or not 0.0 <= float(coverage["coverage_ratio"]) <= 1.0
        or not isinstance(coverage["uncovered_indices"], list)
        or any(
            type(index) is not int or not 0 <= index < source_chars
            for index in coverage["uncovered_indices"]
        )
        or not isinstance(grounding, Mapping)
        or set(grounding)
        != {"objective_term_count", "grounded_term_count", "grounded_terms_sha256"}
        or type(grounding["objective_term_count"]) is not int
        or type(grounding["grounded_term_count"]) is not int
        or not 0 <= grounding["grounded_term_count"] <= grounding["objective_term_count"]
        or not re.fullmatch(r"[0-9a-f]{64}", str(grounding["grounded_terms_sha256"]))
    ):
        raise ValueError("atomic decomposition envelope summary is invalid")
    expected_admissible = (
        bool(atoms)
        and not omitted
        and (
            coverage["meaningful_chars"] == coverage["covered_chars"]
            and float(coverage["coverage_ratio"]) == 1.0
            and coverage["uncovered_indices"] == []
        )
    )
    if value["grade_admissible"] is not expected_admissible:
        raise ValueError("atomic decomposition grading authority is invalid")
    return dict(value)


def validate_atomic_decomposition(
    value: Mapping[str, Any],
    *,
    candidate: str,
    objective: str = "",
) -> dict[str, Any]:
    """Independently validate coverage, dependencies, topology, and hashes."""

    value = validate_atomic_decomposition_envelope(value)
    fields = {
        "schema",
        "source_sha256",
        "objective_sha256",
        "source_chars",
        "atoms",
        "transitions",
        "coverage",
        "dependencies",
        "objective_grounding",
        "grade_admissible",
        "receipt_sha256",
    }
    if set(value) != fields:
        raise ValueError("atomic decomposition fields do not match schema")
    payload = {key: value[key] for key in fields - {"receipt_sha256"}}
    if value["receipt_sha256"] != _canonical_sha256(payload):
        raise ValueError("atomic decomposition receipt commitment mismatch")
    if (
        value["schema"] != ATOMIC_DECOMPOSITION_SCHEMA
        or value["source_sha256"] != _text_sha256(candidate)
        or value["objective_sha256"] != _text_sha256(objective)
        or value["source_chars"] != len(candidate)
    ):
        raise ValueError("atomic decomposition source binding mismatch")

    atoms = value["atoms"]
    transitions = value["transitions"]
    if not isinstance(atoms, list) or len(atoms) > MAX_ATOMS:
        raise ValueError("atomic decomposition atom inventory is invalid")
    if not isinstance(transitions, list) or len(transitions) > MAX_TRANSITIONS:
        raise ValueError("atomic decomposition transition inventory is invalid")

    covered: set[int] = set()
    expected_spans = _candidate_spans(candidate)
    if len(atoms) != len(expected_spans):
        raise ValueError("atomic claim partition differs from source reconstruction")
    expected_ids = [f"a{index:03d}" for index in range(len(atoms))]
    prior_end = -1
    for expected_id, atom, expected_span in zip(
        expected_ids,
        atoms,
        expected_spans,
        strict=True,
    ):
        atom_fields = {
            "atom_id",
            "kind",
            "start",
            "end",
            "chars",
            "text_sha256",
            "dependency_cues",
            "atom_sha256",
        }
        if not isinstance(atom, Mapping) or set(atom) != atom_fields:
            raise ValueError("atomic claim fields do not match schema")
        atom_payload = {key: atom[key] for key in atom_fields - {"atom_sha256"}}
        if atom["atom_sha256"] != _canonical_sha256(atom_payload):
            raise ValueError("atomic claim commitment mismatch")
        start, end = atom["start"], atom["end"]
        expected_start, expected_end, expected_code = expected_span
        if (
            atom["atom_id"] != expected_id
            or atom["kind"] not in {kind.value for kind in AtomKind}
            or type(start) is not int
            or type(end) is not int
            or not 0 <= start < end <= len(candidate)
            or (start, end) != (expected_start, expected_end)
            or start < prior_end
            or end - start > MAX_ATOM_CHARS
            or atom["chars"] != end - start
        ):
            raise ValueError("atomic claim span is invalid")
        fragment = candidate[start:end]
        if (
            atom["text_sha256"] != _text_sha256(fragment)
            or list(_dependency_cues(fragment)) != atom["dependency_cues"]
            or _atom_kind(fragment, code=expected_code).value != atom["kind"]
        ):
            raise ValueError("atomic claim source reconstruction mismatch")
        covered.update(index for index in range(start, end) if not candidate[index].isspace())
        prior_end = end

    transition_ids: set[str] = set()
    linked_cues: set[tuple[str, str]] = set()
    for transition in transitions:
        transition_fields = {
            "transition_id",
            "kind",
            "premise_ids",
            "output_id",
            "cue",
            "transition_sha256",
        }
        if not isinstance(transition, Mapping) or set(transition) != transition_fields:
            raise ValueError("atomic transition fields do not match schema")
        transition_payload = {
            key: transition[key] for key in transition_fields - {"transition_sha256"}
        }
        if transition["transition_sha256"] != _canonical_sha256(transition_payload):
            raise ValueError("atomic transition commitment mismatch")
        transition_id = transition["transition_id"]
        if (
            not isinstance(transition_id, str)
            or transition_id in transition_ids
            or transition["kind"] not in {kind.value for kind in TransitionKind}
            or transition["cue"]
            not in {"conclusion", "support", "contrast", "condition", "reference"}
        ):
            raise ValueError("atomic transition identity is invalid")
        transition_ids.add(transition_id)
        if transition["cue"] == "support":
            linked_cues.add((transition["premise_ids"][0], transition["cue"]))
        else:
            linked_cues.add((transition["output_id"], transition["cue"]))

    if atoms and objective.strip():
        linked_cues.update((atoms[0]["atom_id"], cue) for cue in atoms[0]["dependency_cues"])

    _validate_transition_graph(atoms, transitions)
    meaningful = _meaningful_indices(candidate)
    omitted = [
        atom["atom_id"]
        for atom in atoms
        if any((atom["atom_id"], cue) not in linked_cues for cue in atom["dependency_cues"])
    ]
    expected_coverage = {
        "meaningful_chars": len(meaningful),
        "covered_chars": len(covered),
        "coverage_ratio": round(len(covered) / len(meaningful), 8) if meaningful else 0.0,
        "uncovered_indices": sorted(meaningful - covered)[:64],
    }
    expected_dependencies = {
        "cue_count": sum(len(atom["dependency_cues"]) for atom in atoms),
        "linked_cue_count": sum(len(atom["dependency_cues"]) for atom in atoms) - len(omitted),
        "omitted_dependency_atom_ids": omitted,
    }
    objective_terms = _objective_terms(objective)
    candidate_terms = {word.lower() for word in _WORD_RE.findall(candidate)}
    grounded_terms = sorted(objective_terms & candidate_terms)
    expected_grounding = {
        "objective_term_count": len(objective_terms),
        "grounded_term_count": len(grounded_terms),
        "grounded_terms_sha256": _canonical_sha256(grounded_terms),
    }
    expected_admissible = bool(atoms) and covered == meaningful and not omitted
    if (
        value["coverage"] != expected_coverage
        or value["dependencies"] != expected_dependencies
        or value["objective_grounding"] != expected_grounding
        or value["grade_admissible"] is not expected_admissible
    ):
        raise ValueError("atomic decomposition reconstructed summary mismatch")
    return dict(value)


def decomposition_check(candidate: str, *, objective: str = "") -> dict[str, Any]:
    """Return a bounded task-verifier check without exposing candidate text."""

    try:
        receipt = build_atomic_decomposition(candidate, objective=objective)
    except (TypeError, ValueError) as exc:
        return {
            "applicable": True,
            "valid": False,
            "score": 0.0,
            "failures": [f"atomic_decomposition:{type(exc).__name__}:{exc}"],
            "receipt": None,
        }
    if not candidate.strip():
        return {
            "applicable": False,
            "valid": False,
            "score": None,
            "failures": [],
            "receipt": receipt,
        }
    failures = [
        f"omitted_dependency:{atom_id}"
        for atom_id in receipt["dependencies"]["omitted_dependency_atom_ids"]
    ]
    failures.extend(
        f"uncovered_source_index:{index}" for index in receipt["coverage"]["uncovered_indices"][:8]
    )
    return {
        "applicable": True,
        "valid": bool(receipt["grade_admissible"]),
        "score": 1.0 if receipt["grade_admissible"] else 0.0,
        "failures": failures,
        "receipt": receipt,
    }


def atom_ids(value: Mapping[str, Any]) -> tuple[str, ...]:
    """Expose a stable atom inventory for downstream verifier routing."""

    atoms = value.get("atoms") if isinstance(value, Mapping) else None
    if not isinstance(atoms, list):
        return ()
    return tuple(str(atom.get("atom_id")) for atom in atoms if isinstance(atom, Mapping))
