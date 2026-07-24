from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from core.learning.mistake_locator import (
    MISTAKE_LOCATOR_SCHEMA_V2,
    MistakeLocatorHead,
    MistakeTransitionExample,
    transition_features,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _split(
    relation: str,
    prefix: str,
    domains: tuple[str, str],
    *,
    invert: bool = False,
) -> list[MistakeTransitionExample]:
    rows: list[MistakeTransitionExample] = []
    for trace_index in range(8):
        error_index = trace_index % 4 if trace_index < 4 else None
        trace_id = f"{prefix}-trace-{trace_index}"
        for transition_index in range(4):
            prior = (
                trace_index * 0.02,
                transition_index * 0.03,
                (trace_index + transition_index) * 0.01,
            )
            is_error = transition_index == error_index
            strong = is_error is not invert
            delta = (4.0, -3.5, 3.0) if strong else (0.03, -0.02, 0.01)
            admitted = tuple(value + change for value, change in zip(prior, delta, strict=True))
            rows.append(
                MistakeTransitionExample(
                    example_id=f"{trace_id}-step-{transition_index}",
                    trace_id=trace_id,
                    task_id=f"{prefix}-task-{trace_index}",
                    domain_id=domains[trace_index % len(domains)],
                    relation=relation,
                    mutation_family=("premise_flip" if trace_index % 2 == 0 else "operator_swap"),
                    transition_index=transition_index,
                    transition_count=4,
                    error_index=error_index,
                    prior_hidden=prior,
                    candidate_hidden=admitted,
                    trace_receipt_sha256=_digest(trace_id),
                    outcome_verifier_id=f"verifier-{prefix}",
                )
            )
    return rows


@pytest.fixture
def admitted_head() -> MistakeLocatorHead:
    head = MistakeLocatorHead.fit(
        _split("train", "train", ("logic", "math")),
        _split("in_domain", "cal", ("logic", "math")),
        _split("out_of_domain", "ood", ("code", "planning")),
        hidden_width=8,
        steps=800,
        seed=7,
    )
    assert head.admitted
    return head


def test_feature_map_contains_state_delta_and_magnitude():
    assert transition_features((1.0, 2.0), (4.0, 0.0)) == (
        1.0,
        2.0,
        4.0,
        0.0,
        3.0,
        -2.0,
        3.0,
        2.0,
    )


def test_fit_requires_task_disjoint_and_genuinely_ood_domains():
    train = _split("train", "train", ("logic", "math"))
    in_domain = _split("in_domain", "cal", ("logic", "math"))
    out_of_domain = _split("out_of_domain", "ood", ("code", "planning"))
    out_of_domain = [
        replace(row, domain_id="logic") if row.domain_id == "code" else row for row in out_of_domain
    ]
    with pytest.raises(ValueError, match="OOD domain identities"):
        MistakeLocatorHead.fit(train, in_domain, out_of_domain)


def test_duplicate_trace_evidence_is_rejected():
    train = _split("train", "train", ("logic", "math"))
    duplicate_receipt = train[0].trace_receipt_sha256
    train = [
        replace(row, trace_receipt_sha256=duplicate_receipt)
        if row.trace_id == "train-trace-1"
        else row
        for row in train
    ]
    with pytest.raises(ValueError, match="duplicate trace evidence"):
        MistakeLocatorHead.fit(
            train,
            _split("in_domain", "cal", ("logic", "math")),
            _split("out_of_domain", "ood", ("code", "planning")),
        )


def test_admitted_head_reports_id_and_ood_location_evidence(
    admitted_head: MistakeLocatorHead,
):
    manifest = admitted_head.manifest()
    assert manifest["admitted"] is True
    assert manifest["repair_steering_authorized"] is False
    assert manifest["in_domain_metrics"]["exact_location_accuracy"] >= 0.70
    assert manifest["out_of_domain_metrics"]["error_exact_accuracy"] >= 0.60
    assert manifest["out_of_domain_metrics"]["within_one_accuracy"] >= 0.80
    assert manifest["out_of_domain_metrics"]["no_error_specificity"] >= 0.75
    assert manifest["process_calibration_schema"] == "aura.rlc.process_calibration.v1"
    assert set(manifest["process_calibration"]) == {"in_domain", "out_of_domain"}
    # Global trace admission does not manufacture process authority in sparse
    # domain/depth cells. This fixture is intentionally below cell support.
    assert not any(
        cell["admitted"]
        for relation in manifest["process_calibration"].values()
        for domain in relation.values()
        for cell in domain.values()
    )
    assert admitted_head.probability((0.0, 0.0, 0.0), (4.0, -3.5, 3.0)) > admitted_head.probability(
        (0.0, 0.0, 0.0), (0.03, -0.02, 0.01)
    )


def test_failed_ood_localization_cannot_be_loaded(tmp_path: Path):
    head = MistakeLocatorHead.fit(
        _split("train", "train", ("logic", "math")),
        _split("in_domain", "cal", ("logic", "math")),
        _split(
            "out_of_domain",
            "ood",
            ("code", "planning"),
            invert=True,
        ),
        hidden_width=8,
        steps=800,
        seed=7,
    )
    assert not head.admitted
    assert head.manifest()["failure_reasons"]
    path = tmp_path / "unadmitted.json"
    path.write_text(
        json.dumps(head.to_payload(), sort_keys=True, separators=(",", ":")),
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="failed admission"):
        MistakeLocatorHead.load(
            path,
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )


def test_pinned_artifact_round_trip_and_tamper_refusal(
    tmp_path: Path,
    admitted_head: MistakeLocatorHead,
):
    path = tmp_path / "locator.json"
    digest = admitted_head.save(path)
    loaded = MistakeLocatorHead.load(path, expected_sha256=digest)
    assert loaded.to_payload() == admitted_head.to_payload()

    raw = path.read_bytes()
    path.write_bytes(raw.replace(b'"temperature":', b'"temperature": '))
    with pytest.raises(ValueError, match="SHA-256 differs"):
        MistakeLocatorHead.load(path, expected_sha256=digest)


def test_loader_refuses_symlink(tmp_path: Path, admitted_head: MistakeLocatorHead):
    target = tmp_path / "target.json"
    digest = admitted_head.save(target)
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(OSError):
        MistakeLocatorHead.load(link, expected_sha256=digest)


def test_v2_process_artifact_remains_loadable(tmp_path: Path, admitted_head: MistakeLocatorHead):
    payload = admitted_head.to_payload()
    assert payload["schema"] == MISTAKE_LOCATOR_SCHEMA_V2
    content = {key: payload[key] for key in payload if key != "content_sha256"}
    payload["content_sha256"] = hashlib.sha256(
        json.dumps(
            content,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    path = tmp_path / "legacy-v2.json"
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="ascii",
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = MistakeLocatorHead.load(path, expected_sha256=digest)
    assert "input_representation" not in loaded.manifest()
