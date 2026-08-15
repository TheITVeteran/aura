"""Capability scores were rebuilt from signed booleans, not from grading.

`validate_capability_report` reconstructs every class score from
`correctness_receipt.payload.correct` — a value a pinned verifier signed. The
deterministic graders are in the same module, the answers are in the same
report, and neither was run. Signature agreement proves who said a thing. It
has never proved the thing is true, so a verifier that signed a wrong verdict
produced a score that reproduced perfectly.

The model manifest had the same shape one layer down. `_validate_model_manifest`
checks that the manifest hashes itself, which proves internal consistency and
nothing about the weights that were loaded: every field can be fabricated
together and the self-digest agrees. Nothing ever opened `model_path`. And the
measurement subject is derived FROM the manifest, so on its own it could only
ever agree with it.

Two smaller holes in the same role map: a file could be claimed by two roles —
which matters because the adapter identity check reads `roles["adapters"]`, so a
weights file listed there would be attested as an adapter — and a declared file
could belong to no role at all, shipped with the checkpoint and attested by
nothing.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json

import pytest

import core.brain.frontier_gap as frontier_gap
from core.brain.frontier_gap import (
    MODEL_MANIFEST_RESOLUTION_SCHEMA,
    canonical_json_bytes,
    resolve_model_manifest,
    sha256_json,
)


# ─────────────────────────── the grader runs again


def test_the_graders_are_rerun_in_both_validators():
    source = inspect.getsource(frontier_gap)

    assert source.count("_regrade_against_deterministic_grader(") == 3


def test_a_signed_verdict_that_contradicts_the_grader_is_refused():
    class _Item:
        item_id = "int-3"

        @staticmethod
        def grade(text):
            return text.strip() == "42"

    with pytest.raises(ValueError, match="contradicts the deterministic grader"):
        frontier_gap._regrade_against_deterministic_grader(
            item=_Item(),
            answer="41",
            signed_correct=True,
            subject="capability candidate",
        )


def test_a_signed_verdict_that_agrees_passes():
    class _Item:
        item_id = "int-3"

        @staticmethod
        def grade(text):
            return text.strip() == "42"

    frontier_gap._regrade_against_deterministic_grader(
        item=_Item(),
        answer="42",
        signed_correct=True,
        subject="capability candidate",
    )


def test_a_signed_incorrect_verdict_must_also_agree():
    """The asymmetry matters: signing "wrong" on a right answer suppresses a
    score just as effectively as the reverse inflates one."""

    class _Item:
        item_id = "int-3"

        @staticmethod
        def grade(text):
            return text.strip() == "42"

    with pytest.raises(ValueError, match="signed=False regraded=True"):
        frontier_gap._regrade_against_deterministic_grader(
            item=_Item(),
            answer="42",
            signed_correct=False,
            subject="frontier reference",
        )


def test_the_failure_names_the_item():
    class _Item:
        item_id = "code-7"

        @staticmethod
        def grade(text):
            del text
            return False

    with pytest.raises(ValueError, match="code-7"):
        frontier_gap._regrade_against_deterministic_grader(
            item=_Item(),
            answer="anything",
            signed_correct=True,
            subject="capability candidate",
        )


def test_the_real_graders_are_deterministic_and_executable():
    """The whole fix rests on this: the module holds graders it can rerun."""
    from core.brain.frontier_gap import _exact_integer_grader

    grader = _exact_integer_grader(12)

    assert grader("12") is True
    assert grader("13") is False


# ─────────────────────────── the manifest meets the disk


def _manifest(tmp_path, *, files):
    entries = []
    for name, payload in files.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        entries.append(
            {
                "path": name,
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return {
        "schema": frontier_gap.MODEL_MANIFEST_SCHEMA,
        "model_path": str(tmp_path),
        "files": entries,
        "file_count": len(entries),
        "total_bytes": sum(entry["size"] for entry in entries),
        "roles": {
            "weights": [name for name in files if name.endswith(".safetensors")],
            "configuration": [name for name in files if name.endswith(".json")],
            "tokenizer": [name for name in files if "tokenizer" in name],
            "adapters": [],
        },
    }


def test_a_manifest_that_matches_the_disk_resolves(tmp_path):
    manifest = _manifest(
        tmp_path,
        files={"model.safetensors": b"weights", "config.json": b"{}", "tokenizer.model": b"tok"},
    )

    resolution = resolve_model_manifest(manifest)

    assert resolution["resolved"] is True
    assert resolution["schema"] == MODEL_MANIFEST_RESOLUTION_SCHEMA
    assert resolution["files_present"] == 3
    assert resolution["files_digested"] == 3
    assert resolution["mismatches"] == []


def test_a_fabricated_digest_is_caught(tmp_path):
    manifest = _manifest(tmp_path, files={"config.json": b"{}"})
    manifest["files"][0]["sha256"] = "0" * 64

    resolution = resolve_model_manifest(manifest)

    assert resolution["resolved"] is False
    assert resolution["reason"] == "manifest_does_not_match_disk"
    assert resolution["mismatches"] == ["sha256:config.json"]


def test_a_fabricated_size_is_caught(tmp_path):
    manifest = _manifest(tmp_path, files={"config.json": b"{}"})
    manifest["files"][0]["size"] = 999_999

    resolution = resolve_model_manifest(manifest)

    assert resolution["mismatches"] == ["size:config.json"]


def test_a_file_that_does_not_exist_is_caught(tmp_path):
    manifest = _manifest(tmp_path, files={"config.json": b"{}"})
    manifest["files"].append(
        {"path": "invented.safetensors", "size": 4, "sha256": "1" * 64}
    )

    resolution = resolve_model_manifest(manifest)

    assert resolution["mismatches"] == ["missing:invented.safetensors"]


def test_an_absent_checkpoint_reports_unresolved_not_clean():
    """A report validated on another machine cannot resolve anything, and
    "could not check" is a different answer from "checked and correct"."""
    resolution = resolve_model_manifest(
        {"model_path": "/nonexistent/checkpoint", "files": [{"path": "x", "size": 0}]}
    )

    assert resolution["resolved"] is False
    assert resolution["reason"] == "model_path_absent_on_this_host"


def test_a_huge_file_is_size_checked_without_being_read(tmp_path):
    """Reading a whole checkpoint would take minutes; the receipt says which
    files were digested rather than implying all of them were."""
    manifest = _manifest(tmp_path, files={"model.safetensors": b"x" * 32})
    manifest["files"][0]["size"] = 32

    monkey = frontier_gap._MANIFEST_FULL_DIGEST_MAX_BYTES
    try:
        frontier_gap._MANIFEST_FULL_DIGEST_MAX_BYTES = 8
        resolution = resolve_model_manifest(manifest)
    finally:
        frontier_gap._MANIFEST_FULL_DIGEST_MAX_BYTES = monkey

    assert resolution["resolved"] is True
    assert resolution["files_present"] == 1
    assert resolution["files_digested"] == 0


def test_the_report_validator_can_require_resolution():
    parameters = inspect.signature(frontier_gap.validate_capability_report).parameters

    assert "model_manifest_resolver" in parameters
    assert "require_resolved_model" in parameters
    assert parameters["require_resolved_model"].default is False


def test_requiring_resolution_refuses_an_unresolved_manifest():
    source = inspect.getsource(frontier_gap)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        rendered = ast.get_source_segment(source, node) or ""
        if "was not resolved against the checkpoint" not in rendered:
            continue
        test = ast.get_source_segment(source, node.test) or ""
        assert "require_resolved_model" in test
        assert 'model_resolution.get("resolved") is not True' in test
        return
    raise AssertionError("the resolution requirement was not found")


# ─────────────────────────── the role map is a partition


def _validated(manifest):
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    manifest = dict(manifest)
    manifest["manifest_sha256"] = sha256_json(body)
    return json.loads(canonical_json_bytes(manifest))


def _manifest_body(**roles):
    files = ["model.safetensors", "config.json", "tokenizer.model"]
    entries = [
        {"path": name, "size": 4, "sha256": hashlib.sha256(name.encode()).hexdigest()}
        for name in files
    ]
    return {
        "schema": frontier_gap.MODEL_MANIFEST_SCHEMA,
        "model_path": "/models/test",
        "files": entries,
        "file_count": len(entries),
        "total_bytes": 12,
        "roles": {
            "weights": roles.get("weights", ["model.safetensors"]),
            "configuration": roles.get("configuration", ["config.json"]),
            "tokenizer": roles.get("tokenizer", ["tokenizer.model"]),
            "adapters": roles.get("adapters", []),
        },
    }


def test_a_complete_partition_validates():
    manifest = _validated(_manifest_body())

    assert frontier_gap._validate_model_manifest(manifest)["file_count"] == 3


def test_a_file_claimed_by_two_roles_is_refused():
    """roles["adapters"] is what the adapter identity check reads, so a
    weights file listed there would be attested as an adapter."""
    manifest = _validated(_manifest_body(adapters=["model.safetensors"]))

    with pytest.raises(ValueError, match="claimed by two roles"):
        frontier_gap._validate_model_manifest(manifest)


def test_an_unclassified_file_is_refused():
    manifest = _validated(_manifest_body(tokenizer=[]))

    with pytest.raises(ValueError, match="has no role"):
        frontier_gap._validate_model_manifest(manifest)


def test_the_self_digest_alone_still_proves_only_consistency():
    """Kept as the statement of what the manifest check IS: internally
    consistent, and silent about the weights that were loaded."""
    manifest = _validated(_manifest_body())
    manifest["model_path"] = "/models/somewhere-else-entirely"

    with pytest.raises(ValueError, match="digest mismatch"):
        frontier_gap._validate_model_manifest(manifest)
