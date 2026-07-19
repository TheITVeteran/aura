"""Defect-detector tests for the requirement registry validator."""
from __future__ import annotations

import hashlib


from reqproof_testkit import make_registry_dict, make_requirement
from tools.reqproof.schema import Registry
from tools.reqproof.validate import BLOCKING_ALWAYS, RATCHETED_CLASSES, validate_registry

ALWAYS_TRUE_COMMIT = lambda commit: True  # noqa: E731


def build(requirements: list[dict]) -> Registry:
    return Registry.from_dict(make_registry_dict(requirements))


def run(registry: Registry, tmp_path, **kwargs):
    kwargs.setdefault("commit_exists", ALWAYS_TRUE_COMMIT)
    return validate_registry(registry, root=tmp_path, **kwargs)


def classes(defects) -> set[str]:
    return {defect.defect_class for defect in defects}


class TestGraphDefects:
    def test_clean_registry_has_no_defects(self, tmp_path):
        registry = build([make_requirement()])
        assert run(registry, tmp_path) == []

    def test_exact_duplicate_id(self, tmp_path):
        # The schema's sort check accepts equal adjacent IDs, so the
        # validator must catch outright duplicates. (Case variants are
        # impossible: the ID pattern forbids lowercase entirely.)
        registry = build(
            [make_requirement(id="DUP-A-001"), make_requirement(id="DUP-A-001")]
        )
        defects = run(registry, tmp_path)
        assert "duplicate-id" in classes(defects)

    def test_orphan_references_all_fields(self, tmp_path):
        registry = build(
            [
                make_requirement(
                    depends_on=["GHOST-001"],
                    closure_requires=["GHOST-002"],
                    parent="GHOST-003",
                )
            ]
        )
        defects = [d for d in run(registry, tmp_path) if d.defect_class == "orphan-ref"]
        assert {d.subject for d in defects} == {
            "TEST-001::GHOST-001",
            "TEST-001::GHOST-002",
            "TEST-001::GHOST-003",
        }

    def test_parent_mismatch(self, tmp_path):
        registry = build(
            [
                make_requirement(id="CHILD-001", parent="PARENT-001"),
                make_requirement(id="OTHER-001"),
                make_requirement(id="PARENT-001", kind="parent",
                                 closure_requires=["OTHER-001"]),
            ]
        )
        defects = run(registry, tmp_path)
        assert "parent-mismatch" in classes(defects)

    def test_closure_cycle_detected_with_canonical_fingerprint(self, tmp_path):
        registry = build(
            [
                make_requirement(id="CYC-A-001", kind="parent",
                                 closure_requires=["CYC-B-001"]),
                make_requirement(id="CYC-B-001", kind="parent",
                                 closure_requires=["CYC-A-001"]),
            ]
        )
        defects = [
            d for d in run(registry, tmp_path) if d.defect_class == "closure-cycle"
        ]
        assert len(defects) == 1
        assert defects[0].subject == "CYC-A-001+CYC-B-001"

    def test_dependency_cycle_detected(self, tmp_path):
        registry = build(
            [
                make_requirement(id="DEP-A-001", depends_on=["DEP-B-001"]),
                make_requirement(id="DEP-B-001", depends_on=["DEP-C-001"]),
                make_requirement(id="DEP-C-001", depends_on=["DEP-A-001"]),
            ]
        )
        defects = [
            d for d in run(registry, tmp_path) if d.defect_class == "dependency-cycle"
        ]
        assert len(defects) == 1
        assert defects[0].subject == "DEP-A-001+DEP-B-001+DEP-C-001"

    def test_randomized_cycles_always_detected(self, tmp_path):
        """Seeded property test: a planted directed cycle is always found."""
        import random

        rng = random.Random(715)
        for trial in range(15):
            count = rng.randint(3, 10)
            ids = [f"RND-{trial:02d}-{index:03d}" for index in range(count)]
            cycle_members = rng.sample(ids, rng.randint(2, count))
            edges: dict[str, list[str]] = {req_id: [] for req_id in ids}
            for index, member in enumerate(cycle_members):
                edges[member].append(cycle_members[(index + 1) % len(cycle_members)])
            for req_id in ids:  # sprinkle acyclic noise edges
                for other in ids:
                    if other > req_id and rng.random() < 0.2 and other not in edges[req_id]:
                        edges[req_id].append(other)
            registry = build(
                [
                    make_requirement(id=req_id, depends_on=sorted(set(edges[req_id])))
                    for req_id in sorted(ids)
                ]
            )
            defects = [
                d
                for d in run(registry, tmp_path)
                if d.defect_class == "dependency-cycle"
            ]
            assert defects, f"trial {trial}: planted cycle {cycle_members} not detected"


class TestClosureTruth:
    def test_false_closure_parent_more_closed_than_child(self, tmp_path):
        registry = build(
            [
                make_requirement(id="CHILD-001", state="open", parent="PARENT-001"),
                make_requirement(
                    id="PARENT-001",
                    kind="parent",
                    state="complete",
                    closure_requires=["CHILD-001"],
                    evidence_required=["implementation"],
                ),
            ]
        )
        defects = run(registry, tmp_path)
        assert "false-closure" in classes(defects)

    def test_complete_without_evidence_is_unproven(self, tmp_path):
        registry = build([make_requirement(state="complete")])
        defects = run(registry, tmp_path)
        assert "unproven-closure" in classes(defects)

    def test_complete_with_verified_evidence_passes(self, tmp_path):
        artifact = tmp_path / "artifacts" / "proof.json"
        artifact.parent.mkdir(parents=True)
        artifact.write_text('{"ok": true}', encoding="utf-8")
        sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
        evidence = [
            {
                "evidence_class": cls,
                "ref": "artifacts/proof.json",
                "sha256": sha,
                "commit": "abc1234",
                "recorded_at": "2026-07-16",
            }
            for cls in ("implementation", "test")
        ]
        registry = build([make_requirement(state="complete", evidence=evidence)])
        assert run(registry, tmp_path) == []

    def test_old_artifact_cannot_close_a_requirement(self, tmp_path):
        """Evidence whose content changed under the recorded hash is impossible."""
        artifact = tmp_path / "artifacts" / "proof.json"
        artifact.parent.mkdir(parents=True)
        artifact.write_text("old proof", encoding="utf-8")
        stale_sha = hashlib.sha256(b"different content entirely").hexdigest()
        evidence = [
            {
                "evidence_class": "implementation",
                "ref": "artifacts/proof.json",
                "sha256": stale_sha,
                "commit": "abc1234",
                "recorded_at": "2026-07-16",
            }
        ]
        registry = build(
            [
                make_requirement(
                    state="complete",
                    evidence=evidence,
                    evidence_required=["implementation"],
                )
            ]
        )
        defects = run(registry, tmp_path)
        assert "impossible-evidence" in classes(defects)
        assert "unproven-closure" in classes(defects)

    def test_missing_evidence_file_is_impossible(self, tmp_path):
        evidence = [
            {
                "evidence_class": "implementation",
                "ref": "artifacts/never_written.json",
                "sha256": "0" * 64,
                "commit": "abc1234",
                "recorded_at": "2026-07-16",
            }
        ]
        registry = build([make_requirement(evidence=evidence)])
        defects = run(registry, tmp_path)
        assert "impossible-evidence" in classes(defects)

    def test_unknown_commit_is_impossible(self, tmp_path):
        artifact = tmp_path / "proof.json"
        artifact.write_text("x", encoding="utf-8")
        sha = hashlib.sha256(b"x").hexdigest()
        evidence = [
            {
                "evidence_class": "implementation",
                "ref": "proof.json",
                "sha256": sha,
                "commit": "deadbee",
                "recorded_at": "2026-07-16",
            }
        ]
        registry = build([make_requirement(evidence=evidence)])
        defects = run(registry, tmp_path, commit_exists=lambda commit: False)
        assert "impossible-evidence" in classes(defects)

    def test_contradictory_status_complete_over_open_dependency(self, tmp_path):
        registry = build(
            [
                make_requirement(
                    id="DONE-001",
                    state="complete",
                    depends_on=["OPEN-001"],
                    evidence_required=["implementation"],
                ),
                make_requirement(id="OPEN-001", state="open"),
            ]
        )
        assert "contradictory-status" in classes(run(registry, tmp_path))

    def test_withdrawn_child_under_live_parent_flagged(self, tmp_path):
        registry = build(
            [
                make_requirement(id="GONE-001", state="withdrawn", mandatory=False),
                make_requirement(
                    id="LIVE-001", kind="parent", closure_requires=["GONE-001"]
                ),
            ]
        )
        assert "withdrawn-required" in classes(run(registry, tmp_path))


class TestPolicyCompleteness:
    def test_every_defect_class_has_a_blocking_policy(self):
        from tools.reqproof.validate import DEFECT_CLASSES

        assert set(DEFECT_CLASSES) == set(BLOCKING_ALWAYS) | set(RATCHETED_CLASSES)
        assert not (set(BLOCKING_ALWAYS) & set(RATCHETED_CLASSES))
