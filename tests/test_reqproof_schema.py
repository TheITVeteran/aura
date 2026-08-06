"""Strict-schema tests for the requirement registry (SCOPE-001)."""
from __future__ import annotations

import json
import random

import pytest
from reqproof_testkit import make_registry_dict, make_requirement

from tools.reqproof.schema import (
    EVIDENCE_CLASSES,
    Registry,
    RegistrySchemaError,
    Requirement,
    load_registry,
    write_registry_atomic,
)


class TestStrictParsing:
    def test_round_trip_is_canonical_and_deterministic(self):
        data = make_registry_dict([make_requirement()])
        registry = Registry.from_dict(data)
        assert registry.to_canonical_json() == Registry.from_dict(
            json.loads(registry.to_canonical_json())
        ).to_canonical_json()

    def test_unknown_registry_field_rejected(self):
        data = make_registry_dict([make_requirement()])
        data["progress_percent"] = 99
        with pytest.raises(RegistrySchemaError, match="unknown fields"):
            Registry.from_dict(data)

    def test_unknown_requirement_field_rejected(self):
        data = make_registry_dict([make_requirement()])
        data["requirements"][0]["done"] = True
        with pytest.raises(RegistrySchemaError, match="unknown fields"):
            Registry.from_dict(data)

    def test_missing_requirement_field_rejected(self):
        data = make_registry_dict([make_requirement()])
        del data["requirements"][0]["acceptance"]
        with pytest.raises(RegistrySchemaError, match="missing fields"):
            Registry.from_dict(data)

    @pytest.mark.parametrize(
        "field,value,message",
        [
            ("state", "done", "state"),
            ("kind", "epic", "kind"),
            ("id", "lower-case-001", "pattern"),
            ("id", "NOHYPHEN", "pattern"),
            ("status_date", "July 4", "ISO date"),
            ("risk_weight", 0, "positive"),
            ("proof_weight", -1, "positive"),
            ("weight_provenance", "vibes", "weight_provenance"),
            ("mandatory", "yes", "boolean"),
            ("evidence_required", [], "non-empty"),
            ("evidence_required", ["vibes"], "not in"),
            ("evidence_required", ["test", "test"], "duplicates"),
            ("sources", [], "at least one provenance"),
        ],
    )
    def test_bad_field_values_rejected(self, field, value, message):
        with pytest.raises(RegistrySchemaError, match=message):
            Requirement.from_dict(make_requirement(**{field: value}))

    def test_atomic_requires_acceptance(self):
        with pytest.raises(RegistrySchemaError, match="acceptance criterion"):
            Requirement.from_dict(make_requirement(acceptance=[]))

    def test_acceptance_modalities_are_aligned_and_union_is_derived(self):
        requirement = Requirement.from_dict(
            make_requirement(
                acceptance=["Static proof.", "Live soak proof."],
                acceptance_evidence_required=[
                    ["implementation", "test"],
                    ["implementation", "test", "live", "soak"],
                ],
                evidence_required=["implementation", "test", "live", "soak"],
            )
        )
        assert requirement.required_evidence_for("A1") == (
            "implementation",
            "test",
        )
        assert requirement.required_evidence_for("A2") == (
            "implementation",
            "test",
            "live",
            "soak",
        )
        assert len(requirement.required_evidence_cells()) == 6

        with pytest.raises(RegistrySchemaError, match="align one-to-one"):
            Requirement.from_dict(
                make_requirement(
                    acceptance=["first", "second"],
                    acceptance_evidence_required=[["implementation", "test"]],
                )
            )
        with pytest.raises(RegistrySchemaError, match="canonical union"):
            Requirement.from_dict(
                make_requirement(
                    acceptance=["static", "live"],
                    acceptance_evidence_required=[
                        ["implementation", "test"],
                        ["implementation", "test", "live"],
                    ],
                    evidence_required=["implementation", "test"],
                )
            )

    def test_self_reference_rejected(self):
        with pytest.raises(RegistrySchemaError, match="itself"):
            Requirement.from_dict(make_requirement(depends_on=["TEST-001"]))
        with pytest.raises(RegistrySchemaError, match="itself"):
            Requirement.from_dict(make_requirement(parent="TEST-001"))

    def test_duplicate_dependency_entries_rejected(self):
        with pytest.raises(RegistrySchemaError, match="duplicate"):
            Requirement.from_dict(
                make_requirement(depends_on=["OTHER-001", "OTHER-001"])
            )

    def test_unsorted_requirements_rejected(self):
        data = make_registry_dict(
            [make_requirement(id="A-001"), make_requirement(id="B-001")]
        )
        data["requirements"].reverse()
        with pytest.raises(RegistrySchemaError, match="sorted"):
            Registry.from_dict(data, verify_hash=False)

    def test_evidence_entry_schema_enforced(self):
        bad_evidence = {
            "evidence_class": "test",
            "ref": "artifacts/x.json",
            "sha256": "zz",
            "commit": "abc1234",
            "recorded_at": "2026-07-16",
        }
        with pytest.raises(RegistrySchemaError, match="sha256"):
            Requirement.from_dict(make_requirement(evidence=[bad_evidence]))


class TestTamperDetection:
    def test_hand_edited_state_fails_content_hash(self, tmp_path):
        """Editing the registry file without regeneration must be detected."""
        data = make_registry_dict([make_requirement()])
        path = tmp_path / "registry.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        loaded = load_registry(path)
        assert loaded.requirements[0].state == "open"

        data["requirements"][0]["state"] = "complete"
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(RegistrySchemaError, match="edited without regeneration"):
            load_registry(path)

    def test_missing_content_hash_rejected(self, tmp_path):
        data = make_registry_dict([make_requirement()])
        del data["content_sha256"]
        path = tmp_path / "registry.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(RegistrySchemaError, match="content_sha256"):
            load_registry(path)


class TestAtomicWrite:
    def test_write_then_load_round_trips(self, tmp_path):
        registry = Registry.from_dict(make_registry_dict([make_requirement()]))
        path = tmp_path / "registry.json"
        write_registry_atomic(registry, path)
        assert load_registry(path).to_canonical_json() == registry.to_canonical_json()

    def test_crash_during_write_preserves_previous_registry(self, tmp_path, monkeypatch):
        """A torn write must never replace a complete registry."""
        registry_v1 = Registry.from_dict(make_registry_dict([make_requirement()]))
        path = tmp_path / "registry.json"
        write_registry_atomic(registry_v1, path)
        before = path.read_text(encoding="utf-8")

        registry_v2 = Registry.from_dict(
            make_registry_dict([make_requirement(title="Changed")])
        )

        original_replace = type(path).replace

        def exploding_replace(self, target):
            raise OSError("simulated crash before rename")

        monkeypatch.setattr(type(path), "replace", exploding_replace)
        with pytest.raises(OSError, match="simulated crash"):
            write_registry_atomic(registry_v2, path)
        monkeypatch.setattr(type(path), "replace", original_replace)

        assert path.read_text(encoding="utf-8") == before
        assert load_registry(path).requirements[0].title == "A test requirement"


class TestPropertyStyle:
    def test_randomized_registries_round_trip(self):
        """Seeded pseudo-property test: parse(serialize(x)) == x for varied shapes."""
        rng = random.Random(20260716)
        states = ("open", "in_progress", "blocked", "complete", "deferred", "withdrawn")
        for trial in range(25):
            count = rng.randint(1, 8)
            ids = sorted(f"PROP-{trial:02d}-{index:03d}" for index in range(count))
            requirements = []
            for index, req_id in enumerate(ids):
                deps = [other for other in ids[:index] if rng.random() < 0.3]
                requirements.append(
                    make_requirement(
                        id=req_id,
                        state=rng.choice(states),
                        depends_on=deps,
                        risk_weight=rng.choice([0.5, 1.0, 2.5]),
                        evidence_required=sorted(
                            rng.sample(EVIDENCE_CLASSES, rng.randint(1, 4))
                        ),
                        notes="x" * rng.randint(0, 40),
                    )
                )
            registry = Registry.from_dict(make_registry_dict(requirements))
            reparsed = Registry.from_dict(json.loads(registry.to_canonical_json()))
            assert reparsed.to_canonical_json() == registry.to_canonical_json()
            assert reparsed.compute_content_sha256() == registry.compute_content_sha256()
