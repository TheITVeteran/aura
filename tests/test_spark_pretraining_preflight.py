"""The preflight has to be able to say NOT READY.

A probe suite that always passes is decoration. These tests weaken a real
instrument and require the preflight to notice — which is the only evidence
that a green run means anything.
"""

from __future__ import annotations

import json

import pytest

from tools import spark_pretraining_preflight as preflight


def test_every_instrument_responds_on_a_clean_tree():
    report = preflight.run_self_check()
    assert report["ready"] is True
    assert report["dead_instruments"] == []
    assert report["probes_fired"] == report["probes_run"] == len(preflight.PROBES)


def test_the_report_names_each_probe_and_the_refusal_it_saw():
    report = preflight.run_self_check()
    for row in report["probes"]:
        assert row["probe"]
        assert row["fired"] is True
        assert row["refusal"]


def test_a_weakened_gate_set_is_caught():
    # Simulate the exact regression the probe exists for: the promotion gate
    # report stops refusing an incomplete set.
    def permissive(rows):
        return {"schema": "weakened", "gates": list(rows), "gate_report_sha256": "x"}

    original = preflight._probe_promotion_gate_set

    def probing():
        import core.learning.permanent_distillation as module

        saved = module.gate_report
        module.gate_report = permissive
        try:
            return original()
        finally:
            module.gate_report = saved

    result = probing()
    assert result["fired"] is False
    assert "accepted a known-bad input" in result["detail"]


def test_a_dead_instrument_makes_the_run_not_ready(monkeypatch):
    def always_accepts():
        return {"probe": "fake", "fired": False, "detail": "accepted a known-bad input"}

    monkeypatch.setattr(preflight, "PROBES", (*preflight.PROBES, always_accepts))
    report = preflight.run_self_check()
    assert report["ready"] is False
    assert report["dead_instruments"] == ["fake"]


def test_a_dead_instrument_exits_non_zero(monkeypatch, capsys):
    def always_accepts():
        return {"probe": "fake", "fired": False, "detail": "accepted a known-bad input"}

    monkeypatch.setattr(preflight, "PROBES", (*preflight.PROBES, always_accepts))
    assert preflight.main([]) == 1
    captured = capsys.readouterr()
    assert "DEAD" in captured.out
    assert "NOT READY" in captured.err


def test_a_probe_that_itself_breaks_counts_as_dead(monkeypatch):
    def explodes():
        raise RuntimeError("import moved")

    monkeypatch.setattr(preflight, "PROBES", (explodes,))
    report = preflight.run_self_check()
    assert report["ready"] is False
    assert "probe itself failed" in report["probes"][0]["detail"]


def test_a_clean_run_exits_zero(capsys):
    assert preflight.main([]) == 0
    assert "READY" in capsys.readouterr().out


# --- campaign artifact validation -------------------------------------------


def test_a_campaign_directory_with_no_artifacts_is_not_an_error(tmp_path, capsys):
    assert preflight.main(["--campaign", str(tmp_path)]) == 0
    assert "0 campaign artifact(s) checked" in capsys.readouterr().out


def test_a_valid_promotion_lineage_is_accepted(tmp_path, monkeypatch, capsys):
    import hashlib

    from core.learning.permanent_distillation import (
        artifact_manifest,
        baseline_generation,
    )
    from core.learning.permanent_distillation_registry import write_lineage

    monkeypatch.setenv("AURA_LOG_DIR", str(tmp_path / "logs"))
    record = baseline_generation(
        artifact=artifact_manifest(
            artifact_id="frozen",
            base_model_identity="preflight",
            adapter_identity="frozen",
            files=[
                {
                    "name": "a.bin",
                    "sha256": hashlib.sha256(b"a").hexdigest(),
                    "size_bytes": 1,
                }
            ],
        ),
        provenance={},
        created_at_unix=1_780_000_000,
    )
    write_lineage(tmp_path / "permanent_distillation.json", [record])
    assert preflight.main(["--campaign", str(tmp_path)]) == 0
    assert "1 campaign artifact(s) checked" in capsys.readouterr().out


def test_a_tampered_promotion_lineage_exits_two(tmp_path, monkeypatch, capsys):
    import hashlib

    from core.learning.permanent_distillation import (
        artifact_manifest,
        baseline_generation,
    )
    from core.learning.permanent_distillation_registry import write_lineage

    monkeypatch.setenv("AURA_LOG_DIR", str(tmp_path / "logs"))
    record = baseline_generation(
        artifact=artifact_manifest(
            artifact_id="frozen",
            base_model_identity="preflight",
            adapter_identity="frozen",
            files=[
                {
                    "name": "a.bin",
                    "sha256": hashlib.sha256(b"a").hexdigest(),
                    "size_bytes": 1,
                }
            ],
        ),
        provenance={},
        created_at_unix=1_780_000_000,
    )
    path = tmp_path / "permanent_distillation.json"
    write_lineage(path, [record])
    document = json.loads(path.read_text(encoding="utf-8"))
    document["generations"][0]["provenance"] = {"tampered": True}
    path.write_text(json.dumps(document), encoding="utf-8")

    assert preflight.main(["--campaign", str(tmp_path)]) == 2
    assert "INVALID" in capsys.readouterr().err


def test_a_contaminated_star_lineage_exits_two(tmp_path, capsys):
    import hashlib

    from core.learning.star_iteration_ledger import GENESIS_PARENT, star_iteration

    def fp(tag: str) -> str:
        return hashlib.sha256(tag.encode()).hexdigest()

    first = star_iteration(
        iteration_index=0,
        parent_iteration_sha256=GENESIS_PARENT,
        generated=100,
        verified=20,
        filtered=20,
        filter_reasons={"verifier_rejected": 8},
        training_fingerprints=[fp(f"train-{i}") for i in range(12)],
        training_trace_classes=["direct"],
        holdout_fingerprints=[fp(f"hold-{i}") for i in range(8)],
        holdout_score=0.5,
        trace_gates=[],
        created_at_unix=1_780_000_000,
    )
    leaked = star_iteration(
        iteration_index=1,
        parent_iteration_sha256=first["iteration_sha256"],
        generated=100,
        verified=20,
        filtered=20,
        filter_reasons={"verifier_rejected": 8},
        training_fingerprints=[fp(f"train2-{i}") for i in range(12)],
        training_trace_classes=["direct"],
        # Iteration 0's training set, reappearing as a "fresh" holdout.
        holdout_fingerprints=[fp(f"train-{i}") for i in range(8)],
        holdout_score=0.95,
        trace_gates=[],
        created_at_unix=1_780_000_001,
    )
    (tmp_path / "star_lineage.json").write_text(
        json.dumps([first, leaked]), encoding="utf-8"
    )
    assert preflight.main(["--campaign", str(tmp_path)]) == 2
    assert "INVALID" in capsys.readouterr().err


def test_the_report_can_be_written_for_the_record(tmp_path):
    out = tmp_path / "preflight.json"
    assert preflight.main(["--out", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["schema"] == preflight.PREFLIGHT_SCHEMA
    assert report["ready"] is True
    assert len(report["probes"]) == len(preflight.PROBES)


@pytest.mark.parametrize("index", range(len(preflight.PROBES)))
def test_each_probe_is_individually_alive(index):
    result = preflight.PROBES[index]()
    assert result["fired"] is True, result.get("detail")
