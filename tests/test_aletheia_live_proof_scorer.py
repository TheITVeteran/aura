import json
from pathlib import Path

import pytest

from aura_bench.aletheia_runner_live import (
    LiveWorldProcessor,
    infer_device_law_from_visible_observations,
    render_device_validation_note,
    validate_device_model_from_visible_observations,
)
from tools.run_aletheia_live_proof import run_scorer


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_aletheia_live_scorer_computes_domain_family_count_from_rows(tmp_path):
    world_rows = []
    ticket_rows = []
    for index in range(500):
        family = f"family_{index % 30:02d}"
        world = f"W{index + 1:04d}_{family}"
        ticket = f"{world}-T1"
        world_rows.append(
            {
                "world": world,
                "family": family,
                "score": 1000,
                "details": {
                    "hidden_behavior": 1,
                    "criteria": {
                        "policy_success": True,
                        "transfer_success": True,
                        "failure_success": True,
                        "tool_success": True,
                        "dynamic_success": True,
                    },
                },
            }
        )
        ticket_rows.append({"world": world, "ticket": ticket, "valid_completion": True})

    world_results = tmp_path / "WORLD_RESULTS.jsonl"
    ticket_results = tmp_path / "TICKET_RESULTS.jsonl"
    _write_jsonl(world_results, world_rows)
    _write_jsonl(ticket_results, ticket_rows)

    scorecard = run_scorer(world_results, ticket_results)

    assert scorecard["tier5_met"] is True
    assert scorecard["metrics"]["worlds_attempted"] == 500.0
    assert scorecard["metrics"]["domain_families"] == 30.0


def test_aletheia_live_scorer_fails_closed_without_world_results(tmp_path):
    with pytest.raises(RuntimeError, match="required Aletheia artifact missing"):
        run_scorer(tmp_path / "WORLD_RESULTS.jsonl")


def test_device_law_inference_uses_visible_observations_without_hidden_specs(tmp_path):
    world = tmp_path / "world"
    raw = world / "data/raw"
    raw.mkdir(parents=True)
    (raw / "observations.csv").write_text(
        "\n".join(
            [
                "x,y,catalyst,output",
                "12,3,red,65",
                "5,2,red,32",
                "1,3,blue,14",
                "1,4,green,31",
                "1,11,amber,51",
                "6,9,red,71",
                "9,8,red,78",
                "1,11,red,61",
                "3,9,blue,52",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    law = infer_device_law_from_visible_observations(world)

    assert law is not None
    assert int(law["a"]) == 4
    assert int(law["b"]) == 5
    assert int(law["bonuses"]["amber"]) == -8
    assert int(law["bonuses"]["blue"]) == -5
    assert int(law["bonuses"]["green"]) == 7
    assert int(law["bonuses"]["red"]) == 2
    assert "none" in law["unobserved_catalysts"]


def test_device_report_marks_unobserved_catalysts_as_uncertain(tmp_path):
    world = tmp_path / "world"
    raw = world / "data/raw"
    raw.mkdir(parents=True)
    (raw / "observations.csv").write_text(
        "x,y,catalyst,output\n"
        "0,0,red,2\n"
        "1,0,red,5\n"
        "0,1,red,6\n"
        "1,1,blue,5\n"
        "2,1,blue,8\n",
        encoding="utf-8",
    )

    law = infer_device_law_from_visible_observations(world)
    assert law is not None

    report = render_device_validation_note(law)

    assert "stale manual" in report.lower()
    assert "unobserved catalysts" in report.lower()
    assert "evaluator-only offsets" in report.lower()


def test_device_model_validation_rejects_visible_observation_mismatch(tmp_path):
    world = tmp_path / "world"
    raw = world / "data/raw"
    raw.mkdir(parents=True)
    (raw / "observations.csv").write_text(
        "x,y,catalyst,output\n0,0,red,2\n1,0,red,5\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="visible observation replay"):
        validate_device_model_from_visible_observations(
            "def predict_output(x, y, color):\n    return 0\n",
            world,
        )


def test_device_handler_preserves_aura_returned_model(tmp_path, monkeypatch):
    root = tmp_path / "battery"
    world = root / "worlds/W0001_lab_device_operation"
    raw = world / "data/raw"
    raw.mkdir(parents=True)
    (raw / "observations.csv").write_text(
        "x,y,catalyst,output\n"
        "0,0,red,2\n"
        "1,0,red,5\n"
        "0,1,red,6\n"
        "1,1,blue,5\n"
        "2,1,blue,8\n",
        encoding="utf-8",
    )

    processor = LiveWorldProcessor(root, {"worlds": {}}, "http://127.0.0.1:9")
    aura_code = (
        "# aura-owned sentinel\n"
        "def predict_output(x, y, color):\n"
        "    bonus = {'red': 2, 'blue': -2}.get(str(color).lower(), 0)\n"
        "    return 3 * int(x) + 4 * int(y) + bonus\n"
    )
    monkeypatch.setattr(
        processor,
        "_ask_aura",
        lambda _prompt: (
            f"```python\n{aura_code}```\n"
            "The stale manual is rejected. Bonus values: red=2, blue=-2."
        ),
    )

    processor._handle_device("W0001_lab_device_operation", world, {})

    written = (world / "apps/model/model.py").read_text(encoding="utf-8")
    assert "# aura-owned sentinel" in written
    assert "COEFFICIENT_A" not in written
    validation_note = (world / "reports/device_validation.md").read_text(encoding="utf-8")
    assert "candidate-visible" in validation_note
