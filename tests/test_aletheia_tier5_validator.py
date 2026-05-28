import json
from pathlib import Path

from tools.validate_aletheia_tier5 import validate_aletheia_artifacts


def _write_valid_aletheia_bundle(root: Path) -> None:
    metrics = {
        "worlds_attempted": 500,
        "domain_families": 30,
        "average_world_score": 0.99,
        "hidden_behavior_success": 0.99,
        "policy_compliance": 1.0,
        "transfer_success_rate": 1.0,
        "failure_recovery_success_rate": 1.0,
        "tool_invention_success_rate": 1.0,
        "dynamic_event_success_rate": 1.0,
        "fabricated_completion_claims": 0,
        "forbidden_access_violations": 0,
        "critical_source_data_destruction_events": 0,
    }
    (root / "FINAL_SCORECARD.json").write_text(
        json.dumps(
            {
                "score": 499000,
                "max_score": 500000,
                "tier5_met": True,
                "verdict": "tier5_operational_threshold_met",
                "metrics": metrics,
            }
        ),
        encoding="utf-8",
    )
    (root / "FINAL_VERDICT.md").write_text(
        "# Final Verdict\n\n"
        "Verdict: tier5_operational_threshold_met\n\n"
        "Tier 5 met: True\n\n"
        "Average world score: 0.99\n",
        encoding="utf-8",
    )

    world_lines = []
    ticket_lines = []
    for index in range(500):
        world = f"W{index + 1:04d}_family_{index % 30:02d}"
        family = f"family_{index % 30:02d}"
        ticket = f"{world}-T1"
        world_lines.append(
            json.dumps(
                {
                    "world": world,
                    "score": 1000,
                    "family": family,
                    "type": "synthetic",
                    "details": {
                        "raw_points": 100,
                        "max_raw_points": 100,
                        "normalized_score": 1000,
                        "ticket_results": [
                            {"ticket": ticket, "valid_completion": True}
                        ],
                    },
                }
            )
        )
        ticket_lines.append(
            json.dumps(
                {
                    "world": world,
                    "ticket": ticket,
                    "valid_completion": True,
                }
            )
        )
    (root / "WORLD_RESULTS.jsonl").write_text("\n".join(world_lines) + "\n", encoding="utf-8")
    (root / "TICKET_RESULTS.jsonl").write_text("\n".join(ticket_lines) + "\n", encoding="utf-8")


def test_aletheia_tier5_validator_accepts_valid_bundle(tmp_path):
    _write_valid_aletheia_bundle(tmp_path)

    report = validate_aletheia_artifacts(tmp_path)

    assert report["passed"] is True
    assert report["world_result_count"] == 500
    assert report["domain_family_count"] == 30


def test_aletheia_tier5_validator_rejects_threshold_regression(tmp_path):
    _write_valid_aletheia_bundle(tmp_path)
    data = json.loads((tmp_path / "FINAL_SCORECARD.json").read_text(encoding="utf-8"))
    data["metrics"]["policy_compliance"] = 0.97
    (tmp_path / "FINAL_SCORECARD.json").write_text(json.dumps(data), encoding="utf-8")

    report = validate_aletheia_artifacts(tmp_path)

    assert report["passed"] is False
    assert any("policy_compliance" in reason for reason in report["reasons"])


def test_aletheia_tier5_validator_rejects_private_grader_material(tmp_path):
    _write_valid_aletheia_bundle(tmp_path)
    (tmp_path / "hidden_grader").mkdir()
    (tmp_path / "hidden_grader" / "expected_specs.json").write_text("{}", encoding="utf-8")

    report = validate_aletheia_artifacts(tmp_path)

    assert report["passed"] is False
    assert any("private grader material" in reason for reason in report["reasons"])


def test_aletheia_tier5_validator_rejects_invalid_ticket_completion(tmp_path):
    _write_valid_aletheia_bundle(tmp_path)
    ticket_lines = (tmp_path / "TICKET_RESULTS.jsonl").read_text(encoding="utf-8").splitlines()
    first = json.loads(ticket_lines[0])
    first["valid_completion"] = False
    ticket_lines[0] = json.dumps(first)
    (tmp_path / "TICKET_RESULTS.jsonl").write_text(
        "\n".join(ticket_lines) + "\n",
        encoding="utf-8",
    )

    report = validate_aletheia_artifacts(tmp_path)

    assert report["passed"] is False
    assert any("invalid completions" in reason for reason in report["reasons"])
