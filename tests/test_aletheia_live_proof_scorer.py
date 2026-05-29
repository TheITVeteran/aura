import json
from pathlib import Path

import pytest

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
