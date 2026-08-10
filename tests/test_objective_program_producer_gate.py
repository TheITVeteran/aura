from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS = REPO_ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_objective_program_producer_gate as gate  # noqa: E402


def test_gate_proves_every_family_without_serializing_private_answers() -> None:
    report = gate.run_gate(seeds=(810_013, 810_017))

    assert report["admitted"] is True
    assert report["task_count"] == 42
    assert report["producer_interface"] == {
        "callable": "solve_objective_program",
        "signature": "(objective: 'str') -> 'tuple[str, dict[str, Any]] | None'",
        "candidate_visible_inputs": ["objective"],
        "private_answer_input": False,
    }
    assert all(report["checks"].values())
    assert set(report["family_counts"]) == {
        "stable_nearest_traversal",
        "separated_subset_count",
        "stateful_python_trace",
        "interventional_chain_inference",
        "dependency_deadline_portfolio",
        "bayesian_frequency_update",
        "premise_audit_table",
    }
    wire = json.dumps(report, sort_keys=True)
    assert "expected_payload" not in wire
    assert "blinded_answer" not in wire


def test_gate_rejects_duplicate_seeds() -> None:
    try:
        gate.run_gate(seeds=(42, 42), difficulties=(1,))
    except ValueError as exc:
        assert str(exc) == "producer gate seeds must be non-empty and unique"
    else:  # pragma: no cover - fail message is more useful than pytest.raises here
        raise AssertionError("duplicate seeds were accepted")
