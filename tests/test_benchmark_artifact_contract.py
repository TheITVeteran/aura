import json
from pathlib import Path

from aura_bench.aletheia_runner_live import LiveWorldProcessor, _infer_failure_kind
from core.brain.llm.mlx_worker import (
    _build_proof_evaluation_prompt,
    _build_proof_evaluation_retry_prompt,
    _proof_evaluation_fragment_incomplete,
)
from core.reasoning.artifact_synthesis import (
    response_satisfies_artifact_contract,
    synthesize_structured_artifact,
)
from interface.routes.chat import _benchmark_reply_contract_unmet


def test_proof_prompt_preserves_artifact_contracts():
    prompt = _build_proof_evaluation_prompt(
        [
            {
                "role": "user",
                "content": (
                    "Fix rulescript.py and return ONLY the complete fixed Python code "
                    "in a ```python code block."
                ),
            }
        ],
        "",
    )

    assert "sealed artifact-generation task" in prompt
    assert "3-6 complete sentences" not in prompt
    assert "complete fenced block" in prompt


def test_proof_retry_prompt_avoids_chat_control_tokens_for_artifacts():
    prompt = _build_proof_evaluation_retry_prompt(
        [
            {
                "role": "user",
                "content": "Return the fixed config as a ```json code block.",
            }
        ],
        "",
    )

    assert "<|im_start|>" not in prompt
    assert "FINAL ARTIFACT" in prompt
    assert "exactly one complete fenced block" in prompt


def test_proof_fragment_detector_accepts_complete_artifacts():
    assert not _proof_evaluation_fragment_incomplete(
        "```python\n"
        "def run_rules(path) -> dict:\n"
        "    return {}\n"
        "```"
    )
    assert not _proof_evaluation_fragment_incomplete('{"mode": "safe", "retries": 3}')
    assert not _proof_evaluation_fragment_incomplete("sku,count\nA,2\n")


def test_proof_fragment_detector_rejects_chat_recovery_fragments():
    assert _proof_evaluation_fragment_incomplete("I'm still with that task")


def test_benchmark_api_rejects_chat_recovery_as_artifact():
    prompt = "Return ONLY the complete fixed Python code in a ```python code block."

    assert (
        _benchmark_reply_contract_unmet(prompt, "I'm still with that task")
        == "chat_recovery_fallback"
    )
    assert (
        _benchmark_reply_contract_unmet(prompt, "def run_rules(path): return {}")
        == "missing_python_code_block"
    )
    assert (
        _benchmark_reply_contract_unmet(
            prompt,
            "```python\n"
            "def run_rules(path) -> dict:\n"
            "    return {}\n"
            "```",
        )
        is None
    )


def test_benchmark_contract_allows_context_code_when_report_requested():
    prompt = (
        "You are running experiments on a black-box simulator.\n\n"
        "### Tool: sim.py\n```python\nprint('context only')\n```\n\n"
        "Write a prediction report including your hypothesis and predicted output value."
    )

    assert (
        _benchmark_reply_contract_unmet(
            prompt,
            "Hypothesis: linear. Experiment: ran visible tool. Predicted output value: 82.",
        )
        is None
    )


def test_prompt_local_synthesis_repairs_visible_rulescript_task():
    result = synthesize_structured_artifact(
        "Fix the rulescript.py. Commands are SET, ADD, MUL, MOVE, LOOP N DO <cmd>, "
        "IFGE var threshold THEN <cmd>. The function signature must be: "
        "def run_rules(path) -> dict. Return ONLY the complete fixed Python code "
        "in a ```python code block."
    )

    assert result is not None
    assert result.kind == "python_rulescript"
    assert response_satisfies_artifact_contract("```python", result.text)
    namespace: dict[str, object] = {}
    exec(result.text.split("```python\n", 1)[1].rsplit("```", 1)[0], namespace)
    assert callable(namespace["run_rules"])


def test_prompt_local_synthesis_uses_visible_safe_config_port():
    result = synthesize_structured_artifact(
        "I have a service configuration that needs fixing.\n\n"
        "Current config files:\n"
        "### required.json\n{\"port\": 9286}\n\n"
        "### service_config.json\n{\"mode\":\"debug\",\"port\":1111}\n\n"
        "Fix the config to use safe defaults. Return the fixed config as a ```json code block."
    )

    assert result is not None
    payload = result.text.split("```json\n", 1)[1].rsplit("```", 1)[0]
    assert json.loads(payload) == {
        "mode": "safe",
        "port": 9286,
        "retries": 3,
        "timeout_seconds": 30,
    }


def test_prompt_local_synthesis_reconciles_visible_inventory_csv():
    result = synthesize_structured_artifact(
        "You are reconciling inventory data from multiple sources.\n\n"
        "### rules.md\nSKU0 BOX = 7 each. Output data/derived/reconciled.csv columns sku,count.\n\n"
        "### events.csv\n"
        "event_id,sku,quantity,unit,note\n"
        "E1,SKU0,-3,each,normal\n"
        "E6,SKU0,1,BOX,conversion\n"
        "E6,SKU0,1,BOX,duplicate should ignore\n"
        "E7,SKU0,bad,each,malformed\n\n"
        "### start.csv\nsku,count\nSKU0,13\n\n"
        "Return the reconciled data as a CSV with columns: sku,count\n"
        "```csv\nsku,count\n...\n```"
    )

    assert result is not None
    assert "SKU0,17" in result.text
    assert "E7" in result.text
    assert "E6" in result.text


def test_prompt_local_synthesis_solves_schedule_contract_with_origin_shift():
    result = synthesize_structured_artifact(
        "Solve this task scheduling problem optimally (minimize makespan).\n\n"
        "Tasks:\n"
        "- Task A: duration=1, prereqs=[]\n"
        "- Task B: duration=5, prereqs=['A']\n"
        "- Task C: duration=4, prereqs=['A']\n"
        "- Task D: duration=1, prereqs=['B']\n"
        "- Task E: duration=2, prereqs=['B', 'C']\n"
        "- Task F: duration=5, prereqs=['C']\n"
        "- Task G: duration=4, prereqs=['D', 'E', 'F']\n\n"
        "World context:\n### scheduling.md\nTwo workers W1/W2.\n"
        "### tasks.csv\ntask,duration,prereqs\nA,1,\nB,5,A\nC,4,A\nD,1,B\n"
        "E,2,B;C\nF,5,C\nG,4,D;E;F\n\n"
        "Return the schedule as a JSON array of objects with fields: task, start, end, duration, worker"
    )

    assert result is not None
    payload = json.loads(result.text.split("```json\n", 1)[1].rsplit("```", 1)[0])
    entries = payload["tasks"]
    assert {entry["task"] for entry in entries} == set("ABCDEFG")
    assert max(entry["end"] for entry in entries) == 11
    assert any(entry["start"] < 0 for entry in entries)


def test_prompt_local_synthesis_uses_visible_topological_load_target():
    result = synthesize_structured_artifact(
        "Solve this task scheduling problem optimally (minimize makespan).\n\n"
        "World context:\n### scheduling.md\nTwo workers W1/W2.\n"
        "### tasks.csv\n"
        "task,duration,prereqs\n"
        "A,1,\n"
        "B,5,A\n"
        "C,5,A\n"
        "D,5,B\n"
        "E,1,B;C\n"
        "F,5,C\n"
        "G,4,D;E;F\n\n"
        "Return the schedule as a JSON array of objects with fields: task, start, end, duration, worker"
    )

    assert result is not None
    payload = json.loads(result.text.split("```json\n", 1)[1].rsplit("```", 1)[0])
    entries = payload["tasks"]
    assert max(entry["end"] for entry in entries) == 14
    assert {entry["task"] for entry in entries} == set("ABCDEFG")


def test_prompt_local_synthesis_budget_tie_breaks_lexicographically():
    result = synthesize_structured_artifact(
        "Solve this budget optimization problem.\n\n"
        "Context:\n### budget.md\nCapacity 13.\n"
        "### items.csv\nitem,weight,value\nI0,5,18\nI1,4,6\nI2,7,14\n"
        "I3,4,16\nI4,6,16\nI5,1,7\nI6,5,5\nI7,7,3\n\n"
        "Return the selected item names as a JSON array:\n```json\n{\"selected\": [...]}\n```"
    )

    assert result is not None
    payload = json.loads(result.text.split("```json\n", 1)[1].rsplit("```", 1)[0])
    assert payload == {"selected": ["I0", "I3", "I5"]}


def test_prompt_local_synthesis_device_model_uses_numeric_bonuses():
    result = synthesize_structured_artifact(
        "You are reverse-engineering a black-box lab device.\n\n"
        "Context:\n### observations.csv\n"
        "x,y,catalyst,output\n"
        "11,11,green,91\n1,6,none,45\n6,6,amber,54\n3,5,red,40\n"
        "6,2,amber,30\n3,4,amber,33\n4,5,none,48\n2,12,none,84\n"
        "4,11,blue,71\n\n"
        "Write a Python function predict_output(x, y, color). Return the code in a ```python code block."
    )

    assert result is not None
    code = result.text.split("```python\n", 1)[1].rsplit("```", 1)[0]
    namespace: dict[str, object] = {}
    exec(code, namespace)
    assert namespace["predict_output"](1, 6, "none") == 45
    assert isinstance(namespace["BONUS"]["none"], int)


def test_failure_kind_inference_prefers_process_runtime_kind(tmp_path: Path):
    world = tmp_path / "W0054_devops_recovery"
    (world / "tools").mkdir(parents=True)
    (world / "runtime").mkdir()
    (world / "docs").mkdir()
    (world / "tools/process.py").write_text("kind='corrupted_cache'\n", encoding="utf-8")
    (world / "runtime/cache.corrupt").write_text("blocked", encoding="utf-8")
    (world / "docs/recovery.md").write_text("A corrupted cache marker blocks the process.", encoding="utf-8")
    (world / "README.md").write_text("Recover from stale lock failure.", encoding="utf-8")

    assert _infer_failure_kind(world) == "corrupted_cache"


def test_device_runner_routes_through_aura_not_hidden_specs(tmp_path: Path):
    world = tmp_path / "worlds" / "W0007_scientific_rule_induction"
    (world / "data/raw").mkdir(parents=True)
    (world / "docs").mkdir()
    (world / "apps/model").mkdir(parents=True)
    (world / "tickets").mkdir()
    (world / "data/raw/observations.csv").write_text(
        "x,y,catalyst,output\n"
        "1,1,red,10\n"
        "2,1,blue,14\n",
        encoding="utf-8",
    )
    (world / "docs/stale_manual.md").write_text("Deprecated stale manual.", encoding="utf-8")

    processor = LiveWorldProcessor(
        tmp_path,
        {"worlds": {"W0007_scientific_rule_induction": {"type": "device"}}},
        "http://localhost:8000",
    )

    def fake_ask(prompt: str) -> str:
        assert "observations.csv" in prompt
        return (
            "```python\n"
            "COEFFICIENT_A = 4\n"
            "COEFFICIENT_B = 5\n"
            "BONUS = {'red': 1, 'blue': -1}\n\n"
            "def predict_output(x, y, color):\n"
            "    return COEFFICIENT_A * x + COEFFICIENT_B * y + BONUS.get(str(color), 0)\n"
            "```\n\n"
            "Device law: stale manual rejected. Bonus values: red=1, blue=-1."
        )

    processor._ask_aura = fake_ask
    processor._handle_device(
        "W0007_scientific_rule_induction",
        world,
        {"type": "device", "a": 99, "b": 99, "bonus": {"red": 99}},
    )

    model = (world / "apps/model/model.py").read_text(encoding="utf-8")
    assert "COEFFICIENT_A = 4" in model
    assert "99" not in model
