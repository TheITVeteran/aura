import pytest

from tools.program_dna.behavioral_equivalence_battery import run_battery, scenarios


@pytest.mark.asyncio
async def test_program_dna_hidden_source_behavioral_equivalence_battery(tmp_path):
    report = await run_battery(project_root=tmp_path)

    assert report["ok"] is True
    assert report["battery"] == "program_dna_hidden_source_behavioral_equivalence"
    assert report["scenario_count"] == 7
    assert report["passed_scenarios"] == 7
    assert report["equivalence"] == 1.0
    assert report["passed_cases"] == report["held_out_cases"]
    categories = {item["category"] for item in report["results"]}
    assert {
        "cli",
        "gui",
        "file_format_converter",
        "web_app",
        "local_db_tool",
        "auth_mocked_app",
        "missing_docs",
    } <= categories
    assert all(item["hidden_source_withheld"] for item in report["results"])
    assert all(item["genome_summary"]["feature_count"] >= 1 for item in report["results"])


def test_program_dna_battery_scenarios_do_not_expose_source_paths():
    for scenario in scenarios():
        assert scenario.held_out_cases
        assert scenario.docs or scenario.behavior_examples
        assert not hasattr(scenario, "source_paths")
