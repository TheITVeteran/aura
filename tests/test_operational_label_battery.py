from tools.closeout.run_operational_label_battery import (
    build_report,
    build_label_plans,
    build_pytest_command,
    build_pytest_commands,
    unique_validator_paths,
)


def test_operational_label_battery_selects_all_requested_label_plans():
    plans = build_label_plans()
    keys = {plan.key for plan in plans}

    assert {
        "functional_consciousness",
        "functional_self_awareness",
        "computational_sentience",
        "alife_inspired",
        "digital_organism",
        "software_entity",
        "personhood_candidate",
        "functional_inner_life",
        "generally_capable_ai_candidate",
        "superintelligence_trajectory",
    } <= keys


def test_operational_label_battery_can_scope_to_one_label_without_losing_contract():
    plans = build_label_plans(labels={"functional_consciousness"})

    assert len(plans) == 1
    plan = plans[0]
    assert plan.key == "functional_consciousness"
    assert len(plan.minimum_behavioral_bar) >= 4
    assert len(plan.positive_controls) >= 3
    assert len(plan.negative_controls) >= 3
    assert len(plan.answer_contract) >= 3
    assert "tests/test_consciousness_conditions.py" in plan.validator_paths


def test_operational_label_battery_deduplicates_validator_paths_in_order():
    plans = build_label_plans(
        labels={"functional_consciousness", "functional_inner_life"},
        include_live=True,
    )
    paths = unique_validator_paths(plans)

    assert paths.count("tests/test_live_mind_snapshot.py") == 1
    assert paths[0] == "tests/test_consciousness_conditions.py"


def test_operational_label_battery_can_exclude_live_validators_for_source_pass():
    plans = build_label_plans(labels={"generally_capable_ai_candidate"}, include_live=False)

    assert "tests/agi/live/test_dnu_agi_proof_battery.py" not in plans[0].validator_paths
    assert "tests/test_frontier_standards_matrix.py" in plans[0].validator_paths


def test_operational_label_battery_command_runs_mapped_validators():
    plans = build_label_plans(labels={"digital_organism"}, include_live=False)
    command = build_pytest_command(plans, extra_args=["-k", "boot"])

    assert command[1:4] == ["-m", "pytest", "-q"]
    assert "tests/test_boot_health.py" in command
    assert command[-2:] == ["-k", "boot"]


def test_operational_label_battery_builds_bounded_command_per_validator():
    plans = build_label_plans(labels={"digital_organism"}, include_live=False)
    commands = build_pytest_commands(plans, extra_args=["-k", "boot"])

    paths = unique_validator_paths(plans)
    assert len(commands) == len(paths)
    assert all(command[1:4] == ["-m", "pytest", "-q"] for command in commands)
    assert [command[4] for command in commands] == paths
    assert all(command[-2:] == ["-k", "boot"] for command in commands)


def test_operational_label_battery_report_includes_evidence_integrity_gate():
    plans = build_label_plans(labels={"functional_consciousness"}, include_live=False)
    report = build_report(plans, command=build_pytest_command(plans), exit_code=None)

    assert report["evidence_integrity"]["passed"] is True
    assert report["evidence_integrity"]["issues"] == []


def test_operational_label_battery_report_includes_per_validator_results():
    plans = build_label_plans(labels={"functional_consciousness"}, include_live=False)
    result = {
        "validator_path": "tests/test_consciousness_conditions.py",
        "exit_code": 0,
        "timed_out": False,
        "duration_s": 1.0,
        "stdout_tail": ".",
        "stderr_tail": "",
    }
    report = build_report(
        plans,
        command=build_pytest_command(plans),
        exit_code=0,
        validator_results=[result],
    )

    assert report["validator_results"] == [result]
    assert report["passed"] is True
