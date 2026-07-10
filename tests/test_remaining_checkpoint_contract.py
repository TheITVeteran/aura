from __future__ import annotations

from tools.closeout.remaining_checkpoint_contract import (
    FICTIONAL_CAPABILITY_IMPORTS,
    GAME_AI_PATTERN_IMPORTS,
    PRODUCT_MATURITY_REFERENCES,
    REMAINING_CHECKPOINTS,
    ROOT,
    REQUIRED_FRONTIER_KEYS,
    REQUIRED_LABEL_KEYS,
    report,
)


def _all_acceptance_text() -> str:
    lines: list[str] = []
    for checkpoint in REMAINING_CHECKPOINTS:
        lines.extend((checkpoint.name, checkpoint.purpose, checkpoint.commit_rule))
        for requirement in checkpoint.requirements:
            lines.extend((requirement.key, requirement.description, *requirement.acceptance))
    return "\n".join(lines).lower()


def test_remaining_contract_has_three_specific_checkpoints_not_old_generic_buckets():
    assert [checkpoint.number for checkpoint in REMAINING_CHECKPOINTS] == [9, 10, 11]
    names = [checkpoint.name.lower() for checkpoint in REMAINING_CHECKPOINTS]

    assert "operational labels and frontier standards proof" in names[0]
    assert "launched aura.app live-path reliability and general agency" in names[1]
    assert "boring reliability, learning closure, and final proof" in names[2]


def test_remaining_contract_includes_all_requested_operational_labels():
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
    } <= set(REQUIRED_LABEL_KEYS)


def test_remaining_contract_includes_frontier_and_product_maturity_targets():
    assert {
        "daily_runtime_reliability",
        "humanlike_conversation",
        "sci_fi_ai_capability",
        "phenomenal_building_blocks",
        "frontier_reasoning_outside_model",
        "superintelligence_trajectory",
        "os_control_frontier",
        "nethack_general_environment",
        "generally_capable_ai",
    } <= set(REQUIRED_FRONTIER_KEYS)

    assert {
        "Chrome",
        "Kubernetes",
        "macOS",
        "Postgres",
        "VS Code",
        "EDR",
        "industrial automation",
    } <= set(PRODUCT_MATURITY_REFERENCES)


def test_remaining_contract_includes_cell_choice_and_game_ai_checkpoint():
    requirements = {
        requirement.key: requirement
        for checkpoint in REMAINING_CHECKPOINTS
        for requirement in checkpoint.requirements
    }

    requirement = requirements["cell_choice_game_ai_integration"]
    text = "\n".join((
        requirement.description,
        *requirement.acceptance,
        *requirement.source_paths,
        *requirement.validators,
    )).lower()

    assert "cell spatial-code import" in text
    assert "preference" in text
    assert "choice" in text
    assert "game-ai patterns" in text
    assert "core/adaptation/spatial_receptor_code.py" in requirement.source_paths
    assert "core/agency/subjective_choice.py" in requirement.source_paths
    assert "tests/test_subjective_choice_engine.py" in requirement.validators


def test_live_desktop_truth_serum_is_explicit_acceptance_criteria():
    text = _all_acceptance_text()

    assert "assistant-mode takeover" in text
    assert "raw-model fallback" in text
    assert "full mind health requires kernel, inference, memory, scheduler, tool governance" in text
    assert "ram/thermal bounds" in text
    assert "duplicate kernels" in text
    assert "live artifacts" in text


def test_general_desktop_agency_is_not_task_shaped_demo_scripting():
    text = _all_acceptance_text()

    assert "general capabilities rather than hard-coded demo scripts" in text
    assert "notes, docs, folders, pdfs, browser research" in text
    assert "ai-interlocution" in text
    assert "popup recovery" in text
    assert "reorientation, retry, repair receipts" in text


def test_action_depth_expectation_engine_is_authoritative_remaining_scope():
    requirements = {
        requirement.key: requirement
        for checkpoint in REMAINING_CHECKPOINTS
        for requirement in checkpoint.requirements
    }

    requirement = requirements["action_depth_expectation_engine"]
    text = "\n".join((
        requirement.description,
        *requirement.acceptance,
        *requirement.source_paths,
        *requirement.validators,
    )).lower()

    assert "acceptance criteria" in text
    assert "partial_success" in text
    assert "success_unverified" in text
    assert "ok=true" in text
    assert "core/runtime/skill_contract.py" in requirement.source_paths
    assert "tests/test_action_depth_honesty.py" in requirement.validators


def test_fictional_ai_imports_cover_requested_and_safety_anti_pattern_systems():
    sources = {item.source for item in FICTIONAL_CAPABILITY_IMPORTS}

    assert {
        "JARVIS",
        "EDI",
        "Cortana",
        "Samantha / SAM",
        "MIST",
        "Pantheon UIs",
        "Safe Surf",
        "Jane",
        "Data",
        "Kokoro / Koroko",
        "Caine",
        "GLaDOS",
        "HAL 9000",
        "Skynet / Ultron / MCP / Agent Smith",
        "Culture Minds / Deep Thought",
        "The Machine / TARS / CASE",
    } <= sources


def test_fictional_ai_imports_land_in_existing_organs_not_fictional_silos():
    for item in FICTIONAL_CAPABILITY_IMPORTS:
        assert item.target_organs
        assert "fictional-AI silo" in item.forbidden_shortcut
        for organ in item.target_organs:
            assert "fictional_ai" not in organ
            assert (ROOT / organ).exists(), (item.source, organ)


def test_game_ai_pattern_imports_cover_user_requested_sources_and_real_organs():
    sources = {item.source for item in GAME_AI_PATTERN_IMPORTS}

    assert {
        "WorldBox",
        "Replika",
        "Kingdom Come: Deliverance II NPC systems",
        "Alien: Isolation",
        "Red Dead Redemption 2",
        "Middle-earth: Shadow of Mordor",
        "The Sims",
    } <= sources

    for item in GAME_AI_PATTERN_IMPORTS:
        assert item.mechanism_target
        assert item.production_boundary
        assert item.target_organs
        for organ in item.target_organs:
            assert "game_ai" not in organ
            assert (ROOT / organ).exists(), (item.source, organ)


def test_remaining_contract_source_and_validator_paths_are_currently_mapped():
    payload = report()

    assert payload["summary"]["remaining_checkpoints"] == 3
    assert payload["summary"]["requirements"] >= 7
    assert payload["summary"]["fictional_imports"] >= 16
    assert payload["summary"]["game_ai_imports"] >= 7
    assert payload["summary"]["gaps"] == 0, payload["gaps"]


def test_live_artifacts_remain_separate_from_source_validation():
    live_requirements = {
        requirement.key: requirement.live_artifacts
        for checkpoint in REMAINING_CHECKPOINTS
        for requirement in checkpoint.requirements
        if requirement.live_artifacts
    }

    assert "label_baseline_battery" in live_requirements
    assert "general_desktop_agency" in live_requirements
    assert "learning_substrate_and_final_artifacts" in live_requirements
    assert report(require_live=True)["summary"]["gaps"] >= 0
