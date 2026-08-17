"""The active configuration plane cannot enable a remote model provider."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.config import LLMConfig, SecurityConfig
from core.runtime.settings_schema import (
    DEFAULT_VALUES,
    SCHEMA_BY_KEY,
    SETTINGS_SCHEMA_VERSION,
    migrated_settings_snapshot,
    validate_settings_patch,
)

ROOT = Path(__file__).resolve().parents[1]


def test_config_models_have_no_remote_model_credentials_or_teacher_controls():
    assert {
        "api_key",
        "gemini_api_key",
        "teacher_model",
    }.isdisjoint(LLMConfig.model_fields)
    assert {
        "allow_cloud_teacher_distillation",
        "redact_personal_data_to_model_providers",
    }.isdisjoint(SecurityConfig.model_fields)


def test_boot_configuration_never_reads_the_retired_gemini_secret():
    config_source = (ROOT / "core/config.py").read_text(encoding="utf-8")
    baseline_source = (
        ROOT / "core/orchestrator/initializers/core_baseline.py"
    ).read_text(encoding="utf-8")

    assert "GEMINI_API_KEY" not in config_source
    assert "GEMINI_API_KEY" not in baseline_source
    assert "_gemini_key" not in baseline_source


def test_retired_cloud_setting_is_rejected_and_removed_during_migration():
    key = "model.cloud_fallback_enabled"

    assert SETTINGS_SCHEMA_VERSION == 2
    assert key not in SCHEMA_BY_KEY
    assert key not in DEFAULT_VALUES
    with pytest.raises(KeyError, match="unknown_setting:model.cloud_fallback_enabled"):
        validate_settings_patch({key: True})

    values, unknown = migrated_settings_snapshot({key: True})
    assert key not in values
    assert unknown == (key,)


def test_first_run_wizard_has_no_remote_model_control():
    source = (ROOT / "interface/static/first_run.js").read_text(encoding="utf-8")

    assert "cloud_fallback" not in source
    assert "cloud provider" not in source
    assert "available local model lanes" in source


@pytest.mark.parametrize(
    "relative_path",
    ("MODEL_CARD.md", "AI_SYSTEM_CARD.md", "HARDWARE_PROFILES.md"),
)
def test_current_cards_make_no_gemini_or_cloud_model_claim(relative_path):
    content = (ROOT / relative_path).read_text(encoding="utf-8").lower()

    assert "gemini" not in content
    assert "cloud fallback" not in content
    assert "cloud / external model profile" not in content
