"""CP126 hardening contracts for core/brain/llm/model_registry.py.

The registry decides how much context every lane may use, so a malformed or
hostile artifact must not crash resolution, enable a mode it declared off,
advertise an impossible allocation, or keep serving a stale limit after a
new artifact is promoted under the same name.
"""
from __future__ import annotations

import json

import pytest

from core.brain.llm.model_registry import (
    _MAX_CONTEXT_WINDOW,
    _MIN_CONTEXT_WINDOW,
    _artifact_signature,
    _coerce_bool,
    _context_window_for_artifact,
    _safe_positive_int,
)


@pytest.fixture
def artifact(tmp_path):
    def _write(config=None, tokenizer=None):
        if config is not None:
            (tmp_path / "config.json").write_text(
                config if isinstance(config, str) else json.dumps(config),
                encoding="utf-8",
            )
        if tokenizer is not None:
            (tmp_path / "tokenizer_config.json").write_text(
                json.dumps(tokenizer), encoding="utf-8"
            )
        return tmp_path

    return _write


class TestMalformedArtifactCannotCrashResolution:
    def test_truncated_config_degrades_to_default(self, artifact):
        path = artifact(config='{"max_position_embeddings": ')
        assert _context_window_for_artifact("m", _artifact_signature(path)) == 32768

    def test_non_object_config_degrades_to_default(self, artifact):
        path = artifact(config="[1, 2, 3]")
        assert _context_window_for_artifact("m2", _artifact_signature(path)) == 32768

    def test_non_numeric_values_do_not_raise(self, artifact):
        path = artifact(config={"max_position_embeddings": "lots"})
        assert _context_window_for_artifact("m3", _artifact_signature(path)) == 32768


class TestDeclaredFlagsAreHonored:
    def test_string_false_does_not_enable_sliding_window(self, artifact):
        # bool("false") is True, so the string spelling used to ENABLE the
        # expansion the artifact declared off.
        path = artifact(
            config={
                "max_position_embeddings": 8192,
                "sliding_window": 999_999,
                "use_sliding_window": "false",
            }
        )
        assert _context_window_for_artifact("m4", _artifact_signature(path)) == 8192

    def test_real_true_still_expands(self, artifact):
        path = artifact(
            config={
                "max_position_embeddings": 8192,
                "sliding_window": 16384,
                "use_sliding_window": True,
            }
        )
        assert _context_window_for_artifact("m5", _artifact_signature(path)) == 16384

    def test_coerce_bool_spellings(self):
        for truthy in (True, 1, "true", "TRUE", "yes", "on", "1"):
            assert _coerce_bool(truthy) is True, truthy
        for falsy in (False, 0, "false", "FALSE", "no", "off", "0", ""):
            assert _coerce_bool(falsy) is False, falsy


class TestDeclaredLimitsAreBounded:
    def test_absurd_window_is_capped(self, artifact):
        path = artifact(config={"max_position_embeddings": 10**30})
        assert _context_window_for_artifact("m6", _artifact_signature(path)) == _MAX_CONTEXT_WINDOW

    def test_tokenizer_sentinel_is_capped(self, artifact):
        path = artifact(config={}, tokenizer={"model_max_length": 10**30})
        assert _context_window_for_artifact("m7", _artifact_signature(path)) == _MAX_CONTEXT_WINDOW

    def test_tiny_window_is_floored(self, artifact):
        path = artifact(config={"max_position_embeddings": 8})
        assert _context_window_for_artifact("m8", _artifact_signature(path)) == _MIN_CONTEXT_WINDOW

    def test_safe_positive_int_never_raises(self):
        assert _safe_positive_int(None) == 0
        assert _safe_positive_int("junk") == 0
        assert _safe_positive_int(-5) == 0
        assert _safe_positive_int(4096) == 4096


class TestPromotionInvalidatesTheCache:
    def test_new_artifact_under_same_name_is_not_stale(self, artifact):
        path = artifact(config={"max_position_embeddings": 4096})
        before = _context_window_for_artifact("same", _artifact_signature(path))

        # Promote a new fused artifact at the same logical name.
        path = artifact(config={"max_position_embeddings": 16384})
        after = _context_window_for_artifact("same", _artifact_signature(path))

        assert before == 4096
        assert after == 16384, "a promoted artifact must not keep the old limit"
