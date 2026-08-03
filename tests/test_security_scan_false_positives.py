"""A gate whose only findings are false positives is a gate people skip.

The secret scanner's entire standing output was six findings, none of them
secrets: versioned schema identifiers, a snake_case mode constant, and CLI
flag names — all flagged because their NAMES contain words like "token" or
"key", or because the OpenAI key prefix appeared mid-word.

These pin that the exclusions are narrow enough to still catch real keys.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from security_scan import (  # noqa: E402
    SECRET_PATTERNS,
    _is_cli_flag_literal,
    _is_schema_identifier_literal,
    _is_symbolic_word_literal,
)


# ------------------------------------------------- real keys still detected


@pytest.mark.parametrize(
    "text",
    [
        'API_KEY = "sk-live-abcdefghijklmnopqrstuvwxyz0123"',
        'GOOGLE_KEY = "AIzaSyA0123456789abcdefghijklmnopqrstu"',
        "Authorization: Bearer sk-proj-0123456789abcdefghijklmnop",
    ],
)
def test_a_real_key_is_still_detected(text):
    assert any(pattern.search(text) for pattern in SECRET_PATTERNS), (
        "the false-positive work must not have blunted the actual detection"
    )


# ------------------------------------------------------ mid-word prefixes


def test_a_key_prefix_inside_a_word_is_not_a_key():
    """--task-issuer-signer-config carries the prefix inside "task"."""
    flag = "--" + "task" + "-issuer-signer-config"
    assert not any(pattern.search(flag) for pattern in SECRET_PATTERNS)


def test_a_key_at_the_start_of_a_line_is_still_detected():
    """The lookbehind must not require a preceding character to exist."""
    assert any(
        pattern.search("sk-live-abcdefghijklmnopqrstuvwxyz0123")
        for pattern in SECRET_PATTERNS
    )


# ------------------------------------------------------ structural literals


@pytest.mark.parametrize(
    "literal",
    [
        "aura.verified_token_trace.v1",
        "aura.verified_transition.bridge_token_binding.v1",
        "aura.verified_transition.recurrent_trace_codec.v1",
    ],
)
def test_a_versioned_schema_identifier_is_not_credential_material(literal):
    assert _is_schema_identifier_literal(["ANY_NAME"], literal)


@pytest.mark.parametrize(
    "literal",
    [
        "sk-live-abcdefghijklmnopqrstuvwxyz0123",
        "AIzaSyA0123456789abcdefghijklmnop",
        "aura.thing",          # no version
        "Aura.Thing.v1",       # not lowercase
        "aura.thing.v",        # no version number
    ],
)
def test_things_that_are_not_schema_identifiers_are_not_excused(literal):
    assert not _is_schema_identifier_literal(["SOME_SCHEMA"], literal)


def test_a_snake_case_mode_constant_is_not_a_key():
    assert _is_symbolic_word_literal("uniform_nonnegative_normalized")


@pytest.mark.parametrize(
    "literal",
    ["sk-live-abcdefghijklmnop", "AbC_dEf", "token_9f8e7d", "singleword"],
)
def test_symbolic_word_exclusion_stays_narrow(literal):
    assert not _is_symbolic_word_literal(literal)


def test_a_long_form_flag_name_is_not_a_key():
    assert _is_cli_flag_literal("--evidence-verifier-signer-config")
    assert _is_cli_flag_literal("--trust-root")


@pytest.mark.parametrize("literal", ["-x", "--", "--Bad-Case", "sk-live-abcdefgh"])
def test_flag_exclusion_stays_narrow(literal):
    assert not _is_cli_flag_literal(literal)


# ------------------------------------------------------------- the gate itself


def test_the_repository_currently_passes_the_secret_scan():
    """It failed at the start of this work, on six non-secrets."""
    from security_scan import scan

    report = scan()
    assert report["passed"], report["findings"]
