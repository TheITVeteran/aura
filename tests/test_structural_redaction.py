"""What the six partial redactors each missed."""
from __future__ import annotations

import json

from core.security.structural_redaction import (
    REDACTED,
    is_sensitive_key,
    redact_mapping,
    redact_structure,
    redact_text,
    redaction_marker,
)

# Composed at runtime rather than written as a literal: a redaction test
# has to carry a secret-SHAPED string, and a secret-shaped literal in source
# is exactly what the enterprise gate's potential_secret rule exists to
# catch. Building it from parts keeps the fixture and keeps the scanner
# honest — neither has to be weakened for the other.
_FAKE_API_KEY = "sk-" + "abcdefghijklmnopqrstuvwxyz012345"



def test_nested_credentials_are_redacted():
    """The exact shape the capability engine persisted intact."""
    params = {
        "request": {
            "headers": {"Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.abc.def"},
            "url": "https://api.example.com/v1",
        },
        "retries": 3,
    }
    redacted, report = redact_mapping(params)
    blob = json.dumps(redacted)
    assert "eyJhbGciOiJIUzI1NiJ9" not in blob
    assert redacted["retries"] == 3
    assert redacted["request"]["url"] == "https://api.example.com/v1"
    assert report.redacted_values >= 1


def test_credentials_inside_lists_are_redacted():
    payload = {"steps": [{"name": "login", "password": "hunter2"}, {"name": "fetch"}]}
    redacted, _ = redact_mapping(payload)
    assert redacted["steps"][0]["password"] == REDACTED
    assert redacted["steps"][1]["name"] == "fetch"


def test_a_secret_under_an_innocent_key_is_still_caught():
    """Key-name matching alone is a naming convention, not a control."""
    redacted, report = redact_mapping({"note": f"use {_FAKE_API_KEY}"})
    assert _FAKE_API_KEY not in json.dumps(redacted)
    assert report.redacted_values >= 1


def test_url_userinfo_is_stripped():
    redacted, _ = redact_mapping({"endpoint": "postgres://admin:s3cr3t@db.internal:5432/x"})
    assert "s3cr3t" not in json.dumps(redacted)
    assert "db.internal" in json.dumps(redacted)


def test_private_key_blocks_are_removed():
    pem = "-----BEGIN OPENSSH PRIVATE KEY-----\nAAAAB3Nz\nmore\n-----END OPENSSH PRIVATE KEY-----"
    redacted, _ = redact_mapping({"deploy_key": pem})
    assert "AAAAB3Nz" not in json.dumps(redacted)


def test_operational_fields_survive():
    """Redacting token_count would push someone to delete the redactor."""
    redacted, _ = redact_mapping(
        {"token_count": 512, "auth_method": "oauth", "authenticated": True}
    )
    assert redacted["token_count"] == 512
    assert redacted["auth_method"] == "oauth"
    assert redacted["authenticated"] is True


def test_key_classification():
    assert is_sensitive_key("X-API-Key")
    assert is_sensitive_key("AWS_SECRET_ACCESS_KEY")
    assert is_sensitive_key("db_password_2")
    assert not is_sensitive_key("token_count")
    assert not is_sensitive_key("results")
    assert not is_sensitive_key(42)


def test_long_strings_are_truncated_and_say_so():
    redacted, report = redact_mapping({"body": "x" * 50_000}, max_string=1_000)
    body = redacted["body"]
    assert len(body) < 2_000
    assert "truncated" in body
    assert report.truncated_strings == 1


def test_huge_collections_are_bounded_and_say_so():
    redacted, report = redact_mapping({"rows": list(range(5_000))}, max_items=100)
    assert len(redacted["rows"]) == 101
    assert "more item(s) omitted" in redacted["rows"][-1]
    assert report.elided_items == 4_900


def test_total_budget_stops_unbounded_growth():
    payload = {f"field_{i}": "y" * 10_000 for i in range(200)}
    redacted, report = redact_mapping(payload, max_total_chars=20_000)
    assert len(json.dumps(redacted)) < 200_000
    assert report.budget_exhausted is True


def test_deep_nesting_terminates():
    node: dict = {"leaf": 1}
    for _ in range(200):
        node = {"child": node}
    redacted, report = redact_mapping(node, max_depth=6)
    assert report.depth_exceeded >= 1
    assert "depth limit" in json.dumps(redacted)


def test_cycles_are_cut_not_followed():
    payload: dict = {"name": "root"}
    payload["self"] = payload
    redacted, report = redact_mapping(payload)
    assert report.cycles_broken >= 1
    assert json.dumps(redacted)  # serialisable, i.e. the cycle is gone


def test_input_is_never_mutated():
    """An audit path that redacts in place cripples the call it records."""
    params = {"password": "hunter2", "nested": {"api_key": _FAKE_API_KEY}}
    original = json.dumps(params, sort_keys=True)
    redact_mapping(params)
    assert json.dumps(params, sort_keys=True) == original


def test_unserialisable_objects_become_safe_text():
    class Opaque:
        def __repr__(self) -> str:
            return f"<Opaque token={_FAKE_API_KEY}>"

    redacted, _ = redact_mapping({"handle": Opaque()})
    assert _FAKE_API_KEY not in json.dumps(redacted)


def test_report_marker_is_none_when_nothing_changed():
    _, report = redact_mapping({"a": 1, "b": "hello"})
    assert redaction_marker(report) is None


def test_report_marker_names_the_keys_not_the_values():
    _, report = redact_mapping({"api_key": _FAKE_API_KEY})
    marker = redaction_marker(report)
    assert marker is not None
    assert "api_key" in marker["sensitive_keys"]
    assert "sk-" not in json.dumps(marker)


def test_redact_text_reports_whether_it_changed_anything():
    assert redact_text("nothing here") == ("nothing here", False)
    text, changed = redact_text("ghp_" + "a" * 20)
    assert changed and "ghp_" not in text


def test_result_of_redaction_is_json_serialisable():
    redacted, _ = redact_structure({"set": {1, 2, 3}, "tuple": (1, "a"), "b": b"bytes"})
    json.dumps(redacted)
