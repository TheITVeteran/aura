from __future__ import annotations

from tools.security_scan import _scan_python_ast


def test_security_scan_does_not_flag_parser_token_regex_constants():
    source = r'''
_COUNT_TOKEN_RE = r"(?P<count>\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
ACCESS_TOKEN_PATTERN = r"\baccess\s+token\b"
'''

    assert _scan_python_ast(source, "sample.py") == []


def test_security_scan_flags_high_entropy_credential_assignments():
    source = '''
api_key = "mJ9Qp3vK8xZ2rT7nL4yB6cD1eF5gH0"
access_token = "nV8xY2rT7mJ9Qp3kL4yB6cD1eF5gH0"
password = "Z9xY8wV7uT6sR5qP4oN3mL2kJ1hG0fE"
'''

    findings = _scan_python_ast(source, "sample.py")

    assert [finding["line"] for finding in findings] == [2, 3, 4]
    assert {finding["kind"] for finding in findings} == {"secret_like_literal"}


def test_security_scan_allows_enum_symbol_values_that_mention_tokens_or_auth():
    source = '''
CAPABILITY_TOKEN_ISSUE = "capability_token_issue"
REQUIRE_FRESH_USER_AUTH = "require_fresh_user_auth"
'''

    assert _scan_python_ast(source, "sample.py") == []


def test_security_scan_allows_explicit_placeholders_for_operator_config():
    source = '''
api_key = "your_api_key_goes_here"
access_token = "placeholder-token-set-in-env"
'''

    assert _scan_python_ast(source, "sample.py") == []
