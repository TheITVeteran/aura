#!/usr/bin/env python3
"""Lightweight local security scan for Aura release gates."""
from __future__ import annotations

import json
import re
import ast
import math
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCAN_ROOTS = ("core", "interface", "skills", "tools", "training")
SKIP_PARTS = {"__pycache__", ".git", ".venv", ".venv_aura", "node_modules", "artifacts"}

SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
)
REGEX_LITERAL_MARKERS = ("(?P<", "\\b", "\\s", "\\d", "[", "]", "|", "*", "+", "{", "}")
CREDENTIAL_NAME_WORDS = {
    "apikey",
    "api_key",
    "auth",
    "bearer",
    "credential",
    "credentials",
    "jwt",
    "key",
    "password",
    "secret",
    "token",
}
NON_SECRET_TOKEN_NAME_WORDS = {
    "count",
    "format",
    "lexer",
    "parser",
    "pattern",
    "regex",
    "re",
    "tokenizer",
    "tokens",
}
PLACEHOLDER_MARKERS = (
    "change_me",
    "changeme",
    "dummy",
    "example",
    "fake",
    "not_set",
    "placeholder",
    "sample",
    "test",
    "your_",
)


def scan() -> dict:
    findings: list[dict] = []
    files_scanned = 0
    for root in SCAN_ROOTS:
        base = ROOT / root
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".js", ".ts", ".tsx", ".json", ".yaml", ".yml", ".toml", ".md"}:
                continue
            if any(part in SKIP_PARTS for part in path.parts):
                continue
            files_scanned += 1
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            rel = path.relative_to(ROOT).as_posix()
            findings.extend(_scan_text_literals(text, rel))
            if path.suffix == ".py":
                findings.extend(_scan_python_ast(text, rel))
    return {
        "generated_at": time.time(),
        "files_scanned": files_scanned,
        "findings": findings,
        "passed": not findings,
    }


def _line(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _scan_text_literals(text: str, rel: str) -> list[dict]:
    findings: list[dict] = []
    for pattern in SECRET_PATTERNS:
        for match in pattern.finditer(text):
            findings.append({"kind": "secret_like_literal", "file": rel, "line": _line(text, match.start())})
    return findings


def _scan_python_ast(text: str, rel: str) -> list[dict]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    findings: list[dict] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            findings.extend(_scan_assignment(node, rel))
        elif isinstance(node, ast.AnnAssign):
            findings.extend(_scan_assignment(node, rel))
        elif isinstance(node, ast.Call):
            finding = _scan_call(node, rel)
            if finding:
                findings.append(finding)
    return findings


def _scan_assignment(node: ast.Assign | ast.AnnAssign, rel: str) -> list[dict]:
    names: list[str] = []
    if isinstance(node, ast.Assign):
        for target in node.targets:
            names.extend(_target_names(target))
        value = node.value
    else:
        names.extend(_target_names(node.target))
        value = node.value
    if not names or not isinstance(value, ast.Constant) or not isinstance(value.value, str):
        return []
    literal = value.value.strip()
    if _is_parser_regex_constant(names, literal):
        return []
    if not _has_credential_name(names):
        return []
    if _is_enum_symbol_literal(names, literal):
        return []
    if _is_env_var_name_literal(names, literal):
        return []
    if _is_placeholder_literal(literal):
        return []
    if len(literal) < 20 or literal.startswith("$") or literal.startswith("<"):
        return []
    if _entropy(literal) < 3.0:
        return []
    return [{"kind": "secret_like_literal", "file": rel, "line": getattr(node, "lineno", 1)}]


def _has_credential_name(names: list[str]) -> bool:
    for name in names:
        normalized = name.strip("_").lower()
        if not normalized or "enum" in normalized:
            continue
        if normalized in CREDENTIAL_NAME_WORDS:
            return True
        if "api_key" in normalized or "apikey" in normalized:
            return True
        words = [part for part in re.split(r"[^a-z0-9]+|_", normalized) if part]
        word_set = set(words)
        if word_set & {"secret", "password", "credential", "credentials", "jwt", "bearer", "auth"}:
            return True
        if "token" in word_set and not (word_set & NON_SECRET_TOKEN_NAME_WORDS):
            return True
        if "key" in word_set and (word_set & {"api", "private", "secret", "client", "access"}):
            return True
    return False


def _is_parser_regex_constant(names: list[str], literal: str) -> bool:
    if not any(marker in literal for marker in REGEX_LITERAL_MARKERS):
        return False
    for name in names:
        lowered = name.lower()
        if lowered.endswith("_re") or lowered.endswith("_regex") or "pattern" in lowered:
            return True
        words = {part for part in re.split(r"[^a-z0-9]+|_", lowered) if part}
        if "regex" in words or "parser" in words or "tokenizer" in words:
            return True
        if "token" in words and words & NON_SECRET_TOKEN_NAME_WORDS:
            return True
    return False


def _is_placeholder_literal(literal: str) -> bool:
    lowered = literal.lower()
    return any(marker in lowered for marker in PLACEHOLDER_MARKERS)


def _is_env_var_name_literal(names: list[str], literal: str) -> bool:
    """`FOO_TOKEN_ENV = "AURA_FOO_TOKEN"` names WHERE a secret lives, not the
    secret itself: the variable must say it holds an env-var name and the
    literal must be a SCREAMING_SNAKE identifier, not credential material."""
    if not re.fullmatch(r"[A-Z][A-Z0-9_]{2,}", literal):
        return False
    for name in names:
        words = {part for part in re.split(r"[^a-z0-9]+|_", name.lower()) if part}
        if "env" in words or "envvar" in words:
            return True
    return False


def _is_enum_symbol_literal(names: list[str], literal: str) -> bool:
    if not literal or not re.fullmatch(r"[a-z][a-z0-9_:-]{2,}", literal):
        return False
    normalized_literal = literal.replace("-", "_").replace(":", "_").upper()
    for name in names:
        normalized_name = name.strip("_").upper()
        if normalized_name and normalized_name == normalized_literal:
            return True
    return False


def _scan_call(node: ast.Call, rel: str) -> dict | None:
    name = _call_name(node.func)
    if name == "subprocess.run":
        for keyword in node.keywords:
            if keyword.arg == "shell" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
                return {"kind": "dangerous_runtime_pattern", "file": rel, "line": node.lineno}
    if name == "os.system":
        return {"kind": "dangerous_runtime_pattern", "file": rel, "line": node.lineno}
    if name in {"pickle.load", "pickle.loads"}:
        return {"kind": "dangerous_runtime_pattern", "file": rel, "line": node.lineno}
    return None


def _target_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        return [node.attr]
    if isinstance(node, (ast.Tuple, ast.List)):
        out: list[str] = []
        for item in node.elts:
            out.extend(_target_names(item))
        return out
    return []


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _entropy(value: str) -> float:
    if not value:
        return 0.0
    total = len(value)
    counts = {ch: value.count(ch) for ch in set(value)}
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def main() -> int:
    report = scan()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
