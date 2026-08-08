"""Privacy-preserving hashes that bind authored text to persisted artifacts."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable


def canonical_text(value: object) -> str:
    """Normalize transport-only differences without changing authored words."""

    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def text_sha256(value: object) -> str:
    return hashlib.sha256(canonical_text(value).encode("utf-8")).hexdigest()


def paragraph_sha256s(value: object) -> tuple[str, ...]:
    """Hash non-empty paragraphs so a larger document can prove inclusion."""

    paragraphs = (
        canonical_text(part)
        for part in re.split(r"\n\s*\n", canonical_text(value))
    )
    return tuple(text_sha256(part) for part in paragraphs if part)


def contains_paragraph_hashes(
    container_hashes: Iterable[object],
    required_hashes: Iterable[object],
) -> bool:
    container = {str(value) for value in container_hashes if str(value)}
    required = {str(value) for value in required_hashes if str(value)}
    return bool(required) and required.issubset(container)
