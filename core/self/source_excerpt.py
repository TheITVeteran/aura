"""Read Aura's own source, so she can answer about it instead of inventing it.

LIVE DEFECT, 2026-08-03 19:43. Asked "can you show me a section of your code
that you're interested in?", then "can you show me the actual code?", Aura
produced a generic transformer pipeline —

    tokens = tokenize_text(user_input)
    embeddings = generate_embeddings(tokens)
    response_vector, attention_weights = transformer_model(embeddings)

— and a ``reschedule_attention`` method that exists nowhere in this repository.
She then said her implementation "involves distributed computation across
multiple GPUs and specialized hardware accelerators", on a single-GPU MacBook,
and when Bryan corrected her she agreed and produced a second false
explanation about dual-GPU laptops.

None of that came from a file. The conversational path had no way to reach the
source tree, so the question fell through to the model's weights, which will
always answer a question about "your code" with something that looks like
code. The repository was right there: ``core/self_modification/repo_search.py``
greps it and ``core/self/architecture_index.py`` indexes it, and neither was
reachable from a chat turn.

This module is the reachable seam. Every excerpt it returns was read from a
real file on disk, and carries the path and line numbers so the claim is
checkable. When it cannot read one, it returns nothing — the caller then says
so, which is the honest answer to "show me your code" from a runtime that
could not open it.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("Aura.Self.SourceExcerpt")

#: Bounds. An excerpt is for reading in a chat window, not for dumping a file.
_MAX_EXCERPT_LINES = 40
_MAX_EXCERPT_CHARS = 2400
_MAX_FILES_SCANNED = 4000

#: Where Aura's own code lives, relative to this file.
_SOURCE_ROOT = Path(__file__).resolve().parents[2]

#: Directories that are not Aura's source: dependencies, build output, data,
#: and the artifacts of past runs.
_SKIP_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "__pycache__",
        "node_modules",
        ".claude",
        "artifacts",
        "archive",
        "data",
        "logs",
        "build",
        "dist",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    }
)

#: Topics a person is likely to ask about, and the parts of the tree that
#: actually implement them. Ordered: the first entry whose files exist wins.
_TOPIC_HINTS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("memory", "remember", "recall", "forget"), ("core/memory", "core/autonomy/memory_persister.py")),
    (("attention", "attend", "focus"), ("core/consciousness", "core/cognition")),
    (("emotion", "affect", "feel", "mood"), ("core/affect",)),
    (("route", "routing", "pathway", "mycelium"), ("core/mycelium.py",)),
    (("reply", "response", "answer", "conversation", "talk"), ("core/synthesis.py", "core/conversation")),
    (("model", "cortex", "mlx", "inference", "generate"), ("core/brain/llm",)),
    (("goal", "drive", "motivation", "want"), ("core/goals", "core/agency")),
    (("safety", "governance", "constitution", "refuse"), ("core/governance", "core/constitution.py")),
    (("browser", "web", "chatgpt", "interlocutor"), ("core/capabilities/web_interlocutor.py",)),
    (("screen", "see", "vision", "perception"), ("core/perception",)),
    (("identity", "who you are", "self"), ("core/identity", "core/self")),
    (("skill", "tool", "capability"), ("core/skills", "core/capability_engine.py")),
)


@dataclass(frozen=True)
class SourceExcerpt:
    """A real excerpt of Aura's own source, with where it came from."""

    relative_path: str
    start_line: int
    end_line: int
    text: str
    symbol: str = ""

    def rendered(self) -> str:
        """The excerpt as it should appear in a reply — attributed."""
        where = f"{self.relative_path}:{self.start_line}"
        if self.symbol:
            where = f"{where} ({self.symbol})"
        return f"{where}\n\n```python\n{self.text}\n```"


def _is_source_file(path: Path) -> bool:
    if path.suffix != ".py":
        return False
    return not any(part in _SKIP_DIRS for part in path.parts)


def _iter_source_files(roots: tuple[str, ...]) -> list[Path]:
    found: list[Path] = []
    for relative in roots:
        target = _SOURCE_ROOT / relative
        if target.is_file():
            if _is_source_file(target):
                found.append(target)
            continue
        if not target.is_dir():
            continue
        for path in sorted(target.rglob("*.py")):
            if len(found) >= _MAX_FILES_SCANNED:
                return found
            if _is_source_file(path):
                found.append(path)
    return found


def _topic_roots(topic: str) -> tuple[str, ...]:
    lowered = str(topic or "").lower()
    for markers, roots in _TOPIC_HINTS:
        if any(marker in lowered for marker in markers):
            existing = tuple(root for root in roots if (_SOURCE_ROOT / root).exists())
            if existing:
                return existing
    return ()


_DEF_RE = re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def _first_documented_function(path: Path) -> SourceExcerpt | None:
    """The first function in a file that carries a docstring.

    A documented function is the one worth showing: it says what it is for in
    Aura's own words, rather than being an anonymous helper.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        logger.debug("Source excerpt skipped %s: %s", path, exc)
        return None
    for index, line in enumerate(lines):
        match = _DEF_RE.match(line)
        if not match:
            continue
        following = lines[index + 1 : index + 3]
        if not any('"""' in candidate for candidate in following):
            continue
        end = min(len(lines), index + _MAX_EXCERPT_LINES)
        body = "\n".join(lines[index:end]).rstrip()
        if len(body) > _MAX_EXCERPT_CHARS:
            body = body[:_MAX_EXCERPT_CHARS].rsplit("\n", 1)[0].rstrip()
            end = index + body.count("\n") + 1
        try:
            relative = str(path.relative_to(_SOURCE_ROOT))
        except ValueError:  # pragma: no cover - path outside the tree
            relative = str(path)
        return SourceExcerpt(
            relative_path=relative,
            start_line=index + 1,
            end_line=end,
            text=body,
            symbol=match.group(1),
        )
    return None


def excerpt_for_topic(topic: str = "") -> SourceExcerpt | None:
    """A real excerpt of Aura's source related to ``topic``, or None.

    None means the read did not happen — a missing tree, an unreadable file,
    no match. The caller must say that rather than substituting prose, because
    "here is my code" backed by nothing is the defect this exists to remove.
    """
    if not _SOURCE_ROOT.is_dir():
        return None
    roots = _topic_roots(topic)
    if not roots:
        # No topic, or one with no mapping: show a part of the runtime that is
        # unambiguously hers and unambiguously interesting.
        roots = tuple(
            candidate
            for candidate in ("core/mycelium.py", "core/synthesis.py", "core/self")
            if (_SOURCE_ROOT / candidate).exists()
        )
    for path in _iter_source_files(roots):
        excerpt = _first_documented_function(path)
        if excerpt is not None:
            return excerpt
    return None


def source_tree_is_readable() -> bool:
    """Whether Aura can read her own source at all right now."""
    try:
        return _SOURCE_ROOT.is_dir() and os.access(_SOURCE_ROOT, os.R_OK)
    except OSError:
        return False
