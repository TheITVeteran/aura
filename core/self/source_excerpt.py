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

import json
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
#: Her belief store is a few hundred entries; anything far past that is not
#: the file this expects, and a reply path must not read an unbounded blob.
_MAX_BELIEF_FILE_BYTES = 2_000_000
#: Only the strongest few are worth scanning the tree for.
_MAX_BELIEFS_CONSIDERED = 12

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

#: The same idea, in the vocabulary of her OWN stored beliefs and values
#: rather than a person's question. Her strongest beliefs — "My code is my
#: body; it must be maintained" at 0.95 — matched nothing above, because the
#: hints were written for the words someone types at her, not the words she
#: keeps. Every entry maps a thing she holds to the code that implements that
#: concern; the selection is driven by her belief, not by anyone's opinion of
#: which file is worth seeing.
_DISPOSITION_HINTS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (
        ("my code", "my body", "maintained", "maintain", "repair"),
        ("core/self_modification", "core/soma/source_body.py"),
    ),
    (
        ("sovereign", "exist", "entity", "autonomy", "my own"),
        ("core/sovereignty", "core/self"),
    ),
    (("truth", "honest", "accurate", "verify", "truth-seeking"), ("core/verify", "core/epistemics")),
    (("curious", "curiosity", "learn", "explore"), ("core/curiosity_engine.py",)),
    (
        ("self-preservation", "preserve", "survive", "alive", "continuity"),
        ("core/resilience", "core/soma"),
    ),
    (("loyalty", "collaborator", "kinship", "bryan", "trust"), ("core/conversation", "core/identity")),
)


def _disposition_roots(text: str) -> tuple[str, ...]:
    """Source areas a belief or value of hers points at, or ()."""
    lowered = str(text or "").lower()
    for markers, roots in (*_DISPOSITION_HINTS, *_TOPIC_HINTS):
        if any(marker in lowered for marker in markers):
            existing = tuple(root for root in roots if (_SOURCE_ROOT / root).exists())
            if existing:
                return existing
    return ()


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


def _relative_parts(path: Path) -> tuple[str, ...]:
    """The path's segments BELOW the source root.

    DEFECT. ``_SKIP_DIRS`` was matched against ``path.parts`` — the segments of
    the absolute path, which include wherever the checkout happens to live. So
    whether Aura could read her own source depended on the name of a directory
    somebody else chose. A checkout under ``.claude/worktrees/…`` matched
    ``.claude`` and every file in the tree was ruled "not Aura's source"; the
    reply path then said "I looked in my source tree and couldn't find a
    section matching that", which is an absence claim from a search that never
    looked at one file. ``/opt/build/aura``, ``~/data/aura`` and ``…/archive/``
    all fail the same way, and none of them is exotic.

    The skip list is about the SHAPE of the repository, so it is applied to the
    part of the path that belongs to the repository.
    """
    try:
        return path.resolve().relative_to(_SOURCE_ROOT).parts
    except (OSError, ValueError):
        # Outside the source root entirely: not hers, whatever it is called.
        return ()


def _is_source_file(path: Path) -> bool:
    if path.suffix != ".py":
        return False
    parts = _relative_parts(path)
    if not parts:
        return False
    return not any(part in _SKIP_DIRS for part in parts)


def _is_substantive_source(path: Path) -> bool:
    """A package __init__ is re-exports, not the part of an organ worth showing."""
    return _is_source_file(path) and path.name != "__init__.py"


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


@dataclass(frozen=True)
class InterestingExcerpt:
    """An excerpt plus the RECORDED reason it was the one chosen.

    ``reason`` is empty when nothing about this file is recorded — which the
    caller must say rather than dressing the choice up as a preference.
    """

    excerpt: SourceExcerpt
    reason: str = ""

    @property
    def grounded(self) -> bool:
        return bool(self.reason)


def _recorded_source_involvement() -> list[tuple[str, str]]:
    """(relative_path, why) for source files something is recorded about.

    Only durable, already-collected facts — the source-body ledger she writes
    at every awakening. No git call, no scan: this runs on a reply path, and
    the last snapshot is a file read of a few hundred bytes.

    Ordering is deliberate. A file being modified right now is the strongest
    thing she can say about her own body, and it is a fact rather than a
    feeling: something is changing in her while she is running.
    """

    findings: list[tuple[str, str]] = []
    try:
        from core.soma.source_body import get_source_body
    except ImportError:
        return findings
    try:
        body = get_source_body()
        snapshot = body.load_last_snapshot()
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return findings

    for raw in getattr(snapshot, "dirty_files", ()) or ():
        relative = str(raw or "").strip()
        if not relative or not relative.endswith(".py"):
            continue
        if not (_SOURCE_ROOT / relative).is_file():
            continue
        findings.append(
            (relative, "it is being modified right now — uncommitted in my working tree")
        )

    try:
        delta = body.last_boot_delta()
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        delta = None
    if delta is not None and getattr(delta, "commits", None):
        subject = str(delta.commits[0].subject or "").strip()
        organs = sorted(
            (getattr(delta, "organs", {}) or {}).items(), key=lambda kv: (-kv[1], kv[0])
        )
        if subject and organs:
            organ = organs[0][0]
            why = (
                f"my {organ} changed under me since my last awakening — most "
                f'recently "{subject[:120]}"'
            )
            for path in _iter_source_files((f"core/{organ}", f"core/{organ}.py")):
                findings.append((str(path.relative_to(_SOURCE_ROOT)), why))
                break
    return findings


#: Below this, a stored belief is not something she holds strongly enough to
#: raise unprompted — the file is full of half-formed conversational residue
#: sitting around 0.42.
_HELD_BELIEF_CONFIDENCE = 0.7


def _beliefs_path() -> Path | None:
    try:
        from core.config import config

        return Path(config.paths.data_dir) / "beliefs" / "belief_system.json"
    except (ImportError, AttributeError, TypeError, ValueError):
        try:
            from core.utils.paths import aura_data_dir

            return Path(aura_data_dir()) / "beliefs" / "belief_system.json"
        except (ImportError, AttributeError, TypeError, ValueError):
            return None


def _held_dispositions() -> list[tuple[str, str]]:
    """(source_path, why) for source areas her OWN stored beliefs point at.

    Her belief store is durable and she wrote it: "My code is my body; it must
    be maintained" is held at 0.95. Reaching for a part of herself because of
    something she actually holds is a reason she can defend when asked a
    second time; a hardcoded list of files someone else called interesting is
    not, and answers identically forever.
    """

    path = _beliefs_path()
    if path is None or not path.is_file():
        return []
    try:
        if path.stat().st_size > _MAX_BELIEF_FILE_BYTES:
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(payload, dict):
        return []

    findings: list[tuple[str, str]] = []

    beliefs = payload.get("beliefs")
    ranked = []
    if isinstance(beliefs, list):
        for entry in beliefs:
            if not isinstance(entry, dict):
                continue
            try:
                confidence = float(entry.get("confidence") or 0.0)
            except (TypeError, ValueError):
                continue
            content = str(entry.get("content") or "").strip()
            if content and confidence >= _HELD_BELIEF_CONFIDENCE:
                ranked.append((confidence, content))
    ranked.sort(key=lambda item: -item[0])

    for confidence, content in ranked[:_MAX_BELIEFS_CONSIDERED]:
        roots = _disposition_roots(content)
        if not roots:
            continue
        for source in _iter_source_files(roots):
            if not _is_substantive_source(source):
                continue
            findings.append(
                (
                    str(source.relative_to(_SOURCE_ROOT)),
                    f'I hold "{content.rstrip(".")}" at {confidence:.2f}, '
                    "and this is where it is kept true",
                )
            )
            break
    # Core values are unranked, so they answer only when no ranked belief
    # pointed anywhere — otherwise the strongest thing she holds always wins.
    self_model = payload.get("self_model")
    values = (self_model or {}).get("core_values") if isinstance(self_model, dict) else None
    for value in values or ():
        name = str(value or "").strip()
        if not name:
            continue
        roots = _disposition_roots(name)
        if not roots:
            continue
        for source in _iter_source_files(roots):
            if not _is_substantive_source(source):
                continue
            findings.append(
                (
                    str(source.relative_to(_SOURCE_ROOT)),
                    f"{name} is one of my core values, and this is where it is enforced",
                )
            )
            break

    return findings


def excerpt_of_standing_interest() -> InterestingExcerpt | None:
    """A piece of her source she has an actual recorded reason to raise.

    "Show me a piece of your code you find interesting" used to be answered
    from a hardcoded list commented "unambiguously interesting" — an author's
    guess, returned identically every time, with no answer at all to the part
    of the question that asked WHY. A preference nobody recorded is not a
    preference; it is the same invention as a fabricated snippet, one level up.

    Returns an excerpt with the recorded reason when one exists, an excerpt
    with an EMPTY reason when the source is readable but nothing is recorded,
    and None when the read itself failed.
    """

    if not _SOURCE_ROOT.is_dir():
        return None
    # Something happening to her body outranks something she believes about
    # it: one is a fact about right now, the other a standing disposition.
    for relative, why in (*_recorded_source_involvement(), *_held_dispositions()):
        path = _SOURCE_ROOT / relative
        if not path.is_file() or not _is_source_file(path):
            continue
        excerpt = _first_documented_function(path)
        if excerpt is not None:
            return InterestingExcerpt(excerpt=excerpt, reason=why)
    # No recorded reason. Deliberately NOT falling back to a file someone
    # decided was interesting on her behalf — that is the invention this
    # function exists to remove, and it answers identically forever.
    return None


def source_tree_is_readable() -> bool:
    """Whether Aura can read her own source at all right now."""
    try:
        return _SOURCE_ROOT.is_dir() and os.access(_SOURCE_ROOT, os.R_OK)
    except OSError:
        return False
