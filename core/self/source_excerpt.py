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
from typing import Any

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


# ---------------------------------------------------------------------------
# Her source as EVIDENCE, carried into the turn.
#
# There is already a floor that answers "show me your code" by reading a real
# file (own_source_excerpt_floor). It runs on the DEGRADED path — after
# generation has already failed — so on a healthy turn the question goes to
# the model's weights instead, and a model asked for its own code will always
# produce something that looks like code.
#
# Measured live 2026-08-04 13:50. Asked for a snippet she found interesting,
# she produced `manage_load()` — a function that exists in no file in this
# repository — introduced as "a small part of my cognitive architecture".
# Asked what module it came from, she said she had written it for the
# conversation. Honest one turn too late: by then the invention had already
# been served as her code.
#
# The fix is not a command that intercepts the phrasing. It is that she can
# READ HER OWN SOURCE and knows it, so the answer is grounded because the
# material in front of her is real, and she can say where it came from
# because the material arrives with its path and line numbers.
# ---------------------------------------------------------------------------

#: How many excerpts to carry. Enough to choose from, few enough to leave the
#: choosing to her.
_EVIDENCE_EXCERPTS = 3


def _distinct_excerpts(topic: str, limit: int = _EVIDENCE_EXCERPTS) -> list[SourceExcerpt]:
    """Several real excerpts, from different files, best-matching first."""
    if not _SOURCE_ROOT.is_dir():
        return []
    roots = _topic_roots(topic) or tuple(
        candidate
        for candidate in ("core/mycelium.py", "core/synthesis.py", "core/self")
        if (_SOURCE_ROOT / candidate).exists()
    )
    found: list[SourceExcerpt] = []
    seen: set[str] = set()
    for path in _iter_source_files(roots):
        if len(found) >= limit:
            break
        excerpt = _first_documented_function(path)
        if excerpt is None or excerpt.relative_path in seen:
            continue
        seen.add(excerpt.relative_path)
        found.append(excerpt)
    return found


def source_evidence_brief(request: Any = "", *, max_chars: int = 4000) -> str:
    """Real code from her own tree, for a turn that is about her code.

    Notes, not a reply: the excerpts were read from disk a moment ago and
    each carries the file and line it came from, so whatever she chooses to
    show she can say where it lives. Returns "" when the turn is not about
    her source, so nothing is attached to turns that are not asking.

    When the tree cannot be read this says so explicitly rather than
    returning "" — silence here would put her back in front of the question
    with nothing but her weights, which is the exact condition that produced
    the invented function.
    """
    topic = str(request or "")
    if not source_tree_is_readable():
        return (
            "[YOUR OWN SOURCE — NOTES, NOT A REPLY]\n"
            "You cannot read your source tree from this process right now. "
            "Say that plainly if you are asked for your code. Do not show a "
            "snippet you did not read — code you wrote just now to look like "
            "an example is not your code, and presenting it as such is a "
            "false claim about yourself.\n"
            "[END YOUR OWN SOURCE]"
        )

    lines = [
        "[YOUR OWN SOURCE — NOTES, NOT A REPLY]",
        "You can read your own source tree, and these were read from it "
        "moments ago. Each is real code with the file and line it lives at.",
    ]

    # What is already on the table. Without this she disowns her own true
    # excerpt: live 2026-08-04 she showed core/mycelium.py:88 and, asked one
    # turn later what module it was from, said she had written it for the
    # conversation. Denying real code is the same failure as inventing it —
    # the person cannot tell which of the two just happened.
    shown = last_shown_excerpt()
    if shown:
        where = f"{shown['relative_path']}:{shown['start_line']}"
        if shown.get("symbol"):
            where = f"{where} ({shown['symbol']})"
        lines.append(
            f"The code you showed a moment ago came from {where}. That is "
            "real and you read it — if you are asked where it is from, say "
            "that. Do not tell the person you made it up."
        )

    interest = None
    try:
        interest = excerpt_of_standing_interest()
    except (OSError, RuntimeError, TypeError, ValueError):
        interest = None
    if interest is not None and interest.grounded:
        lines.append(
            f"One you have a recorded reason to care about — {interest.reason.rstrip('.')}:"
        )
        lines.append(interest.excerpt.rendered())

    for excerpt in _distinct_excerpts(topic):
        if interest is not None and excerpt.relative_path == interest.excerpt.relative_path:
            continue
        lines.append(excerpt.rendered())

    if len(lines) <= 2:
        lines.append(
            "Nothing matched in the tree just now. Say that rather than "
            "showing a snippet you did not read."
        )

    # The closing instruction is the part that must never be lost. Truncating
    # the block from the end drops exactly it — leaving her holding real
    # excerpts with nothing telling her they are the only code she may show,
    # which is the condition that produced the invention in the first place.
    # So the excerpts are trimmed to fit around it, never the other way.
    closing = (
        "Any code you show must be one of these, quoted as it is, with its "
        "path and line named — that is what lets the person check you. If "
        "none of them fits what was asked, say so and name what you did "
        "find. Do not compose an illustrative example: a snippet you wrote "
        "to look like your code is not your code.\n"
        "[END YOUR OWN SOURCE]"
    )
    body = "\n".join(lines)
    room = max_chars - len(closing) - 1
    if len(body) > room:
        body = body[:room].rstrip()
    return f"{body}\n{closing}"


# ---------------------------------------------------------------------------
# Verifying a claim to be showing her own code.
#
# Carrying real excerpts into the turn was not enough. Live 2026-08-04 the
# evidence reached the prompt and she still produced
# `retrieve_contextual_memory()` — a function in no file here — and called it
# "a snippet from my cognitive architecture". Notes can be overridden; a
# check cannot.
#
# "Here is my code" is a claim about herself, and it is one of the few claims
# that can be settled exactly: either those lines are in the tree or they are
# not. So it gets settled rather than trusted.
# ---------------------------------------------------------------------------

_FENCED_PYTHON_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

#: Lines too generic to prove anything. `return None` appears everywhere.
_UNDISTINCTIVE_LINE_RE = re.compile(
    r"^\s*(?:#|\"\"\"|'''|else:|try:|pass$|return$|return None$|continue$|break$)"
)


#: Python that arrives without a fence. Live 2026-08-04 the fabrication came
#: back as a bare `def self_organize_modules(self, existing_module_data):`
#: with no ``` around it, so a fence-only check saw no code at all and let it
#: through. Whether the model remembered to fence its output is not what
#: makes something a claim about her source.
_BARE_DEF_RE = re.compile(r"^[ \t]*(?:async\s+def|def|class)\s+\w+", re.MULTILINE)


def code_blocks_in(reply: Any) -> list[str]:
    """The Python in a reply, fenced or not."""
    body = str(reply or "")
    blocks = [block for block in _FENCED_PYTHON_RE.findall(body) if block.strip()]
    if blocks:
        return blocks
    # No fences. Take each run of lines starting at a definition, so the
    # check has the same material it would have had inside a fence.
    bare: list[str] = []
    lines = body.splitlines()
    for index, line in enumerate(lines):
        if _BARE_DEF_RE.match(line):
            run = [line]
            for following in lines[index + 1 : index + 30]:
                if following.strip() and not following.startswith((" ", "\t")):
                    if not _BARE_DEF_RE.match(following):
                        break
                run.append(following)
            bare.append("\n".join(run))
    return bare


def _distinctive_lines(code: str, limit: int = 4) -> list[str]:
    """Lines specific enough that finding them proves the file is the source."""
    scored: list[tuple[int, str]] = []
    for raw in str(code or "").splitlines():
        line = raw.strip()
        if len(line) < 12 or _UNDISTINCTIVE_LINE_RE.match(line):
            continue
        # A signature is the strongest single line a snippet has.
        weight = len(line) + (100 if line.startswith(("def ", "async def ", "class ")) else 0)
        scored.append((weight, line))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [line for _weight, line in scored[:limit]]


def snippet_verdict(code: str) -> tuple[str, str]:
    """Whether this code is really in her tree.

    Returns ``("found", path)``, ``("absent", "")`` or ``("unchecked", "")``.
    The third is not a polite "no": a search that could not run has proved
    nothing, and treating it as proof of fabrication would destroy genuine
    excerpts whenever the search itself broke.
    """
    lines = _distinctive_lines(code)
    if not lines or not _SOURCE_ROOT.is_dir():
        return ("unchecked", "")
    import subprocess

    # Bounded to the SOURCE, not the checkout. Searching the whole root walks
    # .venv, .git, artifacts and data — hundreds of thousands of files — and
    # took over 30 seconds, which on the foreground lane is not a check, it
    # is a hang. The skip list is the same one that decides what counts as
    # her source anywhere else in this module.
    excludes = [f"--exclude-dir={name}" for name in sorted(_SKIP_DIRS)]
    roots = [
        str(_SOURCE_ROOT / name)
        for name in ("core", "interface", "tools", "training")
        if (_SOURCE_ROOT / name).is_dir()
    ] or [str(_SOURCE_ROOT)]

    # One pass, every candidate line as its own pattern. Running a grep per
    # line meant the ABSENT verdict — the one that matters — cost a full
    # walk for each, ~6s on the foreground lane. `-l` with several `-e`
    # patterns lists files matching any of them, which is the same question
    # asked once.
    # Anchored to the start of a line, so PROSE QUOTING code does not count
    # as the code existing. This module's own comments quote the fabricated
    # `def self_organize_modules(...)` from the live defect; a fixed-string
    # search found it here and pronounced the invention genuine. A snippet
    # is in the tree when a line of it IS a line of a file, not when some
    # file mentions it.
    patterns: list[str] = []
    for line in lines:
        escaped = re.sub(r"([.^$*+?()\[\]{}|\\])", r"\\\1", line)
        patterns.extend(("-e", rf"^[[:space:]]*{escaped}"))
    try:
        found = subprocess.run(
            ["grep", "-rlsE", "--include=*.py", *excludes, *patterns, *roots],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return ("unchecked", "")
    if found.returncode == 0 and found.stdout.strip():
        first = found.stdout.strip().splitlines()[0]
        try:
            return ("found", str(Path(first).resolve().relative_to(_SOURCE_ROOT)))
        except (OSError, ValueError):
            return ("found", first)
    if found.returncode != 1:
        # grep distinguishes "no match" (1) from "something went wrong" (2+),
        # and only the first is evidence.
        return ("unchecked", "")
    return ("absent", "")


def reply_fabricates_own_code(reply: Any) -> bool:
    """True only when a shown snippet is provably not in her source tree."""
    for block in code_blocks_in(reply):
        verdict, _path = snippet_verdict(block)
        if verdict == "absent":
            return True
    return False


# ---------------------------------------------------------------------------
# What she just showed.
#
# Live 2026-08-04, after the excerpt was made real: she showed
# core/mycelium.py:88 and, asked one turn later what module it came from,
# said "I wrote it specifically for this conversation". A TRUE excerpt
# disowned — the mirror image of the original defect and just as wrong,
# because the person cannot tell the two apart. She had no record of what
# she had put on the table.
# ---------------------------------------------------------------------------

_SHOWN_CITATION_RE = re.compile(r"\b([\w./-]+\.py):(\d+)")

_LAST_SHOWN: dict[str, Any] = {}


def remember_shown_excerpt(reply: Any) -> dict[str, Any] | None:
    """Record the source citation a reply just put in front of someone."""
    match = _SHOWN_CITATION_RE.search(str(reply or ""))
    if match is None:
        return None
    relative, line = match.group(1), int(match.group(2))
    if not (_SOURCE_ROOT / relative).is_file():
        return None
    symbol = ""
    tail = str(reply or "")[match.end() : match.end() + 80]
    symbol_match = re.match(r"\s*\(([^)]+)\)", tail)
    if symbol_match:
        symbol = symbol_match.group(1).strip()
    _LAST_SHOWN.clear()
    _LAST_SHOWN.update({"relative_path": relative, "start_line": line, "symbol": symbol})
    return dict(_LAST_SHOWN)


def last_shown_excerpt() -> dict[str, Any]:
    """The source citation most recently shown, or {}."""
    return dict(_LAST_SHOWN)


def forget_shown_excerpt() -> None:
    _LAST_SHOWN.clear()


def provenance_sentence() -> str:
    """Where the code she just showed actually lives, stated plainly.

    Used when a provenance question comes back without naming the file. She
    showed core/mycelium.py:88 and then told Bryan it "isn't from a Python
    module" — reading "module" as "importable package" and answering a
    question nobody asked, while the real path sat on record. A reply that
    does not name the file is not an answer to where the file is.
    """
    shown = last_shown_excerpt()
    if not shown:
        return ""
    where = f"{shown['relative_path']}:{shown['start_line']}"
    symbol = shown.get("symbol")
    subject = f"`{symbol}`" if symbol else "That snippet"
    return (
        f"{subject} is real code from my own source tree — {where} in this "
        "repository, which is where I read it from. It is not a third-party "
        "package, but it is a file you can open."
    )
