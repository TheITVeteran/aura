"""Kept procedures as steps: inside the sandbox, a program asks a kept kind for its answer to a sub-problem.

A kept procedure answers a whole request of its kind
(core/learning/procedures_from_solved_examples.py). Larger work is made of
such pieces: a word problem whose middle is a unit conversion, a schedule
whose check is date arithmetic, a new kind of problem whose solution is two
kinds she already keeps, one after the other. So a program running in her
sandbox, whether a procedure her model is writing or a program her reasoning
organ is running (core/brain/executable_reasoning.py), may call
``kept_answer(kind, problem)`` and get the answer her kept procedures for that
kind give, exactly as they would give it to a request.

This module runs inside the sandbox, so it imports only the standard library.
The runtime puts its directory and her book on the child's read path and sets
``BOOK`` before the program runs (``prelude``). Each procedure of the kind runs
in its own namespace; when a kind keeps more than one, they must agree. A
kind that asks for its own answer, directly or through others, is refused
rather than left to recurse until the sandbox's clock runs out.
"""

from __future__ import annotations

import json

#: The book's path; the runtime sets it in the child before the program runs.
BOOK = ""
_LOADED: dict = {}
_ASKING: list = []


def _book() -> dict:
    if not _LOADED:
        with open(BOOK, encoding="utf-8") as handle:
            _LOADED.update(json.load(handle))
    return _LOADED


def kinds() -> list:
    """The kinds that have kept procedures, by name."""
    return sorted(_book().get("procedures") or {})


def kept_answer(kind: str, problem: str) -> str:
    """The answer her kept procedures for ``kind`` give to ``problem``; raises when there is none or they disagree."""
    codes = (_book().get("procedures") or {}).get(kind)
    if not codes:
        raise KeyError(f"no kept procedure for the kind {kind!r}")
    if kind in _ASKING:
        raise RecursionError(f"the kind {kind!r} asks for its own answer through {_ASKING!r}")
    answers = []
    _ASKING.append(kind)
    try:
        for code in codes:
            space: dict = {"__name__": f"kept_{kind}", "kept_answer": kept_answer}
            exec(compile(code, f"<kept procedure {kind}>", "exec"), space)  # noqa: S102 - inside the OS sandbox
            answers.append(str(space["solve"](problem)))
    finally:
        _ASKING.pop()
    if len({" ".join(a.split()).casefold() for a in answers}) != 1:
        raise ValueError(f"the kept procedures for {kind!r} disagree")
    return answers[0]


def prelude(directory: str, book: str) -> str:
    """The lines the runtime puts before a program so that ``kept_answer`` is there to call."""
    return (f"import sys as _aura_sys\n_aura_sys.path.insert(0, {directory!r})\n"
            f"import kept_steps as _aura_kept\n_aura_kept.BOOK = {book!r}\n"
            "from kept_steps import kept_answer\n")


def reference(book: dict, *, example_chars: int = 240) -> str:
    """What a program may call, as her model reads it: each kept kind with one of its problems and answer, shortened."""
    lines = ["kept_answer(kind, problem) returns what the kept procedures for that kind answer. Kinds:"]
    for kind in sorted(book.get("procedures") or {}):
        shown = (book.get("examples") or {}).get(kind) or []
        if shown:
            problem = " ".join(str(shown[0].get("problem", "")).split())
            problem = problem if len(problem) <= example_chars else problem[:example_chars] + " ..."
            lines.append(f"- {kind!r}: e.g. {problem!r} -> {str(shown[0].get('answer', ''))[:80]!r}")
        else:
            lines.append(f"- {kind!r}")
    return "\n".join(lines)
