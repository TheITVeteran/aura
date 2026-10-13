"""How much of each of her replies is about herself, by what was asked.

A count over her stored conversations. A reply's sentences are split on
sentence marks, and a sentence is about her when it has a first-person
singular word in it. A message is about her when it addresses her (a
second-person word or her name). The reading that matters is the share of
her sentences about herself on messages that were about something else.

    usage: how_much_of_a_reply_is_about_her.py [conversations.db] [--since-hours N]
"""
from __future__ import annotations

import re
import sqlite3
import statistics
import sys
import time
from pathlib import Path

_HER = re.compile(r"\b(i|i'm|i've|i'd|i'll|me|my|mine|myself)\b", re.IGNORECASE)
_ADDRESSED = re.compile(r"\b(you|your|yours|yourself|aura)\b", re.IGNORECASE)
_SENTENCE = re.compile(r"[^.!?\n]+[.!?\n]?")


def share_about_her(reply: str) -> float | None:
    sentences = [s for s in _SENTENCE.findall(reply or "") if len(s.split()) >= 3]
    if not sentences:
        return None
    return sum(bool(_HER.search(s)) for s in sentences) / len(sentences)


def read(path: Path, since_hours: float | None = None) -> dict[str, list[float]]:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = con.execute("select role, content, created_at from turns order by created_at").fetchall()
    cutoff = time.time() - since_hours * 3600.0 if since_hours else 0.0
    found: dict[str, list[float]] = {"asked about her": [], "asked about something else": []}
    heard = ""
    for role, content, at in rows:
        if role == "user":
            heard = content or ""
        elif heard and content and at >= cutoff:
            share = share_about_her(content)
            if share is not None:
                found["asked about her" if _ADDRESSED.search(heard) else "asked about something else"].append(share)
            heard = ""
    return found


def main(argv: list[str]) -> int:
    since = None
    args = list(argv)
    if "--since-hours" in args:
        at = args.index("--since-hours")
        since = float(args[at + 1])
        del args[at : at + 2]
    path = Path(args[0]) if args else Path.home() / ".aura" / "data" / "conversations.db"
    for asked, shares in read(path, since).items():
        if not shares:
            print(f"{asked}: no replies")
            continue
        print(
            f"{asked}: {len(shares)} replies, median share of sentences about herself "
            f"{statistics.median(shares):.3f}, mean {statistics.fmean(shares):.3f}, most {max(shares):.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
