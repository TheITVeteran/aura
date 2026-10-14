#!/usr/bin/env python3
"""Give her solved examples a person knows the answers to, as examples she met.

Each row of the file is a problem and the answer a person states for it. They
are noted with the source ``a_person_gave_it`` in her store of met examples
(core/learning/examples_she_meets.py). Nothing is induced here: while she is
idle, her autonomy job groups what she met by meaning and writes a procedure
for a group once each half of its hash split has as many problems as a request
shows. A row of a kind she already keeps is checked against that kind, and a
disagreement retires the kind's procedures.

Usage:
    teach_her_a_kind.py FILE    # JSON lines, or a JSON list, of {"problem": ..., "answer": ...}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def rows_of(text: str) -> list[dict]:
    text = text.strip()
    if text.startswith("["):
        return list(json.loads(text))
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", type=Path)
    args = parser.parse_args()

    from core.learning.examples_she_meets import note_met

    rows = rows_of(args.file.expanduser().read_text(encoding="utf-8"))
    noted = sum(note_met(str(r.get("problem", "")), str(r.get("answer", "")), "a_person_gave_it") for r in rows)
    print(json.dumps({"rows": len(rows), "noted": noted}))
    return 0 if noted == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
