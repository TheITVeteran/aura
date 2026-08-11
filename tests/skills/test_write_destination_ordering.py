"""The path being written TO is not always the first path in the sentence.

LIVE, 2026-08-10. "Count how many .py files are in
/Users/bryan/.aura/live-source/core/introspection, then write that number and
the file names into ~/Documents/aura_probe_count.txt. Tell me the number."

The planner used named_paths[0] and tried to write to the SOURCE directory:

    write_text_file failed: Path is outside Aura's allowed desktop/document
    artifact roots: /Users/bryan/.aura/live-source/core/introspection

The artifact-root guard caught it, which is the only reason a read request did
not become a write into her own source tree. That guard is a backstop, not a
plan — the plan was wrong before it ever reached governance.

The first path in a sentence is whatever the sentence talks about first, which
here is the thing to READ. Two signals separate a destination from a source: a
path introduced by a write verb plus "into"/"to", and a destination for
write_text_file being a file rather than a bare directory.
"""

from __future__ import annotations

import pytest

from core.skills.desktop_task import DesktopTaskSkill

SOURCE = "/Users/bryan/.aura/live-source/core/introspection"
DESTINATION = "~/Documents/aura_probe_count.txt"


def test_the_live_case_puts_the_destination_first() -> None:
    text = (
        f"Count how many .py files are in {SOURCE}, then write that number and "
        f"the file names into {DESTINATION}. Tell me the number."
    )

    ordered = DesktopTaskSkill._ordered_by_write_destination(text, (SOURCE, DESTINATION))

    assert ordered[0] == DESTINATION


def test_order_in_the_sentence_does_not_decide_it() -> None:
    """Same request with the destination named first must be stable."""
    text = f"Write into {DESTINATION} the number of .py files in {SOURCE}."

    ordered = DesktopTaskSkill._ordered_by_write_destination(text, (SOURCE, DESTINATION))

    assert ordered[0] == DESTINATION


def test_a_file_outranks_a_bare_directory_without_a_verb_cue() -> None:
    """Second signal, for phrasings the verb pattern does not catch."""
    text = f"Take {SOURCE} and produce {DESTINATION} from it."

    ordered = DesktopTaskSkill._ordered_by_write_destination(text, (SOURCE, DESTINATION))

    assert ordered[0] == DESTINATION


@pytest.mark.parametrize(
    "paths",
    [(), ("~/Documents/x.txt",)],
)
def test_zero_or_one_path_is_returned_unchanged(paths) -> None:
    ordered = DesktopTaskSkill._ordered_by_write_destination("write hi into ~/Documents/x.txt", paths)

    assert ordered == tuple(paths)


def test_the_planner_consults_the_ordering() -> None:
    """Without the wire the helper is dead code and the plan stays wrong."""
    import inspect

    source = inspect.getsource(DesktopTaskSkill)
    marker = "named_paths = extract_target_paths(text)"
    assert marker in source
    window = source[source.find(marker) : source.find(marker) + 700]
    assert "_ordered_by_write_destination" in window
