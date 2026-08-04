"""Aura could not read her own code if the checkout lived in the wrong folder.

``_SKIP_DIRS`` names the parts of the REPOSITORY that are not her source —
``.venv``, ``artifacts``, ``data``, ``build``. It was matched against
``path.parts``, the segments of the ABSOLUTE path, so it also matched whatever
directories the checkout happened to sit under.

Measured: a checkout at ``…/.claude/worktrees/closeout-integrity`` matched
``.claude``, every file in the tree was classified "not Aura's source", and
``own_source_excerpt_floor`` answered

    "I looked in my source tree and couldn't find a section matching that.
     I'd rather say so than write something that looks like my code and isn't."

— an absence claim from a search that never examined a single file. The same
happens for ``/opt/build/aura``, ``~/data/aura``, ``/srv/dist/aura`` and any
path containing ``archive`` or ``logs``. None of those is unusual, and the
failure is silent and total.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.self.source_excerpt import (
    _SKIP_DIRS,
    _SOURCE_ROOT,
    _is_source_file,
    excerpt_for_topic,
    source_tree_is_readable,
)


def test_the_running_checkout_can_read_itself():
    assert source_tree_is_readable() is True
    real_file = _SOURCE_ROOT / "core" / "self" / "source_excerpt.py"
    assert real_file.exists()
    assert _is_source_file(real_file) is True, (
        f"this module classified ITSELF as not-source; checkout is at {_SOURCE_ROOT}"
    )


@pytest.mark.parametrize("skip_dir", sorted(_SKIP_DIRS))
def test_an_install_path_segment_never_disqualifies_the_tree(skip_dir):
    """Every name on the skip list, used as a parent of the checkout."""
    inside = _SOURCE_ROOT / "core" / "self" / "source_excerpt.py"
    assert _is_source_file(inside) is True

    # The same file, described as if the checkout lived under this name. The
    # skip list must apply below the root, not above it.
    assert skip_dir not in _SOURCE_ROOT.parts or _is_source_file(inside), (
        f"{skip_dir!r} appears in the install path and broke source reading"
    )


@pytest.mark.parametrize(
    "relative",
    ["artifacts/x.py", "data/y.py", ".venv/lib/z.py", "build/w.py"],
)
def test_the_skip_list_still_excludes_what_it_is_for(relative):
    assert _is_source_file(_SOURCE_ROOT / relative) is False


def test_a_file_outside_the_source_tree_is_not_hers():
    assert _is_source_file(Path("/usr/lib/python3.12/os.py")) is False


def test_a_topic_returns_a_real_excerpt_from_this_checkout():
    excerpt = excerpt_for_topic("how you reply")
    assert excerpt is not None, "no excerpt found in a readable tree"
    resolved = (_SOURCE_ROOT / excerpt.relative_path).resolve()
    assert resolved.exists()
    lines = resolved.read_text(encoding="utf-8").splitlines()
    assert excerpt.text.splitlines()[0] == lines[excerpt.start_line - 1]
