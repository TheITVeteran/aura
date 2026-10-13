"""A page gone blank: where the thing on it has shown nothing at all for a while, it has gone, and it is begun again.

A person whose game turns into a plain page, white or black, with no word and no shape on it, waits a moment, clicks
it, and then loads the page again: the thing that was there has broken or gone away. Offline 2026-10-13 a game whose
"submit score" went to a server that was not there turned white, and she played the white page as it happened for
four minutes, eight seconds at a time, clicking its middle in between.

What shows nothing is read off what she reads of the screen while it stands still: no words, no regions of writing,
no shapes that look pressable and no drawn objects. A screen that moves is never blank, and a reading that did not
come back, or was not of a still screen, says nothing either way. Gone is nothing shown in every reading for
``GONE_BLANK_S``; begun again at most ``BEGUN_AGAIN_AT_MOST`` times in one game, for a page that goes blank each time
is not mended by loading it again.

Nothing here knows a game.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["BEGUN_AGAIN_AT_MOST", "GONE_BLANK", "GONE_BLANK_S", "gone_blank", "shows_nothing"]

#: How long the thing on the page may show nothing before it is taken to have gone, in seconds.
GONE_BLANK_S = 15.0
#: Why a run ended where the thing on the page went blank.
GONE_BLANK = "the thing on the page shows nothing any more"
#: How many times one game is begun again for going blank.
BEGUN_AGAIN_AT_MOST = 2


def shows_nothing(observation: Any) -> bool:
    """Whether a reading of a still screen found nothing on it: no words, no writing, no shapes and no objects."""
    if not isinstance(observation, Mapping) or observation.get("ok") is False or observation.get("error"):
        return False
    if "shapes" not in observation or "objects" not in observation:
        return False                                   # not a still screen's reading: what is drawn was not looked for
    return not (str(observation.get("text") or "").strip() or observation.get("layout")
                or observation.get("shapes") or observation.get("objects"))


def gone_blank(keep: dict[str, Any], observation: Any, at: float) -> bool:
    """Whether every reading since one at least ``GONE_BLANK_S`` ago has shown nothing; when that began is kept in
    ``keep`` and forgotten at the first reading that shows anything."""
    if not shows_nothing(observation):
        if isinstance(observation, Mapping) and observation.get("ok") is not False and not observation.get("error"):
            keep.pop("blank_since", None)
        return False
    since = float(keep.setdefault("blank_since", at))
    return at - since >= GONE_BLANK_S
