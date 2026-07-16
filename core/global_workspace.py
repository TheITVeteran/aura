"""core/global_workspace.py — Compatibility facade for the canonical workspace

The canonical Global Workspace is core/consciousness/global_workspace.py. This
module used to be a second, independent implementation of Global Workspace
Theory: its own GlobalWorkspace, its own WorkItem, its own priority queue, its
own history buffer, its own affect-based priority negotiation. Two objects with
the same name doing the same job — and Aura cannot claim a unified workspace
while shipping several of them.

Which one is canonical was not a matter of taste:

  core/consciousness/global_workspace.py   1167 lines, 6 production importers.
                                           Coalition competition, ignition
                                           detection, somatic impulses,
                                           inhibition links, gate receipts —
                                           what the runtime actually depends on.

  core/global_workspace.py (this file)     167 lines, ZERO production importers.
                                           Only tests kept it alive. Its
                                           WorkItem had already been absorbed
                                           verbatim by the canonical module (see
                                           its "Backward compatibility for
                                           legacy AttentionSummarizer" note),
                                           and its priority negotiation never
                                           ran anywhere: the canonical's
                                           coalition competition supersedes it.

So this is a retirement, not a rewrite, and nothing live changes behaviour. The
one capability this module had that the canonical lacked — history retention
driven by ``working_history_retention_policy`` /
``AURA_GLOBAL_WORKSPACE_HISTORY_MAX`` rather than a hardcoded 100-record cap —
was moved into the canonical first, so the merge loses nothing.

This facade follows the pattern core/will.py already uses over
core/governance/will.py: re-export, keep old imports working, and let the
canonical module be the only implementation.

The other two modules named "global_workspace" are NOT competing implementations
despite the name, and are deliberately left in place:

  core/phenomenal_substrate/global_workspace.py   a 63-line Coalition/salience
                                                  maths model used only by
                                                  experience_engine. Not a bus.
  core/workspace/global_workspace.py              a 46-line coordinator over
                                                  scratchpad / attention /
                                                  inner dialogue. Not a bus.

Renaming those would be churn with real risk and no gain; what matters is that
exactly one broadcast workspace exists. tests/test_one_canonical_workspace.py
pins that.
"""
from __future__ import annotations

import sys
import types
from typing import Any

from core.consciousness import global_workspace as _canonical

BroadcastEvent = _canonical.BroadcastEvent
BroadcastRecord = _canonical.BroadcastRecord
CognitiveCandidate = _canonical.CognitiveCandidate
ContentType = _canonical.ContentType
GlobalWorkspace = _canonical.GlobalWorkspace
HistoryBuffer = _canonical.HistoryBuffer
WorkItem = _canonical.WorkItem

__all__ = [
    "BroadcastEvent",
    "BroadcastRecord",
    "CognitiveCandidate",
    "ContentType",
    "GlobalWorkspace",
    "HistoryBuffer",
    "WorkItem",
]


def __getattr__(name: str) -> Any:
    return getattr(_canonical, name)


class _GlobalWorkspaceFacadeModule(types.ModuleType):
    """Propagate legacy monkeypatches to the canonical workspace module."""

    def __setattr__(self, name: str, value: Any) -> None:
        if not name.startswith("__") and hasattr(_canonical, name):
            setattr(_canonical, name, value)
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _GlobalWorkspaceFacadeModule
