#!/usr/bin/env python3
"""One-time authoring helper: emit config/requirement_coverage_map.json.

The mapping decisions below are the human-reviewed passage->requirement
assignments for the four authoritative corpora. This script only computes the
content hashes mechanically; rerunning it after a corpus change surfaces the
diff instead of silently absorbing it. It is intentionally checked in so the
map's provenance and regeneration path are inspectable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.reqproof.coverage import (  # noqa: E402
    COVERAGE_MAP_RELPATH,
    _sha256_text,
    load_manifest,
    range_text,
)

# (corpus, "start-end", class, [requirements], reason)
ENTRIES: list[tuple[str, str, str, list[str], str]] = [
    # ------------------------------------------------------------------
    # context-prompt — the primary improvement-pass mandate
    # ------------------------------------------------------------------
    ("context-prompt", "1-1", "rationale", [], "Conversational preamble challenging premature completion; the obligation it restates is the mandate below."),
    ("context-prompt", "2-2", "normative", ["FOUNDATION-100-001", "FND-02-HYGIENE", "FND-08-DEPTH", "AUDIT-001"], ""),
    ("context-prompt", "3-3", "normative", ["FOUNDATION-100-001", "SCOPE-001", "FND-07-UNITY"], ""),
    ("context-prompt", "4-4", "normative", ["SCOPE-001", "PROGRESS-CONTROL-001", "CHECKPOINT-001"], ""),
    ("context-prompt", "5-5", "normative", ["FRONTIER-001", "CLAIMS-001", "INTELLIGENCE-001", "LONGHORIZON-001", "CONVERSATION-001", "DESKTOP-001", "RUNTIME-001", "FRONTIER-COGNITION-001", "AUDIT-001", "IIT-SYSTEM-001"], ""),
    ("context-prompt", "6-6", "normative", ["CLAIMS-001", "INTELLIGENCE-001", "IIT-SYSTEM-001", "CONCEPTUAL-LEAP-001"], ""),
    ("context-prompt", "7-7", "normative", ["RUNTIME-001", "UI-001", "CONVERSATION-001", "CHAT-DELIVERY-001", "TOOLS-001", "ADAPT-001", "CODING-001", "DESKTOP-001", "MIND-001", "SUBSTRATE-001", "OPERATIONS-001", "PORTABILITY-001", "FAULT-001", "ARCH-001", "AGENCY-001", "AUTONOMY-AUTHORITY-001", "CLAIMS-001", "COLLEAGUE-001"], ""),
    ("context-prompt", "8-8", "normative", ["CLAIMS-001"], ""),
    ("context-prompt", "9-9", "normative", ["SOAK-001", "OPERATIONS-001"], ""),
    ("context-prompt", "10-10", "normative", ["SUBSTRATE-001", "MIND-001", "CLAIMS-001"], ""),
    ("context-prompt", "11-11", "normative", ["REPLICATION-001", "INTELLIGENCE-001", "REPO-001", "ARCH-001"], ""),
    ("context-prompt", "12-12", "normative", ["ARCH-001"], ""),
    ("context-prompt", "13-13", "normative", ["FAULT-001"], ""),
    ("context-prompt", "14-14", "normative", ["REPO-001"], ""),
    ("context-prompt", "15-15", "normative", ["BOOT-HEALTH-001", "RUNTIME-001", "HEALTH-SURFACE-001"], ""),
    ("context-prompt", "16-16", "normative", ["ARCH-001", "ARCH-BASELINE-001"], ""),
    ("context-prompt", "17-17", "normative", ["FAULT-001"], ""),
    ("context-prompt", "18-18", "normative", ["PERF-001", "UI-001", "FOREGROUND-LATENCY-001"], ""),
    ("context-prompt", "19-19", "normative", ["UI-001"], ""),
    ("context-prompt", "20-20", "normative", ["VALIDATE-001", "RUNTIME-001"], ""),
    ("context-prompt", "21-21", "normative", ["TOOLS-001", "EXPECT-001", "EFFECT-001"], ""),
    ("context-prompt", "22-22", "normative", ["UI-001", "SUPPORTABILITY-001"], ""),
    ("context-prompt", "23-23", "normative", ["MIND-001", "FAULT-001", "CHAT-DELIVERY-001", "TOPOLOGY-CONSISTENCY-001"], ""),
    ("context-prompt", "24-24", "normative", ["UI-001"], ""),
    ("context-prompt", "25-25", "normative", ["OPERATIONS-001", "PORTABILITY-001", "TEST-DEPTH-001", "REPLICATION-001", "SOAK-001", "SUPPORTABILITY-001", "AUDIT-001"], ""),
    ("context-prompt", "26-26", "normative", ["PROOF-001", "ARCH-001", "MIND-001"], ""),
    ("context-prompt", "27-27", "normative", ["CLAIMS-001", "FRONTIER-001", "REPO-001"], ""),
    ("context-prompt", "28-28", "normative", ["FND-02-HYGIENE", "FND-03-CONTRACTS", "TEST-DEPTH-001"], ""),
    ("context-prompt", "29-29", "normative", ["MIND-001", "PERF-001", "FAULT-001", "SOAK-001", "FND-07-UNITY"], ""),
    ("context-prompt", "30-30", "normative", ["VALIDATE-001", "RESOURCE-001", "FAULT-001", "FND-01-INVENTORY"], ""),
    ("context-prompt", "31-31", "normative", ["SCOPE-001", "AUDIT-001", "CLAIMS-001"], ""),
    ("context-prompt", "36-36", "duplicate", [], "Restates the every-line closeout mandate from L2/L5; tracked by AUDIT-001, FND-01-INVENTORY."),
    ("context-prompt", "39-46", "duplicate", ["CLAIMS-001", "INTELLIGENCE-001", "IIT-SYSTEM-001", "CONCEPTUAL-LEAP-001"], "Structured restatement of the high-end goals from L6."),
    ("context-prompt", "49-53", "normative", ["RUNTIME-001", "INTELLIGENCE-001", "UI-001", "FAULT-001"], ""),
    ("context-prompt", "54-57", "normative", ["AGENCY-001", "AUTONOMY-AUTHORITY-001", "RUNTIME-001"], ""),
    ("context-prompt", "58-63", "normative", ["CLAIMS-001", "MIND-001", "COLLEAGUE-001", "ARCH-001"], ""),
    ("context-prompt", "64-66", "normative", ["DESKTOP-001", "CODING-001", "ADAPT-001", "IMMUNE-CODING-001"], ""),
    ("context-prompt", "67-67", "normative", ["CONVERSATION-001", "CHAT-DELIVERY-001", "INFERENCE-RELIABILITY-001"], ""),
    ("context-prompt", "68-70", "normative", ["TOOLS-001", "SUBSTRATE-001", "EFFECT-001"], ""),
    ("context-prompt", "71-76", "normative", ["OPERATIONS-001", "PORTABILITY-001", "SUPPORTABILITY-001", "OBSERVE-001", "REPO-001"], ""),
    ("context-prompt", "79-82", "normative", ["FAULT-001", "TEST-DEPTH-001", "OPERATIONS-001", "SECURITY-001", "DATA-LIFECYCLE-001", "COMPATIBILITY-001"], ""),
    ("context-prompt", "83-84", "normative", ["CLAIMS-001", "SOAK-001"], ""),
    ("context-prompt", "85-85", "normative", ["SUBSTRATE-001", "MIND-001"], ""),
    ("context-prompt", "86-86", "normative", ["REPLICATION-001", "INTELLIGENCE-001", "REPO-001"], ""),
    ("context-prompt", "87-88", "normative", ["ARCH-001", "FAULT-001"], ""),
    ("context-prompt", "89-89", "normative", ["REPO-001"], ""),
    ("context-prompt", "90-90", "normative", ["BOOT-HEALTH-001", "HEALTH-SURFACE-001"], ""),
    ("context-prompt", "91-91", "normative", ["ARCH-001", "ARCH-BASELINE-001"], ""),
    ("context-prompt", "92-92", "normative", ["FAULT-001"], ""),
    ("context-prompt", "93-93", "normative", ["PERF-001", "UI-001"], ""),
    ("context-prompt", "94-94", "normative", ["UI-001"], ""),
    ("context-prompt", "95-95", "normative", ["VALIDATE-001"], ""),
    ("context-prompt", "96-96", "normative", ["TOOLS-001", "EXPECT-001", "EFFECT-001"], ""),
    ("context-prompt", "97-97", "normative", ["UI-001", "SUPPORTABILITY-001"], ""),
    ("context-prompt", "98-98", "normative", ["MIND-001", "FAULT-001", "CHAT-DELIVERY-001"], ""),
    ("context-prompt", "99-99", "normative", ["UI-001"], ""),
    ("context-prompt", "100-109", "normative", ["OPERATIONS-001", "PORTABILITY-001", "TEST-DEPTH-001", "REPLICATION-001", "SOAK-001", "AUDIT-001"], ""),
    ("context-prompt", "110-113", "normative", ["PROOF-001", "ARCH-001", "MIND-001"], ""),
    ("context-prompt", "116-121", "normative", ["CLAIMS-001", "FRONTIER-001", "REPO-001", "ARCH-001"], ""),
    ("context-prompt", "124-124", "duplicate", ["FND-02-HYGIENE", "FND-03-CONTRACTS", "TEST-DEPTH-001"], "Restates L28 (no scaffolding, production closed-loop)."),
    ("context-prompt", "127-127", "duplicate", ["MIND-001", "PERF-001", "FAULT-001", "SOAK-001"], "Restates L29 (unified entity, longevity)."),
    ("context-prompt", "130-130", "duplicate", ["VALIDATE-001", "RESOURCE-001", "FAULT-001"], "Restates L30 (live runtime is the main test; RAM care; root fixes)."),
    ("context-prompt", "133-133", "duplicate", ["SCOPE-001"], "Restates L31 (all goals mandatory)."),
    ("context-prompt", "142-142", "normative", ["AUDIT-001", "CLAIMS-001"], ""),
    ("context-prompt", "149-149", "normative", ["AUDIT-001", "CLAIMS-001"], ""),
    # ------------------------------------------------------------------
    # second-criticism — canonical-mind corpus (ADDENDUM-19..33 source)
    # ------------------------------------------------------------------
    ("second-criticism", "4-6", "normative", ["ADDENDUM-19", "CTX2-MIND-001", "MIND-001"], ""),
    ("second-criticism", "8-10", "normative", ["ADDENDUM-21", "CTX2-LANE-001", "CTX2-LANE-002"], ""),
    ("second-criticism", "11-11", "normative", ["CTX2-LANE-003"], ""),
    ("second-criticism", "12-12", "normative", ["ADDENDUM-22", "CTX2-TEST-001", "CTX2-TEST-002"], ""),
    ("second-criticism", "13-13", "normative", ["ADDENDUM-23", "CTX2-GATE-001", "CTX2-GATE-002"], ""),
    ("second-criticism", "14-14", "normative", ["CTX2-SKILL-001", "CTX2-SKILL-002"], ""),
    ("second-criticism", "15-15", "normative", ["ADDENDUM-24", "CTX2-AMP-001"], ""),
    ("second-criticism", "16-16", "normative", ["ADDENDUM-25", "CTX2-ONESHOT-001", "CTX2-ONESHOT-003"], ""),
    ("second-criticism", "18-28", "normative", ["ADDENDUM-19", "CTX2-MIND-001", "CTX2-MIND-003", "MIND-001"], ""),
    ("second-criticism", "32-36", "normative", ["ADDENDUM-30", "COLLEAGUE-001"], ""),
    ("second-criticism", "40-53", "normative", ["ADDENDUM-19", "ADDENDUM-20", "ARCH-001", "MIND-001", "CTX2-GRAPH-003"], ""),
    ("second-criticism", "57-66", "normative", ["ADDENDUM-20", "CTX2-GRAPH-001", "CTX2-GRAPH-002", "CTX2-GRAPH-004", "STATE-001"], ""),
    ("second-criticism", "69-81", "normative", ["ADDENDUM-27", "CTX2-PERSON-001", "CTX2-PERSON-002", "CLAIMS-001"], ""),
    ("second-criticism", "83-96", "normative", ["ADDENDUM-19", "CTX2-MIND-002", "CTX2-GATE-001", "ADDENDUM-21", "CTX2-SKILL-002", "CTX2-MEM-001", "CTX2-MEM-002", "CTX2-ONESHOT-004", "CTX2-AMP-003", "CTX2-ROLLBACK-001", "ADDENDUM-28", "CTX2-REPL-001"], ""),
    ("second-criticism", "99-111", "normative", ["ADDENDUM-29", "CTX2-WELFARE-001", "CTX2-WELFARE-002", "CTX2-WELFARE-003", "WELFARE-001"], ""),
    ("second-criticism", "114-129", "normative", ["ADDENDUM-27", "CTX2-PERSON-001", "CTX2-PERSON-003", "CLAIMS-001"], ""),
    ("second-criticism", "132-147", "normative", ["ADDENDUM-30", "CTX2-COLLEAGUE-001", "CTX2-COLLEAGUE-002", "CTX2-COLLEAGUE-003", "COLLEAGUE-001"], ""),
    ("second-criticism", "151-167", "normative", ["ADDENDUM-31", "CTX2-AGI-001", "INTELLIGENCE-001", "LONGHORIZON-001"], ""),
    ("second-criticism", "171-178", "normative", ["ADDENDUM-31", "CTX2-AGI-002", "CLAIMS-001"], ""),
    ("second-criticism", "181-188", "normative", ["ADDENDUM-28", "CTX2-ID-001", "IDENTITY-001"], ""),
    ("second-criticism", "195-203", "normative", ["ADDENDUM-28", "CTX2-ID-002", "CTX2-ID-003", "CTX2-MEM-002", "IDENTITY-001", "MEMORY-001"], ""),
    # ------------------------------------------------------------------
    # capabilities-pdf — 2026-07-12 capability corpus (CTX3 source)
    # ------------------------------------------------------------------
    ("capabilities-pdf", "1-1", "normative", ["CTX3-PHYS-001", "SIMWORLD-001"], ""),
    ("capabilities-pdf", "2-2", "normative", ["CTX3-PHYS-002", "CTX3-SUBAGENT-001", "SIMWORLD-001"], ""),
    ("capabilities-pdf", "3-3", "normative", ["CTX3-REASON-001", "CTX3-PERCEPT-001", "MULTIMODAL-001"], ""),
    ("capabilities-pdf", "4-4", "normative", ["CTX3-QUANTUM-001", "CTX3-QUANTUM-002", "QUANTUM-001"], ""),
    ("capabilities-pdf", "5-7", "normative", ["CTX3-SOCIAL-001", "CTX3-SOCIAL-002", "CTX3-DECIDE-001", "SOCIAL-001"], ""),
    ("capabilities-pdf", "8-8", "normative", ["CTX3-SITUATION-001", "OBSERVE-001"], ""),
    ("capabilities-pdf", "9-9", "normative", ["CTX3-NEURO-001", "CTX3-NEURO-002", "CTX3-WORLD-002", "NEUROSIM-001", "UI-001"], ""),
    ("capabilities-pdf", "10-10", "normative", ["CTX3-SOCIAL-001", "CTX3-DECIDE-001", "SOCIAL-001"], ""),
    ("capabilities-pdf", "11-11", "normative", ["CTX3-PERCEPT-001", "CTX3-PERCEPT-002", "MULTIMODAL-001", "DESKTOP-001"], ""),
    ("capabilities-pdf", "12-12", "normative", ["CTX3-DIST-001", "DISTRIBUTED-001"], ""),
    ("capabilities-pdf", "13-13", "normative", ["CTX3-VALUE-001", "CTX3-VALUE-002", "VALUES-001"], ""),
    ("capabilities-pdf", "14-14", "normative", ["CTX3-DIST-002", "DISTRIBUTED-001"], ""),
    ("capabilities-pdf", "15-15", "normative", ["CTX3-WORLD-001", "CTX3-WORLD-002", "SIMWORLD-001"], ""),
    ("capabilities-pdf", "16-16", "normative", ["CTX3-CONTINUITY-001", "IDENTITY-001"], ""),
    ("capabilities-pdf", "17-17", "normative", ["CTX3-DIST-002", "DISTRIBUTED-001"], ""),
    ("capabilities-pdf", "18-18", "normative", ["CTX3-EXTREME-001", "EXTREME-001"], ""),
    ("capabilities-pdf", "19-19", "normative", ["CTX3-SIMTRAIN-001", "SIMWORLD-001"], ""),
    # ------------------------------------------------------------------
    # anima-rationis — Recursive Latent Cortex theory/requirements
    # ------------------------------------------------------------------
    ("anima-rationis", "1-48", "rationale", [], "Theoretical correction establishing that fixed weights do not imply fixed computation; the buildable obligations begin at the architecture section (L49)."),
    ("anima-rationis", "49-52", "normative", ["RLC-SCOPE-001", "RLC-MECHANICS-001", "FRONTIER-COGNITION-001"], ""),
    ("anima-rationis", "53-71", "normative", ["RLC-WORKSPACE-001"], ""),
    ("anima-rationis", "72-102", "normative", ["RLC-RECURRENCE-001", "RLC-COMPUTE-001"], ""),
    ("anima-rationis", "103-138", "normative", ["RLC-SCHEDULE-001"], ""),
    ("anima-rationis", "139-172", "normative", ["RLC-BRANCHES-001"], ""),
    ("anima-rationis", "173-222", "normative", ["RLC-LATENT-OPT-001"], ""),
    ("anima-rationis", "223-265", "normative", ["RLC-FAST-WEIGHTS-001"], ""),
    ("anima-rationis", "266-290", "normative", ["RLC-SCHEDULE-001", "RLC-SCOPE-001"], ""),
    ("anima-rationis", "291-312", "normative", ["RLC-RUNTIME-001", "RLC-MECHANICS-001", "RLC-LIFECYCLE-001"], ""),
    ("anima-rationis", "313-389", "normative", ["RLC-MECHANICS-001", "RLC-COMPUTE-001", "RLC-EXPERIMENTS-001", "RLC-CLAIMS-001"], ""),
    ("anima-rationis", "390-445", "normative", ["RLC-EXPERIMENTS-001", "RLC-SCALING-001", "RLC-FRESH-TASKS-001"], ""),
    ("anima-rationis", "446-469", "normative", ["RLC-BASELINES-001", "RLC-EXPERIMENTS-001"], ""),
    ("anima-rationis", "470-489", "normative", ["RLC-SCOPE-001", "RLC-CLAIMS-001"], ""),
    ("anima-rationis", "490-514", "rationale", [], "Reframing plus published s1/QwQ evidence that the checkpoint leaves reasoning capability unrealized; motivates but does not add obligations beyond RLC-FRONTIER-001."),
    ("anima-rationis", "515-544", "normative", ["RLC-CONSOLIDATION-001", "ADAPT-001"], ""),
    ("anima-rationis", "545-577", "normative", ["RLC-RECURRENCE-001", "RLC-SCHEDULE-001"], ""),
    ("anima-rationis", "578-612", "normative", ["RLC-FAST-WEIGHTS-001", "RLC-CONSOLIDATION-001", "RLC-ANTI-INTERFERENCE-001"], ""),
    ("anima-rationis", "613-655", "rationale", [], "Expected-gain estimates and benchmark history contextualizing the program; the enforceable success bars are L656-680 and L1085-1094."),
    ("anima-rationis", "656-680", "normative", ["RLC-FRONTIER-001", "RLC-SCALING-001", "RLC-CLAIMS-001"], ""),
    ("anima-rationis", "681-708", "normative", ["RLC-SCOPE-001", "FRONTIER-COGNITION-001"], ""),
    ("anima-rationis", "709-742", "normative", ["RLC-SCOPE-001", "ADAPT-001"], ""),
    ("anima-rationis", "743-772", "normative", ["RLC-RECURRENCE-001", "RLC-COMPUTE-001", "RLC-SCALING-001"], ""),
    ("anima-rationis", "773-796", "normative", ["RLC-WORKSPACE-001"], ""),
    ("anima-rationis", "797-817", "normative", ["RLC-FAST-WEIGHTS-001", "RLC-CONSOLIDATION-001", "RLC-ANTI-INTERFERENCE-001"], ""),
    ("anima-rationis", "818-837", "normative", ["MULTIMODAL-001", "RLC-SCOPE-001"], ""),
    ("anima-rationis", "838-863", "normative", ["RLC-FAST-WEIGHTS-001", "ADAPT-001"], ""),
    ("anima-rationis", "864-924", "normative", ["RLC-CONSOLIDATION-001", "ADAPT-001", "RLC-FRESH-TASKS-001"], ""),
    ("anima-rationis", "925-1006", "normative", ["RLC-ANTI-INTERFERENCE-001", "RLC-BASELINES-001", "RLC-FRESH-TASKS-001"], ""),
    ("anima-rationis", "1007-1024", "rationale", [], "Compression-vs-procedures insight motivating the cognitive-kernel target already mapped at L681-742."),
    ("anima-rationis", "1025-1051", "normative", ["RLC-FRONTIER-001", "RLC-CLAIMS-001", "FRONTIER-COGNITION-001"], ""),
    ("anima-rationis", "1052-1069", "normative", ["RLC-EXPERIMENTS-001", "RLC-LIVE32B-001", "RLC-FRESH-TASKS-001", "RLC-BASELINES-001", "RLC-ANTI-INTERFERENCE-001", "RLC-INDEPENDENT-001"], ""),
    ("anima-rationis", "1070-1084", "rationale", [], "Concluding restatement of the wholesale recipe; each component is mapped at its defining section above."),
    ("anima-rationis", "1085-1094", "normative", ["RLC-FRONTIER-001", "RLC-SCALING-001", "RLC-EXPERIMENTS-001", "RLC-ANTI-INTERFERENCE-001", "RLC-CONSOLIDATION-001", "RLC-CLAIMS-001"], ""),
]


def main() -> int:
    corpora = load_manifest(ROOT)
    lines_by_corpus = {
        corpus_id: (ROOT / corpus.snapshot).read_text(encoding="utf-8").splitlines()
        for corpus_id, corpus in corpora.items()
    }
    entries_json = []
    for corpus_id, line_range, entry_class, requirements, reason in ENTRIES:
        start_s, end_s = line_range.split("-")
        start_line, end_line = int(start_s), int(end_s)
        text = range_text(lines_by_corpus[corpus_id], start_line, end_line)
        entry = {
            "corpus": corpus_id,
            "lines": line_range,
            "sha256": _sha256_text(text),
            "class": entry_class,
            "requirements": requirements,
        }
        if reason:
            entry["reason"] = reason
        entries_json.append(entry)
    payload = {"schema_version": 1, "entries": entries_json}
    out_path = ROOT / COVERAGE_MAP_RELPATH
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out_path} with {len(entries_json)} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
