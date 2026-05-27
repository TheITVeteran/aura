# Aura Memory Card

## Purpose

This document describes how Aura's memory systems work, how memories influence
future behavior, and how users can control their data.

## Memory Architecture

```
┌──────────────────────────────────────────────────────┐
│                  Memory Hierarchy                     │
│                                                      │
│  ┌────────────┐  ┌────────────┐  ┌───────────────┐  │
│  │  Working   │  │  Episodic  │  │   Semantic    │  │
│  │  Memory    │  │  Memory    │  │   Memory      │  │
│  │ (session)  │  │ (convos)   │  │ (knowledge)   │  │
│  └─────┬──────┘  └─────┬──────┘  └───────┬───────┘  │
│        │               │                 │           │
│        └───────────┬───┴─────────────────┘           │
│                    │                                  │
│           ┌────────▼─────────┐                       │
│           │   ColdStore      │                       │
│           │  (long-term)     │                       │
│           └────────┬─────────┘                       │
│                    │                                  │
│           ┌────────▼─────────┐                       │
│           │ State Snapshots  │                       │
│           │  (backup/audit)  │                       │
│           └──────────────────┘                       │
└──────────────────────────────────────────────────────┘
```

## Memory-Behavior Causality

Memories demonstrably change future behavior through:

1. **Context Assembly**: Relevant memories are retrieved and injected into the
   model context for each turn, directly influencing responses.

2. **Preference Learning**: User preferences stored in memory change response
   style, tool selection, and proactive behavior.

3. **Procedural Memory**: Learned procedures (how to do X for this user) are
   retrieved and followed in similar future situations.

4. **Identity Continuity**: CanonicalSelf state persists across sessions,
   maintaining consistent personality and relationship context.

5. **Error Memory**: Past failures are remembered to avoid repeating them.

## Memory Governance

All memory writes are gated:

```
Candidate Write → Will Decision → Receipt → Storage → Verification
```

- No memory write occurs without a WillReceipt
- Writes are integrity-hashed for tamper detection
- Write provenance (what caused this write) is logged
- Writes can be audited, exported, or deleted

## User Controls

| Action | Command | Effect |
|--------|---------|--------|
| List memories | `make memory-list` | Show all stored memories |
| Search memories | `make memory-search Q="query"` | Find specific memories |
| Export all | `make memory-export` | JSON export of all memory |
| Delete one | `make memory-delete ID=<id>` | Remove specific memory |
| Delete all | `make memory-purge` | Wipe all memories |
| Backup | `make backup` | Full state backup |
| Restore | `make restore BACKUP=<path>` | Restore from backup |
