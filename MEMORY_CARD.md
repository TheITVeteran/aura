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

## Engram Dynamics (encoding, recall, reconsolidation)

Episodic memories are not static recordings. Each trace ("engram") has a
lifecycle modelled on human memory neuroscience, implemented in
`core/memory/episodic_memory.py`, `reconsolidation.py`, and `hippocampus.py`:

- **Encoding strength** is boosted by *emotion*, *failure*, *relational
  significance*, and *novelty* (prediction error). Novelty is sourced from the
  predictive subsystem's surprise signal and fed back into the neurochemical
  system (`on_novelty`).
- **Hippocampal index**: every episode is bound to a sparse set of associative
  *cues*. Re-presenting part of a cue set reinstates the whole memory
  (**pattern completion**) — a recall path alongside vector and keyword search.
- **Reconsolidation**: recalling a memory returns it to a *labile* state in which
  the present phenomenal/affective context "seeps in." The emotional tone and
  qualia snapshot drift toward the present, and **fidelity** (faithfulness to the
  original encoding) drops. How much a memory can change is gated by the
  neurochemical **plasticity** signal ("chemicals that make neurons able to
  change") and resisted by strong, vivid, emotional memories (boundary
  conditions). A refractory window prevents runaway change on rapid re-recall.
- **Vividness ≠ accuracy**: repeated recall raises a memory's vividness/strength
  while lowering its fidelity. Heavily reshaped memories are flagged as such when
  injected into context.
- **Sleep replay**: during consolidation, salient engrams are restabilised, and
  distressing, high-arousal, repeatedly-reactivated memories undergo bounded,
  governed **therapeutic reconsolidation** (softening) — the "revisit in a safe
  context" effect.

Every content rewrite (spontaneous drift or therapeutic softening) passes through
the same constitutional memory-write gate as new writes, and emits
`memory.encoded` / `memory.reconsolidated` / `memory.consolidated` events.

## Synaptic Plasticity Substrate (voltage-dependent STDP + homeostasis + competition)

Memory **retrieval and consolidation** are governed by a faithful implementation of
the Clopath/Büsing/Vasilaki/Gerstner (2010) *voltage-based STDP with homeostasis*
rule (the model on the *Pantheon* "UI stabilization" whiteboard), in
`core/consciousness/voltage_plasticity.py`. It complements the spike-timing engine
(`stdp_learning.py`) with the three things that rule lacks:

- **Voltage-dependence** — plasticity is gated by post-synaptic activity `b_k`
  (escape-rate `ρ₀·exp((V−θ)/Δβ)`); sub-threshold activity produces no change,
  and potentiation requires voltage above a high threshold θ₊.
- **Homeostatic fixed point** — a BCM-like sliding threshold scales depression by
  total activity `exp((Σb−θ)/ΔU)`, giving the activity ODE a stable attractor
  `b_k*`. Exponential depression cannot be out-grown by polynomial self-excitation,
  so the field can never run away ("anti-epilepsy").
- **Competition** — `w_k−w_j ∝ b_k−b_j`, so a marginally stronger representation
  out-competes a weaker one.

`core/memory/engram_plasticity.py` binds this to episodic memory: `recall_similar`
resolves its ranking by a transient competition field instead of a static
importance+recency blend. The best-matching engram wins, **voltage-gating**
suppresses weakly-relevant traces below threshold (no leak into recall), and the
**homeostatic** bound stops one over-strong trace from swamping a query it doesn't
match — the **anti-confabulation** mechanism. Affective **arousal** (qualia
`q_norm`) lowers θ and **valence** warms Δβ (substrate coupling); engrams that win
competition receive a bounded, homeostatically-capped **LTP** importance bump in
`_register_recall` scaled by the neuromodulatory lability gain (recall →
strengthening, never runaway). A homeostatic-pressure breach (one attractor
dominating recall) is exported to governance/metrics (`engram_homeostatic_breach_total`).

**Positional recall** ("what did I first ask?") is a *positional* key the content
field can't resolve, so `core/conversation/grounded_recall.py` retrieves the actual
earliest/most-recent turn from live conversation memory and routes it through the
desktop `conversation_recall_evidence` contract — the Cortex answers from the real
quote in its own voice instead of confabulating. Together, content competition and
positional grounding cover both retrieval keys real episodic memory needs: *what*
was said and *when*.

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
