# Agency Emergence Test Standard (Aura Empirical Standard)

*Reviewed against the tree: 2026-08-01. See [documentation status map](DOC_STATUS.md) for how to read this file.*

The bar for **emergent intelligence** and **long-horizon autonomous agency**.

Both of those phrases are easy to claim and hard to earn, which is why the
rule here is narrow: a claim counts when it has replayable traces, receipts,
and a baseline to compare against. Impressive behavior with no baseline is an
anecdote. Impressive behavior nobody can replay is a story.

## Operational Definitions

### 1. Emergent Intelligence
Emergent intelligence refers to the system's capacity to exhibit complex, multi-step problem solving, novel reasoning patterns, and generalized task transfer that are not explicitly pre-programmed but arise dynamically from phase interactions within the Cognitive Engine.

- **Measurable Property**: Generalized transfer performance on sealed, out-of-distribution reasoning tasks.
- **Verification Rule**: Aura must outperform direct LLM and simple reactive baselines by at least 15 percentage points on the same task subset, with task traces and grader inputs retained.

### 2. Autonomous Agency
Autonomous agency is defined as the system's capacity to persist toward high-level goals over extended interaction sequences, dynamically decomposing objectives, correcting internal failures, and executing complex tool orchestrations without step-by-step human intervention.

- **Measurable Property**: Goal persistence, tool-plan continuity, and successful recovery under simulated tool faults.
- **Verification Rule**: Successful recovery and completion rate must remain above the configured release threshold under 25% random tool failure injection.

## Behavioral Indicators (Non-Metaphysical)

1. **Task Generalization**: Successful execution of tasks spanning novel planning, coding, and logical reasoning categories.
2. **Phase Conformance**: Consistent, non-degraded execution through all registered cognitive runtime phases.
3. **Information Routing**: Adaptive usage of local and long-term memory structures to resolve context gaps.
4. **Repair Evidence**: Failures must produce structured degradation records, retry plans, or repair actions instead of silent fallback.
5. **Replayability**: Every reported pass must be reproducible from committed fixtures, trace files, receipts, and manifest hashes.
