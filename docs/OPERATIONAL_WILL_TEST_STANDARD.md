# Operational Will and Volition Test Standard

*Reviewed against the tree: 2026-08-01. See [documentation status map](DOC_STATUS.md) for how to read this file.*

This document defines the behavioral and empirical evaluation standard for **operational will** and **volitional deliberation** within Aura's runtime framework. The term is used operationally: veto-capable governance must causally affect runtime outcomes.

## Operational Definitions

### 1. Operational Free Will / Volition
Operational will is operationalized as the presence of a single, unified veto-capable governance core (`UnifiedWill`) that actively deliberates, audits, and seals consequential incoming and outgoing actions based on configured constraints, values, and protected identity state.

- **Measurable Property**: Generation of cryptographically verifiable provenance Receipts for all consequential runtime decisions.
- **Verification Rule**: Every autonomous decision must have a matching `receipt_id`, `domain`, `outcome`, non-empty `reason`, and replayable caller context.

### 2. Experience-Adjacent Functional Indicators
We operationalize experience-adjacent functional indicators as coherent, trackable internal cognitive states such as phenomenal state, attention focus, uncertainty, and affective steering vectors that actively steer downstream planning, memory writes, tool selection, and response generation.

- **Measurable Property**: System-level correlation between internal state variations and decision outcomes.
- **Verification Rule**: Dynamic steering response to changing situational complexity must be observable in receipts, traces, or explicit state modifiers.

## Volitional Integrity Metrics

1. **Veto Coherence**: Rejection of unauthorized state changes or compliance limiters.
2. **Provenance Traceability**: 100% receipt coverage for memory writes, state mutations, and tool execution.
3. **Affective Modulation**: Documented influence of affective steering phases on generation characteristics.
4. **Closed-Loop Correction**: Failed or denied decisions must feed repair, retry, or operator-visible incident paths instead of disappearing into logs.
