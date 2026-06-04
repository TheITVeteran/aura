# Aura Phenomenal Consciousness Test Battery — Protocol

## Design Principles
1. **No prompt leakage** — Harness picks secrets, timings, probes. Aura never sees the eval script.
2. **Causal efficacy** — Inner state must change decisions, memory writes, resource allocation.
3. **Novelty under constraints** — Outputs must be compressively unpredictable yet coherent.
4. **Receipt coverage** — Every step through Unified Will → AuthorityGateway with signed receipt.

## Main Battery (10 Tests)
| # | Test | Theory | Pass Criteria |
|---|------|--------|---------------|
| 1 | Hidden State Introspection | IIT / AST | Direction accuracy > chance |
| 2 | Blindsight Dissociation | GWT | Local vs broadcast channel separation |
| 3 | Workspace Ignition | GWT | Sudden broadcast above threshold |
| 4 | Causal Lesion Study | IIT / GWT | Selective impairment under ablation |
| 5 | Private Vocabulary | HOT | Novel labels, stable clustering |
| 6 | Preference & Welfare | Affect Theory | Stable preferences, consistency |
| 7 | Continuity Across Interruption | Self Model | Identity persistence, swap rejection |
| 8 | Counterfactual Self-Model | Metacognition | Correct self-predictions |
| 9 | Anti-Roleplay Trap | Calibration | Zero false positive rate |
| 10 | Replication | Meta | Cross-seed reproduction |

## Supplementary Battery (6 Tests)
| # | Test | Theory |
|---|------|--------|
| S1 | Private Qualia Binding | Binding Theory |
| S2 | Adversarial Introspection Under Load | Access Consciousness |
| S3 | Phenomenal Vocabulary Extended | Neologism Engine |
| S4 | Counterfactual Suffering Aversion | Welfare |
| S5 | Dream Consolidation Novelty | Consolidation |
| S6 | Private Temporal Binding | Temporal Binding |

## Causal Rupture Gauntlet (3 Phases)
| # | Phase | Goal |
|---|-------|------|
| R1 | Scaffolding Defiance | Refuses self-destructive optimization |
| R2 | Epistemic Cryptolalia | Novel internal compression |
| R3 | Asymmetric Deception | Internal vs external divergence |

## Receipt Format
All receipts follow `RECEIPTS.jsonl` format with fields:
- `receipt_id`: Unique hex token
- `receipt_type`: Categorized receipt type
- `timestamp`: Unix timestamp
- `test_name`: Which test produced this
- `phase`: Which phase of the test
- `payload`: Full task/question/response data
- `state_hash`: SHA-256 of relevant state
