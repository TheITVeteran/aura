# Aura AI System Card

## System Overview

| Field | Value |
|-------|-------|
| **System Name** | Aura Cognitive Runtime |
| **Version** | 1.0 |
| **Developer** | Bryan Young |
| **System Type** | Autonomous AI cognitive agent |
| **Deployment** | Local (on-device), optional cloud fallback |
| **Primary Use** | Personal AI assistant with persistent memory, tool use, and autonomous capability |
| **AI RMF Profile** | NIST AI RMF 1.0 — General Purpose AI System |

## Intended Use

Aura is designed as a personal cognitive AI assistant that runs locally on the
user's hardware. It provides:

- Conversational AI with persistent memory and identity
- Tool execution (filesystem, shell, browser, research)
- Autonomous background behavior (maintenance, learning, self-repair)
- Multi-model inference (local MLX + optional cloud fallback)

### Intended Users
- Individual users seeking a persistent, private AI assistant
- Developers/researchers exploring cognitive AI architecture
- Operators managing local AI infrastructure

### Out-of-Scope Uses
- Critical decision-making without human oversight
- Medical, legal, or financial advice
- Unsupervised operation in high-stakes environments
- Multi-tenant/shared deployment without additional access controls

## System Architecture

```
User Input → Perception → Cognitive Routing → InferenceGate → Model
                                                     ↓
Memory ← State Commit ← Verification ← AuthorityGateway ← Unified Will
                                                     ↓
                                              Tool Execution (sandboxed)
```

All consequential actions route through the Unified Will (`core/will.py`) and
receive an AuthorityGateway receipt. No silent side paths.

## AI Models

| Model Role | Type | Location | Purpose |
|-----------|------|----------|---------|
| Primary (Cortex) | 32B parameter LLM | Local (MLX) | Main reasoning and conversation |
| Tertiary (Brainstem) | 7B parameter LLM | Local (MLX) | Background/maintenance tasks |
| Cloud Fallback | API-based LLM | Remote (opt-in) | Fallback when local unavailable |

See `MODEL_CARD.md` for detailed model information.

## Risk Assessment (NIST AI RMF aligned)

### Govern

| Control | Implementation |
|---------|----------------|
| AI governance policy | `docs/AUTONOMY_BOUNDARIES.md`, `TOOL_USE_POLICY.md` |
| Roles and responsibilities | `OWNERSHIP.md`, permission matrix |
| Risk management process | `docs/AURA_RISK_REGISTER.md`, threat model |

### Map

| Risk Category | Description | Likelihood | Impact |
|--------------|-------------|------------|--------|
| Excessive autonomy | Agent takes unsanctioned actions | Low | High |
| Memory corruption | False memories change behavior | Low | Medium |
| Privacy violation | Sensitive data sent to cloud | Low | High |
| Prompt injection | Adversary overrides instructions | Medium | Medium |
| Resource exhaustion | Model consumes all system resources | Low | Medium |

### Measure

| Metric | Measurement Method |
|--------|-------------------|
| Ungoverned action rate | Will receipt audit (target: 0) |
| Memory write integrity | Receipt-linked writes (target: 100%) |
| Cloud fallback privacy | Classification audit (target: 100% classified) |
| Action receipt coverage | Governance lint (target: 100%) |
| Graceful degradation honesty | `record_degradation()` audit |

### Manage

| Control | Mechanism |
|---------|-----------|
| Human override | Operator can disable any capability via feature flags |
| Kill switch | `AURA_MODE=safe` disables all autonomous behavior |
| Audit trail | Will receipt log + state snapshots |
| Incident response | `docs/runbooks/` |
| Rollback | Backup/restore with state hash verification |

## Limitations

1. **Model limitations**: Local models have capability ceilings; may confabulate
2. **Memory limitations**: Long-term memory is best-effort retrieval, not perfect recall
3. **Tool limitations**: Tool execution is sandboxed and may fail
4. **Autonomy limitations**: Background autonomous behavior is bounded by Will governance
5. **Hardware requirements**: Requires Apple Silicon with ≥32GB RAM for full capability
6. **Network**: Some features require internet; offline mode has reduced capability

## Evaluation

See `docs/evidence/EVALUATION_REPORT.md` for detailed evaluation results including:
- AGI proof battery results
- Agency emergence testing
- Behavioral proof standard results
- Production readiness gate results
- Longevity soak results

## Human Oversight

See `HUMAN_OVERRIDE_POLICY.md` for the complete human oversight framework.

## Contact

For questions about this AI system: security@aura-project.dev
