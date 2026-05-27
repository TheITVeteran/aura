# OWASP Top 10 for LLM Applications — Aura Mapping

## Framework

This mapping covers the OWASP Top 10 for Large Language Model Applications
(2025 revision), applied to Aura as an agentic LLM system with tools, memory,
model routing, and autonomous action surfaces.

## Mapping

### LLM01: Prompt Injection

**Risk**: Attacker crafts input to override system instructions or inject
malicious instructions via tool output, files, or web content.

**Aura Controls**:
- `security/sanitizer.py`: Strips control tokens, ChatML markers, role prefixes from all user input
- `core/brain/llm/mlx_worker.py`: `_truncate_role_continuation()` strips model role-drift
- `core/brain/llm/mlx_worker.py`: `_sanitize_telemetry_leakage()` rejects hallucinated paths
- `core/brain/inference_gate.py`: Integrity check on all model output before delivery
- Tool output boundary markers separate tool results from instructions
- Will/AuthorityGateway gates all consequential actions regardless of prompt content

**Status**: ✅ Multi-layer defense

---

### LLM02: Insecure Output Handling

**Risk**: LLM output containing code, scripts, or injection payloads is
executed or displayed unsafely.

**Aura Controls**:
- All model output passes through integrity check before user delivery
- Tool execution results are sandboxed and sanitized
- UI renders model output as text, not executable content
- `core/brain/cognitive/integrity_check.py`: Validates output structure

**Status**: ✅ Mitigated

---

### LLM03: Training Data Poisoning

**Risk**: Model fine-tuning or memory poisoning corrupts behavior.

**Aura Controls**:
- Local models are loaded from verified checksum sources
- Memory writes are gated by Unified Will with receipts
- Memory audit trail enables detection of poisoned entries
- `core/governance/will_gate.py`: All memory mutations require authorization

**Status**: ✅ Mitigated

---

### LLM04: Model Denial of Service

**Risk**: Crafted inputs cause excessive resource consumption.

**Aura Controls**:
- `core/brain/inference_gate.py`: Token budgets per request
- `core/resilience/resource_governor.py`: RAM pressure monitoring
- `core/brain/llm/mlx_worker.py`: Metal semaphore serializes GPU access
- Request timeouts enforced via `Deadline` class
- `core/ops/metabolic_monitor.py`: System-wide resource tracking

**Status**: ✅ Mitigated

---

### LLM05: Supply-Chain Vulnerabilities

**Risk**: Compromised dependencies, model weights, or plugins.

**Aura Controls**:
- Pinned dependency hashes in lockfile
- `pip-audit` / OSV scanner in CI
- SBOM generation via `tools/build_provenance.py`
- Model checksum verification on load
- Skill manifest + signature required in production mode

**Status**: ✅ Mitigated

---

### LLM06: Sensitive Information Disclosure

**Risk**: LLM reveals sensitive data through responses.

**Aura Controls**:
- Privacy classification for cloud fallback decisions
- Log sanitization (no secrets in telemetry)
- `core/state/vault.py`: Encrypted storage for sensitive state
- Cloud fallback requires explicit opt-in with privacy level

**Status**: ✅ Mitigated

---

### LLM07: Insecure Plugin Design

**Risk**: Plugins/skills with excessive permissions or no input validation.

**Aura Controls**:
- Skill contract: manifest, permissions, risk level, sandbox policy
- `security/code_sandbox.py`: Sandboxed execution environment
- `security/sandbox.py`: Process-level isolation
- Each skill declares input/output schema, timeout, resource limits
- Unsigned skills do not load in production mode

**Status**: ✅ Mitigated

---

### LLM08: Excessive Agency

**Risk**: LLM takes consequential actions without proper authorization.

**Aura Controls**:
- `core/will.py`: Unified Will — every consequential action requires a WillDecision
- `core/governance/will_gate.py`: Governance gate with audit coverage
- `core/governance/will_receipt_log.py`: Receipt trail for all approved actions
- `tools/lint_governance.py`: CI lint ensures no ungoverned action paths
- Feature flags gate dangerous capabilities (self-modification, etc.)
- Operator permission matrix controls what Aura may do

**Status**: ✅ Primary defense

---

### LLM09: Overreliance

**Risk**: Users trust LLM outputs without verification.

**Aura Controls**:
- Uncertainty indicators in responses (when confidence is low)
- Explicit refusal when competence boundary is reached
- `core/autonomy/genuine_refusal.py`: Honest refusal rather than confabulation
- Action receipts show what Aura did and why

**Status**: ✅ Mitigated

---

### LLM10: Model Theft

**Risk**: Unauthorized access to model weights or fine-tuning data.

**Aura Controls**:
- Local-only model storage (no cloud exposure)
- Filesystem permissions on model directory
- No model-serving API exposed externally by default

**Status**: ✅ Mitigated (local deployment model)

## Summary

| OWASP LLM Risk | Aura Defense Layer | Primary Control |
|----------------|-------------------|-----------------|
| LLM01 Prompt Injection | Multi-layer | Sanitizer + integrity + Will |
| LLM02 Insecure Output | Output pipeline | Integrity check + UI escaping |
| LLM03 Data Poisoning | Memory governance | Will-gated writes + audit trail |
| LLM04 Model DoS | Resource management | Token budgets + governor + timeouts |
| LLM05 Supply Chain | Build pipeline | Lockfile + SBOM + audit |
| LLM06 Info Disclosure | Privacy controls | Classification + vault + log sanitizer |
| LLM07 Insecure Plugins | Skill contract | Manifest + sandbox + signature |
| LLM08 Excessive Agency | Action governance | Unified Will + AuthorityGateway |
| LLM09 Overreliance | Response quality | Genuine refusal + uncertainty |
| LLM10 Model Theft | Local deployment | No external exposure |
