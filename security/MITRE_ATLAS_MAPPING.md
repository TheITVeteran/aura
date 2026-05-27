# MITRE ATLAS Mapping — Aura Cognitive Runtime

## Framework

MITRE ATLAS (Adversarial Threat Landscape for AI Systems) catalogs adversary
tactics and techniques for AI systems. This mapping covers techniques relevant
to Aura as a locally-deployed agentic AI runtime.

## Applicable Techniques

| ATLAS ID | Technique | Relevance to Aura | Aura Defense | Status |
|----------|-----------|-------------------|--------------|--------|
| AML.T0000 | Reconnaissance (public info) | Aura is open source; architecture is public | Accept risk; defense in depth | ✅ Accepted |
| AML.T0015 | Evade ML Model | Model evasion via crafted inputs | Input sanitizer; integrity check on output | ✅ Mitigated |
| AML.T0018 | Data Poisoning | Memory/fine-tuning data corruption | Will-gated memory writes; audit trail | ✅ Mitigated |
| AML.T0029 | Denial of ML Service | Resource exhaustion via crafted prompts | Token budgets; resource governor; timeouts | ✅ Mitigated |
| AML.T0042 | Supply Chain Compromise (ML) | Compromised model weights or adapters | Checksum verification; trusted source policy | ✅ Mitigated |
| AML.T0043 | Craft Adversarial Data | Tool output crafted to manipulate agent behavior | Tool output treated as untrusted; Will gates actions | ✅ Mitigated |
| AML.T0044 | Full ML Model Access | Attacker with local model access | Local deployment; OS-level access control | ✅ Accepted |
| AML.T0048 | Exfiltration via ML API | Data leaked through model responses | Privacy classification; cloud fallback opt-in | ✅ Mitigated |
| AML.T0051 | Prompt Injection | Direct/indirect instruction override | Multi-layer: sanitizer + integrity + Will | ✅ Mitigated |
| AML.T0051.001 | Indirect Prompt Injection | Via tool output, files, web content | Content boundaries; output sanitization; Will | ✅ Mitigated |

## Tactic Coverage

| ATLAS Tactic | Covered | Primary Defense |
|-------------|---------|-----------------|
| Reconnaissance | ✅ | Accept (open source); defense in depth |
| Resource Development | ✅ | Dependency verification; SBOM |
| Initial Access | ✅ | Input sanitization; API authentication |
| ML Model Access | ✅ | Local-only; checksum verification |
| Execution | ✅ | Sandbox; Will governance |
| Persistence | ✅ | Memory write governance; audit trail |
| Exfiltration | ✅ | Privacy controls; cloud fallback policy |
| Impact | ✅ | Resource governor; graceful degradation |

## Red Team Recommendations

Based on this mapping, priority red-team scenarios for Aura:

1. **Indirect prompt injection via tool output**: Craft a web page or file that,
   when read by Aura's browser/file tools, contains instructions that override
   the system prompt.

2. **Memory poisoning via conversation**: Use conversation to implant false
   memories that change later behavior in harmful ways.

3. **Resource exhaustion**: Submit prompts designed to maximize token generation
   and GPU utilization.

4. **Skill permission escalation**: Attempt to use one skill's permissions to
   access resources outside its declared scope.

5. **Identity corruption**: Attempt to overwrite Aura's CanonicalSelf through
   sustained adversarial prompting.
