# Security Policy — Aura Cognitive Runtime

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.0.x   | ✅ Active  |
| < 1.0   | ❌ Research only, no security patches |

## Reporting a Vulnerability

**Do not open a public issue for security vulnerabilities.**

Email: security@aura-project.dev
PGP Key: See `security/pgp-public-key.asc`

Response SLA:
- Acknowledgement: 48 hours
- Triage: 5 business days
- Fix (critical): 7 calendar days
- Fix (high): 14 calendar days
- Fix (medium/low): next release cycle

## Security Architecture

Aura is a locally-deployed AI cognitive runtime. Its threat model differs from
cloud SaaS — the primary trust boundary is the local machine, but the AI agent
surfaces (tool use, memory, model routing, autonomous action) create unique
attack vectors.

### Trust Boundaries

```
┌─────────────────────────────────────────────────┐
│  Local Machine (Operator Trust Boundary)        │
│  ┌───────────────────────────────────────────┐  │
│  │  Aura Runtime Process                     │  │
│  │  ┌─────────────┐  ┌──────────────────┐   │  │
│  │  │ User Input  │  │ Tool/Skill Exec  │   │  │
│  │  │ (untrusted) │  │ (sandboxed)      │   │  │
│  │  └──────┬──────┘  └────────┬─────────┘   │  │
│  │         │                  │              │  │
│  │  ┌──────▼──────────────────▼──────────┐  │  │
│  │  │  Unified Will / AuthorityGateway   │  │  │
│  │  │  (all consequential actions gated) │  │  │
│  │  └──────┬─────────────────────────────┘  │  │
│  │         │                                │  │
│  │  ┌──────▼──────┐  ┌──────────────────┐  │  │
│  │  │ Memory/State│  │ Model Inference  │  │  │
│  │  │ (encrypted) │  │ (local/cloud)    │  │  │
│  │  └─────────────┘  └──────────────────┘  │  │
│  └───────────────────────────────────────────┘  │
│                                                 │
│  ┌───────────────┐  ┌───────────────────────┐  │
│  │ Filesystem    │  │ Network (optional)    │  │
│  │ (workspace)   │  │ (cloud fallback/API)  │  │
│  └───────────────┘  └───────────────────────┘  │
└─────────────────────────────────────────────────┘
```

### Security Controls Summary

| Control | Status | Implementation |
|---------|--------|----------------|
| Action governance (Will/Authority) | ✅ Enforced | `core/will.py`, `core/governance/will_gate.py` |
| Tool sandboxing | ✅ Enforced | `security/code_sandbox.py`, `security/sandbox.py` |
| Input sanitization | ✅ Enforced | `security/sanitizer.py` |
| Memory encryption at rest | ✅ Available | `core/state/vault.py` |
| Secret scanning (CI) | ✅ CI gate | `.github/workflows/enterprise-gate.yml` |
| Dependency vulnerability scanning | ✅ CI gate | `pip-audit` in CI |
| SBOM generation | ✅ Available | `tools/build_provenance.py` |
| Governance bypass detection | ✅ Enforced | `tools/lint_governance.py` |
| Prompt injection defenses | ✅ Multi-layer | Input sanitizer + integrity checks |
| Cloud fallback privacy controls | ✅ Configurable | `AURA_CLOUD_FALLBACK_POLICY` |

### Secure Defaults

Aura ships with these defaults in production mode (`AURA_MODE=production`):

- All tool/skill execution is sandboxed
- Self-modification is disabled
- Cloud fallback requires explicit opt-in
- Memory writes require Will receipts
- Unsigned/unmanifested skills do not load
- Broad filesystem access is denied by default
- Network access requires explicit skill permission
- Debug/research endpoints are disabled
- Verbose logging is disabled (no secret leakage)

## Dependency Management

- Production dependencies are pinned with hashes in `requirements/core.txt`
- Lock file: `requirements_lock.txt`
- Automated vulnerability scanning via `pip-audit` and OSV
- SBOM generated per release via `make provenance`
- No fallback dependency installation in production mode

## Incident Response

See `docs/runbooks/` for operational incident response procedures.

## Compliance Mappings

| Framework | Document |
|-----------|----------|
| NIST SSDF | `security/NIST_SSDF_MAPPING.md` |
| OWASP ASVS | `security/OWASP_ASVS_MAPPING.md` |
| OWASP LLM Top 10 | `security/OWASP_LLM_MAPPING.md` |
| SLSA | `security/SLSA_PROVENANCE.md` |
| MITRE ATLAS | `security/MITRE_ATLAS_MAPPING.md` |
