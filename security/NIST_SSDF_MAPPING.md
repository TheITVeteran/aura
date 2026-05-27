# NIST Secure Software Development Framework (SSDF) v1.1 — Aura Mapping

## Framework Reference

NIST SP 800-218: Secure Software Development Framework (SSDF) Version 1.1

## Practice Group: Prepare the Organization (PO)

| Practice | Task | Aura Implementation | Status |
|----------|------|---------------------|--------|
| PO.1 | Define security requirements | `security/threat_model.md`; OWASP mappings | ✅ |
| PO.2 | Implement roles and responsibilities | `OWNERSHIP.md`; `CONTRIBUTING.md` | ✅ |
| PO.3 | Implement supporting toolchains | CI/CD with lint, SAST, dependency audit | ✅ |
| PO.4 | Define and use criteria for checks | `specs/QUALITY_GATES.md`; enterprise gate baseline | ✅ |
| PO.5 | Implement and maintain environments | `Dockerfile`; `docker-compose.yml`; `Makefile` | ✅ |

## Practice Group: Protect the Software (PS)

| Practice | Task | Aura Implementation | Status |
|----------|------|---------------------|--------|
| PS.1 | Protect all forms of code | `.gitignore`; branch protection; code review | ✅ |
| PS.2 | Verify integrity of software | SBOM; provenance; lockfile hashes | ✅ |
| PS.3 | Archive and protect releases | `tools/build_provenance.py`; signed releases | ✅ |

## Practice Group: Produce Well-Secured Software (PW)

| Practice | Task | Aura Implementation | Status |
|----------|------|---------------------|--------|
| PW.1 | Design secure software | Threat model; trust boundaries; defense in depth | ✅ |
| PW.2 | Review software design | Architecture docs; `docs/REVIEWER_PACKET.md` | ✅ |
| PW.4 | Reuse secure software | Standard library crypto; no custom primitives | ✅ |
| PW.5 | Create source code following secure practices | Input validation; output encoding; error handling | ✅ |
| PW.6 | Configure compilation/build/packaging securely | Pinned deps; no fallback installs in production | ✅ |
| PW.7 | Review/analyze human-readable code | CI lint; enterprise gate; governance lint | ✅ |
| PW.8 | Test executable code | `make test`; production readiness gate; proof batteries | ✅ |
| PW.9 | Configure runtime environments securely | Secure defaults; production mode hardening | ✅ |

## Practice Group: Respond to Vulnerabilities (RV)

| Practice | Task | Aura Implementation | Status |
|----------|------|---------------------|--------|
| RV.1 | Identify and confirm vulnerabilities | `pip-audit`; OSV scanner; enterprise gate | ✅ |
| RV.2 | Assess, prioritize, and remediate | Risk register; `docs/AURA_RISK_REGISTER.md` | ✅ |
| RV.3 | Analyze root causes | Incident manager; `core/resilience/incident_manager.py` | ✅ |

## Summary

All 19 SSDF practices are addressed. Primary gaps are in continuous
penetration testing (external validation) and formal code review process
documentation.
