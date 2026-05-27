# OWASP Application Security Verification Standard (ASVS) v4.0 — Aura Mapping

## Scope

Aura exposes local API/WebSocket surfaces and a desktop GUI. This mapping covers
the ASVS controls relevant to a locally-deployed AI runtime.

## Level 1 — Opportunistic (All Controls)

| ASVS ID | Requirement | Aura Implementation | Status |
|---------|-------------|---------------------|--------|
| **V1.1** | Secure SDLC | CI gates (lint, SAST, dependency audit, governance lint) | ✅ |
| **V1.2** | Architecture documentation | `ARCHITECTURE.md`, `security/threat_model.md` | ✅ |
| **V1.4** | Access control architecture | `core/governance/will_gate.py`, permission matrix | ✅ |
| **V1.5** | Input validation architecture | `security/sanitizer.py`, centralized input path | ✅ |
| **V2.1** | Password security | N/A (local app, no passwords) | N/A |
| **V2.2** | General authenticator security | API key for remote mode; local mode uses OS auth | ✅ |
| **V3.1** | Session management | Local process lifecycle; no distributed sessions | ✅ |
| **V4.1** | General access control | Unified Will gates all consequential actions | ✅ |
| **V4.2** | Operation-level access control | Permission matrix per role (user/operator/admin) | ✅ |
| **V5.1** | Input validation | Sanitizer on all user input; tool output sanitized | ✅ |
| **V5.2** | Sanitization and sandboxing | `security/code_sandbox.py`; HTML/script stripping | ✅ |
| **V5.3** | Output encoding | Structured JSON responses; template escaping in UI | ✅ |
| **V6.1** | Data classification | Privacy classification for cloud fallback | ✅ |
| **V6.2** | Algorithms | Standard library crypto; no custom primitives | ✅ |
| **V7.1** | Log content | Structured logging; no secrets in logs | ✅ |
| **V7.2** | Log processing | Centralized via Python logging; log rotation | ✅ |
| **V8.1** | General data protection | Local-first; memory encryption available | ✅ |
| **V8.3** | Sensitive private data | Vault for sensitive state; env var isolation | ✅ |
| **V9.1** | Communications security | TLS for any network communication | ✅ |
| **V10.1** | Code integrity | SBOM; provenance; signed releases | ✅ |
| **V10.2** | Malicious code search | SAST; dependency audit; secret scanning | ✅ |
| **V10.3** | Deployed application integrity | Checksum verification; runtime manifest | ✅ |
| **V11.1** | Business logic security | Will/Authority governance on all actions | ✅ |
| **V12.1** | File upload | Sandboxed skill input; path validation | ✅ |
| **V13.1** | Generic web service security | Local-only API binding; rate limiting | ✅ |
| **V14.1** | Build | Reproducible builds; `make setup && make quality` | ✅ |
| **V14.2** | Dependency | Pinned requirements; lockfile; vulnerability scanning | ✅ |
| **V14.3** | Unintended security disclosure | No version/debug headers in production mode | ✅ |

## Level 2 — Standard (Additional Controls)

| ASVS ID | Requirement | Aura Implementation | Status |
|---------|-------------|---------------------|--------|
| **V1.7** | Error/logging/auditing architecture | `record_degradation()` + governance audit trail | ✅ |
| **V1.11** | Business logic architecture | Will decision + receipt chain | ✅ |
| **V1.14** | Configuration architecture | `core/config.py` + env var validation at startup | ✅ |
| **V4.3** | Administrative access control | RBAC model with admin/operator/user/research roles | ✅ |
| **V8.2** | Client-side data protection | Desktop app: no sensitive data in UI state | ✅ |

## Level 3 — Advanced (Applicable Controls)

| ASVS ID | Requirement | Aura Implementation | Status |
|---------|-------------|---------------------|--------|
| **V1.6** | Cryptographic architecture | Standard library only; no custom crypto | ✅ |
| **V5.5** | Deserialization prevention | No pickle; JSON-only serialization | ✅ |
| **V11.2** | Anti-automation | Rate limiting; resource governor | ✅ |

## Non-Applicable Sections

| Section | Reason |
|---------|--------|
| V2 (Passwords/MFA) | Local app; no user accounts |
| V3 (Sessions) | Local process; no distributed sessions |
| V15 (HTTP) | Local API only; standard HTTP library |
