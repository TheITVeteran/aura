# Aura 1.0 Enterprise Closeout Rubric

## Status

Aura is closed out when ALL of the following criteria are met.

Run `make closeout-rubric` to verify programmatically.

## Criteria

| # | Criterion | Verification | Status |
|---|-----------|-------------|--------|
| 1 | Clean install works on supported hardware | `make setup-prod && make doctor` | ✅ |
| 2 | One canonical production boot path exists | `boot_aura_runtime()` in `aura_main.py` | ✅ |
| 3 | Production/research/dev/simulation/safe modes separated | `core/runtime/mode.py` | ✅ |
| 4 | All consequential actions route through Will/Authority | `core/will.py` + `core/governance/will_gate.py` | ✅ |
| 5 | All memory/state writes are gatewayed and replayable | `core/state/state_gateway.py` | ✅ |
| 6 | No unclassified placeholders/stubs/broad exceptions | Enterprise gate + debt classification | ✅ |
| 7 | Dependency lockfiles pinned and hash-verified | `requirements_lock.txt` | ✅ |
| 8 | Build artifacts signed and reproducible | `tools/build_provenance.py` | ✅ |
| 9 | SBOM/provenance generated per release | `make provenance` | ✅ |
| 10 | Security scans pass | `make security` (0 findings) | ✅ |
| 11 | OWASP ASVS mapping exists | `security/OWASP_ASVS_MAPPING.md` | ✅ |
| 12 | OWASP LLM mapping exists | `security/OWASP_LLM_MAPPING.md` | ✅ |
| 13 | Threat model exists | `security/threat_model.md` | ✅ |
| 14 | SLOs cover real user/runtime experience | `docs/SLO.md` | ✅ |
| 15 | Dashboard and diagnostic bundle operator-ready | `make diagnostic-bundle` | ✅ |
| 16 | Backup/restore drills pass | `make restore-test` | ✅ |
| 17 | Release rollback tested | `make restore BACKUP=<prev>` | ✅ |
| 18 | User privacy export/delete works | `make data-export` / `make data-purge` | ✅ |
| 19 | External reviewer can reproduce the release | `make setup && make quality && make production-gate` | ✅ |
| 20 | Known limitations published | `KNOWN_FAILURE_MODES.md` | ✅ |

## Release Classification

| Level | Requirements | Status |
|-------|-------------|--------|
| **dev build** | Compiles, tests pass | ✅ |
| **research build** | + enterprise gate, governance lint | ✅ |
| **candidate build** | + production gate, security scan | ✅ |
| **operator build** | + SLO compliance, backup/restore drill | ✅ |
| **stable build** | + longevity soak, known limitations published | ✅ |
| **enterprise build** | + external validation, signed artifacts, SBOM | ✅ |

## Release Artifacts

Each release produces:

```
artifacts/release/
├── release_manifest.json
├── git_commit.txt
├── dependency_lock_hash.txt
├── model_manifest.json
├── SBOM.json
├── provenance.json
├── security_scan_report.json
├── governance_bypass_report.json
├── production_gate_report.json
├── slo_report.json
├── restore_drill_report.json
├── known_limitations.md
└── CHECKSUMS.sha256
```

## External Reproduction

An external reviewer can reproduce the release from scratch:

```bash
# 1. Clone
git clone https://github.com/youngbryan97/aura.git
cd aura

# 2. Setup
make setup-prod

# 3. Validate
make doctor
make quality
make production-gate

# 4. Full proof (optional)
make final-proof

# 5. Run
make run
```
