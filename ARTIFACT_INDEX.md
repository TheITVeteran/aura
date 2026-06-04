# Aura Artifact Index

This document maps all authoritative verification artifacts produced during the validation of the Aura cognitive agent runtime. All artifacts are stored under `artifacts/current/`.

## Core Reports

| Artifact Path | Purpose / Description |
| :--- | :--- |
| [FINAL_CLOSURE_REPORT.md](file://<AURA_REPO_ROOT>/artifacts/current/FINAL_CLOSURE_REPORT.md) | The final validation summary report, containing the truth run records, changes made, and claim statuses. |
| [final_closure_status.json](file://<AURA_REPO_ROOT>/artifacts/current/final_closure_status.json) | Machine-readable status of all validation gates and overall repo compliance. |
| [enterprise_gate.json](file://<AURA_REPO_ROOT>/artifacts/current/enterprise_gate.json) | Output of the enterprise quality ratchet gate scanning syntax, security vulnerabilities, and wildcard imports. |
| [production_readiness.json](file://<AURA_REPO_ROOT>/artifacts/current/production_readiness.json) | Compliance checklist verification records for all production-readiness controls. |
| [architecture_map.json](file://<AURA_REPO_ROOT>/artifacts/current/architecture_map.json) | Operational surface maps detailing memory writes, state mutations, tool execution, and LLM calls. |
| [production_surface_lint.json](file://<AURA_REPO_ROOT>/artifacts/current/production_surface_lint.json) | Static analysis report of production code for bypasses of canonical gateways or unsafe async tasks. |
| [receipt_coverage.json](file://<AURA_REPO_ROOT>/artifacts/current/receipt_coverage.json) | Audit coverage report verifying that all consequential runtime actions produced valid signed decision receipts. |
| [artifact_consistency.json](file://<AURA_REPO_ROOT>/artifacts/current/artifact_consistency.json) | Consistency validation output checking that no contradiction exists between final metrics, claims, and reports. |
| [final_claim_validation.json](file://<AURA_REPO_ROOT>/artifacts/current/final_claim_validation.json) | Machine-readable output validating the claims of the CLAIMS_MATRIX against empirical receipts. |

## Empirical Proof Bundles

| Directory Path | Purpose / Description |
| :--- | :--- |
| [agi_live/](file://<AURA_REPO_ROOT>/artifacts/current/agi_live/) | Sealed AGI DNU task execution records, task traces, and grading metrics. |
| [agency_emergence_boxed_entity/](file://<AURA_REPO_ROOT>/artifacts/current/agency_emergence_boxed_entity/) | Scorecards, baselines, and ablation comparison traces for emerging agency and volition properties. |
| [external_live_validation/](file://<AURA_REPO_ROOT>/artifacts/current/external_live_validation/) | Traces and grader results for external real-world task scenarios. |
| [longevity_soak/](file://<AURA_REPO_ROOT>/artifacts/current/longevity_soak/) | Resource usage logs, event-loop lag diagnostics, and queue stability files over the longevity soak run. |
