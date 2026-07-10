# Live Harness Proof Report
Run ID: `61cc87c8-104b-4276-80f8-bede90e1b044`
Timestamp: `1783656799.442786`
Commit SHA: `f1bc934b397c7e715cf982274e0e3b3ef9642baa`
Source mode: `git_clean`
Source snapshot SHA-256: `not_applicable`
Platform: `macOS-26.4.1-arm64-arm-64bit`
Python: `3.12.13 (main, Mar  3 2026, 12:39:30) [Clang 17.0.0 (clang-1700.6.3.2)]`

## Positive Controls
Verification that real Aura modules boot and execute as expected.

| Control Point | Status | Description |
|---|---|---|
| Source Identity | PASS | clean git commit or hashed isolated source snapshot |
| Will Boot & Decide | PASS | Boot `UnifiedWill` and route decisions |
| Authority Gateway Routing | PASS | Gate action execution through Unified Will |
| Will Receipt Verification | PASS | Trace and verify cryptographic receipt ID |
| Agency Core Goal Lifecycle | PASS | goal inject and matching lifecycle checks |
| Volition Cooldown / Dedup | PASS | deduplicate identical concurrent volition goals |
| Real Skill Execution | PASS | execute a registered skill via CapabilityEngine |

## Negative Controls
Verification that Aura's runtime detects, fails closed, and protects against adversarial inputs.

| Control Point | Status | Description |
|---|---|---|
| Disabled Will Fail-Closed | PASS | Block actions when Unified Will is offline / degraded |
| Forged Receipt Rejection | PASS | Reject manipulated / forged Will decision receipts |
| Missing Effect Proof | PASS | Fail closure verification if effect is not verified |
| Canary Leak Detection | PASS | Identify answer hashes leaked to logs / data dirs |
| Fake Projected Score | PASS | Reject synthetic benchmark report claims without traces |
| Mock Service Detection | PASS | Audit container registrations for test mocks / stubs |

## Summary
Overall Live Harness Proof Status: **PASSED**
    