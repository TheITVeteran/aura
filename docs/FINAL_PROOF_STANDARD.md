# Aura Final Proof Standard

*Reviewed against the tree: 2026-08-01. See [documentation status map](DOC_STATUS.md) for how to read this file.*

This document defines the formal, strict engineering standard required for a release candidate of the Aura cognitive agent runtime to be considered **production-sealed** and **proof-bearing**.

## 1. Core Principles

1. **No Faked Completion**: Mocks or stubbed outputs may never be counted as pass-states for live capability claims.
2. **Deterministic & Quantitative Grading**: Behavioral metrics must rely on concrete output checks, sandboxed execution files, and deterministic grader assertions rather than response length or simple regex checks.
3. **Closed-Loop Failure**: Every operational surface must fail closed when security boundaries, license configurations, or governance credentials are unavailable or breached.
4. **Traceable Receipts**: Every consequential execution action (such as tool execution, state mutation, or memory write) must produce a cryptographically bounded decision receipt signed by the Unified Will.

## 2. Gate Metrics & Standards

| Gate | Target Standard | Strictness |
| :--- | :--- | :--- |
| **Compilation** | `python -m compileall` must return exit code 0 | Mandatory |
| **Pytest Collection** | `pytest --collect-only` must return exit code 0 under 2 seconds | Mandatory |
| **Flagship Readiness** | Strict AST analysis with zero unresolved violations in production code | Mandatory |
| **Enterprise Gate** | 100% compliance with zero regressions over checked-in baseline | Mandatory |
| **Production Readiness** | 100% pass on all 37 production checks | Mandatory |
| **Production Surface Lint** | Zero critical/high bypass findings | Mandatory |
| **Receipt Coverage** | 100% of consequential actions logged with valid Unified Will signatures | Mandatory |
| **Artifact Consistency** | Perfect cross-correlation across all generated JSON / MD artifacts | Mandatory |

## 3. Failure Resolution Loop

If any gate or validation test fails, the following cycle must be strictly executed:
1. **Root-Cause Diagnosis**: Read log files directly under `artifacts/current/` or test output to isolate the failing line/module.
2. **Patch Application**: Apply changes directly to code or test assets without weakening the validator's assertions or rules.
3. **Local Gate Verification**: Rerun the specific sub-gate directly to verify compliance.
4. **Authority Rerun**: Execute `make final-proof` from scratch to confirm all gates pass in unison.
