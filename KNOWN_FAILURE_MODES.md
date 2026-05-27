# Known Failure Modes — Aura Cognitive Runtime

## Purpose

This document catalogs known failure modes, their likelihood, impact, detection
method, and recovery procedure. Every operator should read this before running
Aura in any production-like setting.

## Critical Failure Modes

### F01: Model fails to load

**Cause**: Insufficient RAM, corrupted weights, missing model files
**Likelihood**: Low (first boot) / Very Low (steady state)
**Impact**: No inference capability
**Detection**: Boot probe failure; health check reports `brainstem: not_initialized`
**Recovery**: `make doctor` → validates model files → re-download if needed
**Runbook**: `docs/runbooks/model_failure.md`

### F02: Worker process crash during inference

**Cause**: GPU memory pressure, MLX runtime error, corrupted prompt
**Likelihood**: Low
**Impact**: Current request fails; auto-recovery spawns new worker
**Detection**: Worker health probe; `record_degradation("mlx_worker", ...)`
**Recovery**: Automatic — InferenceGate spawns new worker within 30s
**Runbook**: `docs/runbooks/worker_crash.md`

### F03: Memory database corruption

**Cause**: Dirty shutdown, disk full, concurrent write race
**Likelihood**: Very Low
**Impact**: Memory retrieval fails; boot may degrade
**Detection**: SQLite integrity check on boot; state hash mismatch
**Recovery**: `make restore` from last backup; WAL replay
**Runbook**: `docs/runbooks/memory_corruption.md`

### F04: Shutdown hangs

**Cause**: Blocked async task, hung worker, deadlocked service
**Likelihood**: Low
**Impact**: Process requires SIGKILL
**Detection**: Shutdown timeout (12s budget); watchdog
**Recovery**: SIGKILL + clean boot; bounded shutdown prevents forever-hang
**Runbook**: `docs/runbooks/shutdown_hang.md`

## High Severity Failure Modes

### F05: Cloud fallback privacy leak

**Cause**: Misconfigured privacy classification; prompt classified as public when sensitive
**Likelihood**: Very Low (defense in depth)
**Impact**: Sensitive data sent to cloud provider
**Detection**: Cloud fallback audit log; privacy classification review
**Recovery**: Disable cloud fallback; audit sent prompts; notify user
**Runbook**: `docs/runbooks/cloud_provider.md`

### F06: Prompt injection succeeds

**Cause**: Novel injection technique bypasses sanitizer + integrity check
**Likelihood**: Low (multi-layer defense)
**Impact**: Aura performs unintended action
**Detection**: Will receipt audit; anomalous action patterns
**Recovery**: Revert affected memory writes; review Will receipt chain
**Runbook**: `docs/runbooks/prompt_injection.md`

### F07: Resource exhaustion (RAM/GPU)

**Cause**: Large context, multiple concurrent requests, memory leak
**Likelihood**: Medium (under load)
**Impact**: Degraded performance; potential OOM kill
**Detection**: Metabolic monitor; resource governor alerts
**Recovery**: Automatic tier demotion; request throttling; restart if needed
**Runbook**: `docs/runbooks/resource_exhaustion.md`

### F08: Background task orphaning

**Cause**: Task creator dies without cleaning up background work
**Likelihood**: Low
**Impact**: Wasted resources; potential stale state
**Detection**: Task tracker orphan detection; hypervisor reaping
**Recovery**: Hypervisor kills orphaned tasks; cleanup on next boot
**Runbook**: `docs/runbooks/orphaned_tasks.md`

## Medium Severity Failure Modes

### F09: Stale memory retrieval

**Cause**: Vector DB index drift; outdated embeddings
**Likelihood**: Medium (over time)
**Impact**: Irrelevant context in responses
**Detection**: Memory retrieval quality metrics; user feedback
**Recovery**: Re-index memory; consolidation cycle

### F10: Identity drift

**Cause**: Sustained adversarial prompting; corrupted CanonicalSelf state
**Likelihood**: Very Low
**Impact**: Aura's personality/identity becomes inconsistent
**Detection**: Identity coherence check; CanonicalSelf hash
**Recovery**: Reset CanonicalSelf from canonical snapshot

### F11: Tool execution timeout

**Cause**: Slow external service; large file operation; network timeout
**Likelihood**: Medium
**Impact**: Individual tool call fails
**Detection**: Timeout enforcement; degradation recording
**Recovery**: Automatic — tool reports failure; Aura retries or explains

### F12: Lock contention/deadlock

**Cause**: Multiple subsystems contending for same resource
**Likelihood**: Low
**Impact**: Request stalls until watchdog releases
**Detection**: Lock watchdog; stall detection
**Recovery**: Automatic — watchdog releases stale locks after threshold

## Low Severity Failure Modes

### F13: Log rotation failure

**Cause**: Disk full; permission error
**Likelihood**: Very Low
**Impact**: Logs stop writing; no data loss
**Detection**: Log write error; disk space monitor
**Recovery**: Free disk space; restart log rotation

### F14: Telemetry emission failure

**Cause**: Metrics endpoint unavailable
**Likelihood**: Low (local deployment)
**Impact**: Missing observability data
**Detection**: Telemetry health check
**Recovery**: Restart telemetry; data gap in dashboard

## Recovery Drill Schedule

| Drill | Frequency | Procedure |
|-------|-----------|-----------|
| Backup/restore | Monthly | `make backup && make restore-test` |
| Dirty shutdown recovery | Quarterly | Kill -9 → verify boot |
| Model re-download | Quarterly | Delete model → verify re-acquisition |
| State corruption recovery | Quarterly | Corrupt test DB → verify recovery |
| Full disaster recovery | Annually | Fresh machine → full install → restore |
