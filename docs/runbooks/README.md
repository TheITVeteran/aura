# Aura Runbooks

Each scenario below has a single runbook documenting symptoms, diagnosis,
safe mitigation, unsafe mitigation, rollback, and verification.

| Scenario | Runbook |
| --- | --- |
| Aura will not boot | [aura-will-not-boot.md](aura-will-not-boot.md) |
| Aura stuck before READY | [aura-stuck-before-ready.md](aura-stuck-before-ready.md) |
| Model fails to load | [model-fails-to-load.md](model-fails-to-load.md) |
| Memory corruption detected | [memory-corruption.md](memory-corruption.md) |
| State vault unavailable | [state-vault-unavailable.md](state-vault-unavailable.md) |
| Event bus degraded | [event-bus-degraded.md](event-bus-degraded.md) |
| Actor crash loop | [actor-crash-loop.md](actor-crash-loop.md) |
| Browser actor leaked | [browser-actor-leaked.md](browser-actor-leaked.md) |
| Self-repair failed | [self-repair-failed.md](self-repair-failed.md) |
| Checkpoint restore failed | [checkpoint-restore-failed.md](checkpoint-restore-failed.md) |
| Governance receipt missing | [governance-receipt-missing.md](governance-receipt-missing.md) |
| Tool timeout storm | [tool-timeout-storm.md](tool-timeout-storm.md) |
| High event loop lag | [high-event-loop-lag.md](high-event-loop-lag.md) |
| Disk full | [disk-full.md](disk-full.md) |
| Dirty shutdown recovery | [dirty-shutdown-recovery.md](dirty-shutdown-recovery.md) |
| Camera unavailable | [camera-unavailable.md](camera-unavailable.md) |
| Microphone unavailable | [microphone-unavailable.md](microphone-unavailable.md) |
| Movie mode broken | [movie-mode-broken.md](movie-mode-broken.md) |
| Worker crash | [worker-crash.md](worker-crash.md) |
| Shutdown hangs | [shutdown-hang.md](shutdown-hang.md) |
| Orphaned background tasks | [orphaned-tasks.md](orphaned-tasks.md) |
| Resource exhaustion (RAM/GPU) | [resource-exhaustion.md](resource-exhaustion.md) |
| Prompt injection | [prompt-injection.md](prompt-injection.md) |
| Excessive agency | [excessive-agency.md](excessive-agency.md) |
| Cloud provider failure | [cloud-provider.md](cloud-provider.md) |
| Research core stalled | [research-core-stalled.md](research-core-stalled.md) |
| Disaster recovery | [disaster-recovery.md](disaster-recovery.md) |
| Pass F maturity risks | [pass-f-maturity-risks.md](pass-f-maturity-risks.md) |

Every runbook is written against fields that `aura doctor --bundle` emits, so
produce the bundle first:

```bash
aura doctor --bundle
```

The failure-mode catalogue that these runbooks resolve is
[KNOWN_FAILURE_MODES.md](../../KNOWN_FAILURE_MODES.md) (F01–F19).
