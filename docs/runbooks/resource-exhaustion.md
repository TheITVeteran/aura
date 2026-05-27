# Runbook: Resource Exhaustion

## Trigger

- RAM usage exceeds 80% of available
- GPU memory pressure detected
- Model inference timeouts increasing
- Process killed by OOM killer

## Diagnosis

1. Check current resource state:
   ```bash
   python -c "from core.ops.metabolic_monitor import get_metabolic_state; print(get_metabolic_state())"
   ```

2. Check resource governor status:
   ```bash
   grep "resource_governor" logs/aura.log | tail -20
   ```

3. Check token budget utilization:
   ```bash
   grep "token_budget" logs/aura.log | tail -10
   ```

## Response

1. **Reduce load**: Kill background tasks
   ```bash
   export AURA_FOREGROUND_ONLY=1
   ```

2. **Restart with lower resource profile**: Reduce model tier

3. **If OOM killed**: Check `dmesg` for OOM details; restart Aura

## Prevention

- Resource governor monitors RAM/GPU pressure
- Token budgets cap generation length
- Metal semaphore serializes GPU access
- Tier demotion under pressure is automatic
- `HARDWARE_PROFILES.md` documents expected resource usage per tier
