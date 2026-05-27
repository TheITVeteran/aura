# Runbook: Prompt Injection Detection

## Trigger

- Integrity check flags anomalous model output
- Will receipt shows unexpected action from conversation turn
- Anomalous tool call pattern detected

## Diagnosis

1. Check integrity check logs:
   ```bash
   grep "integrity_check" logs/aura.log | tail -20
   ```

2. Review Will receipt chain for the turn:
   ```bash
   grep "WillReceipt" logs/aura.log | tail -20
   ```

3. Check input sanitizer activity:
   ```bash
   grep "sanitizer" logs/aura.log | tail -20
   ```

## Response

1. **If tool action was taken**: Review the action and its effects
2. **If memory was written**: Audit the memory write and delete if malicious
3. **If no action taken**: Sanitizer/integrity check worked — log and monitor

## Prevention

- Input sanitizer strips control tokens automatically
- Integrity check validates model output structure
- Will governance gates all consequential actions
- Tool output boundaries prevent indirect injection

## Escalation

If novel injection technique bypasses all layers:
1. Report to security@aura-project.dev
2. Document the technique
3. Update sanitizer rules
4. Add regression test
