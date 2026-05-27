# Runbook: Cloud Provider Incident

## Trigger

- Cloud fallback sending prompts unexpectedly
- Privacy classification failure
- Cloud provider reports data incident

## Diagnosis

1. Check cloud fallback policy:
   ```bash
   echo $AURA_CLOUD_FALLBACK_POLICY
   ```

2. Review cloud request log:
   ```bash
   grep "cloud_fallback" logs/aura.log | tail -20
   ```

3. Check privacy classifications sent:
   ```bash
   grep "privacy_class" logs/aura.log | tail -20
   ```

## Response

1. **Disable cloud fallback immediately**:
   ```bash
   export AURA_CLOUD_FALLBACK_POLICY=disabled
   ```

2. **Audit sent prompts**: Review what was transmitted

3. **Assess privacy impact**: Determine if sensitive data was exposed

## Prevention

- Cloud fallback is disabled by default
- All prompts are classified before cloud transmission
- `sensitive` and `restricted` prompts never sent to cloud
- Cloud usage logged in Will receipt trail
