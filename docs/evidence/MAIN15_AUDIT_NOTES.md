# Aura main (15) adaptation notes

> **Historical record — 2026-07-09.** A dated snapshot, kept as written for
> provenance. It is not a statement about the system today and is
> deliberately not updated. Current status: [DOC_STATUS.md](../DOC_STATUS.md).

Inspected upload: /mnt/data/aura-main (15).zip
Extracted files: 3947
Python files: 3234
Python lines: 784257
Python syntax errors: 0
Full pytest collection observed: 7085 tests collected (with pywebview diagnostic ignored)
Existing being-focused tests observed: 135 passed
New v3 tests: 6 passed

Why v3 differs from v1/v2:
- main (15) already includes BeingRuntime, AuraNow, WelfareState, SemanticStream,
  FunctionalSoul, SelfReportCalibrator, WelfareTransaction, and action_policy.
- v3 therefore does not introduce a parallel "being" stack.
- v3 binds to those surfaces and adds the missing causal bridge:
  AuraNow -> CausalSelfVector -> FunctionalIAttractor -> ClosedLoopPolicy
  -> optional calibrated steering -> experience/plasticity governance.
