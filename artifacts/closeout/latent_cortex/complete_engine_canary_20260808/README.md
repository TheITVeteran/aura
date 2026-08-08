# Complete-engine RLC canary — 2026-08-08

Status: Historical evidence index

This bounded real-checkpoint canary validates the corrected reconciliation
experiment. It does not test resident-32B capability and does not support a
reasoning-gain, frontier, fusion, activation, release, or `WOW Signal` claim.

## Result

- model: `mlx-community/Qwen2.5-1.5B-Instruct-4bit`
- arms: `vanilla`, `vanilla_equal_compute`, `full_stack`
- cells: `21/21`
- elapsed: `92.5s`
- harness faults: `0`
- evidence manifest: valid
- complete full-stack runtime receipts: measured
- paired vanilla floor: held
- right-to-wrong regressions: `0`
- unpromoted byte divergences: `0`
- score: `0/7` for every arm
- decision: `inconclusive_battery_uninformative_ordinary_decode_scored_zero`

## Evidence custody

The full 3.8 MiB evidence bundle, including all seven full-stack runtime
receipts, is retained outside the Git object database at:

`/Users/bryan/.aura/evidence/rlc/complete-engine-canary-20260808-0535`

Its `SHA256SUMS` file covers the source/decode manifest, task commitment,
journal, status, verdict, and each runtime receipt. SHA-256 of that inventory:

`16e37b7ac5c2f367d3f6f68c0443e03ffef67bf8418dcf59c2e58b5ed061c6f1`

The source/decode manifest independently binds the sweep runner and every
Python implementation file under `core/brain/llm/latent_cortex/`. Grading
reopens every runtime receipt, verifies its canonical digest, and reconstructs
the compact causal summary carried in the journal.

## Interpretation

The canary establishes that the complete engine can execute with its required
controls, retain ordinary decode exactly when promotion is not justified, and
produce independently replayable mechanism evidence. Since ordinary decode
scored zero on this intentionally tiny battery, it is statistically and
scientifically uninformative about capability. The resident-32B campaign must
use fresh tasks, both controls, the complete product arm, and the same evidence
contract.
