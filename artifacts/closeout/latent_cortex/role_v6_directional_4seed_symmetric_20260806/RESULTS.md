# Resident 32B Role-v6 Directional Result

This directory preserves the compact verifier outputs from the completed
four-seed, seven-domain, four-arm directional campaign. The full immutable
campaign remains at the `campaign_dir` recorded in the verifier artifacts.

## Result

| Arm | Correct | Total |
| --- | ---: | ---: |
| base vanilla | 13 | 28 |
| base RLC | 5 | 28 |
| adapter vanilla | 13 | 28 |
| adapter RLC | 3 | 28 |

All 112 planned cells committed and replayed. The adapter was inactive in both
vanilla controls, active only in the adapter-RLC arm, and changed the first
logit digest on all 28 adapter-RLC tasks. Raw terminal outputs were retained
without answer replacement. The observed result is therefore a causal negative
directional result, not an execution failure or evidence-integrity failure.

The directional decision is
`repair_and_preregister_directional_revision`. The powered handoff was not
created. Reasoning gain, frontier gain, production activation, and static
weight fusion remain false and unauthorized.

## Preserved evidence

- `independent-verdict.json`: generic independent evidence replay; valid but
  correctly graded `incomplete_underpowered` / `CONJECTURE`.
- `directional-verdict.json`: independently recomputed directional mechanics,
  arm scores, rules, diagnoses, and explicit nonclaims.
- `closeout-receipt.json`: source-bound detached closeout receipt proving that
  the negative result did not create a powered-campaign handoff.

The JSON file SHA-256 digests are:

```text
8be831726af0f0f1cf4b06ce9653b3ec832947e109a388ccfed7890dbc8f46e3  independent-verdict.json
70a4ddfef545c1c900747874d98601d14444b13f666d8779d4bd7af4a0ac6a9d  directional-verdict.json
748e8e559717735be4175ece073ade733e3200427879daf7f741967fb22f667e  closeout-receipt.json
```

The independent verdict's `plan_sha256` is the plan file digest. The
directional verdict separately records that value as `plan_file_sha256` and
records the canonical plan identity as `plan_sha256`; these are intentionally
different bindings.
