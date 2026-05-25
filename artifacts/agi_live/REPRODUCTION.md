# Reproduction Instructions

## Environment
- **Commit SHA:** `c991266ff05ea3a0554b0ee66557e2fc0942f931`
- **Python Version:** `3.12.13 (main, Mar  3 2026, 12:39:30) [Clang 17.0.0 (clang-1700.6.3.2)]`
- **Platform:** `macOS-26.4.1-arm64-arm-64bit`
- **Run ID:** `5269e5ae-f61f-430a-88eb-9b40aabd4181`

## Prerequisites
- Aura source code at the specified commit
- LLM model server running (check port configuration)
- Python environment with all dependencies

## Commands
```bash
cd /path/to/aura-source
git checkout c991266ff05ea3a0554b0ee66557e2fc0942f931
python tools/agi/run_dnu_agi_proof_battery.py
```

## Verification
```bash
python -m pytest tests/agi/live/test_dnu_agi_proof_battery.py -q
```

## Notes
- Task fixtures are sealed under `tests/agi/fixtures/dnu_tasks/`
- Grader salts are in `.grader_salts*.json` files (not task packs)
- Results depend on model server availability and response quality
- Different model versions will produce different results
