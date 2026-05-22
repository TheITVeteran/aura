# Reproduction Instructions

## Environment
- **Commit SHA:** `6b02595f7fb3cc02f48d7b180ba6adebb4b9a19a`
- **Python Version:** `3.12.13 (main, Mar  3 2026, 12:39:30) [Clang 17.0.0 (clang-1700.6.3.2)]`
- **Platform:** `macOS-26.4.1-arm64-arm-64bit`
- **Run ID:** `876340f9-b1ee-45be-92c9-a18014ee44cb`

## Prerequisites
- Aura source code at the specified commit
- LLM model server running (check port configuration)
- Python environment with all dependencies

## Commands
```bash
cd /path/to/aura-source
git checkout 6b02595f7fb3cc02f48d7b180ba6adebb4b9a19a
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
