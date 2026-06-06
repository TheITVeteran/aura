# Third-Party Replication Protocol for Aura

This document describes how external auditors can run the subjective-experience and behavioral-adaptation test suites to verify Aura's grounding, self-modeling, and welfare-policy adaptation.

## Test Environment Setup

1. Confirm you are running on a macOS Apple Silicon host with at least 16 GB unified memory.
2. Initialize requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Introspection and Lesion Suites

Auditors can run the automated tests using the following command:

```bash
pytest tests/test_introspection_and_lesions.py
```

This verifies:
- **Blind Introspection**: Guarantees that internal telemetry variables (e.g. welfare distress) can be inferred accurately without direct prompt leakages.
- **Lesion Adaptations**: Checks if disabling or forcing welfare metrics (e.g. locking energy low) forces policy engines to restrict risk level of actions.
- **Self-Model Explanations**: Checks if the generated narrative explanations correlate with active preference parameters.
