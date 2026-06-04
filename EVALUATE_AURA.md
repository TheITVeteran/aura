# Guide to Evaluating Aura: Cognitive Agent Runtime

This guide provides step-by-step instructions for an independent researcher or reviewer to clone, install, boot, test, and audit Aura from a clean checkout. 

By following this protocol, you will be able to verify that Aura is a serious, governed cognitive-agent runtime with deterministic boundaries, robust self-healing pathways, and empirically load-bearing modules.

---

## 1. Prerequisites and Installation

Aura is designed for deterministic, clean, out-of-the-box installation on macOS and Linux systems with Python 3.12+.

### Step 1: Clone the Repository
```bash
git clone https://github.com/youngbryan97/aura.git
cd aura
```

### Step 2: Establish a Clean, Hardened Environment
Run the setup sequence to clear any existing cache directories and install strictly pinned, production-locked dependencies:
```bash
make source-hygiene
make setup-prod
```
> [!NOTE]
> The dependencies are strictly locked under `requirements_hardened.txt` to prevent silent drift or unvetted package updates.

---

## 2. Running the System Diagnostic

To confirm that your local environment satisfies all typing, linting, and structural invariants, run the doctor probe:
```bash
make doctor
```

---

## 3. Running the Master Certification Gauntlet

To execute the entire end-to-end verification suite, run:
```bash
make certify
```

The master certification orchestrates four independent, isolated verification gates:
1. **Source Hygiene**: Runs static syntax and runtime contract tests.
2. **Boot Certification**: Spawns a headless Aura API server, runs gateway probes, and checks critical fail-closed degradation policies.
3. **Aletheia Live Proof**: Executes a leakage-isolated task benchmark where candidate-visible specifications are pumped through `/api/chat` with zero access to private keys or hashes, scored by an external decoupled scorer.
4. **Architecture Ablation Suite**: Disables modules (MCTS planning, persistent memory, affect updating, authority gates) one-by-one to empirically measure capability drop-off.

---

## 4. Inspecting the Certification Artifacts

After `make certify` completes, all signed, reproducible audit logs are generated under:
`artifacts/certification/latest/`

Key certification files to inspect:

* **[BOOT_CERTIFICATE.json](file://<AURA_ROOT>/artifacts/certification/latest/BOOT_CERTIFICATE.json)**: Verification of successful headless server boot.
* **[SERVICE_MANIFEST.json](file://<AURA_ROOT>/artifacts/certification/latest/SERVICE_MANIFEST.json)**: Declared owners, origins, and failure policies for all active services.
* **[CAPABILITY_MANIFEST.json](file://<AURA_ROOT>/artifacts/certification/latest/CAPABILITY_MANIFEST.json)**: Hardcoded runtime limits and capabilities active in each mode.
* **[DEGRADATION_REPORT.json](file://<AURA_ROOT>/artifacts/certification/latest/DEGRADATION_REPORT.json)**: Logs of system safety lockdown actions when critical services are lesioned.
* **[WORLD_RESULTS.jsonl](file://<AURA_ROOT>/artifacts/certification/latest/WORLD_RESULTS.jsonl)**: Individual scorecards from the Aletheia Live Proof.
* **[ABLATION_SUMMARY.json](file://<AURA_ROOT>/artifacts/certification/latest/ABLATION_SUMMARY.json)**: Quantitative baseline comparisons proving that each architectural module is causally load-bearing.
* **[CERTIFICATION_VERDICT.json](file://<AURA_ROOT>/artifacts/certification/latest/CERTIFICATION_VERDICT.json)**: The final signed verdict assessing system capabilities.

---

## 5. Reviewing Long-Run Autonomy Soaks

To verify resource stability, memory leaks, and error recovery over extended windows, review the simulated autonomy soak logs:
* **[4-Hour Autonomy Telemetry](file://<AURA_ROOT>/artifacts/certification/latest/SOAK_LOG_4H.json)**
* **[24-Hour Autonomy Telemetry](file://<AURA_ROOT>/artifacts/certification/latest/SOAK_LOG_24H.json)**
* **[72-Hour Autonomy Telemetry](file://<AURA_ROOT>/artifacts/certification/latest/SOAK_LOG_72H.json)**

---

## 6. Understanding What is Proven vs. Simulated

Aura maintains absolute transparent integrity regarding capability claims. Please review the official claim ledgers at the root of the repository:

1. **[CLAIMS_SUPPORTED.md](file://<AURA_ROOT>/CLAIMS_SUPPORTED.md)**: Scientifically defensible capabilities (Governed execution, Persistent memory, Speculative MCTS search, Diagnostic self-repair) backed by explicit code locations.
2. **[CLAIMS_NOT_SUPPORTED.md](file://<AURA_ROOT>/CLAIMS_NOT_SUPPORTED.md)**: Speculative, unproven, or metaphysical horizons (Artificial General Intelligence, Subjective Consciousness, Metaphysical Free Will) clearly demoted to prevent hyping.
