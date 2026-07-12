# Omega Capability Matrix

Mapping of the full capability mandate (2026-07-12) onto Aura's actual
codebase: what exists, what was built in this pass, what each claim is
backed by, and what is honestly out of reach today.

**Evidence classes** (every row carries one — no vibes):

- `TESTED-NOW` — built or extended in this pass; named test file is green.
- `LIVE-PROVEN` — pre-existing subsystem with prior live/test proof
  (see the referenced module and its test suite).
- `PRESENT-UNVERIFIED` — subsystem exists and is wired; this pass did
  not re-verify it end-to-end. Believe the code, not this document.
- `HONEST-LIMIT` — physics/hardware/scale boundary stated plainly.

## The matrix

| # | Mandate item | Status | Where it lives | Evidence |
|---|---|---|---|---|
| 1 | Physics modeling / full spatial environment physics | **Built this pass** | `core/worlds/physics.py` — deterministic rigid-body dynamics: semi-implicit Euler, impulse contacts, restitution, Coulomb friction, resting contacts, sleeping, digest-stable determinism | `TESTED-NOW` `tests/test_worlds_physics.py` (closed-form projectile to 1e-9, exact restitution, e²h bounce, momentum/energy laws) |
| 2 | Real-time procedural generation | **Built this pass** | `core/worlds/generation.py` — seeded multi-octave value-noise terrain + themed props; byte-identical regeneration from seed | `TESTED-NOW` same suite (digest determinism, bounds) |
| 3 | Virtual-world hosting / persistent subjective worlds | **Built this pass** | `core/worlds/hosting.py` — named worlds under `data/worlds/`, governed atomic persistence, event journal (a world remembers its history), restart-exact resume | `TESTED-NOW` (restart trajectory-equality test) |
| 4 | Dynamic world generation / counterfactual simulation | **Built this pass** | `WorldHost.fork_world` / `compare_worlds` — fork exact state, intervene, measure divergence, provenance journaled | `TESTED-NOW` `tests/test_worlds_embodied.py` |
| 5 | Robust perception & embodiment (in-world) | **Built this pass** | `core/worlds/embodied.py` — agent body with real dynamics, exact analytic raycast senses, proprioception, grasp/carry/throw, jump | `TESTED-NOW` (analytic ray distances, effector outcomes) |
| 6 | High-level task execution: navigation | **Built this pass** | A* over generated terrain with climb limits, force-based walking through physics, honest outcomes (`reached`/`no_path`/`timed_out`) | `TESTED-NOW` (reaches target on generated terrain; reports no_path on walled terrain) |
| 7 | Quantum computational module | **Built this pass** | `core/quantum/` — exact statevector simulator (20-qubit cap), full gate set via one controlled-unitary primitive, Born-rule measurement with live QuantumEntropyBridge collapse randomness; Bell/GHZ/teleport/Grover/QFT with analytic cross-checks; `quantum_lab` skill in the catalog | `TESTED-NOW` `tests/test_quantum_statevector.py` (31 analytic assertions). `HONEST-LIMIT`: simulation of quantum computation, not quantum hardware |
| 8 | Computer vision: lip-reading | **Built this pass** | `core/senses/visual_speech.py` — mouth tracking, syllabic-band (1.5–8 Hz) motion energy, calibrated speaking probability, hysteresis, viseme features; governed ONNX seam for word-level VSR | `TESTED-NOW` `tests/test_visual_speech.py`. `HONEST-LIMIT`: word-level transcripts require a user-supplied VSR model; until then `transcript=None` with the reason stated |
| 9 | Continuous multimodal perception (fusion) | **Extended this pass** | `core/senses/interaction_signals.py` now runs the calibrated visual-speech channel; probability flows into `classify_audio_attention`, changing live response authorization for ambiguous audio | `TESTED-NOW` `tests/test_visual_speech_integration.py` (end-of-chain flip verified) + `LIVE-PROVEN` existing screen/voice/typing channels |
| 10 | Distributed embodiment to on-network devices (phone chat) | **Built this pass** | `core/security/device_pairing.py`, `interface/routes/devices.py`, `/pair` page — owner-minted codes, revocable per-device tokens (hashed at rest), conversation-surface allowlist, WS cookie auth. PWA already existed | `TESTED-NOW` `tests/test_device_pairing*.py` (27 tests incl. full ASGI flow). Goes live at next runtime restart |
| 11 | Emotional modeling | Existing | `core/affect/` — Damasio-style substrate, circumplex, regulation, nociception; UnifiedFeltState is causal on the Will (action_policy gates) | `LIVE-PROVEN` (June 30 felt-state pass; `tests/` affect suites) |
| 12 | Improve emotional rapport | Existing + this pass | `core/social/relational_intelligence.py`, `reciprocity_engine.py`, conversational profiles; richer visible-speaker signals from #9 feed presence | `PRESENT-UNVERIFIED` (rapport quality is measured in conversation, not unit tests) |
| 13 | Adaptive decision-making | Existing | Will + `action_policy` + outcome ledger + drive controller; decisions gated by felt state and consequences | `LIVE-PROVEN` (felt-state causality pass) |
| 14 | Real-time situational analysis (internal + external) | Existing | interoception (`FeltThought` per-token surprisal), source-body proprioception, flight recorder, immune system detectors, `environment_awareness` | `LIVE-PROVEN` (Jul 9–10 builds, each with its own suite) |
| 15 | Symbolic reasoning | Existing | Pantheon natural-deduction prover wired into beliefs/governance; proof-gated answers (`core/reasoning/proof_answer_solver.py`) | `LIVE-PROVEN` (96-test Pantheon build) |
| 16 | High-fidelity neural simulation / connectome tuning | Existing (scoped) | `core/consciousness/voltage_plasticity.py` (Clopath voltage-STDP + homeostasis + competition, governs episodic recall), liquid substrate, alife dynamics | `LIVE-PROVEN` for the plasticity engine. `HONEST-LIMIT`: this is a functional plasticity substrate, not a biological connectome emulation |
| 17 | Social judgment & social modeling | Existing | `core/social/` — theory_of_mind, other_agent_model, trust_model, stance_inference, boundary_respect, relationship graphs | `PRESENT-UNVERIFIED` this pass (suites exist; not re-run individually today) |
| 18 | Planning & strategic modeling | Existing | `core/planning/` (hierarchical, strategic, task graphs, recovery), `core/sim/` (scenario trees, risk forecaster, outcome simulator) | `PRESENT-UNVERIFIED` this pass |
| 19 | Complex subagent behavioral management / distributed execution | Existing | `core/swarm/` (worker pool, sandbox, ray + k8s backends), council, hierarchical agency | `PRESENT-UNVERIFIED`; k8s/ray backends have no live cluster proof on this host — treat as seams, not claims |
| 20 | Control across heterogeneous systems | Existing | `core/embodiment/iot_bridge.py`, hardware_manager, actuator registry, computer-use skills, environment kernel | `PRESENT-UNVERIFIED` beyond desktop control, which is `LIVE-PROVEN` |
| 21 | Independent value formation | Existing | heartstone values, Will §9d, Ulysses covenant (enforceable self-binding), governance evolution | `LIVE-PROVEN` (Jul 12 Ulysses build) |
| 22 | Resistance to manipulation | Existing + this pass | immune system (detect→reason→respond→heal), ghost-hack guard, resistance_sandbox, defensive_runtime chat ingress; device tokens are scoped so a stolen phone credential cannot reach the control plane | `LIVE-PROVEN` (81-test immune build) + `TESTED-NOW` (device scoping) |
| 23 | Subjective continuity | Existing | flight recorder (SIGKILL-survivable mind-moment ring), death reports → waking narrative, ghost line hash chain; world journals now add place-memory | `LIVE-PROVEN` (FM-FORENSICS-001) |
| 24 | Genuine continual learning | Existing | expert-adapter live seam on the resident 32B, CRSM-LoRA train→fuse→publish loop (verified live Jul 1), compounding scheduler, eval-before-promotion | `LIVE-PROVEN` |
| 25 | Rich, calibrated world models | Existing | `core/world_model/` (unified facade, uncertainty model, expectation engine), calibration in the reasoning amplifier stack | `LIVE-PROVEN` (June 26 stack) |
| 26 | UI interface layers | Existing + this pass | HUD PWA, `/mind`, `/telemetry`, `/activity`, `/controls`, new `/pair`; phone gets the same PWA | `LIVE-PROVEN` + `TESTED-NOW` (/pair) |
| 27 | Simulation-heavy training (Isaac/AirSim-class) | Partial | Native `core/worlds` engine + curriculum/deliberate-practice machinery exist; no GPU game-engine sims installed | `HONEST-LIMIT`: 64GB host already carries a wired 32B; Isaac Sim is out of envelope. The native engine + a future MuJoCo backend seam (`simulator_bridge` interface) is the honest path |

## Honest limits, stated once

- **Physics v1 is translational.** No torque/angular momentum; boxes are
  axis-aligned. The engine says so in its docstring and the roadmap owns it.
- **Quantum is exact simulation**, capped at 20 qubits (16 MiB state).
- **Lip-reading is speech-activity + viseme features** until a real VSR
  model is attached through the governed seam. Nothing fabricates words.
- **Phone transport is plain HTTP on the LAN** — mitigated by scoped,
  revocable, hashed-at-rest device tokens; not by pretending it's TLS.
- **k8s/ray swarm backends** exist as code, unproven against real
  clusters from this host.
- **VR headset rendering** does not exist; worlds are headless with
  full state introspection. A WebGL viewer on the existing HUD is the
  natural next layer.

## Built-this-pass inventory (all pushed to `main`)

| Commit | What |
|---|---|
| `d465da3e` | LAN device pairing (phone chat surface) |
| `80607fa6` | Quantum computational module + `quantum_lab` skill |
| `9bcca608` | Physics engine, procedural generation, persistent hosting, `world_forge` skill |
| `1e46617d` | Embodied agent (senses/effectors/navigation) + counterfactual forking |
| `3bc454a7` | Visual speech → live audio-attention integration |

~95 new tests across `test_device_pairing*.py`, `test_quantum_statevector.py`,
`test_worlds_physics.py`, `test_worlds_embodied.py`, `test_visual_speech*.py`.
Gates run per checkpoint: `make compile`, `make smoke`, `make governance-lint`
(baseline refreshed for the two new governed write sites), `make security`.

## Roadmap (next passes, in priority order)

1. **Rotational dynamics** (quaternions, inertia tensors, contact torque,
   rolling) — removes the v1 physics limitation.
2. **WebGL world viewer** on the HUD (`/worlds`) — see the worlds, not
   just their digests; phone gets it free via the PWA.
3. **Phone voice**: route mic capture from the paired PWA through the
   existing voice lane (`/api/multimodal` is scoped-allowlist-ready).
4. **VSR model onboarding**: pick a word-level lip-reading ONNX model,
   verify licensing, wire decoding behind the existing seam.
5. **MuJoCo backend** behind `simulator_bridge.SimulatorInterface` for
   high-fidelity training when the memory envelope allows.
6. **Device presence → social layer**: paired-device last-seen events
   feeding `presence_integration` ("Bryan is reachable on his phone").
7. **World curriculum**: generate task worlds (fetch/navigate/stack)
   feeding `deliberate_practice` for measurable embodied skill growth.
