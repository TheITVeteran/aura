# Aura Model Card

## Primary Model (Cortex)

| Field | Value |
|-------|-------|
| **Role** | Primary reasoning and conversation |
| **Architecture** | Transformer LLM (32B parameters, Qwen2.5-32B-Instruct) |
| **Runtime** | MLX on Apple Silicon |
| **Quantization** | 8-bit (MLX native); a 4-bit build exists as a legacy / low-memory option |
| **Context Window** | 8192 tokens (configurable) |
| **Inference** | Local, on-device |
| **Fine-tuning** | 8-bit base weights with a personality LoRA applied; the runtime can also promote a fused LoRA delta (`training/fused-model/active.json`) as the live Cortex without a re-quantize |

### Intended Use
Primary model for all user-facing conversation, reasoning, tool planning,
and complex cognitive tasks.

### Limitations
- May confabulate when knowledge is insufficient
- Context window limits multi-turn reasoning depth
- 4-bit quantization trades precision for memory efficiency
- Cannot process images or audio natively

### Ethical Considerations
- Model weights are publicly available base models
- No private/personal data in training
- Prompt injection mitigations applied at runtime layer

---

## Deep Model (Solver)

| Field | Value |
|-------|-------|
| **Role** | Deep-reasoning hot-swap tier for hard problems |
| **Architecture** | Transformer LLM (72B parameters, Qwen2.5-72B-Instruct) |
| **Runtime** | MLX on Apple Silicon |
| **Quantization** | 4-bit (MLX native) |
| **Inference** | Local, on-device |

### Intended Use
Hot-swapped in for the deepest reasoning passes on 64GB-class desktops. It is
the highest-capacity local lane but the slowest (~84s/pass), so it is not the
default foreground model — the 32B Cortex handles standard turns and the Solver
is promoted only when a problem warrants it. Auto-detected/enabled via
`AURA_DEEP_MODEL`.

---

## Background Model (Brainstem)

| Field | Value |
|-------|-------|
| **Role** | Background maintenance, classification, lightweight tasks |
| **Architecture** | Transformer LLM (7B parameters) |
| **Runtime** | MLX on Apple Silicon |
| **Quantization** | 4-bit (MLX native) |
| **Context Window** | 4096 tokens |
| **Inference** | Local, on-device |

### Intended Use
Background tasks: memory consolidation, classification, health probes,
maintenance reasoning. Never used for user-facing responses in production mode.

### Limitations
- Reduced reasoning capability compared to primary
- Not suitable for complex multi-step reasoning
- Background-only; foreground lane isolation prevents interference

---

## Reflex Model

| Field | Value |
|-------|-------|
| **Role** | Fast reflex lane: sub-second acknowledgements, routing, guards |
| **Architecture** | Transformer LLM (1.5B parameters, Qwen2.5-1.5B-Instruct) |
| **Runtime** | MLX on Apple Silicon |
| **Quantization** | 4-bit (MLX native) |
| **Inference** | Local, on-device |

### Intended Use
The lowest-latency local tier. Handles reflexive turns and lightweight
routing/guard decisions when the 32B Cortex is warming or contended, so the
conversation lane can answer immediately instead of waiting on the heavy
model. Never used for substantive full-mind replies.

---

## Cloud Fallback Model

| Field | Value |
|-------|-------|
| **Role** | Fallback when local models unavailable |
| **Provider** | Configurable (OpenAI, Anthropic, etc.) |
| **Activation** | Opt-in only; requires explicit configuration |
| **Privacy** | Prompts classified before transmission |

### Intended Use
Emergency fallback only. Activated when local models are unavailable and
user has explicitly opted in.

### Privacy Controls
- `AURA_CLOUD_FALLBACK_POLICY`: `disabled` (default), `opt-in`, `auto`
- Prompts classified as `public`, `internal`, `sensitive`, `restricted`
- `sensitive` and `restricted` prompts never sent to cloud
- Cloud usage logged in Will receipt trail

---

## Model Verification

All models are verified at load time:
- SHA-256 checksum against `MODEL_MANIFEST.json`
- File integrity check
- Architecture compatibility validation
- Memory footprint validation against hardware profile
