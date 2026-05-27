# Aura Model Card

## Primary Model (Cortex)

| Field | Value |
|-------|-------|
| **Role** | Primary reasoning and conversation |
| **Architecture** | Transformer LLM (32B parameters) |
| **Runtime** | MLX on Apple Silicon |
| **Quantization** | 4-bit (MLX native) |
| **Context Window** | 8192 tokens (configurable) |
| **Inference** | Local, on-device |
| **Fine-tuning** | None (base weights only) |

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

## Tertiary Model (Brainstem)

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
