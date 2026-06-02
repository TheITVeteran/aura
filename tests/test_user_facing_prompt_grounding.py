from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


USER_FACING_PROMPT_FILES = (
    ROOT / "core/phases/response_generation_unitary.py",
    ROOT / "core/phases/response_generation.py",
    ROOT / "core/brain/inference_gate.py",
)


FORBIDDEN_LIVE_PROMPT_FRAGMENTS = (
    "PHENOM:",
    "## PHENOMENOLOGY",
    "Phenomenology:",
    "Inner monologue right now:",
    "phenomenological reality",
    "phenomenal context, not abstraction",
    "Stay in character",
    "Trust your instincts",
    "I am present and aware.",
)


def test_live_prompt_scaffolding_uses_state_grounded_language() -> None:
    joined = "\n".join(path.read_text(encoding="utf-8") for path in USER_FACING_PROMPT_FILES)

    for fragment in FORBIDDEN_LIVE_PROMPT_FRAGMENTS:
        assert fragment not in joined

    assert "STATE_GROUNDING:" in joined
    assert "Functional state telemetry:" in joined
    assert "State-grounded report right now:" in joined
