import numpy as np

from core.brain.llm_interface import LLMInterface


class CognitiveCompressor:
    """Johnson-Lindenstrauss random projection dimensionality reduction."""

    def __init__(self, input_dim: int, target_dim: int, seed: int = 42):
        self.input_dim = input_dim
        self.target_dim = target_dim
        # Deterministic random projection matrix
        rng = np.random.default_rng(seed)
        self.proj = rng.normal(0.0, 1.0 / np.sqrt(target_dim), (target_dim, input_dim))

    def compress(self, vector: np.ndarray) -> np.ndarray:
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector, dtype=np.float32)
        # Handle shape if 1D or 2D
        if vector.ndim == 1:
            return np.dot(self.proj, vector)
        return np.dot(vector, self.proj.T)


class ContextCompressor:
    def __init__(self, model: LLMInterface):
        self.model = model

    async def compress(self, history: str) -> str:
        """Summarize interaction history while preserving key facts."""
        prompt = f"""
Summarize the following interaction history
while preserving important facts.

{history}
"""
        response = await self.model.generate(prompt)
        if hasattr(response, 'content'):
            return response.content
        return str(response)
