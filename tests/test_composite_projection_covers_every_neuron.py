"""Every neuron reaches the composite output, whatever the total size.

CP126 (high), core/brain/hierarchical_brain.py: "Composite projection
silently drops trailing neurons. For vector lengths not divisible by 64,
floor stride covers only 64 times stride values and ignores the remainder;
the comment calls this a hash projection although it is contiguous
averaging."

Both halves were true. With ``stride = composite.size // 64``, a 200-element
composite used stride 3 and pooled the first 192 values — the last eight
neurons contributed nothing, on every call, silently. A 1000-element
composite dropped forty. Any region whose size pushed the total off a
multiple of 64 lost its tail, and the loss is invisible because the output
is still a well-formed 64-vector.

The comment was the second defect: "hash-project" describes a different
operation with different collision behaviour, so anyone reasoning about this
vector from the comment had the wrong model of it.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.brain.hierarchical_brain import HierarchicalBrain


def _project(values: np.ndarray) -> np.ndarray:
    """Drive the real projection through a stubbed region set."""

    class _Region:
        def __init__(self, out):
            self._out = out

        def get_output(self):
            return self._out

    brain = HierarchicalBrain.__new__(HierarchicalBrain)
    brain._regions = {"only": _Region(values)}
    return brain.get_composite_output()


class TestTheTailIsNotDropped:
    @pytest.mark.parametrize("size", [65, 100, 127, 200, 321, 1000, 1023])
    def test_a_change_in_the_last_element_changes_the_output(self, size):
        """The direct test of the defect: if the tail were still ignored,
        editing it would leave the projection identical."""
        base = np.arange(size, dtype=np.float32)
        tail_changed = base.copy()
        tail_changed[-1] = 9999.0
        assert not np.allclose(_project(base), _project(tail_changed)), (
            f"size={size}: the final neuron does not reach the composite"
        )

    @pytest.mark.parametrize("size", [200, 1000])
    def test_the_previously_dropped_span_now_contributes(self, size):
        """Exactly the span floor-stride skipped: size - 64*(size//64)."""
        stride = size // 64
        first_dropped = 64 * stride
        assert first_dropped < size, "this size was never affected"
        base = np.arange(size, dtype=np.float32)
        changed = base.copy()
        changed[first_dropped:] = 9999.0
        assert not np.allclose(_project(base), _project(changed))


class TestTheProjectionStaysWellFormed:
    @pytest.mark.parametrize("size", [1, 63, 64, 65, 200, 1000])
    def test_the_output_is_always_64_dimensional(self, size):
        result = _project(np.arange(size, dtype=np.float32))
        assert result.shape == (64,)
        assert result.dtype == np.float32

    @pytest.mark.parametrize("size", [65, 200, 1000])
    def test_no_bucket_is_empty_or_nan(self, size):
        result = _project(np.arange(size, dtype=np.float32))
        assert np.all(np.isfinite(result))

    def test_a_short_vector_is_padded_not_pooled(self):
        result = _project(np.arange(10, dtype=np.float32))
        assert np.allclose(result[:10], np.arange(10))
        assert np.allclose(result[10:], 0.0)

    def test_an_exact_multiple_is_unchanged_in_behaviour(self):
        """Sizes divisible by 64 were always correct and must stay so."""
        values = np.arange(320, dtype=np.float32)
        expected = np.array([values[i * 5:(i + 1) * 5].mean() for i in range(64)])
        assert np.allclose(_project(values), expected)

    def test_no_regions_returns_a_zero_vector(self):
        brain = HierarchicalBrain.__new__(HierarchicalBrain)
        brain._regions = {}
        assert brain.get_composite_output().shape == (32,)


class TestTheCommentMatchesTheOperation:
    def test_it_no_longer_claims_to_be_a_hash_projection(self):
        import inspect

        source = inspect.getsource(HierarchicalBrain.get_composite_output)
        assert "NOT a hash projection" in source
