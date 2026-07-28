"""The memory panel must read the fields the memory API actually sends.

LIVE DEFECT, 2026-07-27. Bryan opened WHAT SHE REMEMBERS → SEMANTIC and saw
eight boxes, each containing a single colon and nothing else.

The renderer read four fields:

    const key = item.key || item.subject || '';
    const val = item.value || item.predicate || '';

``/api/memory/semantic`` emits ``{id, content, metadata, timestamp}`` and has
never had any of those four. Every row therefore rendered as
``<strong></strong>: `` — structurally valid, completely empty.

This is the same disease as the context assembler's ``"spontaneous:"``
prefix: a consumer's hand-written field list drifted from what the producer
emits, and the symptom was silent blankness rather than an error. Nothing
crashed, nothing logged, and the panel looked like a design choice.

Two guards. The renderer reads the real field, and a row with nothing to
show is dropped rather than drawn — because eight empty boxes read as "this
feature is broken", while an honest "no semantic memories yet" reads as
information.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

AURA_JS = Path(__file__).resolve().parents[1] / "interface" / "static" / "aura.js"


def _semantic_branch() -> str:
    """The semantic rendering branch, read out of the shipped JS."""
    source = AURA_JS.read_text(encoding="utf-8")
    start = source.index("} else if (type === 'semantic') {")
    end = source.index("} else if (type === 'goals') {", start)
    return source[start:end]


def _api_emitted_fields() -> set[str]:
    """Field names the semantic memory response builder actually constructs."""
    from interface.routes import memory as memory_routes

    source = inspect.getsource(memory_routes._build_semantic_memory_response)
    return set(re.findall(r'"([a-z_]+)":', source))


class TestTheRendererReadsRealFields:
    def test_the_api_emits_content(self):
        """Pins the producer side, so the test fails if the API changes too."""
        assert "content" in _api_emitted_fields()

    def test_the_renderer_reads_content(self):
        """The defect: it read key/subject/value/predicate and never content."""
        assert "item.content" in _semantic_branch()

    def test_renderer_and_api_share_at_least_one_field(self):
        """The drift check itself.

        A renderer whose entire field list is disjoint from what the producer
        emits can only ever draw empty rows — which is exactly what shipped.
        """
        branch = _semantic_branch()
        read_fields = set(re.findall(r"item\.([a-z_]+)", branch))
        assert read_fields & _api_emitted_fields(), (
            f"renderer reads {sorted(read_fields)}, API emits "
            f"{sorted(_api_emitted_fields())} — no overlap"
        )


class TestEmptyRowsAreNotDrawn:
    def test_an_unrenderable_row_returns_nothing(self):
        assert "return '';" in _semantic_branch()

    def test_the_whole_panel_falls_back_to_the_empty_state(self):
        """If every item is unrenderable, say so instead of showing blank."""
        source = AURA_JS.read_text(encoding="utf-8")
        assert "had no renderable fields" in source
        guard = source[source.index("had no renderable fields"):]
        assert "mem-empty" in guard[:400]


class TestLegacyShapesStillRender:
    """Over-correction is the opposite failure: older records that genuinely
    use key/value must keep displaying."""

    @pytest.mark.parametrize("field", ["key", "subject", "value", "predicate"])
    def test_the_legacy_fields_are_still_consulted(self, field):
        assert f"item.{field}" in _semantic_branch()
