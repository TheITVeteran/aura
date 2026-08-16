"""GenAI spans in the shape a standard backend can read.

opentelemetry has been installed here for a long time and the codebase emitted
zero gen_ai.* attributes: every model call was traced under names only this
repo understands. Instrumentation nobody else can parse has an audience of one.

The attribute names below are asserted as literals on purpose. They are a
contract with consumers outside this repo, so a rename should break a test
rather than quietly break a dashboard.
"""
from __future__ import annotations

import pytest

from core.observability import histograms
from core.observability.genai_semconv import (
    OPERATION_DURATION_METRIC,
    TOKEN_USAGE_INPUT_METRIC,
    TOKEN_USAGE_OUTPUT_METRIC,
    GenAIAttr,
    GenAIOperation,
    agent_span,
    annotate_response,
    chat_span,
    content_capture_enabled,
    embeddings_span,
    memory_span,
    tool_span,
    workflow_span,
)
from core.observability.tracing import SpanStatus, get_tracer

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _force_sampling(monkeypatch):
    """Head-based sampling would otherwise drop these spans at random."""
    tracer = get_tracer()
    monkeypatch.setattr(tracer, "enabled", True)
    monkeypatch.setattr(tracer, "sample_rate", 1.0)
    return tracer


@pytest.fixture(autouse=True)
def _content_capture_off(monkeypatch):
    monkeypatch.delenv("AURA_GENAI_CAPTURE_CONTENT", raising=False)


# ── span naming ────────────────────────────────────────────────────────────


def test_a_chat_span_is_named_operation_then_model():
    with chat_span("qwen-32b") as span:
        pass

    assert span.name == "chat qwen-32b"


def test_a_tool_span_is_named_execute_tool_then_tool():
    with tool_span("read_file") as span:
        pass

    assert span.name == "execute_tool read_file"


def test_an_agent_span_is_named_invoke_agent_then_agent():
    with agent_span("aura") as span:
        pass

    assert span.name == "invoke_agent aura"


def test_a_workflow_span_is_named_invoke_workflow_then_workflow():
    with workflow_span("nightly-consolidation") as span:
        pass

    assert span.name == "invoke_workflow nightly-consolidation"


def test_a_memory_span_carries_no_id_in_its_name():
    """Ids are high-cardinality and would shatter span-name aggregation."""
    with memory_span(GenAIOperation.SEARCH_MEMORY) as span:
        pass

    assert span.name == "search_memory"


# ── attribute contract ─────────────────────────────────────────────────────


def test_the_attribute_keys_are_the_spec_strings():
    assert GenAIAttr.OPERATION_NAME == "gen_ai.operation.name"
    assert GenAIAttr.PROVIDER_NAME == "gen_ai.provider.name"
    assert GenAIAttr.REQUEST_MODEL == "gen_ai.request.model"
    assert GenAIAttr.RESPONSE_MODEL == "gen_ai.response.model"
    assert GenAIAttr.USAGE_INPUT_TOKENS == "gen_ai.usage.input_tokens"
    assert GenAIAttr.USAGE_OUTPUT_TOKENS == "gen_ai.usage.output_tokens"
    assert GenAIAttr.TOOL_NAME == "gen_ai.tool.name"
    assert GenAIAttr.TOOL_CALL_ID == "gen_ai.tool.call.id"
    assert GenAIAttr.AGENT_NAME == "gen_ai.agent.name"
    assert GenAIAttr.CONVERSATION_ID == "gen_ai.conversation.id"


def test_a_chat_span_carries_the_required_attributes():
    with chat_span(
        "qwen-32b", provider="mlx", conversation_id="c1", temperature=0.7
    ) as span:
        pass

    assert span.attributes["gen_ai.operation.name"] == "chat"
    assert span.attributes["gen_ai.provider.name"] == "mlx"
    assert span.attributes["gen_ai.request.model"] == "qwen-32b"
    assert span.attributes["gen_ai.conversation.id"] == "c1"
    assert span.attributes["gen_ai.request.temperature"] == 0.7


def test_absent_optional_attributes_stay_absent():
    """Present-but-null reads as 'measured, and it was nothing'."""
    with chat_span("qwen-32b") as span:
        pass

    assert "gen_ai.request.temperature" not in span.attributes
    assert "gen_ai.provider.name" not in span.attributes


def test_a_tool_span_carries_its_call_id():
    with tool_span("bash", call_id="call_7", tool_type="function") as span:
        pass

    assert span.attributes["gen_ai.tool.name"] == "bash"
    assert span.attributes["gen_ai.tool.call.id"] == "call_7"
    assert span.attributes["gen_ai.tool.type"] == "function"


def test_response_annotation_lands_on_the_span():
    with chat_span("qwen-32b") as span:
        annotate_response(
            span,
            model="qwen-32b-instruct",
            response_id="r1",
            finish_reasons=["stop"],
            input_tokens=100,
            output_tokens=25,
        )

    assert span.attributes["gen_ai.response.model"] == "qwen-32b-instruct"
    assert span.attributes["gen_ai.response.id"] == "r1"
    assert span.attributes["gen_ai.response.finish_reasons"] == ["stop"]
    assert span.attributes["gen_ai.usage.input_tokens"] == 100
    assert span.attributes["gen_ai.usage.output_tokens"] == 25


# ── errors ─────────────────────────────────────────────────────────────────


def test_a_failing_call_records_the_error_type_and_reraises():
    with pytest.raises(RuntimeError):
        with chat_span("qwen-32b") as span:
            raise RuntimeError("model unavailable")

    assert span.attributes["error.type"] == "RuntimeError"
    assert span.status is SpanStatus.ERROR


def test_the_error_message_does_not_become_a_metric_dimension():
    """error.type is the low-cardinality class name, per spec."""
    with pytest.raises(ValueError):
        with tool_span("bash") as span:
            raise ValueError("some very specific path /srv/data/report.txt failed")

    assert span.attributes["error.type"] == "ValueError"


# ── content capture: the egress boundary ───────────────────────────────────


def test_content_capture_is_off_by_default():
    assert content_capture_enabled() is False


def test_messages_are_not_attached_when_capture_is_off():
    with chat_span("m", input_messages=[{"role": "user", "content": "my secret"}]) as span:
        pass

    assert "gen_ai.input.messages" not in span.attributes


def test_messages_are_attached_when_capture_is_on(monkeypatch):
    monkeypatch.setenv("AURA_GENAI_CAPTURE_CONTENT", "1")

    with chat_span("m", input_messages=[{"role": "user", "content": "hello"}]) as span:
        pass

    assert span.attributes["gen_ai.input.messages"] == [
        {"role": "user", "content": "hello"}
    ]


def test_captured_content_is_redacted(monkeypatch):
    """A span is an egress path. The flag someone flips during an incident is
    exactly when raw conversation would otherwise leak."""
    monkeypatch.setenv("AURA_GENAI_CAPTURE_CONTENT", "1")

    with chat_span(
        "m",
        input_messages=[{"role": "user", "content": "key sk-abcdefghijklmnopqrstuvwx"}],
    ) as span:
        pass

    captured = span.attributes["gen_ai.input.messages"][0]["content"]
    assert "sk-abcdefghijklmnopqrstuvwx" not in captured
    assert "REDACTED" in captured


def test_capture_drops_keys_the_collector_did_not_ask_for(monkeypatch):
    """Only role and content. The last egress boundary audited here scrubbed
    values but not keys."""
    monkeypatch.setenv("AURA_GENAI_CAPTURE_CONTENT", "1")

    with chat_span(
        "m",
        input_messages=[{"role": "user", "content": "hi", "internal_id": "SECRET"}],
    ) as span:
        pass

    assert set(span.attributes["gen_ai.input.messages"][0]) == {"role", "content"}


def test_output_messages_honour_the_same_switch(monkeypatch):
    with chat_span("m") as span:
        annotate_response(span, output_messages=[{"role": "assistant", "content": "hi"}])
    assert "gen_ai.output.messages" not in span.attributes

    monkeypatch.setenv("AURA_GENAI_CAPTURE_CONTENT", "1")
    with chat_span("m") as span2:
        annotate_response(span2, output_messages=[{"role": "assistant", "content": "hi"}])
    assert "gen_ai.output.messages" in span2.attributes


def test_the_switch_is_read_per_call_not_cached_at_import(monkeypatch):
    """Turning capture back off must not require a restart."""
    monkeypatch.setenv("AURA_GENAI_CAPTURE_CONTENT", "1")
    assert content_capture_enabled() is True

    monkeypatch.setenv("AURA_GENAI_CAPTURE_CONTENT", "0")
    assert content_capture_enabled() is False


# ── metrics ────────────────────────────────────────────────────────────────


def test_the_standard_histograms_are_declared_with_owners():
    for name in (
        TOKEN_USAGE_INPUT_METRIC,
        TOKEN_USAGE_OUTPUT_METRIC,
        OPERATION_DURATION_METRIC,
    ):
        histogram = histograms.get_histogram(name)
        assert histogram is not None, f"{name} is not declared"
        assert histogram.spec.owner
        assert histogram.spec.description


def test_token_usage_is_recorded_split_by_direction():
    before_in = histograms.get_histogram(TOKEN_USAGE_INPUT_METRIC).snapshot()["count"]
    before_out = histograms.get_histogram(TOKEN_USAGE_OUTPUT_METRIC).snapshot()["count"]

    with chat_span("m") as span:
        annotate_response(span, input_tokens=10, output_tokens=5)

    assert histograms.get_histogram(TOKEN_USAGE_INPUT_METRIC).snapshot()["count"] == before_in + 1
    assert histograms.get_histogram(TOKEN_USAGE_OUTPUT_METRIC).snapshot()["count"] == before_out + 1


def test_duration_is_recorded_even_when_the_call_fails():
    before = histograms.get_histogram(OPERATION_DURATION_METRIC).snapshot()["count"]

    with pytest.raises(RuntimeError):
        with chat_span("m"):
            raise RuntimeError("boom")

    assert histograms.get_histogram(OPERATION_DURATION_METRIC).snapshot()["count"] == before + 1


# ── guards ─────────────────────────────────────────────────────────────────


def test_a_non_agent_operation_is_refused_by_the_agent_helper():
    with pytest.raises(ValueError, match="not an agent operation"):
        with agent_span("aura", operation=GenAIOperation.CHAT):
            pass


def test_a_non_memory_operation_is_refused_by_the_memory_helper():
    with pytest.raises(ValueError, match="not a memory operation"):
        with memory_span(GenAIOperation.CHAT):
            pass


def test_create_agent_is_a_legal_agent_operation():
    with agent_span("aura", operation=GenAIOperation.CREATE_AGENT) as span:
        pass

    assert span.name == "create_agent aura"


def test_embeddings_span_is_named_and_attributed():
    with embeddings_span("bge-small", provider="local") as span:
        pass

    assert span.name == "embeddings bge-small"
    assert span.attributes["gen_ai.operation.name"] == "embeddings"


def test_nested_spans_share_a_trace():
    with agent_span("aura") as outer:
        with tool_span("bash") as inner:
            pass

    assert inner.trace_id == outer.trace_id
    assert inner.parent_span_id == outer.span_id

def test_a_completed_generation_emits_its_token_usage():
    """genai_semconv defined the histograms and nothing ever fed them."""
    from core.brain.llm.measured_admission import record_generation
    from core.observability import genai_semconv

    seen = []
    original = genai_semconv.histograms.record
    genai_semconv.histograms.record = lambda name, value: seen.append((name, value))
    try:
        record_generation(
            model="cortex",
            prompt_tokens=1200,
            generated_tokens=340,
            prefill_seconds=0.4,
            decode_seconds=2.0,
        )
    finally:
        genai_semconv.histograms.record = original

    recorded = dict(seen)
    assert recorded[genai_semconv.TOKEN_USAGE_INPUT_METRIC] == 1200.0
    assert recorded[genai_semconv.TOKEN_USAGE_OUTPUT_METRIC] == 340.0


def test_a_broken_metric_sink_does_not_break_the_response_path():
    from core.brain.llm.measured_admission import record_generation
    from core.observability import genai_semconv

    original = genai_semconv.histograms.record

    def explode(_name, _value):
        raise ValueError("sink is broken")

    genai_semconv.histograms.record = explode
    try:
        record_generation(
            model="cortex",
            prompt_tokens=10,
            generated_tokens=5,
            prefill_seconds=0.1,
            decode_seconds=0.1,
        )
    finally:
        genai_semconv.histograms.record = original
