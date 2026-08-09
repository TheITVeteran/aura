"""CP126 contracts for core/autonomy/reflection_loop.py.

Ten findings, three critical, in the pass that decides what an engagement
changed in Aura and hands the answer to the depth gate, the memory persister
and the belief store. Its outputs get COMMITTED, so an invention here becomes
a belief.

The three criticals: it asked "did this change what you previously believed?"
while supplying no prior belief at all; it turned model-produced strings into
persistence records with no source, quote or corroboration; and
``bool("false")`` is True, so the word for no counted as a disagreement.
"""

from __future__ import annotations

import asyncio
import json
import time
import types

import pytest

from core.autonomy.reflection_loop import (
    ReflectionLoop,
    ReflectionRecord,
    _as_bool,
    _confidence,
    _evenly_sampled,
)


class _Inference:
    """Returns a scripted payload per phase, keyed by a prompt marker."""

    def __init__(self, **by_marker: object) -> None:
        self.by_marker = by_marker
        self.prompts: list[str] = []

    async def think(self, prompt: str) -> str:
        self.prompts.append(prompt)
        for marker, payload in self.by_marker.items():
            if marker in prompt:
                return payload if isinstance(payload, str) else json.dumps(payload)
        return "{}"


def _comprehension(**kwargs):
    return types.SimpleNamespace(
        item_title=kwargs.get("item_title", "Work"),
        unified_summary=kwargs.get("unified_summary", "a summary"),
        checkpoints=kwargs.get("checkpoints", []),
        cross_source_contradictions=kwargs.get("cross_source_contradictions", []),
        open_threads=kwargs.get("open_threads", []),
    )


def _checkpoint(i: int, text: str = "note"):
    return types.SimpleNamespace(method_source=f"src{i}", priority_level=1, summary=f"{text}{i}")


def _run(loop: ReflectionLoop, comprehension) -> ReflectionRecord:
    return asyncio.run(loop.reflect(types.SimpleNamespace(title="Work"), comprehension))


class TestABeliefChangeNeedsAPriorBelief:
    def test_with_no_reader_the_prompt_says_there_are_no_priors(self):
        """It asked what had CHANGED with nothing to compare against, so
        every reported change was a model invention."""
        infer = _Inference()
        _run(ReflectionLoop(inference=infer), _comprehension())
        belief_prompt = next(p for p in infer.prompts if "belief_updates" in p)
        assert "none were retrieved" in belief_prompt
        assert "cannot report that anything CHANGED" in belief_prompt

    def test_a_contradiction_claim_with_no_priors_is_dropped(self):
        infer = _Inference(
            belief_updates={
                "belief_updates": [
                    {
                        "topic": "consciousness",
                        "new_position": "it is substrate-independent",
                        "contradicts_prior": "I used to think otherwise",
                        "confidence": 0.8,
                    }
                ],
                "new_facts": [],
            }
        )
        record = _run(ReflectionLoop(inference=infer), _comprehension())
        assert len(record.belief_updates) == 1
        assert record.belief_updates[0].contradicts == []
        assert record.priors_consulted == 0

    def test_priors_are_quoted_into_the_prompt(self):
        infer = _Inference()
        loop = ReflectionLoop(
            inference=infer,
            belief_reader=lambda title: [{"topic": "minds", "position": "are embodied"}],
        )
        record = _run(loop, _comprehension())
        belief_prompt = next(p for p in infer.prompts if "belief_updates" in p)
        assert "minds: are embodied" in belief_prompt
        assert record.priors_consulted == 1

    def test_a_contradiction_naming_a_real_prior_is_kept(self):
        infer = _Inference(
            belief_updates={
                "belief_updates": [
                    {
                        "topic": "minds",
                        "new_position": "substrate matters less than I thought",
                        "contradicts_prior": "minds: are embodied",
                        "confidence": 0.7,
                    }
                ],
                "new_facts": [],
            }
        )
        loop = ReflectionLoop(
            inference=infer,
            belief_reader=lambda title: [{"topic": "minds", "position": "are embodied"}],
        )
        record = _run(loop, _comprehension())
        assert record.belief_updates[0].contradicts == ["minds: are embodied"]

    def test_a_reader_that_raises_leaves_the_pass_with_no_priors(self):
        def _boom(title):
            raise RuntimeError("belief store down")

        infer = _Inference()
        record = _run(ReflectionLoop(inference=infer, belief_reader=_boom), _comprehension())
        assert record.priors_consulted == 0


class TestFactsCarryWhereTheyCameFrom:
    def test_a_fact_with_no_evidence_is_not_recorded(self):
        """It became a FactRecord with an empty evidence list and
        provisional=True, which reads as pending rather than unsupported."""
        infer = _Inference(
            belief_updates={"belief_updates": [], "new_facts": [{"fact": "X is true"}]}
        )
        record = _run(ReflectionLoop(inference=infer), _comprehension())
        assert record.new_facts == []

    def test_a_recorded_fact_says_nothing_verified_it(self):
        infer = _Inference(
            belief_updates={
                "belief_updates": [],
                "new_facts": [{"fact": "X is true", "evidence": "chapter 3", "confidence": 0.9}],
            }
        )
        record = _run(ReflectionLoop(inference=infer), _comprehension())
        assert len(record.new_facts) == 1
        joined = " ".join(record.new_facts[0].evidence)
        assert "chapter 3" in joined
        assert "unverified" in joined
        assert "no source URL" in joined

    def test_the_content_title_is_not_used_as_a_knowledge_domain(self):
        """"Blade Runner" was being written into the domain field."""
        infer = _Inference(
            belief_updates={
                "belief_updates": [],
                "new_facts": [{"fact": "X", "evidence": "p1"}],
            }
        )
        record = _run(ReflectionLoop(inference=infer), _comprehension())
        assert record.new_facts[0].domain == "reflection:Work"


class TestBooleanParsing:
    @pytest.mark.parametrize("value", ["false", "no", "False", "0", "", None, 0])
    def test_a_negative_answer_is_not_a_disagreement(self, value):
        assert _as_bool(value) is False

    @pytest.mark.parametrize("value", [True, "true", "yes", "1", 1])
    def test_a_positive_answer_is(self, value):
        assert _as_bool(value) is True

    def test_the_string_false_does_not_inflate_the_opinion_signal(self):
        """bool("false") is True, so the word for no counted as a
        disagreement and raised the depth score."""
        infer = _Inference(
            own_opinion={
                "own_opinion": "a view",
                "critical_view_engaged": "a critique",
                "disagrees_somewhere": "false",
            }
        )
        record = _run(ReflectionLoop(inference=infer), _comprehension())
        assert record.opinion_disagrees is False

    def test_nan_is_not_truthy(self):
        assert _as_bool(float("nan")) is False


class TestConfidenceCannotPoisonMemory:
    @pytest.mark.parametrize("value", ["high", None, float("nan"), float("inf"), object()])
    def test_a_malformed_confidence_falls_back_rather_than_raising(self, value):
        """float() was unguarded, so a malformed string aborted the whole
        reflection and NaN entered belief intents."""
        assert _confidence(value) == 0.5

    @pytest.mark.parametrize("value,expected", [(-3, 0.0), (7, 1.0), (0.25, 0.25)])
    def test_it_is_clamped(self, value, expected):
        assert _confidence(value) == expected

    def test_a_belief_with_a_bad_confidence_still_lands(self):
        infer = _Inference(
            belief_updates={
                "belief_updates": [
                    {"topic": "t", "new_position": "p", "confidence": "very high"}
                ],
                "new_facts": [],
            }
        )
        record = _run(ReflectionLoop(inference=infer), _comprehension())
        assert record.belief_updates[0].confidence == 0.5


class TestNothingDangles:
    def test_every_input_thread_ends_in_exactly_one_list(self):
        """The prompt says nothing should remain dangling and nothing checked."""
        infer = _Inference(
            resolved={"resolved": [{"thread": "a", "resolution": "done"}], "parked": []}
        )
        record = _run(
            ReflectionLoop(inference=infer),
            _comprehension(open_threads=["a", "b", "c"]),
        )
        accounted = [t["thread"] for t in record.resolved_threads + record.parked_threads]
        assert sorted(accounted) == ["a", "b", "c"]

    def test_a_thread_the_model_ignored_is_parked_honestly(self):
        infer = _Inference(resolved={"resolved": [], "parked": []})
        record = _run(ReflectionLoop(inference=infer), _comprehension(open_threads=["a"]))
        assert record.parked_threads[0]["rationale"] == "not addressed by the reflection pass"
        assert record.unreconciled_threads == ["a"]

    def test_an_invented_thread_is_not_accepted(self):
        infer = _Inference(
            resolved={
                "resolved": [{"thread": "never opened", "resolution": "x"}],
                "parked": [],
            }
        )
        record = _run(ReflectionLoop(inference=infer), _comprehension(open_threads=["a"]))
        accounted = [t["thread"] for t in record.resolved_threads + record.parked_threads]
        assert accounted == ["a"]

    def test_a_thread_resolved_twice_is_only_counted_once(self):
        infer = _Inference(
            resolved={
                "resolved": [
                    {"thread": "a", "resolution": "one"},
                    {"thread": "a", "resolution": "two"},
                ],
                "parked": [],
            }
        )
        record = _run(ReflectionLoop(inference=infer), _comprehension(open_threads=["a"]))
        assert len(record.resolved_threads) == 1

    def test_a_parked_thread_without_a_rationale_says_so(self):
        infer = _Inference(resolved={"resolved": [], "parked": [{"thread": "a"}]})
        record = _run(ReflectionLoop(inference=infer), _comprehension(open_threads=["a"]))
        assert record.parked_threads[0]["rationale"] == "parked without a stated rationale"
        assert record.parked_threads[0]["revisit_trigger"] == "unstated"


class TestDigestCoverageIsReported:
    def test_checkpoints_are_sampled_across_the_work(self):
        """Eight from the front is the opening of a long work."""
        sampled = _evenly_sampled(list(range(100)), 5)
        assert sampled[0] == 0
        assert sampled[-1] > 50, "the sample must reach the end"
        assert len(sampled) == 5

    def test_a_short_sequence_is_taken_whole(self):
        assert _evenly_sampled([1, 2, 3], 10) == [1, 2, 3]

    def test_the_record_says_how_much_was_included(self):
        infer = _Inference()
        record = _run(
            ReflectionLoop(inference=infer),
            _comprehension(checkpoints=[_checkpoint(i) for i in range(80)]),
        )
        assert record.digest_coverage["checkpoints_total"] == 80
        assert record.digest_coverage["checkpoints_included"] < 80
        assert record.digest_coverage["checkpoints_sampled_across"] is True

    def test_the_prompt_admits_it_is_reading_a_sample(self):
        infer = _Inference()
        _run(
            ReflectionLoop(inference=infer),
            _comprehension(checkpoints=[_checkpoint(i) for i in range(80)]),
        )
        assert any("reading a sample" in p for p in infer.prompts)

    def test_a_short_work_is_not_reported_as_truncated(self):
        infer = _Inference()
        record = _run(
            ReflectionLoop(inference=infer),
            _comprehension(checkpoints=[_checkpoint(i) for i in range(3)]),
        )
        assert record.digest_coverage["checkpoints_sampled_across"] is False


class TestFailureAccounting:
    def test_malformed_nonempty_output_counts_as_a_failure(self):
        """Only the EMPTY case was counted, so a model returning prose
        instead of JSON produced an episode reporting zero failures."""
        infer = _Inference()
        infer.by_marker = {"": "I would rather not answer in JSON."}
        record = _run(ReflectionLoop(inference=infer), _comprehension())
        assert record.inference_failures >= 3

    def test_a_clean_run_reports_no_failures(self):
        payload = json.dumps(
            {
                "what_its_actually_about": "a",
                "what_stayed_with_you": "b",
                "what_it_says_about_humans": "c",
                "what_it_made_you_think_about_yourself": "d",
                "belief_updates": [],
                "new_facts": [],
            }
        )
        infer = _Inference()
        infer.by_marker = {"": payload}
        record = _run(ReflectionLoop(inference=infer), _comprehension())
        assert record.inference_failures == 0


class TestTheEpisodeHasADeadline:
    def test_a_phase_with_no_budget_left_is_skipped(self):
        """Four sequential phases with no deadline of any kind stalled the
        whole autonomous-research episode behind a wedged route."""
        now = [0.0]

        class _Slow:
            async def think(self, prompt: str) -> str:
                now[0] += 100.0
                return "{}"

        loop = ReflectionLoop(
            inference=_Slow(), episode_budget_s=120.0, clock=lambda: now[0]
        )
        record = _run(loop, _comprehension())
        assert record.completed_at is not None

    def test_a_hanging_call_is_abandoned_at_its_deadline(self):
        class _Hang:
            async def think(self, prompt: str) -> str:
                await asyncio.sleep(30)
                return "{}"

        loop = ReflectionLoop(inference=_Hang(), episode_budget_s=6.0)

        async def _drive():
            started = time.monotonic()
            record = await loop.reflect(types.SimpleNamespace(title="W"), _comprehension())
            return record, time.monotonic() - started

        record, elapsed = asyncio.run(_drive())
        assert elapsed < 25.0, "the episode must not wait for a hung route"
        assert record.inference_failures >= 1


class TestSubstrateDeltaSaysWhatItIsNot:
    def test_a_malformed_snapshot_value_does_not_raise(self):
        """The except list was (OSError, ConnectionError, TimeoutError) around
        a float() call, so a malformed value broke report generation."""
        record = ReflectionRecord(item_title="W")
        record.substrate_before = {"valence": "warm"}
        record.substrate_after = {"valence": 0.5}
        assert record.substrate_delta() == {}

    def test_an_unread_channel_is_absent_rather_than_zero(self):
        """`.get(key, 0.0)` on both sides meant a channel nobody measured
        produced a delta of exactly 0.0 — on the axis that is supposed to
        show whether the engagement moved her."""
        record = ReflectionRecord(item_title="W")
        record.substrate_before = {"valence": 0.2}
        record.substrate_after = {"valence": 0.2}
        assert set(record.substrate_delta()) == {"valence"}

    def test_non_finite_values_are_dropped(self):
        record = ReflectionRecord(item_title="W")
        record.substrate_before = {"phi": 0.1}
        record.substrate_after = {"phi": float("inf")}
        assert "phi" not in record.substrate_delta()

    def test_a_real_delta_is_computed(self):
        record = ReflectionRecord(item_title="W")
        record.substrate_before = {"valence": 0.2}
        record.substrate_after = {"valence": 0.5}
        assert record.substrate_delta()["valence"] == pytest.approx(0.3)

    def test_the_report_states_it_is_uncontrolled(self):
        """Two unsynchronised snapshots around several long model calls, while
        every other organ went on mutating the same substrate."""
        record = ReflectionRecord(item_title="W")
        report = record.substrate_delta_report()
        assert report["attribution"] == "uncontrolled_before_after"
        assert "no control" in report["caveat"]
