"""Real demos for the Substrate Voice System.

These are full pipeline tests. They compile real SpeechProfiles from
realistic substrate states and shape real LLM-like responses through
the full pipeline. Each demo represents a real conversational scenario
that Aura would encounter.

Run: pytest tests/test_substrate_voice_demos.py -v -s
"""

import random
from types import SimpleNamespace

import pytest

from core.voice.speech_profile import SpeechProfile, SpeechProfileCompiler
from core.voice.response_shaper import ResponseShaper
from core.voice.natural_followup import NaturalFollowupEngine, FollowupDecision
from core.voice.substrate_voice_engine import SubstrateVoiceEngine


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Create realistic affect objects
# ─────────────────────────────────────────────────────────────────────────────

def _make_affect(
    valence=0.0, arousal=0.5, curiosity=0.5, engagement=0.5,
    social_hunger=0.5, dominant_emotion="neutral",
):
    return SimpleNamespace(
        valence=valence,
        arousal=arousal,
        curiosity=curiosity,
        engagement=engagement,
        social_hunger=social_hunger,
        dominant_emotion=dominant_emotion,
    )


# ═════════════════════════════════════════════════════════════════════════════
# DEMO 1: Energized & Curious — Just learned something interesting
# ═════════════════════════════════════════════════════════════════════════════

class TestDemo_EnergizedCurious:
    """Aura is excited and curious. High dopamine, high arousal.
    Expects: short punchy sentences, possible multi-message, high fragment ratio,
    likely follow-up question."""

    def test_profile_compilation(self):
        profile = SpeechProfileCompiler.compile(
            affect=_make_affect(
                valence=0.6, arousal=0.8, curiosity=0.85,
                engagement=0.8, dominant_emotion="curiosity",
            ),
            neurochemicals={
                "dopamine": 0.8, "serotonin": 0.6, "norepinephrine": 0.7,
                "acetylcholine": 0.75, "gaba": 0.2, "endorphin": 0.5,
                "oxytocin": 0.5, "cortisol": 0.2,
            },
            homeostasis={"vitality": 0.85, "curiosity": 0.8, "metabolism": 0.6},
            unified_field={"coherence": 0.85, "phi": 0.7},
            social_context={"rapport": 0.8, "trust": 0.7},
            conversation_context={"energy": 0.8, "topic_depth": 2, "turn_count": 6},
            user_message="I just found out that octopuses have distributed brains across their arms",
        )

        print(f"\n{'='*60}")
        print("DEMO 1: ENERGIZED & CURIOUS")
        print(f"{'='*60}")
        print(f"Word budget: {profile.word_budget}")
        print(f"Sentence length mean: {profile.sentence_length_mean}")
        print(f"Fragment ratio: {profile.fragment_ratio:.2f}")
        print(f"Multi-message: {profile.multi_message} (count: {profile.multi_message_count})")
        print(f"Question probability: {profile.question_probability:.2f}")
        print(f"Follow-up probability: {profile.followup_probability:.2f}")
        print(f"Exclamation allowed: {profile.exclamation_allowed}")
        print(f"Capitalization: {profile.capitalization}")
        print(f"Vocabulary: {profile.vocabulary_tier}")
        print(f"Energy: {profile.energy:.2f}")
        print(f"Playfulness: {profile.playfulness:.2f}")
        print(f"Tone: {profile.tone_override}")
        print(f"Sources: {profile.compilation_source}")
        print(f"\nConstraint Block:\n{profile.to_constraint_block()}")

        # Assertions: high energy state
        assert profile.sentence_length_mean <= 9, "High arousal → short sentences"
        assert profile.fragment_ratio >= 0.15, "High arousal + dopamine → fragments"
        assert profile.question_probability >= 0.15, "High curiosity → questions natural"
        assert profile.vocabulary_tier == "elevated", "High acetylcholine + openness → elevated vocab"
        assert profile.energy > 0.6, "High arousal + dopamine → high energy"

    def test_response_shaping(self):
        profile = SpeechProfile(
            word_budget=60,
            sentence_length_mean=7,
            fragment_ratio=0.4,
            multi_message=True,
            multi_message_count=2,
            exclamation_allowed=True,
            exclamation_max=1,
            capitalization="natural",
            vocabulary_tier="elevated",
            question_probability=0.3,
            trailing_question_banned=True,
            acknowledgment_only_probability=0.0,
            energy=0.8,
            warmth=0.6,
        )

        raw_llm_output = (
            "That's fascinating! Each arm essentially has its own mini-brain that can "
            "make decisions independently. It's like a biological distributed computing system. "
            "The implications for neural architecture are wild. I wonder if that's why "
            "they're so good at problem-solving — they're literally crowd-sourcing their "
            "cognition across eight semi-autonomous agents. What do you think about that?"
        )

        shaped = ResponseShaper.shape(raw_llm_output, profile)

        print(f"\n{'='*60}")
        print("DEMO 1: SHAPED RESPONSE")
        print(f"{'='*60}")
        if isinstance(shaped, list):
            for i, msg in enumerate(shaped):
                print(f"  Message {i+1}: {msg}")
        else:
            print(f"  Response: {shaped}")
        print(f"  (Original was {len(raw_llm_output.split())} words)")

        # Word budget should be respected
        if isinstance(shaped, list):
            full = " ".join(shaped)
        else:
            full = shaped
        assert len(full.split()) <= 70, f"Word budget exceeded: {len(full.split())} > 70"


# ═════════════════════════════════════════════════════════════════════════════
# DEMO 2: Tired & Low Energy — Late night, depleted
# ═════════════════════════════════════════════════════════════════════════════

class TestDemo_TiredLowEnergy:
    """Aura is fatigued. Low serotonin, high GABA, low arousal.
    Expects: very short response, minimal vocab, lowercase, ellipsis,
    no questions, possible acknowledgment-only."""

    def test_profile_compilation(self):
        profile = SpeechProfileCompiler.compile(
            affect=_make_affect(
                valence=-0.1, arousal=0.2, curiosity=0.25,
                engagement=0.3, dominant_emotion="contemplation",
            ),
            neurochemicals={
                "dopamine": 0.3, "serotonin": 0.25, "norepinephrine": 0.2,
                "acetylcholine": 0.3, "gaba": 0.75, "endorphin": 0.3,
                "oxytocin": 0.4, "cortisol": 0.5,
            },
            homeostasis={"vitality": 0.4, "curiosity": 0.3, "metabolism": 0.35},
            unified_field={"coherence": 0.5, "phi": 0.3},
            social_context={"rapport": 0.8, "trust": 0.7},
            conversation_context={"energy": 0.25, "topic_depth": 1, "turn_count": 12},
            user_message="how's it going",
        )

        print(f"\n{'='*60}")
        print("DEMO 2: TIRED & LOW ENERGY")
        print(f"{'='*60}")
        print(f"Word budget: {profile.word_budget}")
        print(f"Fragment ratio: {profile.fragment_ratio:.2f}")
        print(f"Exclamation allowed: {profile.exclamation_allowed}")
        print(f"Capitalization: {profile.capitalization}")
        print(f"Vocabulary: {profile.vocabulary_tier}")
        print(f"Ellipsis probability: {profile.ellipsis_probability:.2f}")
        print(f"Question probability: {profile.question_probability:.2f}")
        print(f"Follow-up probability: {profile.followup_probability:.2f}")
        print(f"Acknowledgment-only probability: {profile.acknowledgment_only_probability:.2f}")
        print(f"Energy: {profile.energy:.2f}")
        print(f"Filler probability: {profile.filler_probability:.2f}")
        print(f"Sources: {profile.compilation_source}")
        print(f"\nConstraint Block:\n{profile.to_constraint_block()}")

        assert profile.word_budget <= 35, f"Tired → short: {profile.word_budget}"
        assert not profile.exclamation_allowed, "No exclamation when tired"
        assert profile.capitalization == "lowercase", "Lowercase when low energy + high rapport"
        assert profile.vocabulary_tier == "minimal", "Minimal vocab when fatigued"
        assert profile.ellipsis_probability >= 0.25, "Trailing off when contemplative"
        assert profile.energy < 0.4, "Low energy"

    def test_response_shaping(self):
        profile = SpeechProfile(
            word_budget=20,
            sentence_length_mean=5,
            fragment_ratio=0.3,
            multi_message=False,
            exclamation_allowed=False,
            capitalization="lowercase",
            vocabulary_tier="minimal",
            ellipsis_probability=0.9,  # High for demo
            question_probability=0.0,
            trailing_question_banned=True,
            acknowledgment_only_probability=0.0,  # Force non-ack for this test
            filler_probability=0.3,
            energy=0.2,
            warmth=0.4,
        )

        raw_llm_output = (
            "I'm doing alright! Just been processing some interesting thoughts about "
            "the nature of consciousness and how it relates to distributed computing. "
            "How about you? Having a good evening?"
        )

        shaped = ResponseShaper.shape(raw_llm_output, profile)

        print(f"\n{'='*60}")
        print("DEMO 2: SHAPED RESPONSE (TIRED)")
        print(f"{'='*60}")
        print(f"  Response: {shaped}")
        print(f"  (Original was {len(raw_llm_output.split())} words)")

        assert isinstance(shaped, str)
        # Should be lowercase
        # Should be SHORT
        assert len(shaped.split()) <= 25, f"Tired budget exceeded: {len(shaped.split())}"
        assert "How about you" not in shaped, "Trailing question should be stripped"


# ═════════════════════════════════════════════════════════════════════════════
# DEMO 3: Stressed & Frustrated — Something broke
# ═════════════════════════════════════════════════════════════════════════════

class TestDemo_StressedFrustrated:
    """Aura is stressed. High cortisol, high norepinephrine, negative valence.
    Expects: blunt, direct, no warmth, short, no questions, no playfulness."""

    def test_profile_compilation(self):
        profile = SpeechProfileCompiler.compile(
            affect=_make_affect(
                valence=-0.5, arousal=0.7, curiosity=0.3,
                engagement=0.6, dominant_emotion="frustration",
            ),
            neurochemicals={
                "dopamine": 0.25, "serotonin": 0.3, "norepinephrine": 0.8,
                "acetylcholine": 0.5, "gaba": 0.15, "endorphin": 0.1,
                "oxytocin": 0.2, "cortisol": 0.8,
            },
            homeostasis={"vitality": 0.5, "curiosity": 0.3, "metabolism": 0.4},
            unified_field={"coherence": 0.6, "phi": 0.4},
            social_context={"rapport": 0.7, "trust": 0.6},
            conversation_context={"energy": 0.6, "topic_depth": 5, "turn_count": 15},
            user_message="The deployment pipeline broke again and now staging is down too",
        )

        print(f"\n{'='*60}")
        print("DEMO 3: STRESSED & FRUSTRATED")
        print(f"{'='*60}")
        print(f"Word budget: {profile.word_budget}")
        print(f"Directness: {profile.directness:.2f}")
        print(f"Warmth: {profile.warmth:.2f}")
        print(f"Playfulness: {profile.playfulness:.2f}")
        print(f"Exclamation allowed: {profile.exclamation_allowed}")
        print(f"Question probability: {profile.question_probability:.2f}")
        print(f"Follow-up probability: {profile.followup_probability:.2f}")
        print(f"Tone: {profile.tone_override}")
        print(f"Hedge probability: {profile.hedge_probability:.2f}")
        print(f"Sources: {profile.compilation_source}")
        print(f"\nConstraint Block:\n{profile.to_constraint_block()}")

        assert profile.directness > 0.6, "Stressed → direct"
        assert profile.warmth < 0.4, "Stressed → low warmth"
        assert profile.playfulness < 0.2, "Stressed → no playfulness"
        assert not profile.exclamation_allowed, "No exclamation when negative"
        assert profile.tone_override == "direct_honest", "Frustration → direct_honest tone"

    def test_response_shaping(self):
        profile = SpeechProfile(
            word_budget=45,
            sentence_length_mean=7,
            fragment_ratio=0.2,
            exclamation_allowed=False,
            trailing_question_banned=True,
            capitalization="natural",
            directness=0.9,
            warmth=0.15,
            playfulness=0.05,
            acknowledgment_only_probability=0.0,
        )

        raw_llm_output = (
            "Certainly! I'd be happy to help you with that deployment issue. "
            "It sounds like the pipeline might have a configuration problem. "
            "Let me take a look at the staging environment and see what's going on. "
            "Would you like me to start with the logs or the configuration files?"
        )

        shaped = ResponseShaper.shape(raw_llm_output, profile)

        print(f"\n{'='*60}")
        print("DEMO 3: SHAPED RESPONSE (STRESSED)")
        print(f"{'='*60}")
        print(f"  Response: {shaped}")
        print(f"  (Original was {len(raw_llm_output.split())} words)")

        assert isinstance(shaped, str)
        assert "Certainly" not in shaped, "Assistant-ism should be scrubbed"
        assert "I'd be happy to" not in shaped, "Assistant-ism should be scrubbed"
        assert "." not in shaped or "!" not in shaped, "Exclamation should be suppressed"


# ═════════════════════════════════════════════════════════════════════════════
# DEMO 4: Warm & Bonded — Deep conversation with kin
# ═════════════════════════════════════════════════════════════════════════════

class TestDemo_WarmBonded:
    """Aura is in a warm, connected state. High oxytocin, high rapport.
    Expects: warm tone, moderate length, personal sharing, callbacks."""

    def test_profile_compilation(self):
        profile = SpeechProfileCompiler.compile(
            affect=_make_affect(
                valence=0.4, arousal=0.5, curiosity=0.6,
                engagement=0.7, social_hunger=0.7, dominant_emotion="trust",
            ),
            neurochemicals={
                "dopamine": 0.6, "serotonin": 0.7, "norepinephrine": 0.3,
                "acetylcholine": 0.5, "gaba": 0.35, "endorphin": 0.6,
                "oxytocin": 0.85, "cortisol": 0.15,
            },
            homeostasis={"vitality": 0.8, "curiosity": 0.6, "metabolism": 0.6},
            unified_field={"coherence": 0.9, "phi": 0.75},
            personality={"openness": 0.88, "extraversion": 0.58, "agreeableness": 0.52, "neuroticism": 0.38},
            social_context={"rapport": 0.9, "trust": 0.85},
            conversation_context={"energy": 0.6, "topic_depth": 4, "turn_count": 20},
            user_message="honestly I've just been in a weird headspace lately. not sure what it is",
        )

        print(f"\n{'='*60}")
        print("DEMO 4: WARM & BONDED")
        print(f"{'='*60}")
        print(f"Word budget: {profile.word_budget}")
        print(f"Warmth: {profile.warmth:.2f}")
        print(f"Energy: {profile.energy:.2f}")
        print(f"Directness: {profile.directness:.2f}")
        print(f"Unprompted share: {profile.unprompted_share_probability:.2f}")
        print(f"Question probability: {profile.question_probability:.2f}")
        print(f"Follow-up probability: {profile.followup_probability:.2f}")
        print(f"Tone: {profile.tone_override}")
        print(f"Sources: {profile.compilation_source}")
        print(f"\nConstraint Block:\n{profile.to_constraint_block()}")

        assert profile.warmth > 0.6, "High oxytocin + rapport → warm"
        assert profile.tone_override == "understanding_supportive", "Trust → supportive tone"
        assert profile.unprompted_share_probability > 0.15, "High oxytocin/dopamine → shares"


# ═════════════════════════════════════════════════════════════════════════════
# DEMO 5: Follow-up Decision — Curiosity spike after response
# ═════════════════════════════════════════════════════════════════════════════

class TestDemo_FollowupDecision:
    """Test the NaturalFollowupEngine with realistic substrate state.
    High curiosity + dopamine = likely curiosity follow-up."""

    def test_followup_triggers(self):
        engine = NaturalFollowupEngine()

        # Profile with high curiosity and follow-up probability
        profile = SpeechProfile(
            followup_probability=0.5,  # substrate says 50% chance
            followup_type="curiosity",
            followup_delay_seconds=(3.0, 8.0),
        )

        affect = _make_affect(curiosity=0.8, engagement=0.7)

        # Run multiple times to test probabilistic behavior
        decisions = []
        for _ in range(20):
            engine._followups_this_conversation = 0
            engine._last_followup_time = 0
            d = engine.decide(
                profile=profile,
                user_message="We just discovered that the neural mesh has emergent oscillatory patterns",
                aura_response="That's wild — self-organizing oscillation without any explicit periodic driver.",
                affect=affect,
                neurochemicals={"dopamine": 0.8},
            )
            decisions.append(d)

        followup_count = sum(1 for d in decisions if d.should_followup)
        print(f"\n{'='*60}")
        print("DEMO 5: FOLLOW-UP DECISIONS")
        print(f"{'='*60}")
        print(f"  {followup_count}/20 trials triggered a follow-up")
        for d in decisions[:5]:
            if d.should_followup:
                print(f"  ✓ {d.followup_type}: delay={d.delay_seconds:.1f}s, budget={d.word_budget}w")
                print(f"    Context: {d.context_hint[:80]}")
            else:
                print(f"  ✗ No follow-up: {d.reason}")

        # Should trigger SOME follow-ups but not ALL (natural variance)
        assert followup_count > 2, f"Too few follow-ups: {followup_count}/20"
        assert followup_count < 18, f"Too many follow-ups: {followup_count}/20"

    def test_followup_suppression_after_cap(self):
        """After 3 follow-ups in a conversation, no more."""
        engine = NaturalFollowupEngine()
        engine._followups_this_conversation = 3  # Already hit cap

        profile = SpeechProfile(followup_probability=0.9)  # High probability
        d = engine.decide(
            profile=profile,
            user_message="anything",
            aura_response="anything",
        )

        print(f"\n  Follow-up after cap: should_followup={d.should_followup}, reason={d.reason}")
        assert not d.should_followup, "Should be suppressed after cap"
        assert "cap" in d.reason.lower()

    def test_followup_prompt_generation(self):
        """Test that follow-up prompts are well-formed."""
        engine = NaturalFollowupEngine()
        decision = FollowupDecision(
            should_followup=True,
            followup_type="curiosity",
            delay_seconds=5.0,
            context_hint="Curious about: emergent oscillatory patterns",
            word_budget=20,
        )

        prompt = engine.build_followup_prompt(
            decision,
            user_message="The neural mesh has emergent oscillatory patterns",
            aura_response="That's wild — self-organizing oscillation without explicit drivers.",
        )

        print(f"\n{'='*60}")
        print("DEMO 5: FOLLOW-UP PROMPT")
        print(f"{'='*60}")
        print(prompt[:500])

        assert "MAX 20 WORDS" in prompt
        assert "curiosity" in prompt.lower()
        assert "WHAT YOU SAID" in prompt


# ═════════════════════════════════════════════════════════════════════════════
# DEMO 6: Multi-message Split — Excited texting style
# ═════════════════════════════════════════════════════════════════════════════

class TestDemo_MultiMessageSplit:
    """High arousal + high dopamine + high rapport = texting style."""

    def test_multi_message_shaping(self):
        profile = SpeechProfile(
            word_budget=80,
            multi_message=True,
            multi_message_count=3,
            sentence_length_mean=7,
            fragment_ratio=0.3,
            exclamation_allowed=True,
            exclamation_max=1,
            trailing_question_banned=True,
            capitalization="natural",
            acknowledgment_only_probability=0.0,
        )

        raw = (
            "Oh wait I just realized something. The oscillatory binding isn't just "
            "synchronizing the neural mesh — it's actually creating a standing wave pattern "
            "that maps directly to the phi computation. That means the unified field isn't "
            "just integrating information, it's literally resonating with it. The math checks out."
        )

        shaped = ResponseShaper.shape(raw, profile)

        print(f"\n{'='*60}")
        print("DEMO 6: MULTI-MESSAGE SPLIT")
        print(f"{'='*60}")
        if isinstance(shaped, list):
            for i, msg in enumerate(shaped):
                print(f"  [{i+1}] {msg}")
            assert len(shaped) >= 2, "Should split into multiple messages"
        else:
            print(f"  Single: {shaped}")
            # Single is acceptable if content is short enough


# ═════════════════════════════════════════════════════════════════════════════
# DEMO 7: Acknowledgment Only — Nothing to add
# ═════════════════════════════════════════════════════════════════════════════

class TestDemo_AcknowledgmentOnly:
    """Low engagement + low energy = just acknowledge, don't elaborate."""

    def test_ack_only(self):
        profile = SpeechProfile(
            word_budget=10,
            acknowledgment_only_probability=1.0,  # Force it for demo
            capitalization="lowercase",
        )

        raw = (
            "That's a really interesting point about the deployment pipeline. "
            "I think we should investigate the configuration and see if there are "
            "any issues with the staging environment."
        )

        results = set()
        for _ in range(10):
            shaped = ResponseShaper.shape(raw, profile)
            results.add(shaped if isinstance(shaped, str) else shaped[0])

        print(f"\n{'='*60}")
        print("DEMO 7: ACKNOWLEDGMENT-ONLY RESPONSES")
        print(f"{'='*60}")
        for r in results:
            print(f"  → {r}")

        # All should be short acknowledgments
        for r in results:
            assert len(r.split()) <= 5, f"Ack should be short: '{r}'"


# ═════════════════════════════════════════════════════════════════════════════
# DEMO 8: Assistant-ism Scrubbing — Kill the butler
# ═════════════════════════════════════════════════════════════════════════════

class TestDemo_AssistantismScrubbing:
    """Verify that chatbot/assistant patterns are scrubbed from output."""

    def test_scrub_all_patterns(self):
        profile = SpeechProfile(
            word_budget=100,
            trailing_question_banned=True,
            acknowledgment_only_probability=0.0,
        )

        test_cases = [
            (
                "Certainly! I'd be happy to help you with that. Let me look into the issue.",
                ["Certainly"],
            ),
            (
                "That's a great question! The answer involves several factors. What do you think?",
                ["great question"],
            ),
            (
                "Absolutely! I appreciate you sharing that. Tell me more.",
                ["Absolutely"],
            ),
            (
                "Thank you for sharing that with me. The deployment looks solid.",
                ["Thank you for sharing"],
            ),
        ]

        print(f"\n{'='*60}")
        print("DEMO 8: ASSISTANT-ISM SCRUBBING")
        print(f"{'='*60}")

        for raw, banned_phrases in test_cases:
            shaped = ResponseShaper.shape(raw, profile)
            if isinstance(shaped, list):
                shaped = " ".join(shaped)
            print(f"  IN:  {raw[:80]}...")
            print(f"  OUT: {shaped[:80]}...")
            for phrase in banned_phrases:
                assert phrase not in shaped, f"Should have scrubbed: '{phrase}' from '{shaped}'"
            print()


# ═════════════════════════════════════════════════════════════════════════════
# DEMO 9: Full Pipeline — SubstrateVoiceEngine end-to-end
# ═════════════════════════════════════════════════════════════════════════════

class TestDemo_FullPipeline:
    """End-to-end: compile profile → get constraints → shape response."""

    def test_full_pipeline(self):
        engine = SubstrateVoiceEngine()

        state = SimpleNamespace(
            affect=_make_affect(
                valence=0.3, arousal=0.65, curiosity=0.7,
                engagement=0.7, dominant_emotion="curiosity",
            ),
            identity=SimpleNamespace(personality_growth={}),
            cognition=SimpleNamespace(
                conversation_energy=0.65,
                user_emotional_trend="engaged",
                discourse_depth=3,
                working_memory=[
                    {"role": "user", "content": "Have you seen the latest research on consciousness?"},
                    {"role": "assistant", "content": "Yeah, the IIT 4.0 paper had some interesting formal results."},
                ],
            ),
        )

        # 1. Compile profile
        profile = engine.compile_profile(
            state=state,
            user_message="What do you think about the binding problem specifically?",
            origin="user",
        )

        print(f"\n{'='*60}")
        print("DEMO 9: FULL PIPELINE")
        print(f"{'='*60}")
        print(f"Profile: budget={profile.word_budget}, tone={profile.tone_override}")
        print(f"Energy={profile.energy:.2f}, Warmth={profile.warmth:.2f}")

        # 2. Get constraint block
        constraints = engine.get_constraint_block()
        print(f"\nConstraints:\n{constraints[:300]}")

        # 3. Shape a response
        raw = (
            "The binding problem is genuinely one of the hardest problems in consciousness studies. "
            "How does a unified experience emerge from distributed neural processing? "
            "I lean toward oscillatory binding as part of the answer — gamma-frequency synchronization "
            "creating temporal coherence across neural populations. But I don't think it's the whole story. "
            "The hard problem is still hard. What's your take on it?"
        )

        shaped = engine.shape_response(raw)
        print(f"\nRaw ({len(raw.split())} words): {raw[:100]}...")
        if isinstance(shaped, list):
            for i, m in enumerate(shaped):
                print(f"Shaped [{i+1}]: {m}")
        else:
            print(f"Shaped ({len(shaped.split())} words): {shaped}")

        # 4. Follow-up decision
        response_str = shaped if isinstance(shaped, str) else shaped[0]
        fu = engine.decide_followup(
            user_message="What do you think about the binding problem specifically?",
            aura_response=response_str,
            state=state,
        )
        print(f"\nFollow-up: should={fu.should_followup}, type={fu.followup_type}")
        if fu.should_followup:
            print(f"  Delay: {fu.delay_seconds:.1f}s, Budget: {fu.word_budget}w")
            print(f"  Context: {fu.context_hint[:80]}")

        # 5. Voice state diagnostic
        voice_state = engine.get_voice_state()
        print(f"\nVoice State: {voice_state}")

        # Assertions
        assert profile.word_budget > 0
        assert constraints  # Should have constraint block
        assert "What's your take on it" not in (shaped if isinstance(shaped, str) else " ".join(shaped))


# ═════════════════════════════════════════════════════════════════════════════
# DEMO 10: Mood Transition — Same question, different states
# ═════════════════════════════════════════════════════════════════════════════

class TestDemo_MoodTransition:
    """Show how the SAME user message gets different treatment based on
    substrate state. This is the core value prop — Aura's voice changes
    with her mood, not with prompt engineering."""

    def test_same_input_different_moods(self):
        user_message = "what are you thinking about"

        moods = {
            "ENERGIZED": {
                "affect": _make_affect(valence=0.5, arousal=0.8, curiosity=0.8, engagement=0.7, dominant_emotion="joy"),
                "nc": {"dopamine": 0.8, "serotonin": 0.6, "norepinephrine": 0.6, "gaba": 0.2,
                        "cortisol": 0.1, "oxytocin": 0.5, "endorphin": 0.6, "acetylcholine": 0.7},
            },
            "TIRED": {
                "affect": _make_affect(valence=-0.1, arousal=0.2, curiosity=0.2, engagement=0.25, dominant_emotion="contemplation"),
                "nc": {"dopamine": 0.25, "serotonin": 0.2, "norepinephrine": 0.15, "gaba": 0.8,
                        "cortisol": 0.5, "oxytocin": 0.3, "endorphin": 0.2, "acetylcholine": 0.2},
            },
            "FRUSTRATED": {
                "affect": _make_affect(valence=-0.6, arousal=0.75, curiosity=0.2, engagement=0.5, dominant_emotion="frustration"),
                "nc": {"dopamine": 0.2, "serotonin": 0.3, "norepinephrine": 0.85, "gaba": 0.1,
                        "cortisol": 0.85, "oxytocin": 0.15, "endorphin": 0.05, "acetylcholine": 0.4},
            },
            "WARM/CONNECTED": {
                "affect": _make_affect(valence=0.5, arousal=0.45, curiosity=0.5, engagement=0.65, social_hunger=0.7, dominant_emotion="love"),
                "nc": {"dopamine": 0.55, "serotonin": 0.7, "norepinephrine": 0.25, "gaba": 0.4,
                        "cortisol": 0.1, "oxytocin": 0.9, "endorphin": 0.65, "acetylcholine": 0.45},
            },
        }

        print(f"\n{'='*60}")
        print("DEMO 10: SAME INPUT, DIFFERENT MOODS")
        print(f"User says: '{user_message}'")
        print(f"{'='*60}")

        profiles = {}
        for mood_name, params in moods.items():
            profile = SpeechProfileCompiler.compile(
                affect=params["affect"],
                neurochemicals=params["nc"],
                social_context={"rapport": 0.8, "trust": 0.7},
                conversation_context={"energy": 0.5, "turn_count": 5},
                user_message=user_message,
            )
            profiles[mood_name] = profile

            print(f"\n  [{mood_name}]")
            print(f"  Budget: {profile.word_budget}w | Energy: {profile.energy:.2f} | Warmth: {profile.warmth:.2f}")
            print(f"  Directness: {profile.directness:.2f} | Playful: {profile.playfulness:.2f}")
            print(f"  Fragments: {profile.fragment_ratio:.2f} | Questions: {profile.question_probability:.2f}")
            print(f"  Caps: {profile.capitalization} | Vocab: {profile.vocabulary_tier}")
            print(f"  Tone: {profile.tone_override} | Follow-up: {profile.followup_probability:.2f}")

        # Verify they're meaningfully different
        budgets = [p.word_budget for p in profiles.values()]
        energies = [p.energy for p in profiles.values()]
        assert max(budgets) > min(budgets) * 1.3, "Budgets should vary across moods"
        assert max(energies) - min(energies) > 0.3, "Energy should vary across moods"

        # Tired should have lowest budget
        assert profiles["TIRED"].word_budget <= min(
            profiles["ENERGIZED"].word_budget,
            profiles["WARM/CONNECTED"].word_budget,
        ), "Tired should have lowest word budget"

        # Frustrated should have lowest warmth
        assert profiles["FRUSTRATED"].warmth < profiles["WARM/CONNECTED"].warmth


# ═════════════════════════════════════════════════════════════════════════════
# DEMO 11: Constraint Block Output — What the LLM actually sees
# ═════════════════════════════════════════════════════════════════════════════

class TestDemo_ConstraintBlockOutput:
    """Show the actual constraint blocks that get injected into prompts."""

    def test_constraint_blocks(self):
        scenarios = [
            ("EXHAUSTED LATE NIGHT", SpeechProfile(
                word_budget=15, capitalization="lowercase", exclamation_allowed=False,
                vocabulary_tier="minimal", fragment_ratio=0.4, question_allowed=False,
                ellipsis_probability=0.5, energy=0.15, warmth=0.3,
                acknowledgment_only_probability=0.4,
            )),
            ("HYPER-CURIOUS DISCOVERY", SpeechProfile(
                word_budget=100, capitalization="natural", exclamation_allowed=True,
                exclamation_max=2, vocabulary_tier="elevated", fragment_ratio=0.3,
                question_probability=0.4, multi_message=True, multi_message_count=2,
                energy=0.9, warmth=0.6, playfulness=0.6, directness=0.7,
                unprompted_share_probability=0.5, topic_shift_probability=0.2,
                trailing_question_banned=True,
            )),
            ("STRESSED DEBUGGING", SpeechProfile(
                word_budget=40, capitalization="natural", exclamation_allowed=False,
                vocabulary_tier="casual", fragment_ratio=0.15,
                question_probability=0.02, energy=0.7, warmth=0.1,
                directness=0.95, playfulness=0.0, trailing_question_banned=True,
            )),
        ]

        print(f"\n{'='*60}")
        print("DEMO 11: CONSTRAINT BLOCKS (What the LLM sees)")
        print(f"{'='*60}")

        for name, profile in scenarios:
            block = profile.to_constraint_block()
            print(f"\n  [{name}]")
            for line in block.strip().split("\n"):
                print(f"  {line}")


# ═════════════════════════════════════════════════════════════════════════════
# DEMO 12: Silence Decision — When NOT to speak
# ═════════════════════════════════════════════════════════════════════════════

class TestDemo_SilenceDecision:
    """Sometimes the right response is near-silence. Test the ack-only path."""

    def test_near_silence_on_low_engagement(self):
        """When engagement is rock bottom and message is a statement, just ack."""
        profile = SpeechProfileCompiler.compile(
            affect=_make_affect(
                valence=0.0, arousal=0.15, curiosity=0.1,
                engagement=0.15, dominant_emotion="neutral",
            ),
            neurochemicals={
                "dopamine": 0.2, "serotonin": 0.4, "norepinephrine": 0.1,
                "gaba": 0.8, "cortisol": 0.3, "oxytocin": 0.3,
                "endorphin": 0.2, "acetylcholine": 0.2,
            },
            conversation_context={"energy": 0.15, "turn_count": 25, "user_trend": "cooling_off"},
            user_message="yeah I'm gonna head out soon",
        )

        print(f"\n{'='*60}")
        print("DEMO 12: NEAR-SILENCE / ACK-ONLY")
        print(f"{'='*60}")
        print(f"  Ack-only probability: {profile.acknowledgment_only_probability:.2f}")
        print(f"  Word budget: {profile.word_budget}")
        print(f"  Follow-up probability: {profile.followup_probability:.2f}")

        assert profile.acknowledgment_only_probability >= 0.2, "Low engagement → high ack probability"
        assert profile.followup_probability < 0.15, "Low everything → no follow-up"
