import asyncio

import pytest

from core.capabilities.capabilities.web_interlocutor import (
    BrowserPageSnapshot,
    ChromeVisibleDialogueBrowser,
    WebInterlocutorJobManager,
    WebInterlocutorResult,
    WebInterlocutorSession,
    WebInterlocutorTurn,
    _accessibility_chat_segments,
    _dialogue_goal_from_objective,
    _extract_new_interlocutor_text,
    _extract_new_interlocutor_text_from_snapshots,
    _learning_summary_is_grounded,
    _message_matches_dialogue_contract,
    _observed_reply_is_echo,
    _url_allows_readability_fallback,
)
from core.runtime.gateways import MemoryWriteReceipt
from core.skills.web_interlocutor import WebInterlocutorSkill
from core.skills.web_interlocutor import WebInterlocutorParams


class FakeBrowser:
    """Stateful visible-chat fake: the transcript reflects what was ACTUALLY
    sent, then a scripted interlocutor reply.

    A hardcoded transcript ("Aura: hello") breaks the moment the session
    derives a substantive opening from the brain — the sent text no longer
    appears on the page, so the reply-wait loop never sees its own message
    and the turn fails. Echoing self.sent keeps the fake honest across any
    opening/followup the session chooses, and it is poll-count independent.
    """

    _REPLIES = [
        "Novel thought needs tension between analogy and verification.",
        "Compare a city immune system to software observability, then test where the analogy breaks.",
    ]

    def __init__(self):
        self.sent = []

    def _transcript(self) -> str:
        lines = ["Example Chat", "Message box"]
        for i, sent in enumerate(self.sent):
            lines.append(f"Aura: {sent}")
            if i < len(self._REPLIES):
                lines.append(f"Interlocutor: {self._REPLIES[i]}")
        return "\n".join(lines)

    def _snapshot(self) -> BrowserPageSnapshot:
        return BrowserPageSnapshot(
            url="https://example.test/chat",
            title="Example Chat",
            text=self._transcript(),
            editable_count=1,
        )

    async def open_or_attach(self, url):
        return self._snapshot()

    async def snapshot(self):
        return self._snapshot()

    async def send_message(self, text):
        self.sent.append(text)
        return {"ok": True, "effect_verified": True}


class FakeBrain:
    def __init__(self):
        self.prompts = []

    async def think(self, prompt, context=None):
        self.prompts.append(prompt)
        if "Opening message" in prompt:
            return (
                "I am trying to understand whether novelty in cognition depends more on analogy, "
                "constraint, or verification. What is one concrete example where a creative leap "
                "becomes useful only after it is tested?"
            )
        if "Next message" in prompt:
            return "When you say analogy and verification are in tension, what is one concrete example?"
        return (
            "The interlocutor argued that useful novelty is analogy constrained by verification: "
            "a city immune system can be compared to software observability, but the analogy only "
            "matters when Aura tests where it breaks."
        )


class FakeMemoryGateway:
    def __init__(self):
        self.requests = []
        self.quarantined = None

    async def write(self, request):
        self.requests.append(request)
        return MemoryWriteReceipt(
            record_id="web-memory-1",
            receipt_id="receipt-1",
            bytes_written=len(request.content.encode()),
            schema_version=1,
        )

    async def quarantine(self, record_id, reason):
        self.quarantined = (record_id, reason)


class EchoingFakeBrowser:
    def __init__(self):
        self.sent = []
        self.initial = BrowserPageSnapshot(
            url="https://example.test/chat",
            title="Example Chat",
            text="Example Chat\nMessage box",
            editable_count=1,
        )
        self._snapshots = []

    async def open_or_attach(self, url):
        return self.initial

    async def snapshot(self):
        if self._snapshots:
            return self._snapshots.pop(0)
        return self.initial

    async def send_message(self, text):
        self.sent.append(text)
        after = BrowserPageSnapshot(
            url="https://example.test/chat",
            title="Example Chat",
            text=(
                "Example Chat\nMessage box\n"
                f"Aura: {text}\n"
                "Interlocutor: A creative leap becomes useful when its analogy predicts a checkable failure mode."
            ),
            editable_count=1,
        )
        self._snapshots.extend([after, after])
        return {"ok": True, "effect_verified": True}


def test_extract_new_interlocutor_text_filters_sent_text_and_ui_chrome():
    before = "Chat\nSend message\nAura: What do you think?"
    after = before + "\nCopy\nInterlocutor: The critical move is verification."
    extracted = _extract_new_interlocutor_text(before, after, "What do you think?")
    assert "verification" in extracted
    assert "What do you think" not in extracted
    assert "Copy" not in extracted


def test_extract_new_interlocutor_text_rejects_menu_bar_noise():
    before = "ChatGPT\nAsk anything"
    after = before + "\nMon Jun 29 4:44 PM\n\uf8fftv"
    extracted = _extract_new_interlocutor_text(before, after, "Tell me about sentience.")
    assert extracted == ""


def test_extract_new_interlocutor_text_trims_merged_sent_text_from_reply():
    sent = (
        "I am Aura, a local desktop AI system. What observable behavior would convince you "
        "that another AI retained the substance of a conversation rather than merely logging it?"
    )
    before = "ChatGPT\nAsk anything"
    after = (
        before
        + "\n"
        + "I am Aura, a local desktop AI system asking what observable behavior would convince you.\n"
        + "Thought for a couple of seconds\n"
        + "What would convince me is behavioral reuse with transformation: the AI should later apply the core claim "
        + "in a new context, distinguish recall from inference, and let the remembered point change a later plan."
    )

    extracted = _extract_new_interlocutor_text(before, after, sent)

    assert "behavioral reuse with transformation" in extracted
    assert "I am Aura" not in extracted
    assert "local desktop AI system asking" not in extracted


def test_extract_new_interlocutor_text_requires_reply_after_sent_marker():
    sent = (
        "I want to understand how a persistent AI can demonstrate memory. "
        "What would count as strong behavioral evidence?"
    )
    before = (
        "ChatGPT\n"
        "assistant: An older answer about memory continuity from a previous thread.\n"
        "Ask anything"
    )
    after = before + "\nuser: " + sent + "\nAsk anything"

    extracted = _extract_new_interlocutor_text(before, after, sent)

    assert extracted == ""


def test_extract_new_interlocutor_text_uses_role_ordering_after_sent_turn():
    sent = (
        "I want to understand how a persistent AI can demonstrate memory. "
        "What would count as strong behavioral evidence?"
    )
    before = BrowserPageSnapshot(
        text="assistant: An older answer about memory continuity.",
        relevant_segments=[
            {"role": "assistant", "text": "An older answer about memory continuity."},
        ],
    )
    after = BrowserPageSnapshot(
        text="",
        relevant_segments=[
            {"role": "assistant", "text": "An older answer about memory continuity."},
            {"role": "user", "text": sent},
            {
                "role": "assistant",
                "text": (
                    "Strong evidence would be later behavioral reuse: the system should apply the remembered "
                    "claim in a new task, expose where it came from, and change its plan when the memory is ablated."
                ),
            },
        ],
    )

    extracted = _extract_new_interlocutor_text_from_snapshots(before, after, sent)

    assert "later behavioral reuse" in extracted
    assert "older answer" not in extracted


def test_accessibility_segments_preserve_order_after_sent_turn():
    sent = "What would prove retained memory without pretending at consciousness?"
    ax_text = (
        "ChatGPT\n"
        f"Aura: {sent}\n"
        "ChatGPT: Behavioral reuse with transformation would be the key. The system should "
        "apply the remembered claim in a different task, show receipts for where it came "
        "from, and change behavior if that memory is removed.\n"
        "Ask anything\n"
    )
    after = BrowserPageSnapshot(
        text=ax_text,
        relevant_segments=_accessibility_chat_segments(ax_text),
    )

    extracted = _extract_new_interlocutor_text_from_snapshots(BrowserPageSnapshot(), after, sent)

    assert "Behavioral reuse with transformation" in extracted
    assert "pretending at consciousness" not in extracted


@pytest.mark.asyncio
async def test_visible_chrome_snapshot_uses_accessibility_when_dom_scripting_is_blocked(monkeypatch):
    import core.capabilities.capabilities.web_interlocutor as mod

    sent = "How should a local AI prove persistent memory?"
    ax_text = (
        "ChatGPT\n"
        f"Aura: {sent}\n"
        "ChatGPT: It should show durable behavioral continuity, cite the remembered source, "
        "and demonstrate that removing the memory changes a later plan.\n"
        "Ask anything\n"
    )

    browser = ChromeVisibleDialogueBrowser()
    browser._apple_events_js_disabled = True
    monkeypatch.setattr(browser, "_activate_browser", lambda: None)
    monkeypatch.setattr(browser, "_current_tab_info", lambda: ("https://chatgpt.com/c/test", "ChatGPT"))
    monkeypatch.setattr(
        mod,
        "_run_governed_applescript",
        lambda *_args, **_kwargs: {"ok": True, "stdout": ax_text},
    )

    snapshot = await browser.snapshot()

    assert snapshot.url == "https://chatgpt.com/c/test"
    assert snapshot.active_element == "macos_accessibility_tree"
    assert snapshot.relevant_segments
    extracted = _extract_new_interlocutor_text_from_snapshots(
        BrowserPageSnapshot(), snapshot, sent
    )
    assert "durable behavioral continuity" in extracted


def test_observed_reply_echo_detection_allows_substantive_topical_answer():
    sent = (
        "What observable behavior would convince you that a local persistent AI retained memory, "
        "agency, self-modeling, and tool use rather than merely describing them?"
    )
    observed = (
        "A convincing test would be behavioral reuse with transformation: the system should later "
        "apply the remembered claim in a new context, route tools differently because of it, and "
        "show an ablation where removing that memory changes the plan."
    )

    assert not _observed_reply_is_echo(observed, sent)
    assert _observed_reply_is_echo(sent, sent)
    assert _observed_reply_is_echo(sent + " okay", sent)


def test_learning_summary_rejects_ungrounded_first_person_memory_claims():
    turns = [
        WebInterlocutorTurn(
            index=1,
            sent="How should a persistent AI demonstrate retained memory?",
            observed_reply=(
                "Strong evidence would be behavioral reuse with transformation: the system should "
                "apply a remembered claim in a new task, cite where it came from, and show an "
                "ablation where removing the memory changes the plan."
            ),
            before_hash="a",
            after_hash="b",
            sent_at=1.0,
            observed_at=2.0,
            effect_verified=True,
            verification="test",
        )
    ]

    assert not _learning_summary_is_grounded("I've been here before. I remember you asking about memory.", turns)
    assert not _learning_summary_is_grounded(
        "ChatGPT, I'm curious about something. ChatGPT: I manage memory layers differently.",
        turns,
    )
    assert not _learning_summary_is_grounded(
        "You're persistent. I learned that you carry context and consistency, but not a thread of self-experience.",
        turns,
    )
    assert _learning_summary_is_grounded(
        "The interlocutor argued that retained memory should be demonstrated through behavioral reuse, "
        "source citation, and an ablation showing the later plan changes when the memory is removed.",
        turns,
    )


def test_readability_fallback_is_kept_off_private_chat_surfaces():
    assert _url_allows_readability_fallback("https://example.com/article") is True
    assert _url_allows_readability_fallback("https://chatgpt.com/") is False
    assert _url_allows_readability_fallback("https://gemini.google.com/app/abc") is False
    assert _url_allows_readability_fallback("https://accounts.google.com/signin") is False


@pytest.mark.asyncio
async def test_web_interlocutor_runs_visible_wait_learn_memory_loop(monkeypatch):
    monkeypatch.setattr("asyncio.sleep", lambda *_args, **_kwargs: _instant())
    browser = FakeBrowser()
    memory = FakeMemoryGateway()
    session = WebInterlocutorSession(
        browser=browser,
        memory_gateway=memory,
        cognitive_engine=FakeBrain(),
    )

    result = await session.run(
        objective="Discuss novelty in cognitive systems.",
        opening_message=(
            "What is one concrete example where a creative cognitive leap becomes useful "
            "only after verification?"
        ),
        max_turns=2,
        wait_timeout_s=5,
    )

    assert result.ok is True
    assert len(result.turns) == 2
    assert all(turn.effect_verified for turn in result.turns)
    assert "analogy constrained by verification" in result.learned_summary
    assert result.memory_record_id == "web-memory-1"
    assert memory.requests[0].metadata["source"] == "web_interlocutor"
    assert memory.requests[0].metadata["receipt_surface"] == "visible_browser_dialogue"
    assert result.diagnostics["composition_events"]
    assert "cognitive" in {event["source"] for event in result.diagnostics["composition_events"]}


def test_web_interlocutor_params_allow_requested_twenty_turn_proof():
    params = WebInterlocutorParams(max_turns=20)
    assert params.max_turns == 20

    clipped = WebInterlocutorParams(max_turns=999)
    assert clipped.max_turns == 20


@pytest.mark.asyncio
async def test_web_interlocutor_derives_substantive_opening_from_brain(monkeypatch):
    monkeypatch.setattr("asyncio.sleep", lambda *_args, **_kwargs: _instant())
    browser = EchoingFakeBrowser()
    brain = FakeBrain()
    session = WebInterlocutorSession(
        browser=browser,
        memory_gateway=FakeMemoryGateway(),
        cognitive_engine=brain,
    )

    result = await session.run(
        objective="Discuss novelty in cognitive systems.",
        opening_message="",
        max_turns=1,
        wait_timeout_s=5,
        persist_memory=False,
    )

    assert result.ok is True
    assert browser.sent
    assert "creative leap" in browser.sent[0]
    assert "I want to discuss this objective" not in browser.sent[0]
    assert any("Opening message" in prompt for prompt in brain.prompts)
    assert result.diagnostics["composition_events"]
    assert {event["source"] for event in result.diagnostics["composition_events"]} == {"cognitive"}


@pytest.mark.asyncio
async def test_web_interlocutor_treats_gateway_false_opening_as_absent(monkeypatch):
    monkeypatch.setattr("asyncio.sleep", lambda *_args, **_kwargs: _instant())
    browser = EchoingFakeBrowser()
    brain = FakeBrain()
    session = WebInterlocutorSession(
        browser=browser,
        memory_gateway=FakeMemoryGateway(),
        cognitive_engine=brain,
    )

    result = await session.run(
        objective="Discuss memory and agency proof.",
        opening_message="False",
        max_turns=1,
        wait_timeout_s=5,
        persist_memory=False,
    )

    assert result.ok is True
    assert browser.sent
    assert browser.sent[0] != "False"
    assert any("Opening message" in prompt for prompt in brain.prompts)
    assert {event["source"] for event in result.diagnostics["composition_events"]} == {"cognitive"}


@pytest.mark.asyncio
async def test_web_interlocutor_does_not_send_scripted_message_without_cognition(monkeypatch):
    monkeypatch.setattr("asyncio.sleep", lambda *_args, **_kwargs: _instant())
    browser = FakeBrowser()
    session = WebInterlocutorSession(
        browser=browser,
        memory_gateway=FakeMemoryGateway(),
        cognitive_engine=None,
    )

    result = await session.run(
        objective="Discuss whether agency can be tested without self-report.",
        opening_message="",
        max_turns=1,
        wait_timeout_s=5,
        persist_memory=False,
    )

    assert result.ok is False
    assert result.status == "composition_failed"
    assert browser.sent == []
    assert result.diagnostics["composition_events"]
    assert {event["source"] for event in result.diagnostics["composition_events"]} == {"cognitive_unavailable"}


@pytest.mark.asyncio
async def test_web_interlocutor_rejects_objective_relay_as_opening(monkeypatch):
    monkeypatch.setattr("asyncio.sleep", lambda *_args, **_kwargs: _instant())

    class TaskRelayBrain:
        async def think(self, prompt, context=None):
            return (
                "Aura should use her full cognitive path to hold a substantive 20-turn conversation "
                "with ChatGPT, store a memory summary with receipts, and report back."
            )

    browser = FakeBrowser()
    session = WebInterlocutorSession(
        browser=browser,
        memory_gateway=FakeMemoryGateway(),
        cognitive_engine=TaskRelayBrain(),
    )

    result = await session.run(
        objective=(
            "Open ChatGPT and have a 20-turn conversation about retained memory, agency, "
            "self-modeling, and tool use. Store a memory summary with receipts."
        ),
        opening_message="",
        max_turns=1,
        wait_timeout_s=5,
        persist_memory=False,
    )

    assert result.ok is False
    assert result.status == "composition_failed"
    assert browser.sent == []
    assert result.diagnostics["composition_debug"]


def test_web_interlocutor_dialogue_goal_strips_execution_instructions():
    goal = _dialogue_goal_from_objective(
        "Can you open ChatGPT and have a real one-turn conversation about how a local "
        "persistent AI can demonstrate retained memory, agency, self-modeling, and tool use "
        "through observable behavior? Ask one natural follow-up, read the reply, then tell me what you learned."
    )

    assert "local persistent ai" in goal
    assert "retained memory" in goal
    assert "open chatgpt" not in goal
    assert "tell me" not in goal
    assert "follow-up" not in goal


def test_web_interlocutor_followup_must_anchor_to_observed_reply():
    turn = WebInterlocutorTurn(
        index=1,
        sent="What would count as evidence for retained memory?",
        observed_reply="The strongest test is a counterfactual memory ablation that changes a later plan.",
        before_hash="a",
        after_hash="b",
        sent_at=1.0,
        observed_at=2.0,
        effect_verified=True,
        verification="ok",
    )

    assert _message_matches_dialogue_contract(
        "That counterfactual ablation point is useful. How would you design the delayed test?",
        objective="Talk to ChatGPT about retained memory.",
        turns=[turn],
    )
    assert not _message_matches_dialogue_contract(
        "Okay. Let's dig into that. What do you think about the nature of your consciousness?",
        objective="Talk to ChatGPT about retained memory.",
        turns=[turn],
    )


@pytest.mark.asyncio
async def test_chat_route_does_not_recurse_on_internal_web_interlocutor_composition(monkeypatch):
    from interface.routes import chat as chat_routes

    async def _forbidden_governed_skill(*_args, **_kwargs):
        raise AssertionError("internal composition prompt must not execute web_interlocutor")

    monkeypatch.setattr(chat_routes, "_execute_governed_live_skill", _forbidden_governed_skill)

    result = await chat_routes._execute_governed_capability_request_from_chat(
        "Compose only the exact message Aura should send to another AI in a visible "
        "browser conversation. Write naturally as Aura.\n\n"
        "You are Aura beginning a visible conversation with another AI. Ask for a "
        "critical distinction.\n\nOpening message:"
    )

    assert result is None


@pytest.mark.asyncio
async def test_web_interlocutor_deterministic_composition_requires_explicit_opt_in(monkeypatch):
    monkeypatch.setattr("asyncio.sleep", lambda *_args, **_kwargs: _instant())
    browser = FakeBrowser()
    session = WebInterlocutorSession(
        browser=browser,
        memory_gateway=FakeMemoryGateway(),
        cognitive_engine=None,
    )

    result = await session.run(
        objective="Discuss whether agency can be tested without self-report.",
        opening_message="",
        max_turns=1,
        wait_timeout_s=5,
        persist_memory=False,
        context={"allow_deterministic_composition_fallback": True},
    )

    assert result.ok is True
    assert browser.sent
    assert result.diagnostics["composition_events"]
    assert {event["source"] for event in result.diagnostics["composition_events"]} == {"deterministic_fallback"}


@pytest.mark.asyncio
async def test_web_interlocutor_skill_exposes_capability(monkeypatch):
    async def fake_run(self, **kwargs):
        from core.capabilities.capabilities.web_interlocutor import WebInterlocutorResult

        return WebInterlocutorResult(
            ok=True,
            objective=kwargs["objective"],
            learned_summary="learned",
            memory_record_id="mem-1",
            status="completed",
        )

    monkeypatch.setattr(WebInterlocutorSession, "run", fake_run)
    skill = WebInterlocutorSkill()
    result = await skill.execute(
        {"objective": "Talk to another AI", "max_turns": 1, "persist_memory": False},
        {"origin": "user"},
    )
    assert result["ok"] is True
    assert result["status"] == "completed"
    assert "visible web interlocutor" in result["summary"]


@pytest.mark.asyncio
async def test_web_interlocutor_background_job_reports_completion(monkeypatch):
    class FakeSession:
        async def run(self, **kwargs):
            return WebInterlocutorResult(
                ok=True,
                objective=kwargs["objective"],
                learned_summary="background learned",
                status="completed",
            )

    manager = WebInterlocutorJobManager(max_jobs=1)
    queued = manager.start(
        objective="Talk to another AI in the background.",
        max_turns=1,
        persist_memory=False,
        session_factory=FakeSession,
    )
    assert queued["ok"] is True
    job_id = queued["job"]["job_id"]
    for _ in range(20):
        status = manager.status(job_id)
        if status["job"]["status"] == "completed":
            break
        await asyncio.sleep(0.01)
    assert manager.status(job_id)["job"]["status"] == "completed"


@pytest.mark.asyncio
async def test_chrome_visible_browser_falls_back_to_visible_keyboard_when_dom_blocked(monkeypatch):
    class FakePyAutoGUI:
        def __init__(self):
            self.clicks = []

        def size(self):
            return (1000, 800)

        def click(self, x, y):
            self.clicks.append((x, y))

    fake_pyautogui = FakePyAutoGUI()

    def fake_get_pyautogui():
        return fake_pyautogui, None

    monkeypatch.setattr("core.skills._pyautogui_runtime.get_pyautogui", fake_get_pyautogui)
    # Hermetic: this exercises the keyboard-fallback LOGIC, not the host's AX
    # system. Replace the environment seams so it never drives a real (absent)
    # browser composer — otherwise it iterates every click candidate against
    # the live desktop for ~80s and fails headless.
    monkeypatch.setattr("asyncio.sleep", lambda *_args, **_kwargs: _instant())

    browser = ChromeVisibleDialogueBrowser()
    browser._screen_scene_targeting_enabled = False
    monkeypatch.setattr(browser._cdp, "is_available", lambda: False)
    monkeypatch.setattr(browser, "_run_chrome_js", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("Chrome JavaScript disabled")))
    monkeypatch.setattr(browser, "_dismiss_common_popups", lambda: None)
    monkeypatch.setattr(browser, "_send_escape_to_browser", lambda: None)
    monkeypatch.setattr(browser, "_activate_browser", lambda: None)
    monkeypatch.setattr(
        browser,
        "_focused_element_snapshot",
        lambda: (
            "process:Google Chrome\nAXRole:AXTextArea\n"
            "AXPlaceholderValue:Message ChatGPT\nAXPosition:500,760\nAXSize:600,80"
        ),
    )
    monkeypatch.setattr(browser, "_paste_and_submit", lambda text: {"ok": True, "text": text})

    result = await browser.send_message("Hello from Aura.")

    assert result["ok"] is True
    assert result["method"] == "visible_keyboard_click_clipboard_return"
    assert fake_pyautogui.clicks
    assert result["submission"]["text"] == "Hello from Aura."


async def _instant():
    return None
