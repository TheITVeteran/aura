import asyncio

import pytest

from core.capabilities.web_interlocutor import (
    BrowserPageSnapshot,
    ChromeVisibleDialogueBrowser,
    WebInterlocutorResult,
    WebInterlocutorJobManager,
    WebInterlocutorSession,
    _extract_new_interlocutor_text,
    _url_allows_readability_fallback,
)
from core.runtime.gateways import MemoryWriteReceipt
from core.skills.web_interlocutor import WebInterlocutorSkill


class FakeBrowser:
    def __init__(self):
        self.sent = []
        self.snapshots = [
            BrowserPageSnapshot(
                url="https://example.test/chat",
                title="Example Chat",
                text="Example Chat\nMessage box",
                editable_count=1,
            ),
            BrowserPageSnapshot(
                url="https://example.test/chat",
                title="Example Chat",
                text=(
                    "Example Chat\nMessage box\n"
                    "Aura: hello\n"
                    "Interlocutor: Novel thought needs tension between analogy and verification."
                ),
                editable_count=1,
            ),
            BrowserPageSnapshot(
                url="https://example.test/chat",
                title="Example Chat",
                text=(
                    "Example Chat\nMessage box\n"
                    "Aura: hello\n"
                    "Interlocutor: Novel thought needs tension between analogy and verification."
                ),
                editable_count=1,
            ),
            BrowserPageSnapshot(
                url="https://example.test/chat",
                title="Example Chat",
                text=(
                    "Example Chat\nMessage box\n"
                    "Aura: hello\n"
                    "Interlocutor: Novel thought needs tension between analogy and verification.\n"
                    "Aura: Can you give one concrete example?\n"
                    "Interlocutor: Compare a city immune system to software observability, then test where the analogy breaks."
                ),
                editable_count=1,
            ),
            BrowserPageSnapshot(
                url="https://example.test/chat",
                title="Example Chat",
                text=(
                    "Example Chat\nMessage box\n"
                    "Aura: hello\n"
                    "Interlocutor: Novel thought needs tension between analogy and verification.\n"
                    "Aura: Can you give one concrete example?\n"
                    "Interlocutor: Compare a city immune system to software observability, then test where the analogy breaks."
                ),
                editable_count=1,
            ),
        ]

    async def open_or_attach(self, url):
        return self.snapshots[0]

    async def snapshot(self):
        if len(self.snapshots) > 1:
            return self.snapshots.pop(1)
        return self.snapshots[0]

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
            return "Can you give one concrete example?"
        return "I learned that useful novelty is not free association; it is analogy constrained by verification."


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
        opening_message="hello",
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


@pytest.mark.asyncio
async def test_web_interlocutor_skill_exposes_capability(monkeypatch):
    async def fake_run(self, **kwargs):
        from core.capabilities.web_interlocutor import WebInterlocutorResult

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

    browser = ChromeVisibleDialogueBrowser()
    monkeypatch.setattr(browser._cdp, "is_available", lambda: False)
    monkeypatch.setattr(browser, "_run_chrome_js", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("Chrome JavaScript disabled")))
    monkeypatch.setattr(browser, "_dismiss_common_popups", lambda: None)
    monkeypatch.setattr(browser, "_paste_and_submit", lambda text: {"ok": True, "text": text})

    result = await browser.send_message("Hello from Aura.")

    assert result["ok"] is True
    assert result["method"] == "visible_keyboard_click_clipboard_return"
    assert fake_pyautogui.clicks
    assert result["submission"]["text"] == "Hello from Aura."


async def _instant():
    return None
