"""Phantom Browser Module
Playwright-based "human-like" browser agent for Aura.

Capabilities:
- Dynamic Visibility: Headless (background) vs Headed (interactive)
- Human-like Interaction: Random microsleeps, typing speeds, cursor movements
- Robust Navigation: Handling broken links, backing out, reading content
- Content Extraction: Getting markdown from pages

Usage:
    browser = PhantomBrowser()
    browser.browse("https://aura.internal")
    browser.type("input[name='q']", "Hello World")
    browser.click("input[name='btnK']")
"""
import asyncio
import logging
import random
import re
from typing import Any

from core.runtime.errors import (
    DependencyUnavailable,
    FallbackClassification,
    Severity,
    record_degradation,
)
from core.runtime.runtime_hygiene import get_runtime_hygiene
from core.utils.exceptions import capture_and_log

try:
    from playwright.async_api import Error as PlaywrightError
    from playwright.async_api import TimeoutError as PlaywrightTimeoutError
    from playwright.async_api import Page, async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PlaywrightError = RuntimeError
    PlaywrightTimeoutError = TimeoutError
    Page = Any
    PLAYWRIGHT_AVAILABLE = False

try:
    from playwright_stealth import Stealth
    _STEALTH = Stealth()
    _STEALTH_IMPORT_ERROR = ""
    STEALTH_AVAILABLE = True
except (ImportError, TypeError, ValueError) as stealth_import_error:
    _STEALTH = None
    _STEALTH_IMPORT_ERROR = f"{type(stealth_import_error).__name__}: {stealth_import_error}"
    STEALTH_AVAILABLE = False

logger = logging.getLogger("PhantomBrowser")


def _record_browser_degradation(
    error: BaseException,
    *,
    stage: str,
    action: str,
    severity: Severity = "warning",
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {"stage": stage, "repair_requested": True}
    if extra:
        payload.update(extra)
    record_degradation(
        "phantom_browser",
        error,
        severity=severity,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        extra=payload,
    )

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (X11; Linux x86_64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_3_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3.1 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (iPad; CPU OS 17_3_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3.1 Mobile/15E148 Safari/604.1"
]

_CONTENT_HINT_RE = re.compile(
    r"(article|story|content|entry|main|body|markdown|post|read|chapter|prose|text)",
    re.IGNORECASE,
)
_NOISE_HINT_RE = re.compile(
    r"(nav|footer|header|menu|sidebar|cookie|share|social|comment|promo|banner|breadcrumb|related|recommend|subscribe|login|signup|advert)",
    re.IGNORECASE,
)
_NOISY_LINE_RE = re.compile(
    r"^(?:home|about|menu|privacy|terms|cookies?|share|subscribe|login|sign up|contact|next|previous|advertisement)$",
    re.IGNORECASE,
)


def _clean_extracted_page_text(raw_text: str) -> str:
    seen: set[str] = set()
    cleaned_lines: list[str] = []
    for raw_line in str(raw_text or "").splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        lower = line.lower()
        if _NOISY_LINE_RE.match(line):
            continue
        if len(line) < 18 and not re.match(r"^(chapter|part|section)\b", lower):
            continue
        if lower in seen:
            continue
        seen.add(lower)
        cleaned_lines.append(line)
    return "\n\n".join(cleaned_lines)


def _score_content_block(title: str, block: dict[str, Any]) -> float:
    text = _clean_extracted_page_text(str(block.get("text") or ""))
    if not text:
        return float("-inf")

    tag = str(block.get("tag") or "").lower()
    block_id = str(block.get("id") or "").lower()
    class_name = str(block.get("class_name") or "").lower()
    meta = " ".join(part for part in (tag, block_id, class_name) if part)
    title_tokens = set(re.findall(r"[a-z0-9]+", str(title or "").lower()))
    text_tokens = set(re.findall(r"[a-z0-9]+", text[:1200].lower()))
    token_overlap = len(title_tokens & text_tokens) / max(1, len(title_tokens)) if title_tokens else 0.0

    sentence_count = max(1, len(re.findall(r"[.!?]", text)))
    paragraph_count = int(block.get("paragraph_count") or 0)
    link_density = float(block.get("link_density") or 0.0)
    score = min(len(text) / 180.0, 8.0)
    score += min(sentence_count * 0.12, 2.0)
    score += min(paragraph_count * 0.18, 1.8)
    score += token_overlap * 1.2

    if tag in {"article", "main"}:
        score += 1.2
    if _CONTENT_HINT_RE.search(meta):
        score += 0.8
    if _NOISE_HINT_RE.search(meta):
        score -= 2.4
    score -= min(link_density, 0.8) * 3.0
    return score

class PhantomBrowser:
    """High-fidelity browser agent (Async Version).
    """
    
    def __init__(self, visible: bool = False, browser_type: str = "chromium"):
        self.playwright = None
        self.browser = None
        self.context = None
        self.page: Page | None = None
        self.visible = visible
        self.browser_type = browser_type
        self.is_active = False
        self._homeostasis = None
        self._resource_lock = None
        # Whether background work was actually told to stand down, and the
        # last admission verdict. Both are reported, because "running" and
        # "running with coordination" are different states.
        self._resource_coordinated = False
        self._last_admission: dict[str, Any] = {}
        self._startup_error = ""
        self._startup_failure_count = 0
        self._last_launch_attempts: list[str] = []
        self._stealth_applied = False
        self._stealth_error = _STEALTH_IMPORT_ERROR
        self._driver_pid: int | None = None
        self._driver_registered = False
        
        if not PLAYWRIGHT_AVAILABLE:
            return

    async def ensure_ready(self) -> bool:
        """Public lifecycle method: ensures the browser is started and ready."""
        if not self.is_active:
            await self._start_browser()
        return self.is_active

    #: Below this the host cannot afford a browser's several hundred MB and
    #: handful of processes. Deliberately generous — refusing a user-visible
    #: browse is a real cost, so this protects the machine from a launch that
    #: would push it into swap rather than being frugal for its own sake.
    MIN_AVAILABLE_GB_FOR_BROWSER = 2.0

    def _browser_admission(self) -> dict[str, Any]:
        """Can this host afford to start a browser right now?

        Answers unknown-as-admit on purpose: if memory cannot be measured,
        refusing every browse would break the capability wholesale on any
        platform without the monitor. The check exists to catch a MEASURED
        shortage, which is the case that actually hurt.
        """
        try:
            from core.utils.memory_monitor import get_memory_pressure_snapshot

            snapshot = get_memory_pressure_snapshot()
            available_gb = float(snapshot.available_gb)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "phantom_browser",
                exc,
                severity="info",
                action="admitted the browser because memory pressure could not be measured",
                enforce_failure_policy=False,
            )
            return {"can_admit": True, "reason": "pressure_unmeasured", "available_gb": None}
        if available_gb < self.MIN_AVAILABLE_GB_FOR_BROWSER:
            return {
                "can_admit": False,
                "reason": f"insufficient_memory:{available_gb:.1f}GB_available",
                "available_gb": available_gb,
            }
        return {"can_admit": True, "reason": "", "available_gb": available_gb}

    def get_status(self) -> dict[str, Any]:
        return {
            "active": self.is_active,
            "visible": self.visible,
            "browser_type": self.browser_type,
            "startup_failure_count": self._startup_failure_count,
            "startup_error": self._startup_error[:240],
            "last_launch_attempts": list(self._last_launch_attempts),
            "stealth_available": bool(STEALTH_AVAILABLE),
            "stealth_applied": bool(self._stealth_applied),
            "stealth_error": self._stealth_error[:240],
            "driver_pid": self._driver_pid,
            "driver_registered": bool(self._driver_registered),
            # A browser running WITHOUT resource coordination is a different
            # state from one running with it; reporting only "active" made
            # them look identical.
            "resource_coordinated": bool(self._resource_coordinated),
            "last_admission": dict(self._last_admission),
        }

    async def _start_browser(self) -> bool:
        """Start the Playwright browser asynchronously"""
        try:
            if self.is_active:
                return True

            if not PLAYWRIGHT_AVAILABLE:
                error = DependencyUnavailable("playwright is not installed")
                self._startup_failure_count += 1
                self._startup_error = str(error)
                _record_browser_degradation(
                    error,
                    stage="dependency_check",
                    action="kept phantom browser inactive because Playwright is unavailable",
                    severity="degraded",
                    extra={"browser_type": self.browser_type},
                )
                return False

            # CP126 (medium): "Resource-lock failure is explicitly fail-open.
            # Browser startup continues after homeostatic resource
            # coordination cannot be acquired. There is no admission
            # decision, resource budget, or later reconciliation, so memory-
            # or latency-sensitive runtime periods can still start a full
            # browser while status presents normal readiness."
            #
            # The missing piece was the admission decision, not the lock. A
            # browser is hundreds of megabytes and several processes, and it
            # was launched without anyone asking whether the machine could
            # afford one — on a host already holding a ~20GB resident model.
            admission = self._browser_admission()
            self._last_admission = admission
            if not admission["can_admit"]:
                self._startup_error = f"admission_refused:{admission['reason']}"
                _record_browser_degradation(
                    RuntimeError(f"browser admission refused: {admission['reason']}"),
                    stage="admission",
                    action="refused to start a browser while the host could not afford one",
                    severity="warning",
                    extra={"available_gb": admission.get("available_gb")},
                )
                return False

            # Signal resource lock — heavy background tasks will pause.
            try:
                from core.utils.resource_lock import get_resource_lock
                self._resource_lock = get_resource_lock()
                self._resource_lock.begin_browser_session()
                self._resource_coordinated = True
            except (ImportError, AttributeError, RuntimeError) as lock_exc:
                # Continuing is right — the lock is a courtesy signal to
                # background work, and refusing to browse because a
                # coordination helper is missing would be over-strict. What
                # was wrong is that status then claimed normal readiness, so
                # the uncoordinated state is now reported.
                _record_browser_degradation(
                    lock_exc,
                    stage="resource_lock",
                    action="continued browser startup without resource-lock coordination",
                    severity="warning",
                )
                self._resource_lock = None
                self._resource_coordinated = False

            self.playwright = await async_playwright().start()
            self._register_playwright_driver()

            # RESILIENCE: Build a fallback cascade of browser types.
            # If the configured browser (e.g. Firefox) isn't installed,
            # fall back to chromium which is the most reliably available.
            browser_attempts = [self.browser_type]
            if self.browser_type != "chromium":
                browser_attempts.append("chromium")

            launch_error = None
            self._last_launch_attempts = []
            for bt in browser_attempts:
                self._last_launch_attempts.append(bt)
                try:
                    if bt == "firefox":
                        self.browser = await self.playwright.firefox.launch(headless=not self.visible)
                    elif bt == "webkit":
                        self.browser = await self.playwright.webkit.launch(headless=not self.visible)
                    else:
                        self.browser = await self.playwright.chromium.launch(
                            headless=not self.visible,
                            args=['--disable-blink-features=AutomationControlled']
                        )
                    if bt != self.browser_type:
                        logger.info("✓ Fell back to %s after %s was unavailable.", bt, self.browser_type)
                    launch_error = None
                    break  # Launch succeeded
                except (
                    PlaywrightError,
                    PlaywrightTimeoutError,
                    RuntimeError,
                    AttributeError,
                    TypeError,
                    ValueError,
                ) as launch_exc:
                    self._startup_failure_count += 1
                    self._startup_error = f"{bt}: {launch_exc}"
                    _record_browser_degradation(
                        launch_exc,
                        stage="browser_launch",
                        action="trying next browser fallback after launch attempt failed",
                        severity="warning",
                        extra={
                            "attempted_browser": bt,
                            "configured_browser": self.browser_type,
                            "attempts": list(self._last_launch_attempts),
                        },
                    )
                    launch_error = launch_exc
                    logger.warning("Browser %s failed to launch: %s. Trying next fallback...", bt, launch_exc)

            if launch_error or self.browser is None:
                raise launch_error or RuntimeError("All browser types failed to launch.")

            user_agent = self._get_random_ua()

            self.context = await self.browser.new_context(
                viewport={'width': 1280, 'height': 800},
                user_agent=user_agent
            )
            await self._apply_stealth(self.context)
            self.page = await self.context.new_page()

            self.is_active = True
            self._startup_error = ""
            logger.info("✓ Phantom Browser initialized (Visible: %s, UA: %s...)", self.visible, user_agent[:30])
            return True
        except (
            ImportError,
            PlaywrightError,
            PlaywrightTimeoutError,
            AttributeError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as e:
            self._startup_failure_count += 1
            self._startup_error = f"{type(e).__name__}: {e}"
            _record_browser_degradation(
                e,
                stage="startup",
                action="marked phantom browser inactive and released startup resources after startup failed",
                severity="degraded",
                extra={
                    "browser_type": self.browser_type,
                    "attempts": list(self._last_launch_attempts),
                },
            )
            logger.error("Failed to start browser: %s", e)
            self.is_active = False
            # Release resource lock on failure
            self._release_resource_lock()
            if self.playwright is not None:
                try:
                    await asyncio.wait_for(self.playwright.stop(), timeout=5.0)
                except (RuntimeError, AttributeError, TypeError, ValueError, TimeoutError) as stop_exc:
                    _record_browser_degradation(
                        stop_exc,
                        stage="startup_cleanup",
                        action="left startup cleanup after Playwright stop failed",
                        severity="warning",
                    )
                finally:
                    self.playwright = None
            return False

    def _get_random_ua(self) -> str:
        return random.choice(USER_AGENTS)

    async def _apply_stealth(self, context: Any) -> bool:
        """Apply the installed playwright-stealth API before creating pages."""
        self._stealth_applied = False
        if not STEALTH_AVAILABLE or _STEALTH is None:
            self._stealth_error = _STEALTH_IMPORT_ERROR or "dependency_unavailable"
            logger.warning(
                "playwright-stealth unavailable: %s",
                self._stealth_error,
            )
            return False
        try:
            await _STEALTH.apply_stealth_async(context)
        except (
            PlaywrightError,
            PlaywrightTimeoutError,
            RuntimeError,
            AttributeError,
            TypeError,
            ValueError,
        ) as exc:
            self._stealth_error = f"{type(exc).__name__}: {exc}"
            _record_browser_degradation(
                exc,
                stage="stealth_setup",
                action="continued with standard browser context after stealth setup failed",
                severity="warning",
                extra={"browser_type": self.browser_type},
            )
            logger.warning("Stealth application failed: %s", exc)
            return False
        self._stealth_applied = True
        self._stealth_error = ""
        return True

    def _register_playwright_driver(self) -> bool:
        """Bind Playwright's asyncio-owned driver to Aura's process owner."""
        self._driver_pid = None
        self._driver_registered = False
        try:
            driver = self.playwright._impl_obj._connection._transport._proc
            pid = int(getattr(driver, "pid", 0) or 0)
            if pid <= 0:
                raise RuntimeError("playwright driver pid unavailable")
            get_runtime_hygiene().register_process_handle(
                driver,
                kind="subprocess",
                name="playwright.driver",
                source="core.capabilities.phantom_browser",
                command="playwright driver",
            )
        except (
            AttributeError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            _record_browser_degradation(
                exc,
                stage="driver_registration",
                action="kept browser active but exposed missing driver ownership evidence",
                severity="warning",
                extra={"browser_type": self.browser_type},
            )
            return False
        self._driver_pid = pid
        self._driver_registered = True
        return True

    async def rotate_user_agent(self):
        """Switch to a new context with a different user agent."""
        if not self.is_active:
            await self._start_browser()
            return
            
        logger.info("🔄 Rotating User Agent...")
        ua = self._get_random_ua()
        
        # We need a new context to change the UA effectively
        new_context = await self.browser.new_context(
            viewport={'width': 1280, 'height': 800},
            user_agent=ua
        )
        await self._apply_stealth(new_context)
        old_context = self.context
        old_page = self.page
        self.context = new_context
        self.page = await self.context.new_page()
        
        # Properly close old page and context to prevent resource leaks
        if old_page:
            await self._close_resource("old page", old_page.close, close_timeout=3.0)
        if old_context:
            await self._close_resource("old context", old_context.close, close_timeout=5.0)
        logger.info("✓ User Agent rotated to: %s...", ua[:30])

    async def is_blocked(self) -> bool:
        """Detect if we are hitting a bot-detection page or CAPTCHA."""
        if not self.page:
            return False
        
        content = (await self.page.content()).lower()
        title = (await self.page.title()).lower()
        
        block_signals = [
            "unusual traffic from your computer network",
            "not a robot",
            "captcha",
            "verify you are a human",
            "access to this page has been denied",
            "security check",
            "bot detection",
            "automated requests"
        ]
        
        for signal in block_signals:
            if signal in content or signal in title:
                logger.warning("🚨 Browser Blocked Detected: %s", signal)
                return True
        return False

    async def set_visibility(self, visible: bool):
        """Toggle visibility (requires restart)"""
        if self.visible != visible:
            logger.info("Switching visibility: %s -> %s", self.visible, visible)
            self.visible = visible
            await self.close()
            await self._start_browser()

    async def browse(self, url: str) -> bool:
        """Navigate to a URL"""
        if not self.is_active: 
            await self._start_browser()
            if not self.is_active:
                logger.error("Browser failed to start.")
                return False
            
        if not url.startswith('http'):
            url = 'https://' + url
            
        logger.info("🌐 Navigating to: %s", url)
        try:
            await self.page.goto(url, timeout=30000, wait_until='domcontentloaded')
            await self._human_delay(1, 2)
            return True
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation('phantom_browser', e)
            logger.error("Navigation failed: %s", e)
            return False

    async def click(
        self,
        selector: str | None = None,
        text_match: str | None = None,
    ) -> bool:
        """Human-like click on an element with enhanced robustness."""
        try:
            element = None
            if text_match:
                # Try multiple ways to find text (case-insensitive, contains)
                selectors = [
                    f"text='{text_match}'",
                    f"text=\"{text_match}\"",
                    f"a:has-text('{text_match}')",
                    f"button:has-text('{text_match}')",
                    f"*[role='button']:has-text('{text_match}')"
                ]
                for s in selectors:
                    try:
                        loc = self.page.locator(s).first
                        if await loc.is_visible(timeout=2000):
                            element = loc
                            break
                    except PlaywrightError:
                        continue
                
                if not element:
                    # Fallback to get_by_text with regex for fuzzy match
                    import re
                    try:
                        loc = self.page.get_by_text(re.compile(text_match, re.IGNORECASE)).first
                        if await loc.is_visible(timeout=2000):
                            element = loc
                    except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                        record_degradation('phantom_browser', exc, severity="debug", action="fuzzy text selector failed")
                        logger.debug("Fuzzy text selector failed: %s", exc)
            elif selector:
                element = self.page.locator(selector).first

            if element and await element.is_visible():
                # Scroll into view if needed
                await element.scroll_into_view_if_needed()
                await self._human_delay(0.2, 0.5)
                
                # Human-like mouse movement
                box = await element.bounding_box()
                if box:
                    x = box['x'] + box['width'] / 2 + random.randint(-5, 5)
                    y = box['y'] + box['height'] / 2 + random.randint(-3, 3)
                    await self.page.mouse.move(x, y, steps=15)
                    await self._human_delay(0.1, 0.3)
                
                await element.click()
                logger.info("🖱️ Clicked: %s", selector or text_match)
                await self._human_delay(0.5, 1.5)
                return True
            else:
                logger.warning("Element not found or not visible: %s", selector or text_match)
                return False
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('phantom_browser', e)
            logger.error("Click failed: %s", e)
            return False

    async def type(self, selector: str, text: str) -> bool:
        """Human-like typing"""
        try:
            if not await self.click(selector):
                return False
            
            logger.info("Typing %d characters into selector %s", len(text), selector)
            for char in text:
                await self.page.keyboard.type(char)
                # Random typing delay between keystrokes
                await asyncio.sleep(random.uniform(0.05, 0.15))
            
            await self._human_delay(0.5, 1.0)
            return True
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation('phantom_browser', e)
            logger.error("Typing failed: %s", e)
            return False

    async def scroll(self, direction: str = "down", amount: int = 500) -> bool:
        """Scroll the page in a human-like manner."""
        try:
            steps = 5
            step_amount = amount // steps
            for _ in range(steps):
                if direction == "down":
                    await self.page.mouse.wheel(0, step_amount)
                else:
                    await self.page.mouse.wheel(0, -step_amount)
                await asyncio.sleep(random.uniform(0.1, 0.3))
            await self._human_delay(0.5, 1.0)
            return True
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation('phantom_browser', e)
            logger.error("Scroll failed: %s", e)
            return False

    async def read_content(self) -> str:
        """Extract page content by scoring likely article/content containers."""
        try:
            if not self.page:
                return ""
            
            title = await self.page.title()

            candidate_blocks = await self.page.evaluate("""() => {
                function isVisible(el) {
                    const style = window.getComputedStyle(el);
                    const rect = el.getBoundingClientRect();
                    return style.display !== 'none' &&
                        style.visibility !== 'hidden' &&
                        style.opacity !== '0' &&
                        rect.width > 0 &&
                        rect.height > 0;
                }

                function collectCandidates() {
                    const selectors = [
                        'article',
                        'main',
                        '[role="main"]',
                        '[itemprop="articleBody"]',
                        '.article',
                        '.article-body',
                        '.entry-content',
                        '.post-content',
                        '.story-content',
                        '.story-body',
                        '.main-content',
                        '.content',
                        'section',
                        'body',
                    ];
                    const seen = new Set();
                    const blocks = [];
                    for (const selector of selectors) {
                        const nodes = Array.from(document.querySelectorAll(selector));
                        for (const el of nodes) {
                            if (!el || seen.has(el) || !isVisible(el)) continue;
                            const text = (el.innerText || '').replace(/\\u00a0/g, ' ').trim();
                            if (text.length < 80) continue;
                            seen.add(el);
                            const linkText = Array.from(el.querySelectorAll('a'))
                                .map(a => (a.innerText || '').trim())
                                .join(' ');
                            blocks.push({
                                tag: (el.tagName || '').toLowerCase(),
                                id: (el.id || ''),
                                class_name: (el.className || '').toString(),
                                text,
                                paragraph_count: el.querySelectorAll('p, li, blockquote').length,
                                heading_count: el.querySelectorAll('h1, h2, h3, h4').length,
                                link_density: text.length ? (linkText.length / text.length) : 0,
                            });
                        }
                    }
                    return blocks.slice(0, 24);
                }

                return collectCandidates();
            }""")

            best_text = ""
            best_score = float("-inf")
            for block in list(candidate_blocks or []):
                if not isinstance(block, dict):
                    continue
                score = _score_content_block(title, block)
                text = _clean_extracted_page_text(str(block.get("text") or ""))
                if text and score > best_score:
                    best_score = score
                    best_text = text

            if len(best_text) < 200:
                fallback_text = await self.page.evaluate("() => document.body.innerText")
                fallback_text = _clean_extracted_page_text(str(fallback_text or ""))
                if len(fallback_text) > len(best_text):
                    best_text = fallback_text

            return f"# {title}\n\n{best_text[:60000]}"
        except (OSError, ConnectionError, TimeoutError) as e:
            record_degradation('phantom_browser', e)
            logger.error("Read content failed: %s", e)
            return ""

    async def get_links(self) -> list[dict[str, str]]:
        """Extract all links from page"""
        try:
            if not self.page:
                return []
            links = await self.page.evaluate("""() => {
                return Array.from(document.querySelectorAll('a')).map(a => ({
                    text: a.innerText.trim(),
                    url: a.href
                })).filter(l => l.text && l.url)
            }""")
            return links
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation('phantom_browser', e)
            logger.error("Get links failed: %s", e)
            return []

    async def screenshot(self) -> str | None:
        """Take a screenshot (base64)"""
        try:
            if not self.page:
                return None
            import base64
            bytes_data = await self.page.screenshot()
            return base64.b64encode(bytes_data).decode('utf-8')
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('phantom_browser', e)
            logger.error("Screenshot failed: %s", e)
            return None

    async def _close_resource(self, label: str, close_factory, *, close_timeout: float) -> None:
        try:
            await asyncio.wait_for(close_factory(), timeout=close_timeout)
        except (RuntimeError, asyncio.CancelledError, TimeoutError, AttributeError) as exc:
            record_degradation("phantom_browser", exc)
            logger.debug("Phantom browser %s close failed: %s", label, exc)

    async def close(self):
        """Close browser resources with per-step timeouts to prevent hangs."""
        if self.page:
            await self._close_resource("page", self.page.close, close_timeout=3.0)
        if self.context:
            await self._close_resource("context", self.context.close, close_timeout=5.0)
        if self.browser:
            await self._close_resource("browser", self.browser.close, close_timeout=5.0)
        if self.playwright:
            await self._close_resource("playwright", self.playwright.stop, close_timeout=5.0)
        self.page = None
        self.context = None
        self.browser = None
        self.playwright = None
        self._driver_pid = None
        self._driver_registered = False
        self.is_active = False
        self._release_resource_lock()
        logger.info("Browser closed")

    def _release_resource_lock(self):
        """Release the resource lock so background tasks can resume."""
        lock = getattr(self, '_resource_lock', None)
        if lock:
            lock.end_browser_session()
            self._resource_lock = None
        self._resource_coordinated = False

    async def _human_delay(self, min_s=0.5, max_s=1.5):
        """Random delay to simulate human pause, modulated by homeostasis."""
        if self._homeostasis is None:
            try:
                from core.container import ServiceContainer
                self._homeostasis = ServiceContainer.get("homeostatic_coupling", default=None)
            except (ImportError, AttributeError, RuntimeError) as e:
                record_degradation('phantom_browser', e)
                capture_and_log(e, {'module': __name__})
        
        delay_mod = 1.0
        if self._homeostasis:
            mods = self._homeostasis.get_modifiers()
            # Exhaustion (low vitality) makes her move SLOWER
            if mods.overall_vitality < 0.4:
                delay_mod = 2.5 # Significant fatigue delay
            elif mods.overall_vitality < 0.7:
                delay_mod = 1.5 # Mild fatigue delay
            
            # Urgency makes her move FASTER
            if mods.urgency_flag:
                delay_mod *= 0.6
                
        await asyncio.sleep(random.uniform(min_s, max_s) * delay_mod)

# Integration Helper
async def integrate_phantom_browser(orchestrator):
    """Integrate Phantom Browser into Orchestrator"""
    pb = PhantomBrowser(visible=False)
    await pb.ensure_ready()
    orchestrator.phantom_browser = pb
    logger.info("✅ Phantom Browser integrated")
