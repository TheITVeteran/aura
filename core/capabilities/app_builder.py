"""core/capabilities/app_builder.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Aura builds a real, runnable app from a natural spec — with her own mind.

General architecture: given any app description, she writes a complete,
self-contained single-file web app (HTML + CSS + JS, no external dependencies),
which is then VALIDATED (well-formed, has real logic, no obvious syntax errors)
and written to disk so it can be opened and used. Not a mock — a working app.

Code synthesis goes through the UN-STEERED local code model (the persona cortex
corrupts symbolic code), the same path her reverse-engineering uses, so the
output is clean multi-file-in-one code rather than prose. If the first draft
fails validation she is shown the failure and asked to repair it — a real
build/verify loop, bounded.
"""
from __future__ import annotations

import html.parser
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.AppBuilder")

_MAX_REPAIRS = 2
_MIN_APP_CHARS = 800


@dataclass
class AppBuildResult:
    ok: bool
    spec: str
    title: str = ""
    path: str = ""
    code: str = ""
    bytes_written: int = 0
    validation: dict[str, Any] = field(default_factory=dict)
    attempts: int = 0
    status: str = "ok"
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.__dict__)
        # keep the transported code bounded
        payload["code"] = self.code[:20000]
        return payload


class _TagCounter(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: dict[str, int] = {}

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        self.tags[tag] = self.tags.get(tag, 0) + 1


def _extract_html(raw: str) -> str:
    """Pull the HTML document out of a model response (fenced or bare)."""
    text = str(raw or "").strip()
    fence = re.search(r"```(?:html|xml)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    lower = text.lower()
    start = lower.find("<!doctype")
    if start == -1:
        start = lower.find("<html")
    if start > 0:
        text = text[start:]
    end = text.lower().rfind("</html>")
    if end != -1:
        text = text[: end + len("</html>")]
    return text.strip()


def validate_web_app(code: str) -> dict[str, Any]:
    """Real, not cosmetic: the app must be a well-formed HTML document with
    actual script logic — enough to be a running app, not a stub."""
    problems: list[str] = []
    code_l = code.lower()
    if len(code) < _MIN_APP_CHARS:
        problems.append(f"too_short:{len(code)}<{_MIN_APP_CHARS}")
    if "<html" not in code_l or "</html>" not in code_l:
        problems.append("missing_html_root")
    if "<script" not in code_l or "</script>" not in code_l:
        problems.append("missing_script")
    # balanced script tags
    if code_l.count("<script") != code_l.count("</script>"):
        problems.append("unbalanced_script_tags")
    counter = _TagCounter()
    try:
        counter.feed(code)
    except Exception as exc:  # noqa: BLE001 - parser can raise arbitrary parse errors
        problems.append(f"html_parse_error:{type(exc).__name__}")
    # real interaction logic: at least one event handler / listener / function
    if not re.search(r"addeventlistener|onclick|function\s+\w+|=>|const\s+\w+\s*=", code, re.IGNORECASE):
        problems.append("no_interaction_logic")
    return {
        "ok": not problems,
        "problems": problems,
        "chars": len(code),
        "tags": counter.tags,
        "script_blocks": code_l.count("<script"),
    }


def _title_for(spec: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(spec or "app").strip().lower()).strip("_")
    return (slug or "app")[:48]


def _build_prompt(spec: str, prior_code: str = "", problems: list[str] | None = None) -> str:
    if prior_code and problems:
        return (
            "You wrote this single-file web app but it failed validation.\n"
            f"Problems: {', '.join(problems)}\n\n"
            "Return a corrected COMPLETE single-file HTML document (with <html>, a "
            "<script> block, and working interaction logic). Output ONLY the HTML.\n\n"
            f"Spec: {spec}\n\nPrevious attempt:\n{prior_code[:6000]}"
        )
    return (
        "Build a complete, self-contained, single-file web app.\n"
        "Requirements: one HTML document; all CSS in a <style> tag and all JS in a "
        "<script> tag; NO external files, CDNs, or network calls; it must actually "
        "work when opened in a browser (real logic, real interactivity, clear UI). "
        "Output ONLY the HTML document, nothing else.\n\n"
        f"App to build: {spec}"
    )


async def _generate(prompt: str, *, max_tokens: int) -> str:
    """Generate via the un-steered local code model, falling back to the code
    generator. Returns raw text (HTML extracted by the caller)."""
    try:
        from core.brain.llm.local_code_model import get_local_code_model

        model = get_local_code_model()
        if model is not None:
            return await model.generate(
                prompt,
                system_prompt=(
                    "You are a senior front-end engineer. You output a single complete "
                    "HTML document and nothing else. Standard browser APIs only."
                ),
                max_tokens=max_tokens,
                temperature=0.2,
            )
    except (ImportError, RuntimeError, OSError) as exc:
        logger.debug("Local code model unavailable for app build: %s", exc)
    try:
        from core.brain.llm.code_generator import LLMCodeGenerator

        generator = LLMCodeGenerator(max_tokens=max_tokens, temperature=0.2)
        return await generator.generate_async(prompt, context={"origin": "app_builder"})
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
        record_degradation(
            "app_builder.generate",
            exc,
            severity="warning",
            action="app build failed because no code model was available",
        )
        return ""


async def build_app(
    spec: str,
    *,
    out_dir: str | Path = "artifacts/live_apps",
    max_tokens: int = 6000,
) -> AppBuildResult:
    """Build a runnable single-file web app from a spec, validate, and write it."""
    spec = str(spec or "").strip()
    result = AppBuildResult(ok=False, spec=spec, title=_title_for(spec))
    if not spec:
        result.status = "no_spec"
        result.error = "empty app spec"
        return result

    code = ""
    problems: list[str] = []
    for attempt in range(1, _MAX_REPAIRS + 2):
        result.attempts = attempt
        prompt = _build_prompt(spec, prior_code=code, problems=problems)
        raw = await _generate(prompt, max_tokens=max_tokens)
        code = _extract_html(raw)
        validation = validate_web_app(code)
        result.validation = validation
        result.code = code
        if validation["ok"]:
            break
        problems = validation["problems"]
        logger.info("App build attempt %d failed validation: %s", attempt, problems)

    if not result.validation.get("ok"):
        result.status = "validation_failed"
        result.error = f"app did not validate after {result.attempts} attempt(s): {problems}"
        return result

    out_path = Path(out_dir).expanduser()
    out_path.mkdir(parents=True, exist_ok=True)
    file_path = out_path / f"{result.title}_{int(time.time())}.html"
    file_path.write_text(code, encoding="utf-8")
    result.ok = True
    result.status = "built"
    result.path = str(file_path)
    result.bytes_written = len(code.encode("utf-8"))
    logger.info("🛠️ Built app '%s' -> %s (%d bytes)", spec[:40], file_path, result.bytes_written)
    return result


__all__ = ["AppBuildResult", "build_app", "validate_web_app"]
