#!/usr/bin/env python3
"""Render the shipped Aura shell and prove skill readiness is visible and safe."""

from __future__ import annotations

import functools
import http.server
import json
import threading
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
STATIC_ROOT = ROOT / "interface" / "static"
STATIC_SERVER_ROOT = STATIC_ROOT.parent


class _QuietStaticHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, _format: str, *_args: Any) -> None:
        return


def audit_skill_readiness_ui() -> dict[str, Any]:
    failures: list[str] = []
    report: dict[str, Any] = {
        "failures": failures,
        "schema": "aura.skill_readiness_ui_audit.v1",
    }
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        return {
            **report,
            "failures": [f"playwright_unavailable:{exc}"],
            "ok": False,
        }

    handler = functools.partial(_QuietStaticHandler, directory=str(STATIC_SERVER_ROOT))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            context = browser.new_context(
                service_workers="block",
                viewport={"width": 1440, "height": 1000},
            )
            page = context.new_page()
            page.route(
                "**/api/**",
                lambda route: route.fulfill(
                    status=200,
                    content_type="application/json",
                    body="{}",
                ),
            )
            page.goto(
                f"http://127.0.0.1:{server.server_port}/static/index.html",
                wait_until="domcontentloaded",
                timeout=30_000,
            )
            page.wait_for_function("typeof renderToolCatalog === 'function'")
            page.evaluate(
                """
                () => {
                    const pane = document.getElementById('pane-skills');
                    pane.classList.add('active');
                    pane.style.display = 'block';
                    window.__catalog_xss = false;
                    renderToolCatalog(
                        [{name: 'clock', available: true, active: true, state: 'READY'}],
                        {
                            ready: false,
                            reason: 'catalog_incomplete',
                            missing_live: ['memory_sync'],
                            quarantined_count: 1,
                            quarantined: [{
                                name: 'self_modify',
                                stage: 'constructor',
                                error: '<img src=x onerror="window.__catalog_xss=true"> dependency unavailable'
                            }],
                            execution_preflight: {
                                complete: true,
                                ok: false,
                                failed: ['self_modify']
                            }
                        }
                    );
                }
                """
            )
            blocked = page.evaluate(
                """
                () => {
                    const issues = document.getElementById('tool-catalog-issues');
                    return {
                        catalogState: document.getElementById('tool-catalog-state').textContent,
                        preflightState: document.getElementById('tool-preflight-state').textContent,
                        detail: document.getElementById('tool-catalog-detail').textContent,
                        issuesHidden: issues.hidden,
                        issuesText: issues.textContent,
                        issueCount: issues.querySelectorAll('.tool-catalog-issue').length,
                        injectedImageCount: issues.querySelectorAll('img').length,
                        xssExecuted: window.__catalog_xss === true
                    };
                }
                """
            )
            if blocked["catalogState"] != "BLOCKED":
                failures.append("blocked_catalog_state_not_visible")
            if blocked["preflightState"] != "FAILED":
                failures.append("failed_preflight_state_not_visible")
            for expected in (
                "memory_sync",
                "self_modify",
                "constructor",
                "dependency unavailable",
            ):
                if expected not in blocked["issuesText"]:
                    failures.append(f"issue_detail_missing:{expected}")
            if blocked["issuesHidden"] or blocked["issueCount"] != 3:
                failures.append("issue_ledger_not_rendered")
            if blocked["injectedImageCount"] or blocked["xssExecuted"]:
                failures.append("catalog_issue_html_not_escaped")

            page.set_viewport_size({"width": 390, "height": 844})
            mobile = page.evaluate(
                """
                () => {
                    const issues = document.getElementById('tool-catalog-issues');
                    const rows = [...issues.querySelectorAll('.tool-catalog-issue')];
                    return {
                        containerClientWidth: issues.clientWidth,
                        containerScrollWidth: issues.scrollWidth,
                        rowOverflow: rows.some(row => row.scrollWidth > row.clientWidth + 1),
                        templates: rows.map(row => getComputedStyle(row).gridTemplateColumns)
                    };
                }
                """
            )
            if mobile["containerScrollWidth"] > mobile["containerClientWidth"] + 1:
                failures.append("mobile_issue_container_overflow")
            if mobile["rowOverflow"]:
                failures.append("mobile_issue_row_overflow")

            page.evaluate(
                """
                () => renderToolCatalog(
                    [{name: 'clock', available: true, active: true, state: 'READY'}],
                    {
                        ready: true,
                        reason: 'ready',
                        missing_live: [],
                        quarantined_count: 0,
                        quarantined: [],
                        execution_preflight: {complete: true, ok: true, failed: []}
                    }
                )
                """
            )
            recovered = page.evaluate(
                """
                () => ({
                    catalogState: document.getElementById('tool-catalog-state').textContent,
                    preflightState: document.getElementById('tool-preflight-state').textContent,
                    issuesHidden: document.getElementById('tool-catalog-issues').hidden,
                    issuesText: document.getElementById('tool-catalog-issues').textContent
                })
                """
            )
            if recovered != {
                "catalogState": "READY",
                "preflightState": "VERIFIED",
                "issuesHidden": True,
                "issuesText": "",
            }:
                failures.append("ready_recovery_state_incorrect")

            report.update(
                {
                    "blocked": blocked,
                    "browser": "chromium",
                    "mobile": mobile,
                    "recovered": recovered,
                }
            )
            context.close()
            browser.close()
    except Exception as exc:
        failures.append(f"browser_audit_failed:{type(exc).__name__}:{exc}")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    report["ok"] = not failures
    return report


def main() -> int:
    report = audit_skill_readiness_ui()
    print(
        "AURA_SKILL_READINESS_UI_AUDIT="
        + json.dumps(report, separators=(",", ":"), sort_keys=True)
    )
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
