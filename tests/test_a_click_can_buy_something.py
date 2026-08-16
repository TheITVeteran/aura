"""CP126 ``core/capabilities/phantom_browser.py`` — fifteen findings, three critical.

The object drives a real browser. `browse`, `click`, `type`, `scroll`,
`screenshot` and content extraction were plain methods returning booleans,
and CP126 ``a66d2e59`` found no principal, no scoped authority, no
site-or-action policy, no approval lease and no receipt behind any of
them. A click can buy something, send something, or delete something, and
the caller got back `True`.

``8bf8d32e``: `browse` checked whether the string started with the letters
"http" and otherwise prefixed "https". No parse, no scheme restriction, no
credential rejection, no private-address exclusion, no rebinding defence,
no port policy — every one of which already existed in
`core/runtime/url_policy.py`, which the browser never called.

``c3f1668d``: one mutable page shared by every caller.
"""

from __future__ import annotations

import pytest

from core.capabilities.browser_authority import (
    BrowserAction,
    authorize_browser_action,
    issue_browser_lease,
    origin_of,
    revoke_browser_lease,
)
from core.runtime.url_policy import validate_browser_url


# ── a66d2e59: nothing drives a browser anonymously ──────────────────────────


def test_an_anonymous_caller_cannot_drive_the_browser():
    for action in BrowserAction:
        verdict = authorize_browser_action(action, principal="", url="https://example.com")
        assert verdict.allowed is False, f"{action.value} was permitted with no principal"
        assert "principal" in verdict.reason


def test_reading_needs_only_a_principal():
    verdict = authorize_browser_action(
        BrowserAction.NAVIGATE, principal="bryan", url="https://example.com"
    )
    assert verdict.allowed is True


def test_clicking_needs_a_lease():
    verdict = authorize_browser_action(
        BrowserAction.CLICK, principal="bryan", url="https://example.com/checkout"
    )
    assert verdict.allowed is False
    assert "lease" in verdict.reason, (
        "a click that can complete a purchase was permitted on a principal alone"
    )


def test_typing_needs_a_lease():
    verdict = authorize_browser_action(
        BrowserAction.TYPE, principal="bryan", url="https://example.com/login"
    )
    assert verdict.allowed is False


def test_a_lease_authorizes_the_interaction_it_names():
    lease = issue_browser_lease(
        principal="bryan",
        origin="https://example.com",
        actions={BrowserAction.CLICK},
    )
    try:
        allowed = authorize_browser_action(
            BrowserAction.CLICK,
            principal="bryan",
            url="https://example.com/page",
            lease_id=lease.lease_id,
        )
        assert allowed.allowed is True

        wrong_action = authorize_browser_action(
            BrowserAction.TYPE,
            principal="bryan",
            url="https://example.com/page",
            lease_id=lease.lease_id,
        )
        assert wrong_action.allowed is False
    finally:
        revoke_browser_lease(lease.lease_id)


def test_a_lease_does_not_cross_origins():
    lease = issue_browser_lease(
        principal="bryan", origin="https://example.com", actions={BrowserAction.CLICK}
    )
    try:
        verdict = authorize_browser_action(
            BrowserAction.CLICK,
            principal="bryan",
            url="https://www.iana.org/somewhere",
            lease_id=lease.lease_id,
        )
        assert verdict.allowed is False
        assert "covers" in verdict.reason
    finally:
        revoke_browser_lease(lease.lease_id)


def test_a_lease_does_not_cross_principals():
    lease = issue_browser_lease(
        principal="bryan", origin="https://example.com", actions={BrowserAction.CLICK}
    )
    try:
        verdict = authorize_browser_action(
            BrowserAction.CLICK,
            principal="someone_else",
            url="https://example.com/x",
            lease_id=lease.lease_id,
        )
        assert verdict.allowed is False
    finally:
        revoke_browser_lease(lease.lease_id)


def test_a_lease_is_spent_rather_than_standing():
    lease = issue_browser_lease(
        principal="bryan",
        origin="https://example.com",
        actions={BrowserAction.CLICK},
        interactions=2,
    )
    try:
        for _ in range(2):
            assert authorize_browser_action(
                BrowserAction.CLICK,
                principal="bryan",
                url="https://example.com/x",
                lease_id=lease.lease_id,
            ).allowed is True

        spent = authorize_browser_action(
            BrowserAction.CLICK,
            principal="bryan",
            url="https://example.com/x",
            lease_id=lease.lease_id,
        )
        assert spent.allowed is False, "one approval became standing consent"
        assert "spent" in spent.reason
    finally:
        revoke_browser_lease(lease.lease_id)


def test_an_expired_lease_stops_working():
    lease = issue_browser_lease(
        principal="bryan",
        origin="https://example.com",
        actions={BrowserAction.CLICK},
        ttl_s=1.0,
    )
    lease.expires_at = 0.0
    try:
        verdict = authorize_browser_action(
            BrowserAction.CLICK,
            principal="bryan",
            url="https://example.com/x",
            lease_id=lease.lease_id,
        )
        assert verdict.allowed is False and "expired" in verdict.reason
    finally:
        revoke_browser_lease(lease.lease_id)


# ── 8bf8d32e: the destination goes through policy ───────────────────────────


@pytest.mark.parametrize(
    ("url", "why"),
    [
        ("http://example.com", "scheme"),
        ("ftp://example.com", "scheme"),
        ("https://user:secret@example.com", "credentials"),
        ("https://example.com:8080/x", "port"),
        ("https://exa mple.com", "whitespace"),
    ],
)
def test_a_malformed_or_unsafe_url_is_refused(url, why):
    validated, error = validate_browser_url(url)
    assert validated is None, f"{url} was admitted"
    assert why.split()[0] in error.lower()


@pytest.mark.parametrize(
    "url",
    [
        "https://localhost/x",
        "https://127.0.0.1/x",
        "https://10.0.0.5/x",
        "https://192.168.1.1/x",
        "https://169.254.169.254/latest/meta-data",
    ],
)
def test_the_local_network_is_not_reachable(url):
    validated, error = validate_browser_url(url)
    assert validated is None, f"{url} was admitted; this is the SSRF target"
    assert error


def test_a_public_destination_still_works():
    validated, error = validate_browser_url("https://example.com/page")
    assert validated is not None, error


def test_the_loopback_opt_in_covers_loopback_only(monkeypatch):
    """A local fixture is a real need. The rest of the LAN is not."""
    import importlib

    import core.runtime.url_policy as policy

    monkeypatch.setenv("AURA_BROWSER_ALLOW_LOOPBACK", "1")
    importlib.reload(policy)
    try:
        assert policy.validate_browser_url("http://127.0.0.1:8931/fixture")[0] is not None
        assert policy.validate_browser_url("http://10.0.0.5/x")[0] is None, (
            "the loopback opt-in opened the rest of the local network"
        )
        assert policy.validate_browser_url("http://169.254.169.254/latest")[0] is None
    finally:
        monkeypatch.delenv("AURA_BROWSER_ALLOW_LOOPBACK", raising=False)
        importlib.reload(policy)


def test_the_browser_policy_does_not_inherit_the_fetch_allowlist():
    """A browser is meant to reach arbitrary public sites."""
    from core.runtime.url_policy import validate_fetch_url_static

    fetched, fetch_error = validate_fetch_url_static("https://example.com/x")
    browsed, _ = validate_browser_url("https://example.com/x")
    assert fetched is None and "allowlist" in fetch_error
    assert browsed is not None, (
        "the browser inherited the fetch tool's domain allowlist, which would "
        "reduce it to a handful of sites"
    )


def test_the_ssrf_checks_have_one_implementation():
    """Two copies of the dangerous half is how one of them rots."""
    import inspect

    import core.runtime.url_policy as policy

    browser_source = inspect.getsource(policy.validate_browser_url)
    assert "validate_url_shape" in browser_source
    assert "resolves_to_public_addresses" in browser_source
    fetch_source = inspect.getsource(policy.validate_fetch_url_static)
    assert "validate_url_shape" in fetch_source


# ── 16fd33d1: arriving is checked, and a redirect is revalidated ────────────


def test_the_navigation_verifier_exists_and_revalidates_redirects():
    import ast
    import inspect

    from core.capabilities.phantom_browser import PhantomBrowser

    source = inspect.getsource(PhantomBrowser._verify_arrival)
    assert "authorize_browser_action" in source, (
        "a permitted URL that 302s to a private address defeats the check"
    )
    assert "is_blocked" in source, (
        "browse returned True regardless of a bot block, and the is_blocked "
        "method next to it was never called"
    )
    ast.parse(source.lstrip())


def test_url_normalization_does_not_accept_anything_starting_with_http():
    from core.capabilities.phantom_browser import _normalize_url

    assert _normalize_url("example.com") == "https://example.com"
    assert _normalize_url("https://example.com") == "https://example.com"
    # `startswith('http')` also accepted this.
    assert _normalize_url("httpfoo://example.com").startswith("https://")


# ── ed96f557: an interaction leaves evidence ────────────────────────────────


def test_the_interaction_receipt_records_what_changed():
    import inspect

    from core.capabilities.phantom_browser import PhantomBrowser

    source = inspect.getsource(PhantomBrowser._record_interaction)
    for field in ("url_before", "url_after", "postcondition_verified"):
        assert field in source, f"the interaction receipt has no {field}"


def test_typing_reads_the_field_back():
    import inspect

    from core.capabilities.phantom_browser import PhantomBrowser

    source = inspect.getsource(PhantomBrowser.type)
    assert "_field_value" in source, (
        "success meant the keystrokes were accepted, with no read-back proving "
        "the field holds what was typed"
    )


# ── ae66231a: caller text is matched literally ──────────────────────────────


def test_caller_text_is_escaped_before_it_becomes_a_regex():
    import inspect

    from core.capabilities.phantom_browser import PhantomBrowser

    source = inspect.getsource(PhantomBrowser.click)
    assert "re.escape(text_match)" in source, (
        "caller text was compiled directly as a regular expression, so a "
        "metacharacter could match a broader element than the literal label"
    )


# ── d9990559 / 9fbf83b2 / 5d5a051a: one lifecycle owner, honest cleanup ─────


def test_startup_is_serialized_by_a_lock():
    import inspect

    from core.capabilities.phantom_browser import PhantomBrowser

    source = inspect.getsource(PhantomBrowser.ensure_ready)
    assert "_lifecycle_lock" in source, (
        "several callers could each launch Playwright and a browser while "
        "rotation and close mutated the same references"
    )


def test_a_failed_startup_closes_what_it_created():
    import inspect

    from core.capabilities.phantom_browser import PhantomBrowser

    source = inspect.getsource(PhantomBrowser._start_browser)
    assert "_abandon_partial_startup" in source, (
        "the failure path stopped Playwright and explicitly closed nothing, "
        "so a browser or context created before the failure survived"
    )


def test_cancellation_is_not_swallowed_by_cleanup():
    """Checked on the parse tree: the docstring mentions it too."""
    import ast
    import inspect

    from core.capabilities.phantom_browser import PhantomBrowser

    tree = ast.parse(inspect.getsource(PhantomBrowser._close_resource).lstrip())
    handlers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ExceptHandler)
        and "CancelledError" in ast.dump(node.type or ast.Constant(value=None))
    ]
    assert handlers, "cleanup does not name CancelledError at all"
    for handler in handlers:
        assert any(isinstance(node, ast.Raise) for node in ast.walk(handler)), (
            "CancelledError was absorbed with everything else, so shutdown "
            "could not be interrupted"
        )


def test_close_reports_resources_it_could_not_confirm():
    import inspect

    from core.capabilities.phantom_browser import PhantomBrowser

    source = inspect.getsource(PhantomBrowser.close)
    assert "_close_failures" in source, (
        '"Browser closed" was logged unconditionally, so a live child could '
        "become untracked while the log said otherwise"
    )


# ── d1b5bf25: rotation is atomic ────────────────────────────────────────────


def test_rotation_builds_the_replacement_before_publishing_it():
    import inspect

    from core.capabilities.phantom_browser import PhantomBrowser

    source = inspect.getsource(PhantomBrowser.rotate_user_agent)
    published = source.index("self.context = new_context")
    built = source.index("new_page = await new_context.new_page()")
    assert built < published, (
        "the new context was published before the page and stealth setup "
        "succeeded, so a failure left a broken context and an orphaned session"
    )
    assert "session_replaced_by_user_agent_rotation" in source, (
        "cookies and storage do not migrate and the caller was not told"
    )


# ── 97a07e2a: status is live, not configured ────────────────────────────────


def test_status_reports_the_engine_that_actually_launched():
    from core.capabilities.phantom_browser import PhantomBrowser

    status = PhantomBrowser(visible=False).get_status()
    for field in ("engine_launched", "browser_connected", "page_open", "current_url"):
        assert field in status, f"status has no {field}; it reported configuration only"
    assert status["engine_launched"] == "none"


# ── 5c9be33c: integration says whether it worked ────────────────────────────


def test_integration_returns_whether_the_browser_is_usable():
    import inspect

    from core.capabilities.phantom_browser import integrate_phantom_browser

    source = inspect.getsource(integrate_phantom_browser)
    assert "ready = await pb.ensure_ready()" in source, (
        "ensure_ready's boolean was ignored and success was logged "
        "unconditionally"
    )
    assert "return False" in source


# ── a02663f7 / 808c3430: exports are bounded and labelled ───────────────────


def test_exports_are_bounded():
    from core.capabilities.phantom_browser import PhantomBrowser

    assert PhantomBrowser.MAX_LINKS > 0
    assert PhantomBrowser.MAX_SCREENSHOT_BYTES > 0
    assert PhantomBrowser.MAX_EXTRACT_CHARS > 0


def test_the_screenshot_is_the_viewport_not_the_whole_document():
    import inspect

    from core.capabilities.phantom_browser import PhantomBrowser

    source = inspect.getsource(PhantomBrowser.screenshot)
    assert "full_page=False" in source, (
        "the entire scrollable document was encoded, which is far more of "
        "the person's session than a screenshot request asks for"
    )


def test_extracted_content_carries_its_provenance():
    import inspect

    from core.capabilities.phantom_browser import PhantomBrowser

    source = inspect.getsource(PhantomBrowser.read_content)
    for field in ("final_url", "truncated", "content_sha256", "selected_block", "trust"):
        assert field in source, f"the extraction receipt has no {field}"


def test_links_are_scheme_filtered():
    import inspect

    from core.capabilities.phantom_browser import PhantomBrowser

    source = inspect.getsource(PhantomBrowser.get_links)
    assert 'startswith(("http://", "https://"))' in source, (
        "javascript:, data: and file: links were handed out as destinations"
    )


def test_origin_of_is_scheme_host_and_port():
    assert origin_of("https://example.com/a/b?c=d") == "https://example.com"
    assert origin_of("https://example.com:8443/x") == "https://example.com:8443"
    assert origin_of("not a url") == ""


def test_the_receipts_are_reachable_from_one_place():
    """Five one-line accessors became one method a caller can find."""
    from core.capabilities.phantom_browser import PhantomBrowser

    receipts = PhantomBrowser(visible=False).receipts()
    for field in ("generation", "navigation", "authorization", "interaction", "extraction"):
        assert field in receipts, f"receipts() has no {field}"
    assert receipts["navigation"]["reason"] == "never_navigated"
