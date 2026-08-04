"""A background sweep does not get to call itself the foreground.

CP126 829ffbee. Foreground classification split the origin label on
underscores and intersected the tokens with an allowlist containing user,
admin, api, direct, external and test. Any label CONTAINING one of them
elevated itself: ``background_user`` matched ``user``,
``test_background_sweep`` matched ``test``, ``api_prefetch`` matched ``api``.

Foreground priority decides who gets the resident model when it is contended,
so an origin that can name itself into the foreground is a self-granted
privilege over the turn a person is actually waiting on.
"""
from __future__ import annotations

import pytest

from core.brain.llm import mlx_client

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "origin",
    [
        "user",
        "voice",
        "desktop",
        "desktop-ui",
        "desktop_ui",
        "native-shell",
        "ws",
        "api",
        "admin",
    ],
)
def test_a_real_user_facing_origin_is_still_foreground(origin):
    assert mlx_client._origin_is_user_facing(origin) is True


@pytest.mark.parametrize(
    "origin",
    [
        "background_user",
        "user_background_sweep",
        "test_background_sweep",
        "api_prefetch",
        "prefetch_api",
        "external_crawler",
        "direct_indexer",
        "admin_backfill",
        "autonomous_initiative_loop",
        "self_repair_backlog",
        "long_term_memory",
        "world_monitor",
        "research_cycle",
    ],
)
def test_a_background_origin_cannot_elevate_itself(origin):
    assert mlx_client._origin_is_user_facing(origin) is False


def test_the_match_is_on_the_whole_label():
    """The property, not one example: adding a suffix must not preserve
    foreground status."""
    for base in ("user", "api", "test", "admin", "direct", "external"):
        assert mlx_client._origin_is_user_facing(base) is True
        assert mlx_client._origin_is_user_facing(f"{base}_backfill") is False
        assert mlx_client._origin_is_user_facing(f"nightly_{base}") is False


@pytest.mark.parametrize("origin", ["", None, "   ", "unknown"])
def test_an_absent_origin_is_not_foreground(origin):
    assert mlx_client._origin_is_user_facing(origin) is False


def test_hyphen_and_underscore_spellings_agree():
    assert mlx_client._origin_is_user_facing("desktop-ui") is True
    assert mlx_client._origin_is_user_facing("desktop_ui") is True
    assert mlx_client._origin_is_user_facing("native-shell") is True
    assert mlx_client._origin_is_user_facing("native_shell") is True


def test_case_and_whitespace_do_not_change_the_answer():
    assert mlx_client._origin_is_user_facing("  DESKTOP-UI  ") is True
    assert mlx_client._origin_is_user_facing("  BACKGROUND_USER  ") is False


def test_the_cross_surface_origin_contract_still_holds():
    """Other surfaces compare against the raw set; both spellings stay listed."""
    expected = {"desktop", "desktop-ui", "native-shell", "ws"}

    assert expected <= mlx_client._USER_FACING_ORIGINS
