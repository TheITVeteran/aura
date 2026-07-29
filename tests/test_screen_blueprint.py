"""The screen has a structure, and she can read it.

A person glancing at a screen knows instantly which app is in front, which
window is buried, and where the text box is. Aura had a screenshot and a flat
list of window titles, which answers none of that — so every OS action had to
guess where it was aiming.

These tests hold the blueprint to the standard a glance meets: exact
occlusion rather than an estimate, front-to-back order that means what it
says, and — the part that matters most — an unreadable screen that reports
itself as unreadable instead of as an empty one.
"""

from __future__ import annotations

import pytest

from core.perception.screen_blueprint import (
    ScreenBlueprint,
    WindowFrame,
    _rect_union_area_within,
    capture_blueprint,
)


def _window(app: str, x: int, y: int, w: int, h: int, z: int, **kwargs) -> WindowFrame:
    return WindowFrame(
        app=app,
        title=kwargs.pop("title", app),
        pid=kwargs.pop("pid", 1000 + z),
        window_id=kwargs.pop("window_id", 500 + z),
        x=x,
        y=y,
        width=w,
        height=h,
        layer=0,
        alpha=1.0,
        z_index=z,
        **kwargs,
    )


class TestOcclusionIsMeasuredNotGuessed:
    """"Chrome is covering Notes" has to be a measurement.

    If it is a guess, then so is every decision made from it — including
    whether it is safe to type.
    """

    def test_a_window_fully_behind_another_is_fully_covered(self) -> None:
        assert _rect_union_area_within((0, 0, 100, 100), [(0, 0, 100, 100)]) == 10_000

    def test_overlapping_covers_are_counted_once(self) -> None:
        # Two covers overlap in the middle. Summing their areas would report
        # 130% coverage, which is how an estimate ends up claiming a visible
        # window is hidden.
        covered = _rect_union_area_within(
            (0, 0, 100, 100), [(0, 0, 60, 100), (40, 0, 100, 100)]
        )
        assert covered == 10_000

    def test_a_gap_between_covers_stays_visible(self) -> None:
        covered = _rect_union_area_within(
            (0, 0, 100, 100), [(0, 0, 40, 100), (60, 0, 100, 100)]
        )
        assert covered == 8_000

    def test_a_window_elsewhere_covers_nothing(self) -> None:
        assert _rect_union_area_within((0, 0, 100, 100), [(500, 500, 600, 600)]) == 0

    def test_partial_overlap_is_exact(self) -> None:
        assert _rect_union_area_within((0, 0, 100, 100), [(50, 50, 150, 150)]) == 2_500

    def test_a_cover_smaller_than_the_window_leaves_the_rest(self) -> None:
        assert _rect_union_area_within((0, 0, 10, 10), [(2, 2, 4, 4)]) == 4


class TestTheBlueprintAnswersWhatAPersonWouldAsk:
    def test_front_to_back_order_names_the_frontmost_app(self) -> None:
        blueprint = ScreenBlueprint(
            windows=(
                _window("Google Chrome", 0, 0, 1000, 800, 0),
                _window("Notes", 0, 0, 1000, 800, 1, visible_fraction=0.0,
                        covered_by=("Google Chrome",)),
            ),
            frontmost_app="Google Chrome",
        )
        assert blueprint.apps == ("Google Chrome", "Notes")
        assert blueprint.is_app_frontmost("Google Chrome")
        assert not blueprint.is_app_frontmost("Notes")

    def test_a_buried_window_is_open_but_not_visible(self) -> None:
        """Both facts are true at once and they are not the same fact.

        Notes being open is why "open Notes" should not launch it again;
        Notes being invisible is why typing into it would look to the person
        like nothing happened.
        """
        blueprint = ScreenBlueprint(
            windows=(
                _window("Google Chrome", 0, 0, 1000, 800, 0),
                _window("Notes", 0, 0, 1000, 800, 1, visible_fraction=0.0,
                        covered_by=("Google Chrome",)),
            ),
            frontmost_app="Google Chrome",
        )
        assert blueprint.windows_for("Notes")
        assert not blueprint.app_is_visible("Notes")
        assert "hidden behind Google Chrome" in blueprint.windows[1].describe()

    def test_a_half_covered_window_reports_how_much_is_left(self) -> None:
        frame = _window("Notes", 0, 0, 100, 100, 1, visible_fraction=0.5,
                        covered_by=("Google Chrome",))
        assert frame.is_visible
        assert "50% visible" in frame.describe()

    def test_naming_an_app_matches_how_people_name_it(self) -> None:
        blueprint = ScreenBlueprint(
            windows=(_window("Google Chrome", 0, 0, 100, 100, 0),),
            frontmost_app="Google Chrome",
        )
        assert blueprint.windows_for("Chrome")
        assert blueprint.is_app_frontmost("Chrome")


class TestUnreadableIsNotEmpty:
    """The failure that would matter most.

    If a blueprint that could not be taken looked like a screen with nothing
    on it, she would state — with total confidence — that no windows are
    open, to someone looking at nine of them. Every other guarantee here is
    worth less than this one.
    """

    def test_an_unavailable_blueprint_says_so(self) -> None:
        blueprint = ScreenBlueprint(
            unavailable=True, unavailable_reason="the window list could not be read"
        )
        assert blueprint.windows == ()
        assert "can't read the screen layout" in blueprint.describe()
        assert "no application windows" not in blueprint.describe()

    def test_a_genuinely_empty_screen_says_something_different(self) -> None:
        blueprint = ScreenBlueprint(windows=(), frontmost_app="")
        assert "no application windows" in blueprint.describe()

    def test_capture_never_raises_when_the_window_server_is_gone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import core.perception.screen_blueprint as module

        monkeypatch.setattr(module, "_CACHED", None)

        def _explode() -> list:
            raise RuntimeError("no window server")

        monkeypatch.setattr(module, "_raw_window_list", _explode)
        blueprint = capture_blueprint(fresh=True)
        assert blueprint.unavailable
        assert "RuntimeError" in blueprint.unavailable_reason

    def test_an_unavailable_blueprint_does_not_claim_an_app_is_missing(self) -> None:
        blueprint = ScreenBlueprint(unavailable=True, unavailable_reason="no access")
        # No windows are known, so no window can be asserted visible — but
        # nothing here should read as "Notes is not running" either.
        assert not blueprint.app_is_visible("Notes")
        assert not blueprint.is_app_frontmost("Notes")
        assert blueprint.unavailable


class TestSystemFurnitureIsNotAnOpenApp:
    def test_the_menu_bar_and_status_items_are_filtered_out(self) -> None:
        import core.perception.screen_blueprint as module

        assert not module._is_real_window(
            {"kCGWindowLayer": 25, "kCGWindowBounds": {"Width": 40, "Height": 33}}
        )
        assert not module._is_real_window(
            {"kCGWindowLayer": 0, "kCGWindowBounds": {"Width": 10, "Height": 10}}
        )
        assert not module._is_real_window(
            {
                "kCGWindowLayer": 0,
                "kCGWindowAlpha": 0.0,
                "kCGWindowBounds": {"Width": 800, "Height": 600},
            }
        )
        assert module._is_real_window(
            {
                "kCGWindowLayer": 0,
                "kCGWindowAlpha": 1.0,
                "kCGWindowBounds": {"Width": 800, "Height": 600},
            }
        )


class TestTheBlueprintIsCheapEnoughToUseInALoop:
    def test_capture_is_cached_for_a_burst_of_questions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The frontmost check polls. If each poll forked a subprocess, the
        wait would cost more than the thing it is waiting for — which is
        exactly the bug this replaced."""
        import core.perception.screen_blueprint as module

        monkeypatch.setattr(module, "_CACHED", None)
        calls = {"n": 0}

        def _counted() -> list:
            calls["n"] += 1
            return []

        monkeypatch.setattr(module, "_raw_window_list", _counted)
        capture_blueprint(fresh=True)
        capture_blueprint()
        capture_blueprint()
        assert calls["n"] == 1
