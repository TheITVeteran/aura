"""Late numerical correspondence reaches control only through current matched pixels."""
import threading
from concurrent.futures import Future

import numpy as np
import pytest

from core.perception.learned_tracking import corrections_while_points_keep_moving as correcting
from core.perception.points_that_keep_their_motion import SeenPoint
from core.runtime.lockdep import checked_thread_semaphore


def picture(at, *, generation=0, empty=False):
    pixels = np.full((64, 80, 3), 25, np.uint8)
    if not empty:
        pixels[16:48, 20:52] = np.random.default_rng(71).integers(30, 240, (32, 32, 3), dtype=np.uint8)
    return correcting._Picture(pixels, pixels.copy(), at, generation)


def marks(at):
    return tuple(SeenPoint(i, float(x), float(y), float(x), float(y), at, at, at, .01, 1)
                 for i, (x, y) in enumerate(((26, 23), (42, 23), (26, 39), (42, 39))))


def test_late_correspondence_is_measured_through_every_retained_capture():
    transport = correcting.visible_transport(marks(1.), [picture(1.), picture(1.1), picture(1.2)], 256_000)
    assert transport is not None and len(transport.points) == 4
    assert all(point.at == 1.2 and point.before == 1.1 and point.pictures == 3 for point in transport.points)
    assert correcting.visible_transport(marks(1.), [picture(1.), picture(1.1, empty=True)], 256_000) is None


@pytest.mark.parametrize('departure', ['missing-anchor', 'generation', 'gap'])
def test_absent_or_changed_picture_history_cannot_renew_an_old_correction(departure):
    frames = [picture(1.), picture(1.1)]
    if departure == 'missing-anchor':
        frames[0] = picture(.9)
    elif departure == 'generation':
        frames[1] = picture(1.1, generation=1)
    else:
        frames[1] = picture(2.)
    assert correcting.visible_transport(marks(1.), frames, 256_000) is None


def test_a_slow_teacher_does_not_block_visible_tracking_or_queue_another_picture(monkeypatch):
    raw = Future()
    submitted = []
    slots = checked_thread_semaphore('test.learned_point_corrections', 1)
    monkeypatch.setattr(correcting, '_SLOTS', slots)

    def submit(fn, captured, reset, **kwargs):
        submitted.append(captured)
        return raw

    monkeypatch.setattr(correcting, 'submit_blocking_io', submit)
    points = correcting.CorrectionsWhilePointsMove(lambda: None, limit=16)
    first, second = picture(1.), picture(1.1)
    assert points.see_source(first.source, first.working, first.at)
    assert points.see_source(second.source, second.working, second.at)
    assert len(submitted) == 1 and not slots.acquire(blocking=False)
    raw.set_result(correcting._Refinement(marks(1.), points._operation.pictures[0]))
    assert slots.acquire(blocking=False)
    slots.release()


def test_a_reset_while_work_is_running_forbids_late_publication(monkeypatch):
    raw = Future()
    monkeypatch.setattr(correcting, '_SLOTS', checked_thread_semaphore('test.reset_point_corrections', 1))
    monkeypatch.setattr(correcting, 'submit_blocking_io', lambda *_args, **_kwargs: raw)
    points = correcting.CorrectionsWhilePointsMove(lambda: None, limit=16)
    first = picture(1.)
    points.see_source(first.source, first.working, first.at)
    points.clear()
    raw.set_result(correcting._Refinement(marks(1.), points._operation.pictures[0]))
    points._receive()
    assert points.correction_status == 'correction expired' and points._transport is None


def test_an_old_failed_correction_cannot_disable_the_new_generation(monkeypatch):
    raw = Future()
    monkeypatch.setattr(correcting, '_SLOTS', checked_thread_semaphore('test.failed_old_point_corrections', 1))
    monkeypatch.setattr(correcting, 'submit_blocking_io', lambda *_args, **_kwargs: raw)
    points = correcting.CorrectionsWhilePointsMove(lambda: None, limit=16)
    first = picture(1.)
    points.see_source(first.source, first.working, first.at)
    points.clear()
    raw.set_exception(RuntimeError('a failed old scene'))
    points._receive()
    assert points.correction_status == 'correction expired' and points._failed_generation is None


def test_fast_pictures_continue_on_the_actual_worker_while_the_teacher_is_busy():
    release, began = threading.Event(), threading.Event()

    class SlowTeacher:
        def clear(self):
            pass

        def see_source(self, source, working, at):
            began.set()
            assert release.wait(2.)
            return ()

    points = correcting.CorrectionsWhilePointsMove(SlowTeacher, limit=16)
    first = picture(1.)
    try:
        points.see_source(first.source, first.working, first.at)
        assert began.wait(1.)
        raw = points._operation.future
        for index in range(1, 5):
            seen = picture(1. + .05 * index)
            points.see_source(seen.source, seen.working, seen.at)
        assert points._at == 1.2 and not raw.done()
        assert any(point.pictures >= 5 for point in points.points)
    finally:
        release.set()
        points._operation.future.result(timeout=2.)


def test_visible_transport_has_a_registered_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert correcting._visible_correction_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'late_points_need_visible_transport')
    assert check.predict(None) and any(c.test == check.name for c in suite.claims())
