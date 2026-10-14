"""Model construction and operation cannot evade ownership or headroom checks."""
from __future__ import annotations

import hashlib
import select
import subprocess
import sys
import threading
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest

from core.perception.learned_tracking import models
from core.perception.learned_tracking.admitted_model_memory import AdmittedTrackingBackend
from core.runtime import optional_compute as capacity

pytestmark = pytest.mark.unit


def observer(*, available=True, percent=40., free=8 * 1024**3):
    return SimpleNamespace(memory=lambda **_kw: SimpleNamespace(available=available, percent=percent,
                                                               available_bytes=free))


@pytest.mark.parametrize('sample', [observer(available=False), observer(percent=float('nan')),
                                   observer(percent=88.), observer(free=1024)])
def test_unobserved_or_reserved_capacity_cannot_start_optional_work(tmp_path, sample):
    with pytest.raises(capacity.OptionalComputeBusy):
        with capacity.optional_compute(path=tmp_path/'slot', observer=sample):
            pytest.fail('unadmitted work ran')
    assert capacity._optional_headroom_invariant() == ()


def test_an_owned_thread_cannot_queue_another_optional_worker(tmp_path):
    path = tmp_path/'slot'
    errors = []

    def attempt():
        try:
            with capacity.optional_compute(path=path, observer=observer()):
                errors.append('admitted')
        except capacity.OptionalComputeBusy:
            errors.append('deferred')

    with capacity.optional_compute(path=path, observer=observer()):
        thread = threading.Thread(target=attempt)
        thread.start()
        thread.join(2.)
        assert not thread.is_alive() and errors == ['deferred']
    with capacity.optional_compute(path=path, observer=observer()):
        pass


def test_process_ownership_is_shared_and_os_exit_returns_the_permit(tmp_path):
    path = tmp_path/'slot'
    # The separate process takes the actual OS permit independently of the
    # Python helper; a process-local mutex cannot make this test pass.
    script = ('import fcntl,sys; f=open(sys.argv[1],"a+"); '
              'fcntl.flock(f,fcntl.LOCK_EX); print("owned",flush=True); sys.stdin.readline()')
    child = subprocess.Popen([sys.executable, '-c', script, str(path)], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert select.select([child.stdout], [], [], 5.)[0]
        assert child.stdout.readline().strip() == 'owned'
        with pytest.raises(capacity.OptionalComputeBusy):
            with capacity.optional_compute(path=path, observer=observer()):
                pytest.fail('another process owner was ignored')
        child.communicate('\n', timeout=5.)
        assert child.returncode == 0
        with capacity.optional_compute(path=path, observer=observer()):
            pass
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate(timeout=5.)


def test_exception_returns_the_permit_without_waiting_or_queued_work(tmp_path):
    path = tmp_path/'slot'
    with pytest.raises(ValueError):
        with capacity.optional_compute(path=path, observer=observer()):
            raise ValueError('invented operation failure')
    with capacity.optional_compute(path=path, observer=observer()):
        pass


def test_optional_work_cannot_create_an_uninitialized_runtime_directory(tmp_path):
    target = tmp_path/'not-created'/'slot'
    with pytest.raises(capacity.OptionalComputeBusy, match='not initialized'):
        with capacity.optional_compute(path=target, observer=observer()):
            pytest.fail('optional work created its own runtime root')
    assert not target.parent.exists()


class Lease:
    active = True

    def __init__(self):
        self.released = []

    def release(self, **kwargs):
        self.released.append(kwargs)
        self.active = False
        return True


def test_memory_reservation_precedes_construction_and_survives_until_disposal(monkeypatch, tmp_path):
    from core.perception.learned_tracking import admitted_model_memory as owned

    weights = tmp_path/'weights'
    weights.write_bytes(b'independent fake weights')
    events, lease = [], Lease()
    backend = SimpleNamespace(start=lambda *a: events.append('started'), close=lambda: events.append('closed'))
    monkeypatch.setattr(owned, 'admit_tracking_memory', lambda *_a: events.append('reserved') or lease)
    monkeypatch.setattr(models, '_construct_backend', lambda *_a, **_k: events.append('loaded') or backend)
    monkeypatch.setattr(capacity, 'optional_headroom_reason', lambda **_kw: None)
    monkeypatch.setattr(capacity, 'optional_compute_path', lambda: tmp_path/'slot')
    managed = models.load_backend('online_tapir', weights, expected_sha256=hashlib.sha256(weights.read_bytes()).hexdigest())
    assert events == ['reserved', 'loaded'] and not lease.released
    managed.start(None, ())
    assert events == ['reserved', 'loaded', 'started'] and not lease.released
    managed.close()
    assert events[-1] == 'closed' and len(lease.released) == 1
    managed.close()
    assert len(lease.released) == 1


def test_refused_model_admission_cannot_allocate_weights(monkeypatch, tmp_path):
    from core.perception.learned_tracking import admitted_model_memory as owned

    weights = tmp_path/'weights'
    weights.write_bytes(b'fake weights')

    def refuse(*_a):
        raise RuntimeError('host reservation refused')

    monkeypatch.setattr(owned, 'admit_tracking_memory', refuse)
    monkeypatch.setattr(models, '_construct_backend', lambda *_a, **_kw: pytest.fail('allocated before admission'))
    with pytest.raises(RuntimeError, match='reservation refused'):
        models.load_backend('online_tapir', weights, expected_sha256=hashlib.sha256(weights.read_bytes()).hexdigest())


@pytest.mark.parametrize('name', ['online_tapir', 'sam2'])
def test_direct_real_weight_construction_cannot_bypass_the_memory_fence(tmp_path, name):
    from core.runtime.model_lane_control import ModelLaneControlError

    weights = tmp_path/'weights'
    weights.write_bytes(b'fake real-file checkpoint')
    with pytest.raises(ModelLaneControlError, match='requires_active'):
        if name == 'online_tapir':
            from core.perception.learned_tracking.tapir_backend import OnlineTapirBackend

            OnlineTapirBackend(weights)
        else:
            from core.perception.learned_tracking.sam2_backend import Sam2Backend

            Sam2Backend(weights)._model()


def test_failed_model_construction_releases_the_reservation(monkeypatch, tmp_path):
    from core.perception.learned_tracking import admitted_model_memory as owned

    weights = tmp_path/'weights'
    weights.write_bytes(b'fake weights')
    lease = Lease()
    monkeypatch.setattr(owned, 'admit_tracking_memory', lambda *_a: lease)
    monkeypatch.setattr(capacity, 'optional_headroom_reason', lambda **_kw: None)
    monkeypatch.setattr(capacity, 'optional_compute_path', lambda: tmp_path/'slot')

    def broken(*_a, **_kw):
        raise ValueError('model loader failed')

    monkeypatch.setattr(models, '_construct_backend', broken)
    with pytest.raises(ValueError, match='loader failed'):
        models.load_backend('online_tapir', weights, expected_sha256=hashlib.sha256(weights.read_bytes()).hexdigest())
    assert len(lease.released) == 1


def test_async_device_work_finishes_before_its_permit_is_released(monkeypatch, tmp_path):
    events = []
    backend = SimpleNamespace(track=lambda: events.append('dispatched'), close=lambda: None)
    managed = AdmittedTrackingBackend(backend, Lease(), name='invented', device='mps')
    fake = SimpleNamespace(mps=SimpleNamespace(synchronize=lambda: events.append('finished')))
    monkeypatch.setitem(sys.modules, 'torch', fake)
    monkeypatch.setattr(capacity, 'optional_headroom_reason', lambda **_kw: None)
    monkeypatch.setattr(capacity, 'optional_compute_path', lambda: tmp_path/'slot')
    managed.track()
    assert events == ['dispatched', 'finished']
    managed._lease.active = False
    with pytest.raises(capacity.OptionalComputeBusy):
        managed.track()
    assert events == ['dispatched', 'finished']


def test_expected_model_backpressure_keeps_fast_points_and_allows_a_retry(monkeypatch):
    from core.perception.learned_tracking import corrections_while_points_keep_moving as correcting
    from tests.test_learned_corrections_do_not_hold_up_visible_tracking import picture

    points = correcting.CorrectionsWhilePointsMove(lambda: None, limit=16)
    p = picture(1.)
    raw = Future()
    monkeypatch.setattr(correcting, 'submit_blocking_io', lambda *_a, **_kw: raw)
    assert points.see_source(p.source, p.working, p.at)
    raw.set_exception(capacity.OptionalComputeBusy('foreground owns capacity'))
    points._receive()
    assert points._failed_generation is None and points.correction_status.startswith('correction deferred')
    assert points._operation is None and points.points


def test_direct_tracking_admission_refusal_preserves_the_actual_fast_observer(monkeypatch):
    from core.perception.learned_tracking import corrections_while_points_keep_moving as correcting
    from core.perception.learned_tracking.selection import selected_motion_eyes

    def refuse(*_a, **_kw):
        raise capacity.OptionalComputeBusy('another protected model owns the host')

    monkeypatch.setenv('AURA_PLAY_POINT_BACKEND', 'online_tapir')
    monkeypatch.setenv('AURA_PLAY_POINT_WEIGHTS', '/invented-local-weights')
    monkeypatch.setenv('AURA_PLAY_POINT_MODE', 'direct')
    monkeypatch.setattr(models, 'load_backend', refuse)
    eyes = selected_motion_eyes()
    assert eyes.point_tracking_requested == 'online_tapir'
    assert isinstance(eyes.points, correcting.CorrectionsWhilePointsMove)
    assert eyes.points._teacher is None and eyes.points.correction_status.startswith('correction deferred')


def test_direct_tracker_backpressure_during_play_preserves_current_fast_pixels():
    from core.perception.learned_tracking.contracts import PointResult
    from core.perception.learned_tracking.points_for_the_motion_eyes import PointsForMotionEyes
    from tests.test_learned_corrections_do_not_hold_up_visible_tracking import picture

    class BusyBackend:
        busy = False

        def start(self, frame, _seeds):
            if self.busy:
                raise capacity.OptionalComputeBusy('host unavailable')
            return PointResult('invented', frame.epoch, frame.number, frame.at, frame.shape, (), 0., 'seeded')

        def track(self, _frame):
            raise capacity.OptionalComputeBusy('host unavailable')

    backend = BusyBackend()
    points = PointsForMotionEyes(backend, limit=16, fallback_when_busy=True)
    first, second, third = picture(1.), picture(1.1), picture(1.2)
    points.see_source(first.source, first.working, first.at)
    observed = points.see_source(second.source, second.working, second.at)
    assert observed and all(p.at == 1.1 and p.number >= 1 << 52 for p in observed)
    assert points.status.startswith('point model deferred') and not points._slots
    backend.busy = True
    assert points.see_source(third.source, third.working, third.at)
    assert not points._slots  # A blocked start cannot become a valid tracking session.


def test_retained_corrections_do_not_copy_full_resolution_frames(monkeypatch):
    from core.perception.learned_tracking import corrections_while_points_keep_moving as correcting

    raw = Future()
    monkeypatch.setattr(correcting, 'submit_blocking_io', lambda *_a, **_kw: raw)
    points = correcting.CorrectionsWhilePointsMove(lambda: None, pixel_budget=16_000, limit=16)
    source = np.random.default_rng(871).integers(0, 255, (600, 800, 3), dtype=np.uint8)
    points.see_source(source, source[::5, ::5], 1.)
    assert points._retained[0].source.size <= 16_000 * 3
    raw.set_result(correcting._Refinement((), points._operation.pictures[0]))
    points.close()


def test_the_capacity_claim_is_registered():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'optional_compute_preserves_unobserved_headroom')
    assert check.predict(None) and any(c.test == check.name for c in suite.claims())
