import time
from types import SimpleNamespace

import pytest

import core.volition as volition_module
from core.volition import VolitionEngine


@pytest.fixture
def orchestrator():
    return OrchestratorProbe()

@pytest.fixture
def engine(orchestrator, monkeypatch, tmp_path):
    paths = SimpleNamespace(
        home_dir=tmp_path,
        brain_dir=tmp_path / "brain",
        data_dir=tmp_path / "data",
    )
    paths.brain_dir.mkdir(parents=True, exist_ok=True)
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(volition_module.config, "paths", paths)
    monkeypatch.setattr(
        "core.will.get_will",
        lambda: SimpleNamespace(
            decide=lambda *args, **kwargs: SimpleNamespace(
                is_approved=lambda: True,
                receipt_id="will_unit_test",
            ),
            verify_receipt=lambda _receipt_id: True,
        ),
    )
    return VolitionEngine(orchestrator)


class StatusProbe:
    def __init__(self, running=True):
        self.running = running


class DriveProbe:
    def __init__(self, name="Connection", urgency=0.5):
        self.name = name
        self.urgency = urgency


class SoulProbe:
    def __init__(self):
        self.drive = DriveProbe()

    def get_dominant_drive(self):
        return self.drive


class ProjectStoreProbe:
    def __init__(self):
        self.active_projects = []

    def get_active_projects(self):
        return list(self.active_projects)


class StrategicPlannerProbe:
    def __init__(self):
        self.next_task = None

    def get_next_task(self, _project_id):
        return self.next_task


class OrchestratorProbe:
    def __init__(self):
        self.status = StatusProbe()
        self.cognitive_engine = object()
        self.project_store = ProjectStoreProbe()
        self.strategic_planner = StrategicPlannerProbe()
        self.soul = SoulProbe()
        self.conversation_history = []


class ChoiceSequence:
    def __init__(self, values):
        self.values = list(values)

    def __call__(self, _options):
        if self.values:
            return self.values.pop(0)
        raise AssertionError("choice sequence exhausted")


def test_init(engine, orchestrator):
    assert engine.orchestrator == orchestrator
    assert engine.brain == orchestrator.cognitive_engine
    assert engine.boredom_threshold == 45
    assert not engine.is_dreaming
    assert engine._consecutive_idle_cycles == 0
    assert "the future of bio-computing" in engine.general_interests


@pytest.mark.asyncio
async def test_tick_skip_not_running(engine):
    engine.orchestrator.status.running = False
    assert await engine.tick(current_goal=None) is None


@pytest.mark.asyncio
async def test_tick_skip_has_goal(engine):
    engine.last_activity_time = 0.0
    engine._consecutive_idle_cycles = 5
    
    assert await engine.tick(current_goal="Working on a user request") is None
    
    assert engine.last_activity_time > 0.0
    assert engine._consecutive_idle_cycles == 0


@pytest.mark.asyncio
async def test_search_for_autonomous_goals_boredom(engine):
    # Fast forward time to trigger boredom
    engine.last_activity_time = time.time() - 100
    engine.boredom_threshold = 45
    
    goals = await engine._search_for_autonomous_goals()
    assert len(goals) > 0
    
    boredom_origin = goals[0].get("origin")
    assert boredom_origin in [
        "intrinsic_duty", 
        "intrinsic_duty_strategic", 
        "intrinsic_reflection", 
        "intrinsic_curiosity", 
        "intrinsic_fun",
        "intrinsic_evolution"
    ]


@pytest.mark.asyncio
async def test_generate_duty_goal_strategic(monkeypatch, engine):
    monkeypatch.setattr(volition_module.random, "random", lambda: 0.1)
    engine.orchestrator.project_store.active_projects = [
        SimpleNamespace(name="Test Project", id="proj_1")
    ]
    engine.orchestrator.strategic_planner.next_task = SimpleNamespace(
        description="Test task description",
        id="task_1",
    )
    
    goal = await engine._generate_duty_goal()
    assert goal is not None
    assert goal["origin"] == "intrinsic_duty_strategic"
    assert "Test Project" in goal["objective"]


def test_notify_activity(engine):
    engine.last_activity_time = 0.0
    engine._consecutive_idle_cycles = 10
    engine.unanswered_speak_count = 3
    engine.speak_backoff_multiplier = 4.0
    
    engine.notify_activity()
    
    assert engine.last_activity_time > 0.0
    assert engine._consecutive_idle_cycles == 0
    assert engine.unanswered_speak_count == 0
    assert engine.speak_backoff_multiplier == 1.0


def test_check_soul_drives_connection(engine):
    # Set urgency high enough to trigger connection
    engine.orchestrator.soul.drive.name = "Connection"
    engine.orchestrator.soul.drive.urgency = 0.9
    engine.last_speak_time = 0.0  # Force cooldown pass
    
    goal = engine._check_soul_drives()
    assert goal is not None
    assert goal["origin"] == "intrinsic_connection"
    assert goal["speak"] is True
    # Generating an outreach CANDIDATE is not speaking: the candidate can
    # still lose selection or be refused by the Will, so the counters must
    # not advance until the action is actually admitted.
    assert engine.unanswered_speak_count == 0
    assert engine.speak_backoff_multiplier == 1.0

    engine._commit_goal_effects(goal)
    assert engine.unanswered_speak_count == 1
    assert engine.speak_backoff_multiplier > 1.0
    assert engine.last_speak_time > 0.0


def test_check_soul_drives_connection_silenced(engine):
    engine.orchestrator.soul.drive.name = "Connection"
    engine.orchestrator.soul.drive.urgency = 0.9
    
    engine.unanswered_speak_count = engine.max_unanswered_before_silence
    
    goal = engine._check_soul_drives()
    assert goal is None  # Suppressed due to unanswered messages


def test_load_interests_empty(engine):
    (volition_module.config.paths.data_dir / "interests.json").write_text("{}", encoding="utf-8")
    engine.general_interests = []
    engine.load_interests()
    
    assert "the future of bio-computing" in engine.general_interests


def test_generate_impulse(monkeypatch, engine):
    monkeypatch.setattr(volition_module.random, "choices", lambda *_args, **_kwargs: ["question"])
    monkeypatch.setattr(
        volition_module.random,
        "choice",
        ChoiceSequence(["Ask the user what they think about {topic}.", "the future of bio-computing"]),
    )
    engine.general_interests = ["the future of bio-computing"]
    
    now = time.time()
    impulse = engine._generate_impulse(now)
    
    assert impulse is not None
    assert impulse["origin"] == "impulse_question"
    assert "bio-computing" in impulse["objective"]
    assert impulse["speak"] is True
    # Counters advance on admission, not on generation.
    assert engine.unanswered_speak_count == 0

    engine._commit_goal_effects(impulse)
    assert engine.unanswered_speak_count == 1

def test_generate_impulse_silenced(monkeypatch, engine):
    monkeypatch.setattr(volition_module.random, "choices", lambda *_args, **_kwargs: ["question"])
    engine.unanswered_speak_count = engine.max_unanswered_before_silence
    
    impulse = engine._generate_impulse(time.time())
    
    assert impulse is not None
    assert impulse["speak"] is False
    assert "[Internal thought" in impulse["objective"]


@pytest.mark.asyncio
async def test_generate_duty_goal_fallback(engine):
    engine.orchestrator.strategic_planner = None
    task_dir = volition_module.config.paths.brain_dir / "phase"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.md").write_text("- [ ] Refactor Core\n", encoding="utf-8")

    goal = await engine._generate_duty_goal()

    assert goal is not None
    assert goal["origin"] == "intrinsic_duty"
    assert "Refactor Core" in goal["objective"]


def test_generate_reflection_goal(monkeypatch, engine):
    monkeypatch.setattr(
        volition_module.random,
        "choice",
        lambda _options: "Reflect on my own thinking patterns",
    )
    goal = engine._generate_reflection_goal()

    assert goal is not None
    assert goal["origin"] == "intrinsic_reflection"
    assert "Reflect on my own thinking patterns" in goal["objective"]


def test_generate_curiosity_goal_educational(monkeypatch, engine):
    engine.general_interests = ["robotics"]
    monkeypatch.setattr(
        volition_module.random,
        "choice",
        ChoiceSequence(["robotics", "Research the history of {topic}."]),
    )
    goal = engine._generate_curiosity_goal("educational")

    assert goal is not None
    assert goal["origin"] == "intrinsic_curiosity"
    assert "robotics" in goal["objective"]


def test_generate_curiosity_goal_fun(monkeypatch, engine):
    engine.fun_interests = ["origami"]
    monkeypatch.setattr(
        volition_module.random,
        "choice",
        ChoiceSequence(["origami", "Spend some time {topic} just for fun."]),
    )
    goal = engine._generate_curiosity_goal("fun")

    assert goal is not None
    assert goal["origin"] == "intrinsic_fun"
    assert "origami" in goal["objective"]

def test_check_soul_drives_competence(engine):
    """The competence drive still fires; what it WANTS is now grounded.

    It used to emit one fixed self-diagnosis sentence every time. It now
    consults the faculty model first, so the origin is whichever grounded
    target it found — falling back to the generic sweep only when the
    self-model has nothing measurable to want.
    """
    engine.orchestrator.soul.drive.name = "Competence"
    engine.orchestrator.soul.drive.urgency = 0.6

    goal = engine._check_soul_drives()
    assert goal is not None
    assert goal["origin"].startswith("intrinsic_competence")
    assert goal["objective"]

def test_check_soul_drives_curiosity(engine):
    engine.orchestrator.soul.drive.name = "Curiosity"
    engine.orchestrator.soul.drive.urgency = 0.7
    
    goal = engine._check_soul_drives()
    assert goal is not None
    assert goal["origin"] == "intrinsic_curiosity"


def test_scan_roadmap(engine):
    phase_dir = volition_module.config.paths.brain_dir / "phase-1"
    phase_dir.mkdir(parents=True, exist_ok=True)
    (phase_dir / "task.md").write_text("# Phase 1\nSome text", encoding="utf-8")
    engine.brain_base = volition_module.config.paths.brain_dir

    milestones = engine._scan_roadmap()
    assert milestones == ["Phase 1"]


def test_check_roadmap(monkeypatch, engine):
    engine.milestones = ["Phase 1", "Phase 2"]
    monkeypatch.setattr(volition_module.random, "random", lambda: 0.01)
    
    goal = engine._check_roadmap()
    
    assert goal is not None
    assert goal["origin"] == "intrinsic_evolution"
    assert "Phase 2" in goal["objective"]

def test_select_and_parse_goal(engine):
    goals = [
        {"objective": "Impulse goal", "origin": "impulse_question", "id": "1"},
        {"objective": "Strategic duty", "origin": "intrinsic_duty_strategic", "id": "2"}
    ]
    
    selected = engine._select_and_parse_goal(goals)
    # Strategic duty always overrides
    assert selected["origin"] == "intrinsic_duty_strategic"
    
    # Test selecting the first if no strategic
    goals = [
        {"objective": "Impulse goal", "origin": "impulse_question", "id": "1"},
    ]
    selected = engine._select_and_parse_goal(goals)
    assert selected["origin"] == "impulse_question"


##
