from radarsim.core.clock import SimulationClock


def test_step_advances_time_by_timestep() -> None:
    clock = SimulationClock(timestep=0.5, duration=2.0)
    assert clock.step() == 0.5
    assert clock.time == 0.5
    assert clock.step() == 1.0


def test_is_finished_false_before_duration() -> None:
    clock = SimulationClock(timestep=1.0, duration=3.0)
    clock.step()
    clock.step()
    assert not clock.is_finished()


def test_is_finished_true_at_duration() -> None:
    clock = SimulationClock(timestep=1.0, duration=3.0)
    for _ in range(3):
        clock.step()
    assert clock.is_finished()


def test_is_finished_handles_float_accumulation() -> None:
    # 0.1 + 0.1 + 0.1 != 0.3 exactly in floating point; is_finished must
    # tolerate that instead of running one extra/short step.
    clock = SimulationClock(timestep=0.1, duration=0.3)
    for _ in range(3):
        clock.step()
    assert clock.is_finished()


def test_reset_zeroes_time() -> None:
    clock = SimulationClock(timestep=1.0, duration=5.0)
    clock.step()
    clock.step()
    clock.reset()
    assert clock.time == 0.0
    assert not clock.is_finished()
