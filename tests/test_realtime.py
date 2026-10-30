"""RealTimeSimulator: the simulator paced on a clock. The tests drive it with a fake clock, so they are exact."""
import threading
import time

import numpy as np
import pytest

import tops.dynamic as dps
from tops.realtime import RealTimeSimulator
from tops.simulator import Simulator

DT = 5e-3
RATE = 50  # frames per second: a frame is 4 steps of 5 ms, i.e. 20 ms, so the simulation runs in real time
PERIOD = 1/RATE
T_END = 0.199999  # 40 steps, 10 frames


class FakeClock:
    """A clock that only moves when the simulator sleeps or when a step 'takes' time."""
    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, delay):
        self.sleeps.append(delay)
        self.now += delay

    def advance(self, seconds):
        self.now += seconds


def k2a_ps():
    import tops.ps_models.k2a as model_data
    ps = dps.PowerSystemModel(model=model_data.load())
    ps.init_dyn_sim()
    return ps


def simulator(clock, frame_costs=None, t_end=T_END, **kwargs):
    """A RealTimeSimulator on a fake clock, whose frames take frame_costs[frame] seconds (0 if not given)."""
    sim = RealTimeSimulator(k2a_ps(), dt=DT, rate=RATE, t_end=t_end, clock=clock, sleep=clock.sleep, **kwargs)
    steps = []
    costs = frame_costs or {}

    def step_took_time(sim):
        steps.append(sim.sol.t)
        if len(steps) % sim.steps_per_frame == 0:  # the end of a frame
            clock.advance(costs.get(len(steps)//sim.steps_per_frame - 1, 0))

    sim.interface_functions['took_time'] = step_took_time
    sim.steps = steps
    return sim


def test_a_frame_is_as_many_steps_as_make_it_real_time():
    sim = RealTimeSimulator(k2a_ps(), dt=DT, rate=50)
    assert sim.steps_per_frame == 4 and sim.realtime_factor == pytest.approx(1)
    assert RealTimeSimulator(k2a_ps(), dt=DT, rate=100).steps_per_frame == 2
    assert RealTimeSimulator(k2a_ps(), dt=DT, rate=1000).steps_per_frame == 1  # at least one step

    scaled = RealTimeSimulator(k2a_ps(), dt=DT, rate=50, steps_per_frame=8)
    assert scaled.steps_per_frame == 8 and scaled.realtime_factor == pytest.approx(2)  # twice as fast as real time


@pytest.mark.parametrize('kwargs', [{'rate': 0}, {'rate': -1}, {'steps_per_frame': 0}, {'max_lag': 0}, {'max_lag': -3}])
def test_nonsense_is_refused(kwargs):
    with pytest.raises(ValueError):
        RealTimeSimulator(k2a_ps(), **kwargs)


def test_the_frames_are_one_period_apart_and_the_simulation_ends_at_t_end():
    clock = FakeClock()
    sim = simulator(clock)
    start = clock.now
    sim.main_loop()

    assert sim.frame == 10 and len(sim.steps) == 40
    assert clock.now - start == pytest.approx(10*PERIOD)
    assert clock.sleeps == [pytest.approx(PERIOD)]*10
    assert sim.schedule_time == pytest.approx(10*PERIOD) and sim.lag_resets == 0


def test_the_time_a_frame_takes_is_slept_less_so_that_the_schedule_does_not_drift():
    clock = FakeClock()
    sim = simulator(clock, {frame: 0.013 for frame in range(10)})  # each frame takes 13 of its 20 ms
    start = clock.now
    sim.main_loop()

    assert clock.sleeps == [pytest.approx(PERIOD)] + [pytest.approx(0.007)]*9  # not 0.007, 0.014, ...
    assert clock.now - start == pytest.approx(10*PERIOD + 0.013)  # the schedule, and the last frame


def test_a_small_lag_is_caught_up_without_resetting_the_schedule():
    clock = FakeClock()
    sim = simulator(clock, {3: 0.1})  # frame 3 takes 5 periods, far less than the 25 that are tolerated
    start = clock.now
    sim.main_loop()

    assert sim.lag_resets == 0
    # Frames 0 to 3 wait their period, the five that follow are run at once, the last one waits again
    assert clock.sleeps == [pytest.approx(PERIOD)]*5
    assert clock.now - start == pytest.approx(10*PERIOD)  # back on the schedule


def test_a_stall_beyond_max_lag_resets_the_schedule_and_time_is_dropped_not_replayed():
    clock = FakeClock()
    sim = simulator(clock, {2: 1.0}, t_end=0.399999)  # 20 frames; frame 2 takes 50 periods, 25 are tolerated
    start = clock.now
    sim.main_loop()

    assert sim.lag_resets == 1 and len(sim.steps) == 80
    # Three frames wait, the frame after the stall runs at once (the reset), then every frame waits its period again
    assert clock.sleeps == [pytest.approx(PERIOD)]*19
    # The stall is on the clock once; the 49 periods it overran are not run again
    assert clock.now - start == pytest.approx(69*PERIOD)


def test_without_max_lag_a_stall_is_caught_up_in_a_burst():
    clock = FakeClock()
    sim = simulator(clock, {2: 1.0}, t_end=0.999999, max_lag=None)  # 50 frames
    sim.main_loop()

    assert sim.lag_resets == 0
    assert clock.sleeps == [pytest.approx(PERIOD)]*3  # then the 47 frames that remain run back to back


def test_the_steps_are_the_same_as_the_simulators():
    clock = FakeClock()
    sim = simulator(clock)
    sim.main_loop()

    reference = Simulator(k2a_ps(), dt=DT, t_end=T_END)
    steps = []
    reference.interface_functions['steps'] = lambda s: steps.append(s.sol.t)
    reference.main_loop()
    assert len(steps) == len(sim.steps) == 40
    assert np.array_equal(sim.sol.x, reference.sol.x)


# Speed, pause and stop
def wait_until(condition, timeout=5.0):
    end = time.monotonic() + timeout
    while not condition() and time.monotonic() < end:
        time.sleep(0.005)
    return condition()


def test_the_frames_come_faster_at_a_higher_speed():
    clock = FakeClock()
    sim = simulator(clock, speed=2)
    start = clock.now
    sim.main_loop()

    assert clock.sleeps == [pytest.approx(PERIOD/2)]*10
    assert clock.now - start == pytest.approx(5*PERIOD)
    assert len(sim.steps) == 40  # the same simulation, in half the time


def test_set_speed_changes_the_period_of_the_frames_that_follow():
    clock = FakeClock()
    sim = simulator(clock)

    def slow_down(sim):
        if len(sim.steps) == 20:  # the end of the fifth frame
            sim.set_speed(0.5)

    sim.interface_functions['slow_down'] = slow_down
    sim.main_loop()

    assert clock.sleeps == [pytest.approx(PERIOD)]*5 + [pytest.approx(2*PERIOD)]*5
    assert sim.speed == 0.5


@pytest.mark.parametrize('speed', [0, -1, float('nan')])
def test_a_speed_that_is_not_positive_is_refused(speed):
    with pytest.raises(ValueError, match='speed'):
        RealTimeSimulator(k2a_ps(), speed=speed)
    sim = RealTimeSimulator(k2a_ps())
    with pytest.raises(ValueError, match='speed'):
        sim.set_speed(speed)
    assert sim.speed == 1


def test_a_thread_runs_the_simulation_to_its_end():
    clock = FakeClock()
    sim = simulator(clock)
    sim.start()
    sim.join(timeout=10)
    assert not sim.is_alive() and sim.error is None
    assert sim.frame == 10 and len(sim.steps) == 40


def test_a_simulation_is_started_once():
    sim = RealTimeSimulator(k2a_ps(), rate=1, t_end=0.01)
    sim.start()
    with pytest.raises(RuntimeError, match='started'):
        sim.start()
    sim.stop()
    sim.join(timeout=10)


def test_stop_ends_a_simulation_that_is_waiting_for_its_next_frame():
    sim = RealTimeSimulator(k2a_ps(), dt=DT, rate=0.1)  # a frame every 10 s, no end
    sim.start()
    time.sleep(0.05)
    started = time.monotonic()
    sim.stop()
    sim.join(timeout=5)
    assert not sim.is_alive()
    assert time.monotonic() - started < 2  # not the 10 s of the sleep
    assert sim.stopped()


def test_a_simulation_stopped_before_it_starts_runs_no_frame():
    clock = FakeClock()
    sim = simulator(clock)
    sim.stop()
    sim.main_loop()
    assert sim.frame == 0 and sim.steps == []


def test_pause_holds_the_simulation_at_a_frame_boundary_and_resume_goes_on_without_lag():
    clock = FakeClock()
    sim = simulator(clock)
    sim.interface_functions['pause'] = lambda sim: sim.pause() if len(sim.steps) == 8 else None  # end of frame 2

    sim.start()
    assert wait_until(lambda: sim.paused and sim.frame == 2)
    time.sleep(0.05)
    assert sim.frame == 2 and len(sim.steps) == 8  # it is waiting, not running

    clock.advance(5.0)  # the pause lasts 5 s of the clock
    sim.resume()
    sim.join(timeout=10)

    assert sim.error is None and sim.frame == 10 and sim.lag_resets == 0
    assert clock.sleeps == [pytest.approx(PERIOD)]*10  # the pause was not made up for
    assert sim.schedule_time == pytest.approx(10*PERIOD)  # and is not on the schedule


def test_stop_ends_a_paused_simulation():
    clock = FakeClock()
    sim = simulator(clock)
    sim.interface_functions['pause'] = lambda sim: sim.pause() if len(sim.steps) == 8 else None
    sim.start()
    assert wait_until(lambda: sim.paused and sim.frame == 2)
    sim.stop()
    sim.join(timeout=5)
    assert not sim.is_alive() and sim.frame == 2


def test_an_error_in_the_thread_is_kept_and_ends_the_simulation():
    clock = FakeClock()
    sim = simulator(clock)

    def fail(sim):
        if len(sim.steps) == 6:
            raise RuntimeError('boom')

    sim.interface_functions['fail'] = fail
    sim.start()
    sim.join(timeout=10)
    assert not sim.is_alive()
    assert isinstance(sim.error, RuntimeError) and str(sim.error) == 'boom'
    assert len(sim.steps) == 6
