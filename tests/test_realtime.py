"""RealTimeSimulator: the simulator paced on a clock. The tests drive it with a fake clock, so they are exact."""
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
