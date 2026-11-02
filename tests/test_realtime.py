"""RealTimeSimulator: the simulator paced on a clock. The tests drive it with a fake clock, so they are exact."""
import threading
import time

import numpy as np
import pytest

import tops.dynamic as dps
from k2a_variants import TRAFO, initialized, trafo_model
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


def simulator(clock, frame_costs=None, t_end=T_END, ps=None, **kwargs):
    """A RealTimeSimulator on a fake clock, whose frames take frame_costs[frame] seconds (0 if not given)."""
    sim = RealTimeSimulator(ps or k2a_ps(), dt=DT, rate=RATE, t_end=t_end, clock=clock, sleep=clock.sleep, **kwargs)
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


# Events submitted while it runs
def record_when_applied(sim, ps, what):
    """Records, after every step, what(ps), as a list with one entry per step."""
    seen = []
    sim.interface_functions['seen'] = lambda sim: seen.append(what(ps))
    return seen


def line_out(ps):
    return not ps.lines['Line'].connected[list(ps.lines['Line'].par['name']).index('L7-8-1')]


def test_a_submitted_event_is_applied_just_before_the_next_frame():
    clock = FakeClock()
    sim = simulator(clock)
    seen = record_when_applied(sim, sim.ps, line_out)
    sim.interface_functions['submit'] = lambda sim: (
        sim.submit('lines', 'Line', 'L7-8-1', 'disconnect') if len(sim.steps) == 6 else None)  # in frame 1 (steps 5-8)
    sim.main_loop()

    assert seen[:8] == [False]*8 and seen[8:] == [True]*32  # in force from the first step of frame 2, never mid-frame


def test_events_are_applied_in_the_order_they_were_submitted_with_their_arguments():
    clock = FakeClock()
    sim = simulator(clock)

    def submit(sim):
        if len(sim.steps) == 2:
            sim.submit('lines', 'Line', 'L7-8-1', 'disconnect')
            sim.submit('loads', 'Load', 'L1', 'set_power', P=1000.5)
            sim.submit('loads', 'Load', 'L1', 'set_power', P=1100.0, Q=90.0)
            sim.submit('lines', 'Line', 'L7-8-1', 'connect')

    sim.interface_functions['submit'] = submit
    sim.main_loop()

    load = sim.ps.loads['Load']
    i = list(load.par['name']).index('L1')
    assert load.par['P'][i] == 1100.0 and load.par['Q'][i] == 90.0
    assert not line_out(sim.ps)  # disconnected, then connected again
    assert sim.rejected == []


def test_a_line_trip_a_transformer_trip_and_a_load_change_are_applied_between_frames():
    clock = FakeClock()
    ps = initialized(trafo_model('Trafo'))
    sim = simulator(clock, ps=ps)
    seen = record_when_applied(sim, ps, lambda ps: (
        line_out(ps), not ps.trafos['Trafo'].connected[-1], float(ps.loads['Load'].par['P'][0])))

    def submit(sim):
        if len(sim.steps) == 2:
            sim.submit('lines', 'Line', 'L7-8-1', 'disconnect')
            sim.submit('trafos', 'Trafo', TRAFO, 'disconnect')
            sim.submit('loads', 'Load', 'L1', 'set_power', P=900.0)

    sim.interface_functions['submit'] = submit
    sim.main_loop()

    assert seen[:4] == [(False, False, 967.0)]*4  # frame 0: nothing yet
    assert seen[4:] == [(True, True, 900.0)]*(len(seen) - 4)  # from frame 1, all three at once


def test_events_may_be_submitted_from_another_thread_while_the_simulation_is_paused():
    clock = FakeClock()
    sim = simulator(clock)
    seen = record_when_applied(sim, sim.ps, line_out)
    sim.interface_functions['pause'] = lambda sim: sim.pause() if len(sim.steps) == 8 else None
    sim.start()
    assert wait_until(lambda: sim.paused and sim.frame == 2)

    submitter = threading.Thread(target=sim.submit, args=('lines', 'Line', 'L7-8-1', 'disconnect'))
    submitter.start()
    submitter.join(timeout=5)
    assert seen == [False]*8  # nothing is applied while it is paused
    sim.resume()
    sim.join(timeout=10)

    assert sim.error is None and seen[:8] == [False]*8 and seen[8:] == [True]*32


def test_a_submitted_short_circuit_on_a_bus_takes_its_arguments():
    clock = FakeClock()
    sim = simulator(clock)
    sim.interface_functions['submit'] = lambda sim: (
        sim.submit('buses', '', 'B8', 'short_circuit', y=100) if len(sim.steps) == 1 else None)
    sim.main_loop()
    assert sim.ps.bus_events.y_fault[list(sim.ps.buses['name']).index('B8')] == 100


@pytest.mark.parametrize('args, kwargs, error', [
    (('lines', 'Line', 'NoSuchLine', 'disconnect'), {}, KeyError),
    (('lines', 'Line', 'L7-8-1', 'explode'), {}, ValueError),
    (('lines', 'NoSuchModel', 'L7-8-1', 'disconnect'), {}, KeyError),
    (('nowhere', 'Line', 'L7-8-1', 'disconnect'), {}, KeyError),
    (('loads', 'Load', 'L1', 'set_power'), {'volts': 1}, TypeError),
])
def test_an_event_that_cannot_exist_is_refused_in_the_thread_that_submits_it(args, kwargs, error):
    clock = FakeClock()
    sim = simulator(clock)
    with pytest.raises(error):
        sim.submit(*args, **kwargs)
    sim.main_loop()
    assert sim.frame == 10 and sim.rejected == [] and not line_out(sim.ps)


def test_an_event_that_fails_when_applied_is_kept_in_rejected_and_the_simulation_goes_on():
    clock = FakeClock()
    sim = simulator(clock)
    sim.interface_functions['submit'] = lambda sim: (
        [sim.submit('loads', 'Load', 'L1', 'set_power', P=float('nan')),
         sim.submit('lines', 'Line', 'L7-8-1', 'disconnect')] if len(sim.steps) == 2 else None)
    sim.main_loop()

    assert sim.error is None and sim.frame == 10
    (event, error), = sim.rejected
    assert event[:4] == ('loads', 'Load', 'L1', 'set_power') and list(event[4]) == ['P']  # nan is not equal to nan
    assert isinstance(error, ValueError) and 'P must be a finite number' in str(error)
    assert line_out(sim.ps)  # the event after it was applied all the same


# on_frame and the divergence guard
def test_on_frame_is_called_after_each_frame_with_its_place_on_the_schedule():
    clock = FakeClock()
    frames = []
    sim = simulator(clock, on_frame=lambda sim: frames.append((sim.frame, len(sim.steps), sim.schedule_time, sim.lag)))
    sim.main_loop()

    assert [frame[:2] for frame in frames] == [(k, 4*k) for k in range(1, 11)]
    assert [frame[2] for frame in frames] == [pytest.approx(k*PERIOD) for k in range(1, 11)]
    assert all(frame[3] == 0 for frame in frames)


def test_on_frame_can_stop_the_simulation():
    clock = FakeClock()
    sim = simulator(clock, on_frame=lambda sim: sim.stop() if sim.frame == 3 else None)
    sim.main_loop()
    assert sim.frame == 3 and len(sim.steps) == 12


def test_an_error_in_on_frame_ends_the_simulation_and_is_kept_when_it_runs_in_a_thread():
    def on_frame(sim):
        if sim.frame == 2:
            raise RuntimeError('boom')

    with pytest.raises(RuntimeError, match='boom'):
        simulator(FakeClock(), on_frame=on_frame).main_loop()

    sim = simulator(FakeClock(), on_frame=on_frame)
    sim.start()
    sim.join(timeout=10)
    assert isinstance(sim.error, RuntimeError) and sim.frame == 2


def test_a_state_that_is_not_finite_ends_the_simulation_before_on_frame_sees_it():
    clock = FakeClock()
    frames = []
    sim = simulator(clock, on_frame=lambda sim: frames.append(sim.frame))
    sim.interface_functions['diverge'] = lambda sim: (
        sim.sol.y.__setitem__(0, np.nan) if len(sim.steps) == 6 else None)  # in frame 1

    with pytest.raises(ArithmeticError, match='not finite'):
        sim.main_loop()
    assert frames == [1]  # frame 1 was the last that was fine, the diverged frame 2 never reached on_frame


def test_a_bus_voltage_above_max_voltage_ends_the_simulation():
    def overvoltage(sim):
        if len(sim.steps) == 8:  # the last step of frame 2
            sim.sol.v[0] = 10

    sim = simulator(FakeClock())
    sim.interface_functions['overvoltage'] = overvoltage
    with pytest.raises(ArithmeticError, match='voltage'):
        sim.main_loop()
    assert sim.frame == 2

    unlimited = simulator(FakeClock(), max_voltage=None)  # the check is for the caller to want
    unlimited.interface_functions['overvoltage'] = overvoltage
    unlimited.main_loop()
    assert unlimited.frame == 10


def test_the_guard_ends_a_simulation_run_by_a_thread_and_keeps_the_error():
    sim = simulator(FakeClock())
    sim.interface_functions['overvoltage'] = lambda sim: sim.sol.v.__setitem__(0, 10) if len(sim.steps) == 8 else None
    sim.start()
    sim.join(timeout=10)
    assert isinstance(sim.error, ArithmeticError) and 'voltage' in str(sim.error)


def test_a_healthy_simulation_passes_the_guard_at_every_frame():
    sim = simulator(FakeClock())
    sim.main_loop()
    assert sim.frame == 10 and np.abs(sim.sol.v).max() < sim.max_voltage == 5.0


@pytest.mark.parametrize('max_voltage', [0, -1])
def test_a_max_voltage_that_is_not_positive_is_refused(max_voltage):
    with pytest.raises(ValueError, match='max_voltage'):
        RealTimeSimulator(k2a_ps(), max_voltage=max_voltage)
