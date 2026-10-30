"""A simulator that runs in real time, without any user interface.

The simulation runs in frames of a few steps, one every 1/rate seconds of the clock. Frames are on a schedule that
does not drift: the time a frame takes is not added to the wait for the next one. A simulation that falls behind
catches up, but only up to max_lag: after a longer stall the schedule starts over from now and the time that was lost is
dropped, not replayed in a burst.

    sim = RealTimeSimulator(ps, dt=5e-3, rate=50)
    sim.main_loop()
"""
import time

from tops.simulator import Simulator


class RealTimeSimulator(Simulator):
    def __init__(self, ps, dt=5e-3, rate=50.0, steps_per_frame=None, max_lag=25, clock=time.monotonic,
                 sleep=time.sleep, **kwargs):
        """
        :param rate: Frames per second of the clock.
        :param steps_per_frame: Steps of dt in a frame. The default makes the simulation run in real time, i.e. the
            frame is as much simulated time as the clock time between two frames. Another number is a fixed time scale,
            see realtime_factor.
        :param max_lag: How far behind its schedule, in frame periods, the simulation may fall before the schedule
            starts over from now. None: never, the simulation always catches up.
        :param clock: Function returning the time in seconds, and sleep: function waiting for that many seconds.
            Other than the default, e.g. a clock that is moved by hand, they make the pacing testable.
        :param kwargs: Passed on to Simulator (t_end, events, ...).

        Attributes: frame (frames run), schedule_time (the time of the last frame on the schedule, in seconds from the
        start), lag (seconds the last frame was behind its schedule), lag_resets (times the schedule was started over).
        """
        if rate <= 0:
            raise ValueError(f'rate must be positive, got {rate}.')
        if steps_per_frame is not None and steps_per_frame < 1:
            raise ValueError(f'steps_per_frame must be at least 1, got {steps_per_frame}.')
        if max_lag is not None and max_lag <= 0:
            raise ValueError(f'max_lag must be positive, or None, got {max_lag}.')
        super().__init__(ps, dt=dt, **kwargs)
        self.rate = rate
        self.steps_per_frame = max(1, round(1/(rate*dt))) if steps_per_frame is None else steps_per_frame
        self.max_lag = max_lag
        self._clock = clock
        self._sleep = sleep
        self.frame = 0
        self.schedule_time = 0.0
        self.lag = 0.0
        self.lag_resets = 0

    @property
    def realtime_factor(self):
        """Simulated seconds per second of the clock: 1 is real time."""
        return self.steps_per_frame*self.dt*self.rate

    def main_loop(self):
        period = 1/self.rate
        start = due = self._clock()
        while not self.stopped() and self.sol.t < self.t_end:
            due += period
            delay = due - self._clock()
            self.lag = max(0.0, -delay)
            if delay > 0:
                self._sleep(delay)
            elif self.max_lag is not None and -delay > self.max_lag*period:
                due = self._clock()
                self.lag_resets += 1
            self.schedule_time = due - start
            for _ in range(self.steps_per_frame):
                if self.stopped() or self.sol.t >= self.t_end:
                    break
                self.make_simulation_step()
            self.frame += 1
