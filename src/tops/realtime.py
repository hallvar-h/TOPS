"""A simulator that runs in real time, without any user interface.

The simulation runs in frames of a few steps, one every 1/rate seconds of the clock. Frames are on a schedule that
does not drift: the time a frame takes is not added to the wait for the next one. A simulation that falls behind
catches up, but only up to max_lag: after a longer stall the schedule starts over from now and the time that was lost is
dropped, not replayed in a burst.

    sim = RealTimeSimulator(ps, dt=5e-3, rate=50)
    sim.main_loop()                    # or sim.start() to run it in a thread, sim.pause(), sim.stop() ...
"""
import inspect
import queue
import threading
import time

from tops.simulator import Simulator


class RealTimeSimulator(Simulator):
    def __init__(self, ps, dt=5e-3, rate=50.0, steps_per_frame=None, speed=1.0, max_lag=25, clock=time.monotonic,
                 sleep=None, **kwargs):
        """
        :param rate: Frames per second of the clock.
        :param steps_per_frame: Steps of dt in a frame. The default makes the simulation run in real time, i.e. the
            frame is as much simulated time as the clock time between two frames. Another number is a fixed time scale,
            see realtime_factor.
        :param speed: Frames come speed times as fast as rate says, so 2 is twice as fast as real time. See set_speed.
        :param max_lag: How far behind its schedule, in frame periods, the simulation may fall before the schedule
            starts over from now. None: never, the simulation always catches up.
        :param clock: Function returning the time in seconds, and sleep: function waiting for that many seconds, which
            by default stop() interrupts. Other than the default, e.g. a clock that is moved by hand, they make the
            pacing testable.
        :param kwargs: Passed on to Simulator (t_end, events, ...).

        Attributes: frame (frames run), schedule_time (the time of the last frame on the schedule, in seconds from the
        start), lag (seconds the last frame was behind its schedule), lag_resets (times the schedule was started over),
        error (the exception that ended a simulation run by start(), if one did), rejected (events from submit() that
        failed when they were applied, as pairs of the event and the exception).
        """
        if rate <= 0:
            raise ValueError(f'rate must be positive, got {rate}.')
        if steps_per_frame is not None and steps_per_frame < 1:
            raise ValueError(f'steps_per_frame must be at least 1, got {steps_per_frame}.')
        if max_lag is not None and max_lag <= 0:
            raise ValueError(f'max_lag must be positive, or None, got {max_lag}.')
        self._check_speed(speed)
        super().__init__(ps, dt=dt, **kwargs)
        self.rate = rate
        self.speed = speed
        self.steps_per_frame = max(1, round(1/(rate*dt))) if steps_per_frame is None else steps_per_frame
        self.max_lag = max_lag
        self._stop_event = threading.Event()
        self._running = threading.Event()
        self._running.set()
        self._thread = None
        self._submitted = queue.SimpleQueue()
        self._clock = clock
        self._sleep = self._stop_event.wait if sleep is None else sleep
        self.error = None
        self.rejected = []
        self.frame = 0
        self.schedule_time = 0.0
        self.lag = 0.0
        self.lag_resets = 0

    @property
    def realtime_factor(self):
        """Simulated seconds per second of the clock: 1 is real time."""
        return self.steps_per_frame*self.dt*self.rate

    @staticmethod
    def _check_speed(speed):
        if not speed > 0:  # also catches nan
            raise ValueError(f'speed must be positive, got {speed}. To hold the simulation, use pause().')

    def set_speed(self, speed):
        """Change the speed, from now on; it takes effect at the next frame."""
        self._check_speed(speed)
        self.speed = speed

    @property
    def paused(self):
        return not self._running.is_set()

    def pause(self):
        """Hold the simulation before its next frame, until resume() or stop()."""
        self._running.clear()

    def resume(self):
        """Go on after pause(): the schedule starts over from now, so the pause is not made up for."""
        self._running.set()

    def stop(self):
        """End the simulation, also if it is waiting for its next frame or paused. A stopped simulator stays stopped."""
        self._stop_event.set()
        self._running.set()

    def stopped(self):
        return self._stop_event.is_set()

    def submit(self, category, model, unit, action, **kwargs):
        """Queue an event, to be applied as ps.apply_event(category, model, unit, action, **kwargs) just before the
        next frame, in the order of submission. May be called from any thread.

        An event that cannot exist (unknown unit, action, model or arguments) raises here, in the caller's thread. One
        that fails only when it is applied, e.g. a power that is not a number, is kept in rejected, and the simulation
        goes on.
        """
        events = self.ps.get_events(category, model)
        events.check(unit, action)
        inspect.signature(getattr(events, action)).bind(unit, **kwargs)
        self._submitted.put((category, model, unit, action, kwargs))

    def _apply_submitted(self):
        while not self._submitted.empty():
            event = self._submitted.get_nowait()
            category, model, unit, action, kwargs = event
            try:
                self.ps.apply_event(category, model, unit, action, **kwargs)
            except Exception as error:
                self.rejected.append((event, error))

    def start(self):
        """Run main_loop in a thread. An exception that ends it is kept in error."""
        if self._thread is not None:
            raise RuntimeError('The simulation has already been started.')
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            self.main_loop()
        except Exception as error:
            self.error = error

    def join(self, timeout=None):
        """Wait for the thread started by start() to end."""
        if self._thread is not None:
            self._thread.join(timeout)

    def is_alive(self):
        return self._thread is not None and self._thread.is_alive()

    def main_loop(self):
        start = due = self._clock()
        while not self.stopped() and self.sol.t < self.t_end:
            if self.paused:
                paused_at = self._clock()
                self._running.wait()
                due = self._clock()
                start += due - paused_at
                continue
            period = 1/(self.rate*self.speed)
            due += period
            delay = due - self._clock()
            self.lag = max(0.0, -delay)
            if delay > 0:
                self._sleep(delay)
            elif self.max_lag is not None and -delay > self.max_lag*period:
                due = self._clock()
                self.lag_resets += 1
            self.schedule_time = due - start
            self._apply_submitted()
            for _ in range(self.steps_per_frame):
                if self.stopped() or self.sol.t >= self.t_end:
                    break
                self.make_simulation_step()
            self.frame += 1
