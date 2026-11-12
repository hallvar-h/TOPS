# Real-time simulation

`RealTimeSimulator` runs a simulation on the clock, without any user
interface. It is a `Simulator` that advances in frames of a few steps, one
frame every `1/rate` seconds. It can run in a thread, so that your own code
can pause it, change its speed or send it events while it runs.

```python
from tops.realtime import RealTimeSimulator

sim = RealTimeSimulator(ps, dt=5e-3, rate=50)   # 50 frames per second, of 4 steps: real time
sim.start()                                     # runs in a thread
sim.submit("lines", "Line", "L7-8-1", "disconnect")  # applied just before the next frame
sim.pause()
sim.resume()
sim.set_speed(2)                                # twice as fast as real time
sim.stop()
sim.join()
```

`sim.main_loop()` runs the simulation in the calling thread instead, until
`t_end` is reached.

## Frames and speed

By default the frame is as much simulated time as the clock time between two
frames, so the simulation runs in real time. With `dt=5e-3` and `rate=50` a
frame is 4 steps. Give `steps_per_frame` for another fixed time scale; the
resulting ratio is `sim.realtime_factor`. `speed` makes the frames come
faster or slower than `rate` says, and `set_speed` changes it while running.

## Staying on schedule

The frames are on a schedule that does not drift: the time a frame takes is
not added to the wait for the next one. A simulation that falls behind
catches up, but only up to `max_lag` frame periods. After a longer stall the
time lost is dropped instead of being replayed in a burst, and
`sim.lag_resets` counts how often that happened. `sim.lag` is how far behind
the last frame was. `pause()` and `resume()` also start the schedule over, so
a pause is not made up for.

## Sending events

`submit(category, model, unit, action, **kwargs)`, or
`submit(event_function, unit, **kwargs)` such as
`sim.submit(ps.lines.Line.events.disconnect, "L7-8-1")`, queues an event from
any thread. The events are applied in the order of submission, just before
the next frame, with the same arguments as `ps.apply_event`, see
[Events](events.md). An event that cannot exist, such as an unknown unit or
action, raises in the caller's thread. One that fails only when it is
applied, such as a power that is not a number, is kept in `sim.rejected` and
the simulation goes on.

## Reading the state and ending the run

`on_frame` is called with the simulator after each frame. It is the place to
read the state, and to `pause()` or `stop()`. An exception in it ends the
simulation.

A state that is not finite, or a bus voltage above `max_voltage`, ends the
simulation with an `ArithmeticError`. When it runs in a thread the exception
is kept in `sim.error`. Set `max_voltage=None` to not limit the voltages.

```python
def show(sim):
    print(sim.sol.t, abs(sim.sol.v).max())

sim = RealTimeSimulator(ps, dt=5e-3, rate=50, t_end=10, on_frame=show)
sim.start()
sim.join()
if sim.error is not None:
    raise sim.error
```
