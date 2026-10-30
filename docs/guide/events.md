# Events

Events are discrete changes during a simulation, such as tripping a line or
applying a short circuit. You list them in an **event table** and hand it to
the `Simulator`. They are not part of the model data, so the same model can
be run with different event sequences.

## The event table

An event table has the columns `t`, `category`, `model`, `unit` and
`action`. Like model tables, it is a list with a header row, or a pandas
DataFrame:

```python
events = [
    ["t",  "category", "model", "unit",   "action"],
    [1.0,  "buses",    "",      "B7",     "short_circuit"],
    [1.1,  "buses",    "",      "B7",     "clear_short_circuit"],
    [1.1,  "lines",    "Line",  "L7-8-1", "disconnect"],
]
```

| Column | Meaning |
|---|---|
| `t` | Time in seconds. Rows don't need to be sorted. |
| `category` | `"buses"`, or a model category such as `"lines"`. |
| `model` | The model type within the category, e.g. `"Line"`. Ignored for `"buses"`. |
| `unit` | Name of the bus or unit. |
| `action` | What happens, see the table below. |

An event is applied at the first time step where the simulation time is at
least `t`. With a time step `dt`, the event can therefore come up to `dt`
late.

## Available events

| Category | Model | Actions | Effect |
|---|---|---|---|
| `buses` | | `short_circuit`, `clear_short_circuit` | Connects or removes a fault admittance (1e6 pu) from the bus to ground. |
| `lines` | `Line` | `disconnect`, `connect` | Removes the line from the network or puts it back. |

Applying an event twice in a row, such as disconnecting a line that is
already open, has no effect.

## Example: a line trip

The two-area system has two parallel lines between buses 7 and 8. Tripping
one of them weakens the tie between the areas, and the system settles at a
new operating point after some oscillations.

```python exec="on" source="above" session="events" result="text"
import tops.dynamic as dps
import tops.ps_models.k2a as model_data
from tops.simulator import Simulator, ResultKeeper

ps = dps.PowerSystemModel(model=model_data.load())
ps.init_dyn_sim()

events = [
    ["t", "category", "model", "unit",   "action"],
    [1.0, "lines",    "Line",  "L7-8-1", "disconnect"],
]

sim = Simulator(ps, dt=1e-2, t_end=5, events=events)
res = ResultKeeper(sim, angle=ps.gen.GEN.angle, p_tie=ps.lines.Line.p_from)
sim.interface_functions["results"] = res.update
sim.main_loop()

names = [str(name) for name in ps.lines.Line.par["name"]]
print("Lines:", names)
```

```python exec="on" source="above" session="events" html="1"
--8<-- "plot.py"
import numpy as np
import matplotlib.pyplot as plt

p_tie = np.array(res.store["p_tie"])
fig, ax = plt.subplots(figsize=(7, 3.5))
for name in ["L7-8-1", "L7-8-2"]:
    ax.plot(res.t, p_tie[:, names.index(name)], label=name)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Active power from bus 7 [pu]")
ax.legend()
plt.show()
```

After the trip, `L7-8-1` carries nothing and `L7-8-2` takes over its share
of the power transfer.

## Checking events before the run

`Simulator` checks the whole event table when it is created. A misspelt
unit or action raises an error straight away, rather than in the middle of a
long run:

```python exec="on" source="above" result="text"
import tops.dynamic as dps
import tops.ps_models.k2a as model_data
from tops.simulator import Simulator

ps = dps.PowerSystemModel(model=model_data.load())
ps.init_dyn_sim()

try:
    Simulator(ps, events=[
        ["t", "category", "model", "unit",  "action"],
        [1.0, "lines",    "Line",  "L7-8",  "disconnect"],
    ])
except KeyError as error:
    print("KeyError:", error)
```

## Applying an event directly

Outside an event table, `ps.apply_event` applies an event immediately. This
is useful in interface functions or in your own loop, and it accepts extra
keyword arguments, such as a different fault admittance:

```python
ps.apply_event("lines", "Line", "L7-8-1", "disconnect")
ps.apply_event("buses", None, "B7", "short_circuit", y=50)  # weaker fault
ps.apply_event("buses", None, "B7", "clear_short_circuit")
```

## Events in your own loop

`tops.simulator.EventSchedule` reads the same tables. Call
`apply_due(ps, t)` before every solver step, as in
[Writing your own loop](simulation.md#writing-your-own-loop).

## Adding events to a model

Events belong to models. A model class lists its events by setting
`events_class` to a subclass of `tops.events.ModelEvents`. That subclass
names the allowed actions in `actions` and has one method for each. A
method may change its own model, and returns a list of *effects* on the rest
of the system (currently `AdmittanceChange`). `Line` and `LineEvents` in
`tops/dyn_models/lines.py` are a short, complete example.
