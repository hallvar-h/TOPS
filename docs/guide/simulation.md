# Running a simulation

A TOPS simulation always has the same three parts:

1. a **`PowerSystemModel`**, initialised with `init_dyn_sim()`,
2. a **solver** that steps the states forward in time, and
3. a **loop** that calls the solver and does something at every step:
   applying events, changing inputs, storing results.

The `Simulator` class provides the loop. You can also write the loop
yourself, as shown [at the end of this page](#writing-your-own-loop).

The examples on this page use `sm_ib`, the single machine against an
infinite bus, because it simulates quickly.

## The Simulator

```python
from tops.simulator import Simulator

sim = Simulator(ps, dt=5e-3, t_end=10, events=None)
sim.main_loop()
```

| Argument | Meaning |
|---|---|
| `ps` | An initialised `PowerSystemModel`. |
| `dt` | Time step in seconds (default 5 ms). |
| `t_end` | End time in seconds. With the default, `inf`, the loop runs until stopped. |
| `solver` | Solver class, `ModifiedEulerDAE` by default. See [Solvers](#solvers). |
| `events` | An event table: see the Quickstart and the Events guide. |

At every step, `Simulator` applies the events that are due, advances the
solver by `dt`, then calls each function in `sim.interface_functions` with
the simulator as argument. That dictionary is where you hook in your own
code.

## Recording results

`ResultKeeper` is an interface function that stores results. Give it named
functions with the signature `f(x, v)`. Every model output method works,
for example `ps.gen.GEN.speed` or `ps.lines.Line.p_from`. It returns one
array per time step, with one value per unit.

```python exec="on" source="above" session="sim" result="text"
import numpy as np
import tops.dynamic as dps
import tops.ps_models.sm_ib as model_data
from tops.simulator import Simulator, ResultKeeper

ps = dps.PowerSystemModel(model=model_data.load())
ps.init_dyn_sim()

sim = Simulator(ps, dt=1e-2, t_end=10)
res = ResultKeeper(sim, v_t=ps.gen.GEN.v_t_abs, q_e=ps.gen.GEN.q_e)
sim.interface_functions["results"] = res.update
```

`res.get_dataframe()` returns **all** states as a pandas DataFrame indexed
by time, with a `(unit, state)` column for each state, for example
`("G1", "speed")`.

## Changing inputs during the simulation

Model inputs that aren't connected to another model can be set at any time
with `set_input(name, value)`. Here an interface function raises the voltage
reference of the AVR by 0.05 pu at t = 1 s. The `bias` input of the `SEXS`
exciter is added to its voltage set-point.

```python exec="on" source="above" session="sim" result="text"
avr = ps.avr.SEXS
bias_0 = avr.bias(ps.x0, ps.v0).copy()

def voltage_step(sim):
    if sim.sol.t >= 1.0:
        avr.set_input("bias", bias_0 + 0.05)

sim.interface_functions["voltage_step"] = voltage_step
sim.main_loop()

df = res.get_dataframe()
print(df[[("G1", "speed"), ("G1", "e_q_t")]].iloc[::200])
```

```python exec="on" source="above" session="sim" html="1"
--8<-- "plot.py"
import matplotlib.pyplot as plt

v_t = np.array(res.store["v_t"])[:, 0]   # column 0 is G1, column 1 the infinite bus
q_e = np.array(res.store["q_e"])[:, 0]

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 4))
ax1.plot(res.t, v_t)
ax1.set_ylabel("Terminal voltage [pu]")
ax2.plot(res.t, q_e)
ax2.set_ylabel("Reactive power [pu]")
ax2.set_xlabel("Time [s]")
plt.show()
```

## Solvers

The solvers are in `tops.solvers`. `Simulator` passes the model's
differential *and* algebraic equations to the solver, so it needs one of the
**DAE** solvers:

| Solver | Method | Works with `Simulator` |
|---|---|---|
| `ModifiedEulerDAE` | Heun's method (2nd order), network solved every step. The default. | yes |
| `EulerDAE` | Forward Euler (1st order), network solved every step. | yes |
| `ModifiedEuler`, `Euler`, `SimpleRK4` | ODE solvers for `f(t, x)`, e.g. `ps.ode_fun`. | no |

All of them use a fixed time step. 5–10 ms is usually a good choice for
electromechanical dynamics.

## Writing your own loop

For full control, step the solver yourself. The solver takes the state
derivatives, the algebraic solver, the start time, the initial states and
the initial voltages. `EventSchedule` gives you the same event tables as the
`Simulator`.

```python exec="on" source="above" result="text"
import tops.dynamic as dps
import tops.solvers as dps_sol
import tops.ps_models.sm_ib as model_data
from tops.simulator import EventSchedule

ps = dps.PowerSystemModel(model=model_data.load())
ps.init_dyn_sim()

events = EventSchedule([
    ["t",  "category", "model", "unit", "action"],
    [1.0,  "buses",    "",      "B1",   "short_circuit"],
    [1.05, "buses",    "",      "B1",   "clear_short_circuit"],
])
events.check(ps)   # fail early if a unit or action does not exist

sol = dps_sol.ModifiedEulerDAE(
    ps.state_derivatives, ps.solve_algebraic, 0, ps.x0.copy(), ps.v0.copy(), max_step=1e-2
)

t, speed = [], []
while sol.t < 3:
    events.apply_due(ps, sol.t)
    sol.step()
    t.append(sol.t)
    speed.append(ps.gen.GEN.speed(sol.x, sol.v)[0])

print(f"Max speed deviation of G1: {max(speed):.4f} pu at t = {t[speed.index(max(speed))]:.2f} s")
```

`sol.x` holds the states and `sol.v` the bus voltages of the current step.
