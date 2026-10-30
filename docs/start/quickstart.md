# Quickstart

This page runs a complete dynamic simulation: a three-phase short circuit in
Kundur's two-area system. You will

1. load one of the bundled test systems,
2. solve the power flow and initialise the dynamic models,
3. simulate a fault with the `Simulator`, and
4. plot the generator speeds.

Every block below is executed when the documentation is built, so the output
you see is what the code really produces. Run the blocks in order in a script
or notebook and you get the same result.

## 1. Load a model

A *model* in TOPS is a plain Python dictionary that describes the buses,
lines, generators, controllers and so on. Several test systems come with the
package in `tops.ps_models`. Here we use `k2a`, the two-area system from
Kundur's *Power System Stability and Control*.

```python exec="on" source="above" session="quickstart" result="text"
import tops.dynamic as dps
import tops.ps_models.k2a as model_data

model = model_data.load()
print(list(model.keys()))
```

## 2. Build and initialise the power system

`PowerSystemModel` turns the dictionary into a simulation model.
`init_dyn_sim()` solves the power flow and sets every dynamic model to a
steady state that matches it.

```python exec="on" source="above" session="quickstart" result="text"
ps = dps.PowerSystemModel(model=model)
ps.init_dyn_sim()

print(f"{ps.n_bus} buses, {ps.n_states} states")
print("Generators:", ps.gen.GEN.par["name"])
print("Bus voltages [pu]:", abs(ps.v0).round(3))
```

Dynamic models are grouped by category (`gen`, `avr`, `gov`, `pss`, `lines`,
`loads`, ...) and then by model type, so `ps.gen.GEN` is the set of all
generators of type `GEN`.

## 3. Simulate a short circuit

Events are given as a table with a header row. Here bus `B7` is short
circuited at t = 1 s and the fault is cleared 100 ms later. The
`ResultKeeper` records what you ask for at every time step: here the speed of
each generator.

```python exec="on" source="above" session="quickstart" result="text"
from tops.simulator import Simulator, ResultKeeper

events = [
    ["t", "category", "model", "unit", "action"],
    [1.0, "buses",    "",      "B7",   "short_circuit"],
    [1.1, "buses",    "",      "B7",   "clear_short_circuit"],
]

sim = Simulator(ps, dt=1e-2, t_end=5, events=events)
res = ResultKeeper(sim, gen_speed=ps.gen.GEN.speed)
sim.interface_functions["results"] = res.update
sim.main_loop()

print(f"Simulated {len(res.t)} time steps up to t = {res.t[-1]:.2f} s")
```

## 4. Plot the results

`res.t` holds the time stamps and `res.store["gen_speed"]` one array of
generator speeds (deviation from synchronous speed, in pu) per time step.

```python exec="on" source="above" session="quickstart" html="1"
--8<-- "plot.py"
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(res.t, res.store["gen_speed"], label=ps.gen.GEN.par["name"])
ax.set_xlabel("Time [s]")
ax.set_ylabel("Speed deviation [pu]")
ax.legend()
plt.show()
```

The generators move in pairs: G1 and G2 in one area, G3 and G4 in the other,
with the first area leading. The slow oscillation of one group relative to the
other, at well below 1 Hz, is the inter-area mode that makes `k2a` a popular
test case.
