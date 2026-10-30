# User-defined models

TOPS is meant to be extended. You can write your own model classes without
touching the package, put them in a *user model library*, and pass that
library to `PowerSystemModel`:

```python
ps = dps.PowerSystemModel(model=model, user_mdl_lib=my_lib)
```

When TOPS loads `"avr": {"MyAVR": [...]}`, it looks for `my_lib.avr.MyAVR`
first and then for `tops.dyn_models.avr.MyAVR`. A user model can therefore
also replace a built-in model of the same name.

## Example: a simple AVR

This proportional voltage regulator drives a first-order exciter with
limits. It uses two pieces from TOPS:

- **`DAEModel`**, the base class of all dynamic models.
- **`AVR`**, a mixin from `tops.dyn_models.avr` that defines the standard
  connections of an exciter: the terminal voltage, set-point and PSS signal
  come from the generator named in the `gen` column, and the output goes to
  the generator's field voltage `E_f`.

The model itself has three parts. `input_list` names the inputs.
`add_blocks` builds the control diagram from blocks in
`tops.dyn_models.blocks`. `init_from_connections` sets the `bias` input so
that the model starts in steady state.

```python exec="on" source="above" session="user" result="text"
from types import SimpleNamespace

from tops.dyn_models.avr import AVR
from tops.dyn_models.blocks import TimeConstantLims
from tops.dyn_models.utils import DAEModel


class SimpleAVR(DAEModel, AVR):
    """Proportional AVR with a limited first-order exciter."""

    def input_list(self):
        return ["v_setp", "v_t", "v_pss", "bias"]

    def add_blocks(self):
        p = self.par
        self.exciter = TimeConstantLims(T=p["T_e"], V_min=p["E_min"], V_max=p["E_max"])
        self.exciter.input = lambda x, v: p["K"] * (
            self.v_setp(x, v) - self.v_t(x, v) + self.v_pss(x, v) + self.bias(x, v)
        )
        self.output = lambda x, v: self.exciter.output(x, v)

    def init_from_connections(self, x0, v0, output_0):
        # Pick the bias that makes the exciter output equal the initial field voltage
        self._input_values["bias"] = self.exciter.initialize(x0, v0, output_0["output"]) / self.par["K"]


# A user library only needs one attribute per category
my_lib = SimpleNamespace(avr=SimpleNamespace(SimpleAVR=SimpleAVR))
```

Use it in a model like any built-in type. The columns of the table become
`self.par`:

```python exec="on" source="above" session="user" result="text"
import numpy as np
import tops.dynamic as dps
import tops.ps_models.sm_ib as model_data
from tops.simulator import Simulator, ResultKeeper

model = model_data.load()
model["avr"] = {"SimpleAVR": [
    ["name", "gen", "K", "T_e", "E_min", "E_max"],
    ["AVR1", "G1",  50,  0.5,   -5,      5],
]}

ps = dps.PowerSystemModel(model=model, user_mdl_lib=my_lib)
ps.init_dyn_sim()

# In steady state all derivatives are (close to) zero
print(f"Largest initial derivative: {abs(ps.ode_fun(0, ps.x0)).max():.1e}")
```

Simulate a short fault and see the exciter hit its ceiling:

```python exec="on" source="above" session="user" html="1"
--8<-- "plot.py"
import matplotlib.pyplot as plt

events = [
    ["t",  "category", "model", "unit", "action"],
    [1.0,  "buses",    "",      "B1",   "short_circuit"],
    [1.05, "buses",    "",      "B1",   "clear_short_circuit"],
]
sim = Simulator(ps, dt=1e-2, t_end=4, events=events)
res = ResultKeeper(sim, v_t=ps.gen.GEN.v_t_abs, e_f=ps.avr.SimpleAVR.output)
sim.interface_functions["results"] = res.update
sim.main_loop()

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 4))
ax1.plot(res.t, np.array(res.store["v_t"])[:, 0])
ax1.set_ylabel("Terminal voltage [pu]")
ax2.plot(res.t, np.array(res.store["e_f"])[:, 0])
ax2.set_ylabel("Field voltage [pu]")
ax2.set_xlabel("Time [s]")
plt.show()
```

## Organising a library as a package

For real projects, put the models in a package with one module per
category, as in `examples/user_models/user_lib/`:

```text
user_lib/
├── __init__.py      # from . import avr, pll, vsc
├── avr.py           # class SEXS_mod(DAEModel, AVR): ...
├── pll.py
└── vsc.py
```

```python
import user_lib
ps = dps.PowerSystemModel(model=model, user_mdl_lib=user_lib)
```

## Where to look next

- `tops/dyn_models/avr.py`, `gov.py`, `pss.py`: complete controllers built
  from blocks.
- `tops/dyn_models/blocks.py`: the blocks themselves, each a small
  `DAEModel` with an `initialize` method.
- `tops/dyn_models/pll.py` and `vsc.py`: models attached to a bus instead of
  a generator (`bus_ref_spec`). The VSC also injects current into the network
  (`current_injections`).
