# Dynamic models

Every unit in a TOPS model is an instance of a model class in
`tops.dyn_models`. The category key in the model dictionary (`avr`, `gov`,
...) selects the module, and the model type selects the class. For example,
`"avr": {"SEXS": [...]}` uses `tops.dyn_models.avr.SEXS`.

The parameters of a model are the columns of its table. The easiest way to
see which columns a model needs is to look at a bundled system or an
example that uses it, listed in the last column below.

## Catalogue

```python exec="on"
import importlib
import inspect

from tops.dyn_models.utils import DAEModel

# (category, class) -> (description, where to find a working parameter table)
catalogue = {
    ("gen", "GEN"): ("Synchronous machine, 6th-order (sub-transient) model.", "all bundled systems"),
    ("avr", "SEXS"): ("Simplified excitation system: lead-lag, gain and limited time constant.", "`k2a`, `ieee39`"),
    ("avr", "SEXS_PI"): ("SEXS with an added PI controller.", "`tests/test_models_init.py`"),
    ("avr", "IEEET1"): ("IEEE type 1 exciter. Saturation is not included yet.", "`tests/test_models_init.py`"),
    ("avr", "SCRX"): ("Bus- or solid-fed SCR exciter. Negative field current logic is missing.", "`tests/test_models_init.py`"),
    ("gov", "TGOV1"): ("Steam turbine governor.", "`k2a`, `ieee39`"),
    ("gov", "HYGOV"): ("Hydro turbine governor, with optional backlash. Some limiters are missing.", "`examples/dyn_models/hygov_backlash.py`"),
    ("gov", "IEESGO"): ("IEEE standard steam turbine-governor model.", "`tests/test_validation/plots_k2a_ieesgo.py`"),
    ("pss", "STAB1"): ("Speed-input power system stabiliser: washout and two lead-lags.", "`k2a`, `ieee39`"),
    ("loads", "Load"): ("Constant impedance load (the default for `loads`).", "`k2a`"),
    ("loads", "ConstPowerLoad"): ("Constant power load, solved in the algebraic equations.", "`examples/dyn_models/const_power_loads.py`"),
    ("loads", "DynamicLoad"): ("Load with conductance and susceptance as inputs (`g_setp`, `b_setp`).", "`examples/dyn_models/dyn_loads.py`"),
    ("loads", "DynamicLoadFiltered"): ("`DynamicLoad` with low-pass filtered inputs.", "`examples/sde_solver/sde_solver_1.py`"),
    ("lines", "Line"): ("Pi-model line (the default for `lines`). Supports `connect`/`disconnect` events.", "all bundled systems"),
    ("trafos", "Trafo"): ("Two-winding transformer (the default for `transformers`).", "`k2a`"),
    ("trafos", "DynTrafo"): ("Transformer with ratio inputs, e.g. for tap changers.", "`examples/dyn_models/tap_changer_trafo.py`"),
    ("shunts", "Shunt"): ("Fixed shunt (the default for `shunts`).", "`k2a`"),
    ("vsc", "VSC"): ("Grid-following voltage source converter with PLL and P/Q PI control.", "`examples/dyn_models/vsc.py`"),
    ("pll", "PLL1"): ("Phase-locked loop with filtered angle measurement.", "`examples/dyn_models/pll.py`"),
    ("pll", "PLL2"): ("PLL with a PI controller.", "`examples/dyn_models/pll.py`"),
}

# Every model class in the library must be in the catalogue, and the other
# way round, so this page fails to build when a model is added or renamed.
found = set()
for category in {category for category, _ in catalogue}:
    module = importlib.import_module(f"tops.dyn_models.{category}")
    for name, cls in inspect.getmembers(module, inspect.isclass):
        if cls.__module__ == module.__name__ and issubclass(cls, DAEModel):
            found.add((category, name))
missing, stale = found - set(catalogue), set(catalogue) - found
assert not missing and not stale, f"Update the catalogue. Missing: {missing}, stale: {stale}"

print("| Category | Model | Description | Example parameters |")
print("|---|---|---|---|")
for (category, name), (description, example) in catalogue.items():
    print(f"| `{category}` | `{name}` | {description} | {example} |")
```

The `vsc` and `pll` categories have no bundled system. Add them to a model
yourself, as in the example scripts.

## Inputs and outputs

Models are connected through **inputs** and **outputs**. Each controller
names its generator in the `gen` column. When the model is initialised,
TOPS connects, for example, the generator's terminal voltage to the AVR's
`v_t` input and the AVR's output to the generator's field voltage `E_f`.

Any method of a model with the signature `f(x, v)` returns one value per
unit and can be recorded with `ResultKeeper`. Useful ones:

| Model | Outputs |
|---|---|
| `GEN` | `speed`, `angle`, `p_e`, `q_e`, `v_t_abs`, `e_q_t`, `e_d_t`, `i` |
| `Line` | `p_from`, `q_from`, `p_to`, `q_to`, `i_from`, `i_to` |
| `Load` | `p`, `q` |

Inputs that aren't connected to another model, such as the `bias` of an
AVR or the `g_setp` of a `DynamicLoad`, can be changed during a simulation
with `set_input`. See
[Changing inputs](simulation.md#changing-inputs-during-the-simulation).

## Building blocks

Controllers are made of reusable blocks in `tops.dyn_models.blocks`, such as
`Gain`, `TimeConstant`, `TimeConstantLims`, `LeadLag`, `Washout`,
`PIRegulator`, `Integrator`, `Saturation` and `Backlash`. Reading `SEXS` in
`tops/dyn_models/avr.py`, about 20 lines, is the quickest way to see how
they fit together, and the starting point for writing your own model.
