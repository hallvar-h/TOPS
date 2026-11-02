# Included test systems

TOPS ships with a few standard test systems in `tops.ps_models`. Each is a
module with a `load()` function that returns the model dictionary, so it
takes two lines to start from one:

```python
import tops.ps_models.ieee39 as model_data
model = model_data.load()
```

The table is generated from the bundled models when this page is built, so
the counts are always up to date.

```python exec="on"
import importlib

import tops.ps_models

systems = {
    "sm_ib": "One synchronous machine against an infinite bus. The smallest system to experiment with.",
    "sm_ib_kundur": "Machine against infinite bus with a 2200 MVA, 60 Hz machine from Kundur's book.",
    "sm_load": "One synchronous machine feeding a passive load. No controllers.",
    "k2a": "Kundur's two-area system, the classic case for inter-area oscillations. Used in the Quickstart.",
    "k2a_val": "`k2a` with the controller settings used to validate TOPS against PowerFactory.",
    "ieee39": "IEEE 39-bus (New England) system, based on the PowerFactory version. G1 has no controllers.",
    "ieee39_all_ctrl": "`ieee39` with AVR, governor and PSS on all ten generators.",
    "n44": "Nordic 44-bus system, a reduced model of the Nordic synchronous grid.",
    "ieee68": "IEEE 68-bus (New England / New York) system. No controllers.",
}


def count(table):
    """Number of units in a table (rows after the header) or in a {model: table} dict."""
    if isinstance(table, dict):
        return sum(len(t) - 1 for t in table.values())
    return len(table) - 1


def controllers(model):
    names = []
    for key in ("avr", "gov", "pss"):
        names += [f"{name} ({len(t) - 1})" for name, t in model.get(key, {}).items()]
    return ", ".join(names) or "none"


print("| Module | Buses | Lines | Generators | Controllers | f [Hz] | Description |")
print("|---|---:|---:|---:|---|---:|---|")
for name, description in systems.items():
    model = importlib.import_module(f"tops.ps_models.{name}").load()
    print(
        f"| `{name}` | {count(model['buses'])} | {count(model.get('lines', []))} "
        f"| {count(model.get('generators', {}))} | {controllers(model)} "
        f"| {model['f']} | {description} |"
    )
```

In `sm_ib` and `sm_ib_kundur` the infinite bus is a second generator, `IB`,
with a very large inertia, which is why they count two generators.

Controller names are the model types (`SEXS` exciter, `TGOV1` governor,
`STAB1` stabiliser) with the number of units in brackets.

!!! warning "ieee68"

    `ieee68` does not initialise at the moment: the dynamic initialisation
    gives NaN because of a singular matrix. The test suite marks it as a known
    failure. Use the other systems until it is fixed.

## Models from JSON files

A model can also be stored as JSON with the same structure as the
dictionary. Pass the path to `PowerSystemModel`. Names of bundled files are
found without a path, so this loads the JSON copy of `k2a`:

```python
import tops.dynamic as dps
ps = dps.PowerSystemModel(model="k2a.json")
```

## Starting your own

The easiest way to build a new system is to copy the module closest to what
you need, for example `sm_ib.py` for a single machine or `k2a.py` for a
multi-machine system, and edit the tables. The [Model format](../guide/model-format.md) page explains
the tables.
