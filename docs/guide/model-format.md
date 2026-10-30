# Model format

A TOPS model is a Python dictionary. A few keys hold system data. All the
other keys hold **tables**: a list whose first row is the header (the
parameter names) and whose following rows are one unit each. This keeps a
whole power system readable in one file.

## A complete example

This is the bundled `sm_ib` system: one generator connected through a line
to an infinite bus. The infinite bus is modelled as a second generator, `IB`,
with a very large inertia.

```python
model = {
    "base_mva": 900,      # system base power [MVA]
    "f": 50,              # nominal frequency [Hz]
    "slack_bus": "B2",

    "buses": [
        ["name", "V_n"],  # V_n: nominal voltage [kV]
        ["B1",   20],
        ["B2",   20],
    ],

    "lines": [
        ["name", "from_bus", "to_bus", "length", "S_n", "V_n", "unit", "R",  "X",  "B"],
        ["L1-2", "B1",       "B2",     25,       900,   20,    "PF",   1e-4, 1e-3, 0],
    ],

    "generators": {
        "GEN": [
            ["name", "bus", "S_n",  "V_n", "P",  "V", "H",    "D", "X_d", "X_q", "X_d_t", "X_q_t", "X_d_st", "X_q_st", "T_d0_t", "T_q0_t", "T_d0_st", "T_q0_st"],
            ["G1",   "B1",  900,    20,    600,  1,   6.5,    0,   1.8,   1.7,   0.3,     0.3,     0.2,      0.2,      8.0,      0.6,      0.05,      0.05],
            ["IB",   "B2",  100000, 20,    -600, 1,   999999, 0,   1.8,   1.8,   0.3,     0.3,     0.2,      0.2,      6.67,     6.67,     0.15,      0.15],
        ],
    },

    "gov": {
        "TGOV1": [
            ["name", "gen", "R",  "D_t", "V_min", "V_max", "T_1", "T_2", "T_3"],
            ["GOV1", "G1",  0.05, 0.02,  0,       1,       0.5,   2,     2],
        ],
    },

    "avr": {
        "SEXS": [
            ["name", "gen", "K", "T_a", "T_b", "T_e", "E_min", "E_max"],
            ["AVR1", "G1",  100, 2.0,   10.0,  0.1,   -10,     10],
        ],
    },
}
```

## Top-level keys

| Key | Content |
|---|---|
| `base_mva` | System base power in MVA. Per-unit values without their own base use it. |
| `f` | Nominal frequency in Hz. |
| `slack_bus` | Name of the slack bus for the power flow. Optional: if left out, the bus of the first generator is used. |
| `buses` | Table with `name` and `V_n` (nominal voltage, kV). Required. |
| `lines`, `transformers`, `loads`, `shunts` | One table each. Each table uses one standard model (`Line`, `Trafo`, `Load`, `Shunt`). |
| `generators`, `avr`, `gov`, `pss`, ... | `{model_type: table}` dictionaries, because there are several model types to choose from. |

`transformers` and `generators` are aliases. After loading, the categories
are called `trafos` and `gen`, which is why you access the generators as
`ps.gen`.

## Two ways to write a category

**A single table** works for `lines`, `transformers`, `loads` and `shunts`,
and uses the standard model:

```python
"loads": [
    ["name", "bus", "P",  "Q"],
    ["L1",   "B7",  967,  100],
],
```

**A `{model_type: table}` dictionary** selects the model class by name. It is
required for generators and controllers, and it lets you mix model types in
one category:

```python
"loads": {
    "Load":           [["name", "bus", "P", "Q"], ["L1", "B7", 967, 100]],
    "ConstPowerLoad": [["name", "bus", "P", "Q"], ["L2", "B9", 500,  50]],
},
```

The model type is looked up as a class in `tops.dyn_models.<category>`, for
example `tops.dyn_models.avr.SEXS`. If you pass a user model library
(`PowerSystemModel(model, user_mdl_lib=...)`), it is searched first.

## Units and conventions

- **Powers** (`P`, `Q`, `S_n`) are in MW, Mvar and MVA. Generator `P` is the
  power-flow set-point, and `V` the voltage set-point in pu.
- **Generator parameters** (`X_d`, `X_d_t`, ...) are in pu on the machine
  rating (`S_n`, `V_n`). Time constants are in seconds, and `H` in seconds
  on `S_n`. If `S_n` or `V_n` is 0, the system base or the bus voltage is
  used.
- **Controllers** refer to their generator by name in the `gen` column. A
  generator without a row in `avr`, `gov` or `pss` has constant field
  voltage, mechanical power or no stabiliser.
- **Lines** take the impedance unit from the `unit` column:

    | `unit` | `R`, `X` | `B` |
    |---|---|---|
    | `p.u.` (or no column) | pu per km. If `S_n` and `V_n` are given, they are the base; otherwise the system base is used. | pu per km |
    | `PF` | Ω per km | µS per km (as in PowerFactory) |
    | `Ohm` | Ω per km | S per km |

    The values are multiplied by `length` (1 if the column is missing).
- **Loads** with the standard `Load` model become constant impedances in the
  dynamic simulation, computed from `P`, `Q` and the power-flow voltage. Use
  `ConstPowerLoad` for constant power.

## Using pandas DataFrames

Any table can also be a `pandas.DataFrame` with one column per parameter.
This is handy when the data comes from a spreadsheet or a CSV file:

```python exec="on" source="above" result="text"
import numpy as np
import pandas as pd
import tops.dynamic as dps
import tops.ps_models.sm_ib as model_data

model = model_data.load()
model["buses"] = pd.DataFrame({"name": ["B1", "B2"], "V_n": [20, 20]})

ps = dps.PowerSystemModel(model=model)
ps.power_flow()
print("Bus voltages [pu]:", abs(ps.v0).round(4))
print("Bus angles [deg]:", np.angle(ps.v0, deg=True).round(2))
```

## Editing a bundled model

`load()` returns a new dictionary on every call, so you can change it freely.
Tables are plain lists: find the column by name in the header row.

```python exec="on" source="above" result="text"
import tops.ps_models.k2a as model_data

model = model_data.load()
gen_table = model["generators"]["GEN"]
h = gen_table[0].index("H")

for row in gen_table[1:]:
    row[h] *= 0.5        # halve the inertia of every generator

print([row[h] for row in gen_table[1:]])
```

## Validation

`PowerSystemModel` raises an error for input it can't use, such as a model
file that doesn't exist or events placed inside the model data. A model type
that isn't found gives a warning, and that table is skipped, so check the
warnings if a unit seems to be missing.
