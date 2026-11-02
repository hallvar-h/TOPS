# Examples

The [`examples/`](https://github.com/hallvar-h/TOPS/tree/main/examples)
folder of the repository has complete scripts for most features. Run them
from a clone, for example `python examples/dyn_sim/line_outage_var.py`.

Many of the older scripts step the solver in their own loop and change the
network directly. They work, but for new code the `Simulator` with an event
table, as in this documentation, is shorter.

## Notebooks

| Notebook | Content |
|---|---|
| [k2a-simple_dyn_sim.ipynb](https://github.com/hallvar-h/TOPS/blob/main/examples/notebooks/k2a-simple_dyn_sim.ipynb) | Dynamic simulation of the two-area system. |
| [modal_analysis.ipynb](https://github.com/hallvar-h/TOPS/blob/main/examples/notebooks/modal_analysis.ipynb) | Linearisation, eigenvalues and mode shapes. |

Both run in the browser on Binder, no installation needed:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/hallvar-h/TOPS/HEAD?filepath=examples%2Fnotebooks)

## Scripts

### Simulation and events

| Script | Shows |
|---|---|
| [`dyn_sim/line_outage_var.py`](https://github.com/hallvar-h/TOPS/blob/main/examples/dyn_sim/line_outage_var.py) | Line disconnection and reconnection through a `Simulator` event table (`ieee39`). |
| [`interfaces/results_events.py`](https://github.com/hallvar-h/TOPS/blob/main/examples/interfaces/results_events.py) | Events plus `ResultKeeper`, and results as a DataFrame (`ieee39`). |
| [`interfaces/sim.py`](https://github.com/hallvar-h/TOPS/blob/main/examples/interfaces/sim.py) | The minimal `Simulator` run. |
| [`dyn_sim/short_circuit.py`](https://github.com/hallvar-h/TOPS/blob/main/examples/dyn_sim/short_circuit.py) | Short circuit at a generator bus, own solver loop. |
| [`dyn_sim/line_outage.py`](https://github.com/hallvar-h/TOPS/blob/main/examples/dyn_sim/line_outage.py) | Disconnecting a line, own solver loop. |
| [`dyn_sim/line_outage_cont.py`](https://github.com/hallvar-h/TOPS/blob/main/examples/dyn_sim/line_outage_cont.py) | Contingency screening: trips each line of `ieee39` in turn. |

### Modal analysis

| Script | Shows |
|---|---|
| [`modal_analysis/plot_eigs.py`](https://github.com/hallvar-h/TOPS/blob/main/examples/modal_analysis/plot_eigs.py) | Eigenvalue plot and mode shapes of `k2a`. |

### Dynamic models

| Script | Shows |
|---|---|
| [`dyn_models/const_power_loads.py`](https://github.com/hallvar-h/TOPS/blob/main/examples/dyn_models/const_power_loads.py) | Mixing `Load` and `ConstPowerLoad`. |
| [`dyn_models/dyn_loads.py`](https://github.com/hallvar-h/TOPS/blob/main/examples/dyn_models/dyn_loads.py) | `DynamicLoad` with conductance and susceptance as inputs. |
| [`dyn_models/tap_changer_trafo.py`](https://github.com/hallvar-h/TOPS/blob/main/examples/dyn_models/tap_changer_trafo.py) | `DynTrafo` with changing ratio, like a tap changer. |
| [`dyn_models/trafos.py`](https://github.com/hallvar-h/TOPS/blob/main/examples/dyn_models/trafos.py) | Transformer currents during a short circuit. |
| [`dyn_models/hygov_backlash.py`](https://github.com/hallvar-h/TOPS/blob/main/examples/dyn_models/hygov_backlash.py) | `HYGOV` hydro governor with backlash. |
| [`dyn_models/vsc.py`](https://github.com/hallvar-h/TOPS/blob/main/examples/dyn_models/vsc.py) | Adding a voltage source converter to `k2a`. |
| [`dyn_models/pll.py`](https://github.com/hallvar-h/TOPS/blob/main/examples/dyn_models/pll.py) | `PLL1` and `PLL2` frequency estimates during a fault. |

### User-defined models

| Script | Shows |
|---|---|
| [`user_models/pll.py`](https://github.com/hallvar-h/TOPS/blob/main/examples/user_models/pll.py), [`user_models/vsc.py`](https://github.com/hallvar-h/TOPS/blob/main/examples/user_models/vsc.py) | Loading models from the `user_models/user_lib/` package. See [User-defined models](user-models.md). |

### Stochastic simulation

| Script | Shows |
|---|---|
| [`sde_solver/sde_solver_1.py`](https://github.com/hallvar-h/TOPS/blob/main/examples/sde_solver/sde_solver_1.py) | Random load variations with `EulerDAE_SDE` (`tops.solvers_sde`). |
| [`sde_solver/sde_solver_2.py`](https://github.com/hallvar-h/TOPS/blob/main/examples/sde_solver/sde_solver_2.py) | Random disturbances acting on generator speeds. |
