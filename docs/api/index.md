# API reference

These pages are generated from the source code with
[mkdocstrings](https://mkdocstrings.github.io/). Large parts of TOPS have no
docstrings yet, so many entries show only the signature and the source.
Docstrings in the reST style (`:param name: ...`) are rendered as they are
added.

| Module | Content |
|---|---|
| [`tops.dynamic`](dynamic.md) | `PowerSystemModel` |
| [`tops.simulator`](simulator.md) | `Simulator`, `EventSchedule`, `ResultKeeper`, `InterfacerDirect` |
| [`tops.events`](events.md) | `ModelEvents`, `BusEvents`, `AdmittanceChange` |
| [`tops.model_data`, `tops.power_flow`](model-data.md) | Loading model data, power flow |
| [`tops.solvers`, `tops.solvers_sde`](solvers.md) | Time-step solvers |
| [`tops.modal_analysis`](modal-analysis.md) | `PowerSystemModelLinearization` |
| [`tops.plotting`](plotting.md) | `plot_eigs`, `plot_mode_shape` |
| [`tops.dyn_models.*`](dyn-models.md) | Generators, controllers, loads, lines, transformers, converters |
| [`tops.dyn_models.blocks`](blocks.md) | Control blocks |

For an overview of the models with descriptions, see the
[Dynamic models](../guide/dyn-models.md) guide.
