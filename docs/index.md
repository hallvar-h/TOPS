# TOPS

**T**iny **O**pen **P**ower System **S**imulator: dynamic power system
simulation in plain Python, built to be read, run and modified.

```python exec="on" source="above" result="text"
import tops.dynamic as dps
import tops.ps_models.k2a as model_data

ps = dps.PowerSystemModel(model=model_data.load())
ps.init_dyn_sim()          # power flow + initialisation of all dynamic models

print(f"{ps.n_bus} buses, {len(ps.gen.GEN.par)} generators, {ps.n_states} states")
```

## What it does

- **Power flow**: Newton-Raphson.
- **Dynamic simulation**: time-domain RMS (phasor) simulation with
  generators, exciters, governors, stabilisers, loads, converters and PLLs.
- **Events**: short circuits and line switching, from a simple table.
- **Modal analysis**: numerical linearisation, eigenvalues and mode shapes.
- **Extensible**: write your own models from reusable control blocks.

The only dependencies are numpy, scipy, pandas and matplotlib.

## Get started

<div class="grid cards" markdown>

-   :material-download: **[Install](start/install.md)**

    `pip install tops`, or from source.

-   :material-play-circle: **[Quickstart](start/quickstart.md)**

    A fault simulation in the two-area system, in four steps.

-   :material-table: **[Model format](guide/model-format.md)**

    How a power system is described as tables.

-   :material-sine-wave: **[Modal analysis](guide/modal-analysis.md)**

    Find oscillation modes and their damping.

</div>

## Citing

If you use TOPS in your research, please cite
[this paper](https://ieeexplore.ieee.org/document/9494770). See
[About](about.md) for the BibTeX entry.
