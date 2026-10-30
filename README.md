# TOPS (**T**iny **O**pen **P**ower System **S**imulator)

[![Docs](https://github.com/hallvar-h/TOPS/actions/workflows/docs.yml/badge.svg)](https://github.com/hallvar-h/TOPS/actions/workflows/docs.yml)
[![Tests](https://github.com/hallvar-h/TOPS/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/hallvar-h/TOPS/actions/workflows/python-app.yml)

**Note**: This repository was previously called DynPSSimPy.


This is a package for performing dynamic power system simulations in Python. The aim is to provide a simple and lightweight tool which is easy to install, run and modify, to be used by researchers and in education. Performance is not the main priority. The only dependencies are numpy, scipy, pandas and matplotlib (the core functionality only uses numpy and scipy).

The package is being developed as part of ongoing research, and thus contains experimental features. Use at your own risk!

Some features:
- Newton-Rhapson power flow
- Dynamic time domain simulation (RMS/phasor approximation)
- Linearization, eigenvalue analysis/modal analysis

# Installation
The package can be installed using pip, as follows:

`pip install tops`

# Events
Events change the system during a simulation. Apply one at once with `ps.apply_event(category, model, unit, action, **kwargs)`, or give a table of them to the `Simulator`:

```python
events = [
    ['t', 'category', 'model', 'unit', 'action', 'kwargs'],
    [1.0, 'lines', 'Line', 'L7-8-1', 'disconnect', None],
    [1.2, 'lines', 'Line', 'L7-8-1', 'connect', None],
    [2.0, 'loads', 'Load', 'L1', 'set_power', {'P': 900.0}],
]
sim = Simulator(ps, dt=5e-3, t_end=5, events=events)
```

The `kwargs` column, the arguments of the action, is optional. The events are:
- Lines and transformers (`Line`, `Trafo`, `DynTrafo`): `connect` and `disconnect`.
- Loads (`Load`, `ConstPowerLoad`): `set_power(P=None, Q=None)`, in MW and Mvar.
- Buses: `short_circuit` and `clear_short_circuit`.

A line or transformer table may have a boolean `connected` column, to start a branch out of service; it can be connected later. A bus that has lost all its admittance, such as a bay node whose switches are all open, is solved as 0 V and gives a `DeadBusWarning` (silence it with `warnings.filterwarnings`, or set `ps.pin_dead_buses = False`). See `examples/events/`.

# Real-time simulation
`RealTimeSimulator` runs a simulation on the clock, without any user interface, in frames of a few steps:

```python
from tops.realtime import RealTimeSimulator

sim = RealTimeSimulator(ps, dt=5e-3, rate=50)  # 50 frames per second, of 4 steps: real time
sim.start()                                    # runs in a thread
sim.submit('lines', 'Line', 'L7-8-1', 'disconnect')  # an event, applied just before the next frame
sim.pause()
sim.resume()
sim.set_speed(2)                               # twice as fast as real time
sim.stop()
sim.join()
```

The frames are on a schedule that does not drift. A simulation that falls behind catches up, but only up to `max_lag` frame periods: after a longer stall the time lost is dropped instead of being replayed in a burst. A state that is not finite, or a bus voltage above `max_voltage`, ends it with an `ArithmeticError`, kept in `sim.error` when it runs in a thread. `on_frame` is called with the simulator after each frame.

# Contributing and releases
Pull request titles must follow the [Conventional Commits](https://www.conventionalcommits.org/) format. Use `fix: ...` for bug fixes, `feat: ...` for features, and a `!` such as `feat!: ...` for breaking changes. Pull requests should be squash-merged using the pull request title as the commit message.

[Release Please](https://github.com/googleapis/release-please) maintains a release pull request on `main`. That pull request updates `CHANGELOG.md` and the package version in `pyproject.toml`. Because Release Please uses the repository `GITHUB_TOKEN`, run the **Tests** workflow manually against the generated release branch before merging it. Merging the release pull request creates a version tag and GitHub release, builds and validates the distributions, and starts the protected `pypi` environment deployment.

# Development
The project uses [uv](https://docs.astral.sh/uv/). To set up an environment and run the tests:

```
uv sync
uv run pytest            # fast tests
uv run pytest -m slow    # full-length validation against PowerFactory
uv run pytest -m ""      # everything (as in CI)
```

# Citing
If you use this code for your research, please cite [this paper](https://ieeexplore.ieee.org/document/9494770).

# Example notebooks
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/hallvar-h/TOPS/HEAD?filepath=examples%2Fnotebooks)

# Contact
[Hallvar Haugdal](mailto:hallvhau@gmail.com)
