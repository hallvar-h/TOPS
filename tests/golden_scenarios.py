"""Scenarios behind the v0.4.0 golden file (tests/data/v040_k2a.npz).

Shared by make_golden.py, which writes the file, and test_regression_v040.py, which compares against it.
"""
import numpy as np

import tops.dynamic as dps
from tops.simulator import Simulator

EVENT_COLUMNS = ['t', 'category', 'model', 'unit', 'action']
LINE_TRIP = [EVENT_COLUMNS,
             [0.1, 'lines', 'Line', 'L7-8-1', 'disconnect'],
             [0.6, 'lines', 'Line', 'L7-8-1', 'connect']]
DT = 5e-3
SAMPLE_TIMES = (0.25, 0.5, 1.0)


def k2a_ps():
    import tops.ps_models.k2a as model_data
    ps = dps.PowerSystemModel(model=model_data.load())
    ps.init_dyn_sim()
    return ps


def run(ps, events=None, t_end=1.0):
    """States of a simulation at SAMPLE_TIMES."""
    sim = Simulator(ps, dt=DT, t_end=t_end, events=events)
    samples = {}

    def sample(sim):
        for t in SAMPLE_TIMES:
            if abs(sim.sol.t - t) < DT/2:
                samples[t] = sim.sol.x.copy()

    sim.interface_functions['sample'] = sample
    sim.main_loop()
    return np.array([samples[t] for t in SAMPLE_TIMES])


def snapshot():
    ps = k2a_ps()
    return {
        'y_bus_lf': ps.y_bus_lf,
        'y_bus_red': ps.y_bus_red.toarray(),
        'v_0': ps.v_0,
        'x_0': ps.x0,
        'x_free': run(ps),
        'x_line_trip': run(k2a_ps(), events=LINE_TRIP),
    }
