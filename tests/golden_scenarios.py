"""Scenarios behind the v0.4.0 golden file (tests/data/v040_k2a.npz).

Shared by make_golden.py, which writes the file, and test_regression_v040.py, which compares against it.

The golden values are what a user sees through the public API: the network matrices, the bus voltages and the
outputs of the generators. They are not the raw state vector, whose order and names are internal to the models
(a refactor may change both without changing a single result).
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


def outputs(ps, x, v):
    """Bus voltages and the speed, angle and electrical power of the generators."""
    gen = ps.gen['GEN']
    return {'v': np.array(v), 'speed': gen.speed(x, v).copy(), 'angle': gen.angle(x, v).copy(),
            'p_e': gen.P_e(x, v).copy()}


def run(ps, events=None, t_end=1.0):
    """The outputs of a simulation at SAMPLE_TIMES, one array per output."""
    sim = Simulator(ps, dt=DT, t_end=t_end, events=events)
    samples = {}

    def sample(sim):
        for t in SAMPLE_TIMES:
            if abs(sim.sol.t - t) < DT/2:
                samples[t] = outputs(ps, sim.sol.x, sim.sol.v)

    sim.interface_functions['sample'] = sample
    sim.main_loop()
    return {key: np.array([samples[t][key] for t in SAMPLE_TIMES]) for key in samples[SAMPLE_TIMES[0]]}


def prefixed(prefix, arrays):
    return {f'{prefix}_{key}': value for key, value in arrays.items()}


def snapshot():
    ps = k2a_ps()
    return {
        'y_bus_lf': ps.y_bus_lf,
        'y_bus_red': ps.y_bus_red.toarray(),
        'v_0': ps.v_0,
        **prefixed('init', outputs(ps, ps.x0, ps.v_0)),
        **prefixed('free', run(ps)),
        **prefixed('trip', run(k2a_ps(), events=LINE_TRIP)),
    }
