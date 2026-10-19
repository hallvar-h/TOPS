"""Compare TOPS simulations with PowerFactory results.

Short simulations run by default; the full-length ones are marked slow (run with `pytest -m slow`).
"""
import pathlib
from collections import defaultdict

import pytest
from scipy.integrate import RK45

import tops.dynamic as dps
try:
    from .validation_functions import load_pf_res, compute_error
except ImportError:
    from validation_functions import load_pf_res, compute_error

DATA_DIR = pathlib.Path(__file__).parent
MAX_STEP = 5e-3
MAX_ERROR = 0.02


def simulate(ps, t_end, event):
    """Integrate with RK45, calling event(ps, t) after each step, and return states per time step."""
    sol = RK45(ps.ode_fun, 0, ps.x0, t_end, max_step=MAX_STEP)
    result = defaultdict(list)
    t = 0
    while t < t_end:
        sol.step()
        t = sol.t
        event(ps, t)
        result['Global', 't'].append(t)
        for desc, state in zip(ps.state_desc, sol.y):
            result[tuple(desc)].append(state)
    return result


def short_circuit(t_start, t_stop):
    """Short circuit at the terminal of the first generator."""
    def event(ps, t):
        sc_bus_idx = ps.gen['GEN'].bus_idx_red['terminal'][0]
        ps.y_bus_red_mod[(sc_bus_idx,) * 2] = 1e6 if t_start <= t <= t_stop else 0
    return event


def load_step(ps, t):
    """Add 10 % of the first load's conductance after t = 1 s."""
    load = ps.loads['Load']
    load_bus_idx = load.bus_idx_red['terminal'][0]
    if t > 1:
        ps.y_bus_red_mod[(load_bus_idx,) * 2] = load.y_load.real[0] * 0.1


def k2a_with_gov(gov):
    """K2A validation model where the governor on G1 is replaced by `gov`."""
    import tops.ps_models.k2a_val as model_data
    model = model_data.load()
    model['gov'] = {
        'TGOV1': [model['gov']['TGOV1'][0], *model['gov']['TGOV1'][2:]],
        **gov,
    }
    return model


def k2a():
    import tops.ps_models.k2a_val as model_data
    return model_data.load()


def ieee39():
    import tops.ps_models.ieee39 as model_data
    return model_data.load()


def k2a_hygov():
    return k2a_with_gov({'HYGOV': [
        ['name',    'gen',  'R',    'r',    'T_f',  'T_r',  'T_g',  'A_t',  'T_w',  'q_nl', 'D_turb', 'g_min', 'V_elm', 'g_max', 'P_N'],
        ['HYGOV1',  'G1',   0.04,   0.1,    0.1,    10,     0.5,    1,      1,      0.01,   0.01,     0,       0.15,    1,       0],
    ]})


def k2a_ieesgo():
    return k2a_with_gov({'IEESGO': [
        ['name',     'gen', 'T_1', 'T_2', 'K_1', 'T_3', 'T_4', 'K_3', 'T_6', 'K_2', 'T_5', 'P_N', 'P_min', 'P_max'],
        ['IEESGO-1', 'G1',  0.3,   1.,    10.,   1.,    0.1,   0.4,   0.2,   0.4,   1.,    0.,    0.,      1.],
    ]})


# (case, model, PowerFactory result folder, event, short t_end, full t_end)
CASES = [
    ('k2a', k2a, 'k2a', short_circuit(1, 1.1), 2, 10),
    ('ieee39', ieee39, 'ieee39', short_circuit(1, 1.05), 2, 10),
    ('k2a_hygov', k2a_hygov, 'k2a_hygov', load_step, 5, 30),
    ('k2a_ieesgo', k2a_ieesgo, 'k2a_ieesgo', load_step, 5, 30),
]

PARAMS = [
    param
    for case, model, folder, event, t_short, t_full in CASES
    for param in [
        pytest.param(model, folder, event, t_short, id=f'{case}-{t_short}s'),
        pytest.param(model, folder, event, t_full, id=f'{case}-{t_full}s', marks=pytest.mark.slow),
    ]
]


@pytest.mark.parametrize('model, folder, event, t_end', PARAMS)
def test_against_powerfactory(model, folder, event, t_end):
    ps = dps.PowerSystemModel(model=model())
    ps.init_dyn_sim()

    pf_res = load_pf_res(str(DATA_DIR / folder / 'powerfactory_res.csv'))
    result = simulate(ps, t_end, event)

    assert compute_error(ps, result, pf_res, MAX_STEP) <= MAX_ERROR
