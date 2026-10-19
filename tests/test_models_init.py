"""Check that power system models initialize to a steady state (all state derivatives ~ 0)."""
import importlib
import json

import numpy as np
import pytest

import tops.dynamic as dps

TOL = 1e-8


def max_derivative(ps):
    ps.init_dyn_sim()
    return max(abs(ps.ode_fun(0, ps.x0)))


@pytest.mark.parametrize('name', [
    'sm_ib',
    'sm_ib_kundur',
    'sm_load',
    'k2a',
    'k2a_val',
    'ieee39',
    'ieee39_all_ctrl',
    'n44',
    pytest.param('ieee68', marks=pytest.mark.xfail(reason='Initializes to NaN (singular matrix); pre-existing issue')),
])
def test_bundled_model_initializes(name):
    model = importlib.import_module(f'tops.ps_models.{name}').load()
    ps = dps.PowerSystemModel(model=model)
    assert max_derivative(ps) < TOL


def k2a_avr(mdl_name, header, row):
    """One AVR row per generator of K2A, with the given parameters."""
    rows = [[f'AVR{i}', f'G{i}', *row(i)] for i in range(1, 5)]
    return {mdl_name: [['name', 'gen', *header], *rows]}


AVRS = {
    'SEXS': k2a_avr(
        'SEXS', ['K', 'T_a', 'T_b', 'T_e', 'E_min', 'E_max'],
        lambda i: [100, 2.0, 10.0, 0.5, -3, 3]),
    'SEXS_PI': k2a_avr(
        'SEXS_PI', ['K_p', 'K_i', 'T_a', 'T_b', 'K', 'T_e', 'E_min', 'E_max', 'T_ext'],
        lambda i: [1.0, 0.5, 2.0, 10.0, 100, 0.5, -3, 3, 0.1]),
    'IEEET1': k2a_avr(
        'IEEET1', ['T_r', 'K_a', 'T_a', 'V_rmax', 'V_rmin', 'K_e', 'T_e', 'E_1', 'E_2', 'S_e1', 'S_e2', 'K_f', 'T_f'],
        lambda i: [0.02, 200, 0.05, 10, -10, 1.0, 0.5, 3.0, 4.0, 0.1, 0.3, 0.05, 1.0]),
    # Mix of bus-fed (C_switch = 0) and separately fed (C_switch = 1) units
    'SCRX': k2a_avr(
        'SCRX', ['T_a', 'T_b', 'K', 'T_e', 'E_min', 'E_max', 'C_switch'],
        lambda i: [2.0, 10.0, 100, 0.5, -3, 3, i % 2]),
}


@pytest.mark.parametrize('avr', AVRS.values(), ids=AVRS.keys())
def test_avr_initializes(avr):
    import tops.ps_models.k2a as model_data
    model = model_data.load()
    model['avr'] = avr
    ps = dps.PowerSystemModel(model=model)
    assert max_derivative(ps) < TOL


def test_load_bundled_json_by_name():
    ps = dps.PowerSystemModel(model='k2a.json')
    assert len(ps.gen['GEN'].par) == 4
    assert max_derivative(ps) < TOL


def test_json_file_matches_dict(tmp_path):
    import tops.ps_models.k2a as model_data
    model = model_data.load()
    path = tmp_path / 'model.json'
    path.write_text(json.dumps(model))

    ps_json = dps.PowerSystemModel(model=str(path))
    ps_dict = dps.PowerSystemModel(model=model_data.load())
    ps_json.init_dyn_sim()
    ps_dict.init_dyn_sim()

    np.testing.assert_allclose(ps_json.x0, ps_dict.x0)
