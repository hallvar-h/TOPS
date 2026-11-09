"""A load of zero power initializes to a zero admittance, not NaN."""
import numpy as np
import pytest

import tops.dynamic as dps
import tops.solvers as dps_sol
import tops.ps_models.k2a as model_data

EXTRA_PARAMETERS = {'DynamicLoadFiltered': {'T_g': 1, 'T_b': 1}}


def k2a_with_zero_load(load_model):
    """k2a with its loads as load_model, and an extra load of zero power at bus B7."""
    model = model_data.load()
    header, *rows = model['loads']
    rows = rows + [['L0', 'B7', 0, 0, 'Z']]
    extra = EXTRA_PARAMETERS.get(load_model, {})
    model['loads'] = {load_model: [header + list(extra), *[row + list(extra.values()) for row in rows]]}
    return model


@pytest.mark.parametrize('load_model', ['Load', 'DynamicLoad', 'DynamicLoadFiltered', 'ConstPowerLoad'])
def test_zero_power_load_initializes_and_simulates(load_model):
    ps = dps.PowerSystemModel(model=k2a_with_zero_load(load_model))
    ps.init_dyn_sim()
    assert max(abs(ps.ode_fun(0, ps.x0))) < 1e-10

    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, ps.x0, 0.1, 5e-3, n_dim=ps.n_states)
    while sol.t < 0.1:
        sol.step()
    assert np.isfinite(sol.y).all() and np.isfinite(sol.v).all()

    load = ps.loads[load_model]
    if hasattr(load, 'p'):
        p = load.p(sol.y, sol.v)
        assert np.isfinite(p).all() and p[-1] == 0
