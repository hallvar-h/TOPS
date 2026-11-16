"""The factorisation of the admittance matrix is reused by solve_algebraic, and must follow every change to it."""
import numpy as np
import scipy.sparse.linalg as sp_linalg

import tops.dynamic as dps


def initialized_k2a():
    import tops.ps_models.k2a as model_data
    ps = dps.PowerSystemModel(model=model_data.load())
    ps.init_dyn_sim()
    return ps


def direct_solution(ps, x):
    """Bus voltages solved without the factorisation."""
    i_inj = np.zeros(ps.n_bus_red, dtype=complex)
    for mdl in ps.mdl_instructions['current_injections']:
        bus_idx_red, i_inj_mdl = mdl.current_injections(x, None)
        np.add.at(i_inj, bus_idx_red, i_inj_mdl)
    return sp_linalg.spsolve(ps.y_bus_red + ps.y_bus_red_mod, i_inj)


def test_factorisation_is_reused():
    ps = initialized_k2a()
    v = ps.solve_algebraic(0, ps.x0)
    lu = ps._y_bus_lu_cache[1]
    assert lu is not None
    ps.solve_algebraic(0, ps.x0)
    assert ps._y_bus_lu_cache[1] is lu
    assert np.allclose(v, direct_solution(ps, ps.x0))


def test_in_place_change_of_y_bus_red_mod_is_followed():
    # As in examples/dyn_sim/short_circuit.py
    ps = initialized_k2a()
    ps.solve_algebraic(0, ps.x0)
    bus = ps.gen['GEN'].bus_idx_red['terminal'][0]

    ps.y_bus_red_mod[(bus,) * 2] = 1e6
    v_fault = ps.solve_algebraic(0, ps.x0)
    assert abs(v_fault[bus]) < 1e-3
    assert np.allclose(v_fault, direct_solution(ps, ps.x0))

    ps.y_bus_red_mod[(bus,) * 2] = 0
    assert np.allclose(ps.solve_algebraic(0, ps.x0), direct_solution(ps, ps.x0))
    assert np.allclose(ps.solve_algebraic(0, ps.x0), ps.v0)


def test_events_are_followed():
    ps = initialized_k2a()
    v_0 = ps.solve_algebraic(0, ps.x0)

    ps.apply_event(ps.lines['Line'].events.disconnect, 'L7-8-1')
    v_open = ps.solve_algebraic(0, ps.x0)
    assert not np.allclose(v_open, v_0)
    assert np.allclose(v_open, direct_solution(ps, ps.x0))

    ps.apply_event(ps.lines['Line'].events.connect, 'L7-8-1')
    assert np.allclose(ps.solve_algebraic(0, ps.x0), v_0)
