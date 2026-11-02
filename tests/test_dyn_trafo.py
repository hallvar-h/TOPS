"""DynTrafo: a transformer whose ratios are inputs."""
import numpy as np

import tops.dynamic as dps


def k2a_with(kind):
    import tops.ps_models.k2a as model_data
    model = model_data.load()
    header, *rows = model.pop('transformers')
    if kind == 'DynTrafo':
        header, rows = [*header, 'ratio_from', 'ratio_to'], [[*row, 1, 1] for row in rows]
    model['trafos'] = {kind: [header, *rows]}
    return model


def test_dyn_trafo_model_initializes_to_a_steady_state():
    ps = dps.PowerSystemModel(model=k2a_with('DynTrafo'))
    ps.init_dyn_sim()
    assert max(abs(ps.ode_fun(0, ps.x0))) < 1e-8


def test_dyn_trafo_at_unit_ratios_gives_the_network_of_a_trafo():
    trafo, dyn_trafo = (dps.PowerSystemModel(model=k2a_with(kind)) for kind in ('Trafo', 'DynTrafo'))
    trafo.init_dyn_sim()
    dyn_trafo.init_dyn_sim()
    assert np.allclose(trafo.v_0, dyn_trafo.v_0)
    assert np.allclose(trafo.solve_algebraic(0, trafo.x0), dyn_trafo.solve_algebraic(0, dyn_trafo.x0))
