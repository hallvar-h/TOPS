import numpy as np
import tops.dynamic as dps


def k2a_with_hygov(extra_par=None):
    """k2a with a HYGOV on G1 (and no other governors)."""
    import tops.ps_models.k2a as model_data

    par = {
        'name': 'HYGOV1', 'gen': 'G1', 'R': 0.04, 'r': 0.1, 'T_f': 0.1, 'T_r': 10, 'T_g': 0.5,
        'A_t': 1, 'T_w': 1, 'q_nl': 0.01, 'D_turb': 0.01, 'V_elm': 0.15, 'P_N': 0,
        **(extra_par or {}),
    }
    model = model_data.load()
    model['gov'] = {'HYGOV': [list(par.keys()), list(par.values())]}
    return model


def test_hygov_init():
    ps = dps.PowerSystemModel(model=k2a_with_hygov())
    ps.init_dyn_sim()
    assert max(abs(ps.ode_fun(0, ps.x0))) < 1e-8


def test_hygov_gate_limits():
    # Defaults when g_min/g_max are not given
    ps = dps.PowerSystemModel(model=k2a_with_hygov())
    pi_reg = ps.gov['HYGOV'].pi_reg
    assert np.all(pi_reg.par['x_min'] == 0)
    assert np.all(pi_reg.par['x_max'] == 1)

    # g_min/g_max from the model dict are used
    ps = dps.PowerSystemModel(model=k2a_with_hygov({'g_min': 0.05, 'g_max': 0.95}))
    pi_reg = ps.gov['HYGOV'].pi_reg
    assert np.all(pi_reg.par['x_min'] == 0.05)
    assert np.all(pi_reg.par['x_max'] == 0.95)


if __name__ == '__main__':
    test_hygov_init()
    test_hygov_gate_limits()
