import tops.dynamic as dps


def vsc_model():
    import tops.ps_models.k2a as model_data

    model = model_data.load()
    model["vsc"] = {
        "VSC": [
            ["name", "T_pll", "T_i", "bus", "P_K_p", "P_K_i", "Q_K_p", "Q_K_i", "P_setp", "Q_setp"],
            ["VSC1", 0.1, 1, "B8", 0.1, 0.1, 0.1, 0.1, 100, 100],
        ]
    }
    return model


def test_vsc_init():
    ps = dps.PowerSystemModel(model=vsc_model())
    ps.init_dyn_sim()
    assert max(abs(ps.ode_fun(0, ps.x_0))) < 1e-10

    vsc = ps.vsc["VSC"]
    assert abs(vsc.P(ps.x_0, ps.v_0)[0] - 100) < 1e-6
    assert abs(vsc.Q(ps.x_0, ps.v_0)[0] - 100) < 1e-6
