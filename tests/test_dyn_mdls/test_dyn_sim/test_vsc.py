import numpy as np
import tops.dynamic as dps
import tops.solvers as dps_sol


def test_vsc_power_control():
    import tops.ps_models.k2a as model_data

    model = model_data.load()
    model["vsc"] = {
        "VSC": [
            ["name", "T_pll", "T_i", "bus", "P_K_p", "P_K_i", "Q_K_p", "Q_K_i", "P_setp", "Q_setp"],
            ["VSC1", 0.1, 1, "B8", 0.1, 0.1, 0.1, 0.1, 100, 100],
        ]
    }

    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()
    vsc = ps.vsc["VSC"]

    t_end = 5
    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, ps.x_0, t_end, v0=ps.v_0, max_step=5e-3)

    t = 0
    while t < t_end:
        if t >= 1:
            vsc.set_input("P_setp", 500)
        sol.step()
        x = sol.y
        v = sol.v
        t = sol.t

    # Power injected into the grid, from the terminal voltage and the current of the converter.
    # The grid frequency drifts in this case (k2a has no governors), so the PLL angle lags the
    # bus voltage angle; the power must still be controlled to the setpoints.
    bus = vsc.bus_idx_red["terminal"]
    s_inj = np.sqrt(3) * v[bus] * vsc.sys_par["bus_v_n"][bus] * np.conj(vsc.I_inj(x, v))
    assert abs(s_inj.real[0] - 500) < 1
    assert abs(s_inj.imag[0] - 100) < 1
