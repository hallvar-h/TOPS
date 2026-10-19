import numpy as np
import tops.dynamic as dps
import tops.solvers as dps_sol


def test_dyn_load():
    # Load model
    import tops.ps_models.k2a as model_data
    model = model_data.load()
    model['loads'] = {'DynamicLoad': model['loads']}

    # Power system model
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    t_end = 5

    # Solver
    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, ps.x_0, ps.v_0, t_end, max_step=5e-3)

    # Run simulation
    t = 0
    while t < t_end:
        # Change load setpoints
        if t >= 1:
            ps.loads['DynamicLoad'].set_input('g_setp', [1, 2])
            ps.loads['DynamicLoad'].set_input('b_setp', [-0.1, -0.1])

        # Simulate next step
        sol.step()
        x = sol.y
        v = sol.v
        t = sol.t

    assert np.linalg.norm(ps.loads['DynamicLoad'].y_load(x, v).real - [1, 2]) < 1e-2
    assert np.linalg.norm(ps.loads['DynamicLoad'].y_load(x, v).imag - [-0.1, -0.1]) < 1e-2
