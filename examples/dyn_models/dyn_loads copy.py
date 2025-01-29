import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import tops.dynamic as dps
import tops.solvers as dps_sol
import importlib
importlib.reload(dps)
import numpy as np

if __name__ == '__main__':

    # Load model
    import tops.ps_models.k2a as model_data
    importlib.reload(model_data)
    model = model_data.load()
    model['loads'] = {'DynamicLoad': model['loads']}

    # Power system model
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()
    print(max(abs(ps.state_derivatives(0, ps.x_0, ps.v_0))))

    #########################
    from scipy.sparse import linalg as sp_linalg
    from scipy.sparse import diags as sp_diags
    self = ps
    v_red = self.v0.copy()
    self.s_inj = np.zeros_like(v_red, dtype=complex)
    self.s_inj[0] = 0.1
    y_bus = self.y_bus_red
    x = self.x0

    tol = 1e-10
    max_it = 10
    error = 10 * tol
    it = 0

    i_inj = np.zeros(self.n_bus_red, dtype=complex)
    for mdl in self.mdl_instructions['current_injections']:
        bus_idx_red, i_inj_mdl = mdl.current_injections(x, None)
        np.add.at(i_inj, bus_idx_red, i_inj_mdl)


    # t_tot = time.time()
    # t_spsolve_cum0 = 0
    while error > tol and it < max_it:
        s_v2_diag = np.conj(sp_diags(self.s_inj / v_red ** 2))
        A = y_bus + s_v2_diag
        b = np.conj(self.s_inj / v_red) + i_inj + s_v2_diag * v_red
        v_red = sp_linalg.spsolve(A, b)
        error = np.linalg.norm(y_bus.dot(v_red) - np.conj(self.s_inj / v_red) - i_inj)
        it += 1
        print(it)

    if error > tol:
        print('Warning: Solution of algebraic equations did not converge.')

    # return v_red
    ##########################
if False:
    t_end = 20
    x_0 = ps.x_0.copy()

    # Solver
    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x_0, t_end, max_step=5e-3)

    # Initialize simulation
    t = 0
    res = defaultdict(list)
    t_0 = time.time()

    sc_bus_idx = ps.gen['GEN'].bus_idx_red['terminal'][0]

    # Run simulation
    while t < t_end:
        sys.stdout.write("\r%d%%" % (t/(t_end)*100))

        if 1 <= t:
            ps.loads['DynamicLoad'].set_input('g_setp', 1.3, 0)
        if 10 <= t:
            ps.loads['DynamicLoad'].set_input('b_setp', -0.2, 1)

        # Simulate next step
        result = sol.step()
        x = sol.y
        v = sol.v
        t = sol.t

        dx = ps.ode_fun(0, ps.x_0)

        # Store result
        res['t'].append(t)
        res['gen_speed'].append(ps.gen['GEN'].speed(x, v).copy())
        res['v'].append(v.copy())
        res['load_P'].append(ps.loads['DynamicLoad'].P(x, v).copy())
        res['load_Q'].append(ps.loads['DynamicLoad'].Q(x, v).copy())

    print('Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))

    fig = plt.figure()
    plt.plot(res['t'], np.abs(res['v']))
    plt.xlabel('Time [s]')
    plt.ylabel('Bus voltage')

    
    fig = plt.figure()
    plt.plot(res['t'], np.abs(res['load_P']))
    plt.xlabel('Time [s]')
    plt.ylabel('MW')

    fig = plt.figure()
    plt.plot(res['t'], np.abs(res['load_Q']))
    plt.xlabel('Time [s]')
    plt.ylabel('MVA')
    
    plt.show()
    